"""SIR-Nano training loop.

    python -m training.train --config configs/sir_nano_smoke.yaml

Non-negotiables baked into this file (each exists because its absence would make a reported number
meaningless):

  * one seed drives init, data order, and evaluation order; `--seed` overrides the config for the
    reproducibility harness;
  * the tokenizer artifact is verified against the tokenized data's recorded sha256 before the
    first step — a silent tokenizer/data mismatch is unrecoverable after training;
  * validation is computed on a FIXED window list in a fixed order, so `val_loss` from two runs is
    a comparison of models, not of sampling luck;
  * loss is logged with the token count it was measured over; perplexity is exp(mean CE) computed
    from that same sum, never averaged across batch means;
  * the run writes train_log.json containing config, git commit, environment, parameter count,
    measured tokens/sec, peak RSS, and a `provisional` flag inherited from the dataset report;
  * checkpoints are written to `runs/` and are git-ignored; nothing here ever uploads weights.

Mixed precision is enabled only when CUDA is actually present: running fp16 on CPU changes numerics
without saving time, which would poison the reproducibility criterion.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.config import SirModelConfig  # noqa: E402
from model.transformer import SirNano  # noqa: E402
from sir_paths import load_config, rel, resolve  # noqa: E402
from training import checkpoint as ckpt  # noqa: E402
from training.dataset import WindowDataset, collate, load_tokenized  # noqa: E402


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
def set_seed(seed: int, threads: int | None = None) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    info: dict[str, Any] = {"seed": seed}
    if threads:
        torch.set_num_threads(int(threads))
        info["threads"] = int(threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        info["deterministic_algorithms"] = True
    except Exception as e:  # some ops have no deterministic kernel; report it, never pretend
        info["deterministic_algorithms"] = f"unavailable: {type(e).__name__}: {e}"
    info["benchmark"] = bool(torch.backends.cudnn.benchmark) if torch.cuda.is_available() else False
    return info


def make_lr_lambda(warmup: int, total_steps: int, min_lr_ratio: float):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_lr_ratio + (1 - min_lr_ratio) * cos

    return fn


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: SirNano,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
    seq_len: int,
) -> dict[str, Any]:
    model.eval()
    total_tokens = 0
    nll_sum = 0.0
    t0 = time.perf_counter()
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(amp, device):
            out = model(x, targets=y)
        loss = out["loss"].float()
        ntok = int((y != model.cfg.pad_token_id).sum().item())
        nll_sum += float(loss.item()) * ntok
        total_tokens += ntok
        seen += 1
        if max_batches and seen >= max_batches:
            break
    dt = max(1e-9, time.perf_counter() - t0)
    if total_tokens == 0:
        raise RuntimeError("evaluation covered zero non-pad tokens: the val split or padding is wrong")
    mean_nll = nll_sum / total_tokens
    return {
        "batches": seen,
        "tokens_scored": total_tokens,
        "val_loss": round(mean_nll, 6),
        "val_perplexity": round(math.exp(min(80.0, mean_nll)), 4),
        "eval_seconds": round(dt, 3),
        "eval_tokens_per_sec": round(total_tokens / dt, 1),
        "seq_len": seq_len,
    }


def autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
@dataclass
class RunArtifacts:
    run_dir: Path
    log_path: Path
    final_ckpt: Path
    best_ckpt: Path


def prepare(cfg: dict[str, Any], cli: argparse.Namespace) -> tuple[SirNano, dict[str, Any], dict[str, Any], dict[str, Any]]:
    mcfg = SirModelConfig.from_dict(cfg.get("model") or {})
    tcfg = cfg.get("tokenizer") or {}
    if tcfg.get("chosen_vocab_size") and int(tcfg["chosen_vocab_size"]) != mcfg.vocab_size:
        print(
            f"note: model vocab_size={mcfg.vocab_size} but the bake-off chose {tcfg['chosen_vocab_size']} "
            "(model config wins; retokenize if that was not intended)",
            file=sys.stderr,
        )

    processed = resolve((cfg.get("paths") or {}).get("processed_dir", "data/processed/latest"))
    tokdir = processed / "tokenized"
    tok_spec_path: Path | None = None
    chosen = resolve((cfg.get("paths") or {}).get("tokenizer_dir", "tokenizer/artifacts/latest")) / "chosen.json"
    if chosen.exists():
        tok_spec_path = resolve(json.loads(chosen.read_text(encoding="utf-8"))["spec"])
    if cli.tokenizer_spec:
        tok_spec_path = resolve(cli.tokenizer_spec)

    data_dir = tokdir / ("val" if cli.eval_only else "train")
    if not (data_dir / "meta.json").exists() and not cli.eval_only:
        raise SystemExit(
            f"no tokenized train split at {rel(data_dir)} — run: python -m training.tokenize_corpus --config <cfg>"
        )

    # tokenizer artifact hash must match what the token stream was produced with
    train_meta = json.loads((tokdir / "train" / "meta.json").read_text(encoding="utf-8")) if (tokdir / "train" / "meta.json").exists() else {}
    if tok_spec_path and train_meta:
        spec = json.loads(tok_spec_path.read_text(encoding="utf-8"))
        now = ckpt.sha256_file(tok_spec_path.parent / Path(spec["path"]).name)
        if train_meta.get("tokenizer_sha256") not in (None, "missing") and now != train_meta["tokenizer_sha256"]:
            raise SystemExit(
                "tokenizer artifact changed after tokenization; retokenize before training "
                f"(recorded {str(train_meta.get('tokenizer_sha256'))[:12]}…, current {now[:12]}…)"
            )
        mcfg.vocab_size = int(spec.get("vocab_size") or mcfg.vocab_size)

    if tok_spec_path:
        from tokenizer.api import SirTokenizer

        tok_obj = SirTokenizer.load(tok_spec_path)
        if int(tok_obj.vocab_size) != int(mcfg.vocab_size):
            print(
                f"note: model vocab_size {mcfg.vocab_size} -> {tok_obj.vocab_size} (tokenizer's actual "
                "vocabulary). The embedding table must cover every token id the data contains.",
                file=sys.stderr,
            )
            mcfg.vocab_size = int(tok_obj.vocab_size)

    model = SirNano(mcfg)
    param_report = model.num_parameters()
    target_report = mcfg.check_target(param_report["total"])
    param_report["estimate"] = mcfg.estimate_vs_actual(param_report["total"])
    return model, mcfg.to_dict(), {"params": param_report, "target_check": target_report}, {
        "tokenizer_spec": str(tok_spec_path) if tok_spec_path else None,
        "train_meta": train_meta,
        "tokdir": str(tokdir),
    }


def build_loader(
    tokens_dir: Path,
    seq_len: int,
    batch_size: int,
    tokenizer_sha: str | None,
    shuffle: bool,
    seed: int,
    max_windows: int | None,
    stride: int | None = None,
) -> tuple[DataLoader, Any]:
    tokens, meta, ranges = load_tokenized(tokens_dir, expect_tokenizer_sha=tokenizer_sha)
    ds = WindowDataset(
        tokens,
        seq_len=seq_len,
        ranges=ranges,  # windows stay inside documents; see training/dataset.py
        stride=stride,
        shuffle=shuffle,
        seed=seed,
        max_windows=max_windows,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0,  # worker processes would consume RNG state in a machine-dependent order
        drop_last=True,
        pin_memory=False,
    )
    return loader, (ds, meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train SIR-Nano from a config file.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--seed", type=int, help="override config seed (used by the reproducibility harness)")
    ap.add_argument("--max-steps", type=int, help="override config max_steps")
    ap.add_argument("--run-dir", help="override output dir; default paths.run_dir/<experiment>")
    ap.add_argument("--resume", default=None, const="auto", nargs="?", help="auto|<path>: continue from a checkpoint")
    ap.add_argument("--eval-only", action="store_true", help="build the model, evaluate once, write no checkpoint")
    ap.add_argument("--device", help="cpu|cuda|auto")
    ap.add_argument("--tokenizer-spec", help="explicit tokenizer spec.json")
    ap.add_argument("--tag", default="", help="suffix for the run dir name")
    args = ap.parse_args(argv)

    cfg_path = resolve(args.config)
    cfg = load_config(cfg_path)
    exp = (cfg.get("experiment") or {}).get("name", "unnamed")
    tcfg = cfg.get("training") or {}
    paths = cfg.get("paths") or {}
    runtime = cfg.get("runtime") or {}
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))

    run_dir = resolve(args.run_dir) if args.run_dir else resolve(paths.get("run_dir", f"runs/{exp}"))
    if args.tag:
        run_dir = run_dir.parent / f"{run_dir.name}-{args.tag.strip('-')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_info = set_seed(seed, runtime.get("threads"))

    dev_pref = (args.device or runtime.get("device") or "auto").lower()
    device = torch.device("cuda" if (dev_pref == "auto" and torch.cuda.is_available()) else ("cpu" if dev_pref == "auto" else dev_pref))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("config asks for cuda but cuda is unavailable — refusing to fall back silently")
    amp = bool(runtime.get("amp", False)) and device.type == "cuda"
    if bool(runtime.get("amp", False)) and device.type != "cuda":
        print("note: amp requested but device is CPU -> disabled (CPU autocast would change numerics for no gain)")
        amp = False
    if bool(runtime.get("compile", False)):
        print("note: runtime.compile=true ignored in v0.1 (torch.compile changes numerics/first-run cost; not enabled)")

    model, model_cfg, counts, tokinfo = prepare(cfg, args)
    model.to(device)

    processed = resolve(paths.get("processed_dir", "data/processed/latest"))
    tokdir = Path(tokinfo["tokdir"])
    seq_len = int(tcfg.get("seq_len") or model_cfg["max_seq_len"])
    if seq_len > int(model_cfg["max_seq_len"]):
        raise SystemExit(f"training seq_len {seq_len} exceeds model context {model_cfg['max_seq_len']}")
    batch_size = int(tcfg.get("batch_size", 16))
    accum = max(1, int(tcfg.get("accumulation_steps", 1)))
    tok_sha = (tokinfo.get("train_meta") or {}).get("tokenizer_sha256")

    eval_batches = int(tcfg.get("eval_batches", 0)) or None
    window_stride = int(tcfg.get("window_stride", 0)) or None
    train_loader, (train_ds, train_meta) = build_loader(
        tokdir / "train", seq_len, batch_size, tok_sha, shuffle=True, seed=seed, max_windows=None, stride=window_stride
    )
    val_loader, (val_ds, val_meta) = build_loader(
        tokdir / "val", seq_len, batch_size, tok_sha, shuffle=False, seed=seed, max_windows=None, stride=None
    )

    if len(train_ds) == 0:
        raise SystemExit("train dataset yielded zero windows: the tokenized corpus is shorter than one sequence")
    for name, d in (("train", train_ds), ("val", val_ds)):
        w = d.starvation_warning()
        if w:
            print(f"WARNING[{name}]: {w}", file=sys.stderr)

    # fixed validation order: materialise once so every eval in a run (and between runs) scores the same windows
    val_windows = list(val_ds)
    if eval_batches:
        val_windows = val_windows[: eval_batches]
    val_loader_fixed = DataLoader(val_windows, batch_size=batch_size, collate_fn=collate, num_workers=0, drop_last=True)

    batches_per_epoch = max(1, len(train_ds) // (batch_size * accum))
    if args.max_steps is not None:
        steps = int(args.max_steps)
        step_source = "--max-steps override"
    elif tcfg.get("max_steps"):
        steps = int(tcfg["max_steps"])
        step_source = "config training.max_steps"
    elif tcfg.get("epochs"):
        steps = max(1, int(tcfg["epochs"]) * batches_per_epoch)
        step_source = f"config training.epochs x {batches_per_epoch} batches/epoch"
    else:
        steps = 100
        step_source = "default 100 (neither max_steps nor epochs set)"
    lr = float(tcfg.get("lr", 3e-4))
    min_lr = float(tcfg.get("min_lr", lr / 10))
    betas = tuple(tcfg.get("betas") or (0.9, 0.95))
    decay = float(tcfg.get("weight_decay", 0.0))
    clip = float(tcfg.get("grad_clip", 1.0))

    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay_params if (p.ndim < 2 or "norm" in name) else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": decay}, {"params": nodecay_params, "weight_decay": 0.0}],
        lr=lr,
        betas=betas,  # type: ignore[arg-type]
        eps=1e-8,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(int(tcfg.get("warmup_steps", 0)), steps, min_lr / max(1e-12, lr)))

    start_step = 0
    resume_info: dict[str, Any] = {"resumed": False}
    if args.resume:
        target = run_dir / "latest.pt" if args.resume == "auto" else resolve(args.resume)
        if Path(target).exists():
            loaded, state = ckpt.load(target, map_location=device.type)
            if loaded is not None:
                model.load_state_dict(state["model_state"])
                start_step = int(state.get("step", 0))
                if "optimizer_state" in state:
                    optimizer.load_state_dict(state["optimizer_state"])
                resume_info = {"resumed": True, "from": str(target), "at_step": start_step, "ckpt_val_loss": state.get("validation_loss")}
                print(f"resumed from {rel(Path(target))} at step {start_step}")
        elif args.resume != "auto":
            raise SystemExit(f"--resume {target} does not exist")

    ds_report = processed / "dataset_report.json"
    provisional = bool(json.loads(ds_report.read_text(encoding="utf-8")).get("provisional", False)) if ds_report.exists() else None

    log: list[dict[str, Any]] = []
    header = {
        "tool": "training/train.py",
        "experiment": exp,
        "config": rel(cfg_path),
        "config_sha256": ckpt.sha256_file(cfg_path)[:16],
        "model_config": model_cfg,
        "counts": counts,
        "seed": seed,
        "seed_info": seed_info,
        "device": str(device),
        "amp": amp,
        "optimizer": {"type": "AdamW", "lr": lr, "min_lr": min_lr, "betas": list(betas), "weight_decay": decay, "grad_clip": clip},
        "schedule": {"warmup": int(tcfg.get("warmup_steps", 0)), "total_steps": steps, "type": tcfg.get("schedule", "cosine")},
        "batch": {"per_step": batch_size, "accumulation": accum, "effective_sequences": batch_size * accum, "seq_len": seq_len, "window_stride": window_stride or seq_len + 1},
        "data": {
            "train_windows": len(train_ds),
            "train_docs_total": int(train_ds.docs_total),
            "train_docs_too_short": int(train_ds.docs_too_short),
            "train_docs_usable": int(train_ds.docs_total) - int(train_ds.docs_too_short),
            "train_docs_too_short": int(train_ds.docs_too_short),
            "val_windows": len(val_ds),
            "train_tokens": int(train_meta.n_tokens),
            "val_tokens": int(val_meta.n_tokens),
            "language_token_counts": train_meta.language_counts,
            "train_docs": int(train_meta.n_docs),
            "provisional": provisional,
        },
        "tokenizer": tokinfo.get("tokenizer_spec"),
        "git": ckpt.git_info(),
        "environment": ckpt.environment_info(),
        "resume": resume_info,
        "step_source": step_source,
        "batches_per_epoch": batches_per_epoch,
        "status_note": (
            "pipeline-validation run: small model on a small fixture corpus. Loss curves here prove the "
            "machinery works; they say nothing about SIR's capability."
        ),
    }
    print(f"run: {exp} -> {rel(run_dir)} | device={device} | amp={amp} | steps from {step_source}")
    print(f"model: {SirModelConfig.from_dict(model_cfg).summary()} | params={counts['params']['total']:,}")
    print(f"tokens: train={header['data']['train_tokens']:,} val={header['data']['val_tokens']:,} | windows={len(train_ds)} | steps={steps}")

    best = {"val_loss": float("inf"), "step": -1}
    t_start = time.perf_counter()
    tokens_seen = 0
    step = start_step
    iterator: Iterator = iter(train_loader)
    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro = 0
        for _ in range(accum):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(amp, device):
                out = model(x, targets=y)
            loss = out["loss"] / accum
            loss.backward()
            accum_loss += float(out["loss"].detach().float().item())
            tokens_seen += int(x.numel())
            micro += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)) if clip else None
        optimizer.step()
        sched.step()
        step += 1
        if step % int(tcfg.get("log_every", 25)) == 0 or step == start_step + 1:
            dt = time.perf_counter() - t_start
            entry = {
                "step": step,
                "train_loss": round(accum_loss / max(1, micro), 5),
                "lr": round(sched.get_last_lr()[0], 8),
                "tokens_seen": tokens_seen,
                "tokens_per_sec": round(tokens_seen / max(1e-9, dt), 1),
                "grad_norm": None if grad_norm is None or not math.isfinite(grad_norm) else round(grad_norm, 3),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            }
            log.append(entry)
            print(f"  step {step:5d} | loss {entry['train_loss']:.4f} | lr {entry['lr']:.2e} | {entry['tokens_per_sec']:,.0f} tok/s | rss {entry['peak_rss_mb']:.0f} MB")
        if int(tcfg.get("eval_every", 0)) and (step % int(tcfg["eval_every"]) == 0 or step == steps):
            ev = evaluate_model(model, val_loader_fixed, device, amp, eval_batches, seq_len)
            entry = {"step": step, **ev}
            log.append(entry)
            print(f"  step {step:5d} | VAL loss {ev['val_loss']:.4f} | ppl {ev['val_perplexity']:.2f} | {ev['tokens_scored']} tokens")
            if ev["val_loss"] < best["val_loss"]:
                best = {"val_loss": ev["val_loss"], "step": step, "val_perplexity": ev["val_perplexity"]}
                info = ckpt.save(
                    run_dir / "best.pt", model, model_cfg=model_cfg, step=step, seed=seed, val_loss=ev["val_loss"],
                    run_config=cfg, tokenizer_spec=tokinfo.get("tokenizer_spec"), optimizer=optimizer, scheduler=sched,
                    extra={"counts": counts, "provisional": provisional, "eval": ev},
                )
                log.append({"step": step, "checkpoint": info})
            if step < steps:
                ckpt.save(
                    run_dir / f"step-{step}.pt", model, model_cfg=model_cfg, step=step, seed=seed, val_loss=ev["val_loss"],
                    run_config=cfg, tokenizer_spec=tokinfo.get("tokenizer_spec"), optimizer=optimizer, scheduler=sched,
                )
                ckpt.rotate(run_dir, int((cfg.get("checkpoint") or {}).get("keep_last", 3)))

    final_eval = evaluate_model(model, val_loader_fixed, device, amp, eval_batches, seq_len)
    saved = ckpt.save(
        run_dir / "final.pt", model, model_cfg=model_cfg, step=step, seed=seed, val_loss=final_eval["val_loss"],
        run_config=cfg, tokenizer_spec=tokinfo.get("tokenizer_spec"), optimizer=optimizer, scheduler=sched,
        extra={"counts": counts, "provisional": provisional, "eval": final_eval, "best": best},
    )
    if (run_dir / "final.pt").exists():
        (run_dir / "latest.pt").write_bytes((run_dir / "final.pt").read_bytes())
    out = {
        **header,
        "final_eval": final_eval,
        "best": best,
        "wallclock_seconds": round(time.perf_counter() - t_start, 2),
        "tokens_seen": tokens_seen,
        "mean_train_tokens_per_sec": round(tokens_seen / max(1e-9, time.perf_counter() - t_start), 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "checkpoint": saved,
        "log": log,
    }
    (run_dir / "train_log.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"final: val_loss={final_eval['val_loss']:.4f} ppl={final_eval['val_perplexity']:.2f} | best={best}")
    print(f"checkpoint: {rel(Path(saved['path']))} ({saved['bytes'] // 1024} KiB, sha256 {saved['sha256'][:12]}…)")
    print(f"log: {rel(run_dir / 'train_log.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

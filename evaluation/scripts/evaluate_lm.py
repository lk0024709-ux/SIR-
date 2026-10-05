"""Language-model evaluation on held-out documents: loss, perplexity, throughput, memory.

    python -m evaluation.scripts.evaluate_lm --checkpoint runs/sir_nano_smoke/final.pt \
        --config configs/sir_nano_smoke.yaml --out evaluation/results/lm_eval.json

Two things this deliberately does not hide:

  * Per-document perplexity here is NOT identical to the trainer's windowed validation loss: the
    trainer scores fixed non-overlapping windows, this script scores whole documents. Both numbers
    are reported and their agreement is measured (`trainer_agreement`), because a large gap means the
    evaluation harness and the training harness disagree about the data — a real failure mode.
  * Every number is written to JSON by this script. README and docs/model-card.md quote the JSON. No
    human retypes a benchmark figure anywhere in this project.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inference.engine import load_for_inference, perplexity_from_ids  # noqa: E402
from sir_paths import rel, resolve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a SIR-Nano checkpoint on held-out documents.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--val", help="override val.jsonl path (default: paths.processed_dir from config)")
    ap.add_argument("--max-docs-per-language", type=int, default=60)
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "results" / "lm_eval.json"))
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths") or {}
    val_path = Path(args.val) if args.val else resolve(paths.get("processed_dir", "data/processed/latest")) / "val.jsonl"
    if not val_path.exists():
        print(f"val file missing: {rel(val_path)} — run the data pipeline first", file=sys.stderr)
        return 2

    model, tok, state = load_for_inference(args.checkpoint)
    device = next(model.parameters()).device
    ctx = int(model.cfg.max_seq_len)
    docs = [json.loads(l) for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not docs:
        print("val split is empty", file=sys.stderr)
        return 2

    per_lang: dict[str, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    scored = 0
    tokens_scored = 0
    t0 = time.perf_counter()
    counts: dict[str, int] = defaultdict(int)
    for d in docs:
        lang = str(d.get("language", "unknown"))
        if counts[lang] >= args.max_docs_per_language:
            continue
        counts[lang] += 1
        ids = tok.encode(d["text"])
        if len(ids) < 4:
            continue
        r = perplexity_from_ids(model, ids, int(model.cfg.pad_token_id), device=str(device), seq_len=ctx)
        if not math.isfinite(r["nll"]):
            continue
        row = {**r, "language": lang, "topic": str(d.get("topic", "")), "id": str(d.get("id")), "chars": len(d["text"])}
        per_lang[lang].append(row)
        all_rows.append(row)
        scored += 1
        tokens_scored += int(r["tokens"])
    dt = time.perf_counter() - t0

    def agg(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {"docs": 0, "nll": None, "perplexity": None}
        nll = sum(r["nll"] * r["tokens"] for r in rows) / max(1, sum(r["tokens"] for r in rows))
        ppl_vals = [r["perplexity"] for r in rows]
        return {
            "docs": len(rows),
            "tokens": int(sum(r["tokens"] for r in rows)),
            "nll": round(nll, 6),
            "perplexity_token_weighted": round(math.exp(min(80.0, nll)), 4),
            "perplexity_mean_of_docs": round(statistics.mean(ppl_vals), 4),
            "perplexity_median_of_docs": round(statistics.median(ppl_vals), 4),
        }

    overall = agg(all_rows)
    by_lang = {k: agg(v) for k, v in sorted(per_lang.items())}
    trainer_val = state.get("validation_loss")
    agreement = None
    if trainer_val is not None and overall.get("nll") is not None:
        delta = (overall["nll"] - trainer_val) / trainer_val
        agreement = {
            "trainer_val_loss": round(float(trainer_val), 6),
            "harness_val_nll": overall["nll"],
            "relative_delta": round(delta, 5),
            "note": "expected to differ (windowed vs per-document scoring); a |delta| > 0.15 means the harnesses disagree about the data",
            "suspicious": bool(abs(delta) > 0.15),
        }

    params = int(sum(p.numel() for p in model.parameters()))
    payload = {
        "tool": "evaluation/scripts/evaluate_lm.py",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": rel(resolve(args.checkpoint)),
        "config": rel(resolve(args.config)),
        "experiment": (cfg.get("experiment") or {}).get("name"),
        "val_file": rel(val_path),
        "val_docs_present": len(docs),
        "provisional": (state.get("extra") or {}).get("provisional"),
        "model": {
            "parameters": params,
            "parameter_breakdown": (state.get("extra") or {}).get("counts", {}),
            "context_length": ctx,
            "vocab_size": int(model.cfg.vocab_size),
            "step": state.get("step"),
            "seed": state.get("seed"),
            "architecture": model.cfg.summary(),
        },
        "results": {
            **overall,
            "eval_seconds": round(dt, 3),
            "eval_tokens_per_sec": round(tokens_scored / max(1e-9, dt), 1),
            "forward_tokens_per_sec_estimate": round(tokens_scored / max(1e-9, dt), 1),
        },
        "per_language": by_lang,
        "trainer_agreement": agreement,
        "memory": {
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "torch_cpu_alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1) if torch.cuda.is_available() else None,
        },
        "reading": (
            "perplexity here is a pipeline measurement on a small, non-natural held-out set. It is not "
            "a capability result and must not be compared with numbers from other projects."
        ),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"docs scored: {overall['docs']} | tokens: {overall.get('tokens')} | ppl(token-weighted): {overall.get('perplexity_token_weighted')}")
    for k, v in by_lang.items():
        print(f"  {k:<10} docs={v['docs']:<4} ppl={v.get('perplexity_token_weighted')}")
    if agreement:
        print(f"trainer vs harness: {agreement['trainer_val_loss']} vs {agreement['harness_val_nll']} (delta {agreement['relative_delta']:+.2%})")
    print(f"peak RSS: {payload['memory']['peak_rss_mb']} MB | eval {payload['results']['eval_tokens_per_sec']} tok/s")
    print(f"wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

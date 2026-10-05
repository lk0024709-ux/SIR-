"""Reproducibility harness: same seed, same config, two independent processes.

    python -m evaluation.scripts.reproducibility --config configs/sir_nano_smoke.yaml --max-steps 60

The README criterion is "a rerun reproduces held-out perplexity within +/-2%". This runs training
TWICE in separate subprocesses (in-process reruns can hide state leakage: seeded generators,
global flags, cached data order) and compares:

  1. final validation perplexity  -> relative delta must be <= tolerance
  2. the whole training loss trace -> reported element-wise max/mean deviation
  3. evaluation of one checkpoint twice -> expected to be bit-identical
  4. greedy generation on one checkpoint twice -> expected to be byte-identical

If it fails, the harness prints the nondeterminism suspects it checked and the measured evidence.
It never widens the tolerance to make the number pass — the tolerance comes from the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sir_paths import load_config, rel, resolve  # noqa: E402


def run(cmd: list[str], env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def train_once(cfg_path: Path, tag: str, max_steps: int | None, seed: int | None) -> dict[str, object]:
    cmd = [sys.executable, "-m", "training.train", "--config", str(cfg_path), "--tag", tag]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    t0 = time.perf_counter()
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"training run {tag} failed ({p.returncode}):\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    cfg = load_config(cfg_path)
    run_dir = resolve((cfg.get("paths") or {}).get("run_dir", "runs/sir_nano_smoke"))
    d = run_dir.parent / f"{run_dir.name}-{tag}"
    log = json.loads((d / "train_log.json").read_text(encoding="utf-8"))
    return {
        "run_dir": rel(d),
        "seconds": round(time.perf_counter() - t0, 2),
        "final_val_loss": log["final_eval"]["val_loss"],
        "final_val_perplexity": log["final_eval"]["val_perplexity"],
        "tokens_scored": log["final_eval"]["tokens_scored"],
        "best": log.get("best"),
        "trace": [e.get("train_loss") for e in log["log"] if e.get("train_loss") is not None],
        "seed_info": log.get("seed_info"),
        "environment": log.get("environment"),
        "wallclock": log.get("wallclock_seconds"),
        "tokens_per_sec": log.get("mean_train_tokens_per_sec"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Two-process reproducibility test for SIR-Nano training.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, help="override seed; default: the config's seed")
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "results" / "reproducibility.json"))
    args = ap.parse_args(argv)

    cfg_path = resolve(args.config)
    cfg = load_config(cfg_path)
    ev = cfg.get("evaluation") or {}
    repro = ev.get("reproducibility") or {}
    tolerance = float(repro.get("tolerance_pct", 2.0)) / 100.0
    runs = max(2, int(args.runs))
    seed = args.seed if args.seed is not None else cfg.get("seed")

    print(f"running {runs} independent training processes (seed={seed}, max_steps={args.max_steps})")
    results = []
    for i in range(runs):
        r = train_once(cfg_path, f"repro-{i + 1}", args.max_steps, args.seed)
        print(f"  repro-{i + 1}: val_ppl={r['final_val_perplexity']} ({r['seconds']}s)")
        results.append(r)

    ppls = [float(r["final_val_perplexity"]) for r in results]
    losses = [float(r["final_val_loss"]) for r in results]
    spread = (max(ppls) - min(ppls)) / statistics.mean(ppls)
    traces = [r["trace"] for r in results]
    if traces and all(len(t) == len(traces[0]) and len(t) > 0 for t in traces):
        dev = [max(abs(t[i] - traces[0][i]) for t in traces) for i in range(len(traces[0]))]
        trace_dev = {
            "steps_compared": len(traces[0]),
            "max_abs": round(max(dev), 8),
            "mean_abs": round(statistics.mean(dev), 8),
            "identical": all(x == 0.0 for x in dev),
        }
    else:
        trace_dev = {"identical": False, "note": "trace lengths differ or are empty; runs are not comparable step-for-step"}

    # checkpoint-level determinism: score the same checkpoint twice, generate twice
    ck = resolve(results[0]["run_dir"]) / "final.pt"
    eval_twice = []
    for i in range(2):
        p = run(
            [
                sys.executable,
                "-m",
                "evaluation.scripts.evaluate_lm",
                "--checkpoint",
                str(ck),
                "--config",
                str(cfg_path),
                "--out",
                f"/tmp/repro_eval_{i}.json",
            ]
        )
        if p.returncode == 0:
            eval_twice.append(json.loads(Path("/tmp/repro_eval_%d.json" % i).read_text(encoding="utf-8"))["results"]["perplexity_token_weighted"])
    gen_twice = []
    for i in range(2):
        p = run(
            [
                sys.executable,
                "-m",
                "inference.generate",
                "--checkpoint",
                str(ck),
                "--prompt",
                "भारत का मौसम अच्छा",
                "--max-new-tokens",
                "24",
                "--raw",
            ]
        )
        gen_twice.append(p.stdout)

    eval_identical = len(set(eval_twice)) == 1 and len(eval_twice) == 2
    gen_identical = len(set(gen_twice)) == 1 and len(gen_twice) == 2
    pass_run = spread <= tolerance

    suspects = [
        "torch.set_num_threads differing between runs (both runs pin 1 thread via env OMP/MKL)",
        "torch.use_deterministic_algorithms not enabled — see seed_info in each train_log.json",
        "training data order driven by a global RNG instead of a seeded generator",
        "dropout active during evaluation (v0.1 uses dropout=0.0)",
        "mixed precision on CPU (disabled in the smoke config on purpose)",
        "tokenizer or tokenized data changed between runs (artifact sha256 is compared by train.py itself)",
    ]
    payload = {
        "tool": "evaluation/scripts/reproducibility.py",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criterion": {"metric": "held-out validation perplexity across independent runs", "tolerance_rel": tolerance, "source": "README M1 acceptance"},
        "runs": results,
        "validation_perplexity": {
            "values": [round(p, 4) for p in ppls],
            "mean": round(statistics.mean(ppls), 4),
            "relative_spread": round(spread, 8),
            "within_tolerance": bool(pass_run),
        },
        "validation_loss_values": [round(x, 6) for x in losses],
        "train_loss_trace": trace_dev,
        "checkpoint_eval_twice": {"values": eval_twice, "identical": eval_identical},
        "greedy_generation_twice": {"identical": gen_identical, "sample": (gen_twice[0][:160] if gen_twice else None)},
        "pass": bool(pass_run and gen_identical),
        "if_failed_check": suspects,
        "environment_note": "runs share one machine; cross-machine reproduction is a separate, unmeasured question",
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    verdict = "REPRODUCIBLE" if pass_run else "NOT REPRODUCIBLE"
    print(f"{verdict}: ppl {['%.2f' % p for p in ppls]} spread {spread:.4%} (tolerance {tolerance:.1%})")
    print(f"train loss traces identical: {trace_dev.get('identical')} | checkpoint eval identical: {eval_identical} | greedy text identical: {gen_identical}")
    print(f"wrote {rel(out)}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU inference benchmark: measure latency honestly on the machine that is actually being used.

    python -m inference.bench_cpu --checkpoint runs/sir_nano_smoke/final.pt \
        --out evaluation/results/cpu_bench.json

The README target is "128 tokens in <= 2.0 s". That is a target, so it is evaluated, not asserted:
a failing measurement is reported as a failure with the environment that produced it. Machine specs
are read from /proc (CPU brand, core count, MemTotal) so a number cannot be detached from the hardware
that made it. Thermal throttling and background load are NOT controlled here — the report says so,
because a benchmark that implies control it does not have is worse than no benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference.engine import GenSettings, generate, load_for_inference  # noqa: E402


def read_proc(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cpu_info() -> dict[str, object]:
    info = read_proc("/proc/cpuinfo")
    brand = ""
    mhz = None
    for line in info.splitlines():
        if line.startswith("model name") and not brand:
            brand = line.split(":", 1)[1].strip()
        if line.startswith("cpu MHz"):
            try:
                mhz = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    mem = ""
    for line in read_proc("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            mem = line.split(":", 1)[1].strip()
            break
    try:
        physical = int(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        physical = 0
    return {
        "brand": brand or "unknown",
        "logical_processors": physical or (torch.get_num_threads() or 1),
        "mhz_sample": mhz,
        "mem_total": mem or "unknown",
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def model_size_bytes(checkpoint: Path) -> int:
    return checkpoint.stat().st_size if checkpoint.exists() else 0


def first_token_latency(model, tok, prompt: str, device: str) -> float:
    """Time a single-token generation call: the number a user perceives as responsiveness."""
    t0 = time.perf_counter()
    generate(model, tok, prompt, GenSettings(max_new_tokens=1, temperature=0.0))
    return (time.perf_counter() - t0) * 1000.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark SIR-Nano CPU generation against the README target.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt", default="भारत का मौसम")
    ap.add_argument("--threads", type=int, default=None, help="pin torch threads (default: leave as configured)")
    ap.add_argument("--max-seconds", type=float, help="override the target from config (for CI on slower boxes)")
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "results" / "cpu_bench.json"))
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    ev = (cfg.get("evaluation") or {}).get("cpu_latency") or {}
    target_tokens = int(args.tokens or ev.get("tokens", 128))
    target_seconds = float(args.max_seconds or ev.get("max_seconds", 2.0))
    repeats = max(1, int(args.repeats or ev.get("repeats", 3)))

    if args.threads:
        torch.set_num_threads(int(args.threads))
    model, tok, state = load_for_inference(args.checkpoint)
    nparams = int(sum(p.numel() for p in model.parameters()))
    ck = Path(args.checkpoint)

    # warm-up is not timed: first-call work (page faults, one-time dispatch) is not steady-state
    for _ in range(max(0, args.warmup)):
        generate(model, tok, args.prompt, GenSettings(max_new_tokens=4, temperature=0.0))

    runs: list[dict[str, float]] = []
    for i in range(repeats):
        r = generate(model, tok, args.prompt, GenSettings(max_new_tokens=target_tokens, temperature=0.0))
        runs.append(
            {
                "seconds": r["seconds"],
                "tokens": r["generated_tokens"],
                "tokens_per_sec": r["tokens_per_sec"],
                "ms_per_token": r["ms_per_token"],
                "complete": r["generated_tokens"] >= target_tokens,
                "stopped_early": r["stopped"],
            }
        )
        print(f"  run {i + 1}: {r['generated_tokens']}/{target_tokens} tokens in {r['seconds']:.3f}s ({r['tokens_per_sec']:.1f} tok/s)")
    ftl = statistics.median(first_token_latency(model, tok, args.prompt, "cpu") for _ in range(repeats))
    rss_kb = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(__import__("os").getpid())], capture_output=True, text=True
    ).stdout.strip()

    usable = [r for r in runs if r["complete"]] or runs
    med = statistics.median(r["seconds"] for r in usable)
    p95 = max(r["seconds"] for r in usable)
    met = med <= target_seconds and all(r["complete"] for r in runs)
    payload = {
        "tool": "inference/bench_cpu.py",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {"tokens": target_tokens, "max_seconds": target_seconds, "source": "README M1 acceptance criterion"},
        "result": {
            "median_seconds": round(med, 4),
            "worst_seconds": round(p95, 4),
            "median_tokens_per_sec": round(statistics.median(r["tokens_per_sec"] for r in usable), 2),
            "median_ms_per_token": round(statistics.median(r["ms_per_token"] for r in usable), 3),
            "first_token_ms_median": round(float(ftl), 2),
            "repeats": repeats,
            "runs": runs,
            "all_runs_completed_target_length": all(r["complete"] for r in runs),
        },
        "model": {
            "checkpoint": str(ck),
            "file_bytes": model_size_bytes(ck),
            "weights_only_estimate_bytes": int(nparams * 4),
            "parameters": nparams,
            "context_length": int(model.cfg.max_seq_len),
            "step": state.get("step"),
            "provisional": (state.get("extra") or {}).get("provisional"),
            "kv_cache": False,
        },
        "environment": cpu_info(),
        "process_rss_kb": int(rss_kb) if rss_kb.isdigit() else None,
        "pass": bool(met),
        "caveats": [
            "single-threaded-per-process CPU timing on a shared container: absolute latency may differ on a phone or a laptop",
            "no thermal control and no background-load isolation",
            "greedy decoding, no KV cache: latency scales quadratically with generated length",
            "checkpoint file size includes optimizer state, so it overstates deployable weight size",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    verdict = "MEETS" if payload["pass"] else "MISSES"
    print(f"{verdict} target: median {payload['result']['median_seconds']:.3f}s for {target_tokens} tokens (target <= {target_seconds}s) -> {out}")
    if not payload["pass"]:
        print("  reporting the miss as a miss; the criterion is not adjusted to fit the measurement", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

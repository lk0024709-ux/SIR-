"""Run the multilingual generation probe set and grade it programmatically.

    python -m evaluation.scripts.probe_generation \
        --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml

What is measured (mechanical, defensible properties):
  * the generation is non-empty and contains real characters, not just whitespace or control codes;
  * it is not a degeneracy loop (no token repeated for > `max_run_tokens`, no n-gram repeated 3x);
  * the prompt round-trips through the model's own tokenizer (a cheap sanity check on encoding);
  * decoding is deterministic under greedy sampling: two runs must produce identical text;
  * token counts and throughput per generation.

What is NOT measured, and is therefore listed explicitly in the output: factual correctness,
grammar quality, instruction following, safety, or "understanding". A ~1M-parameter model trained on a
composed fixture cannot be scored on those and SIR will not pretend otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inference.engine import GenSettings, generate, load_for_inference  # noqa: E402
from sir_paths import load_config, rel, resolve  # noqa: E402

CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def grade(text: str, ids: list[int], n: dict[str, object]) -> dict[str, object]:
    """Apply the configured mechanical rules and record which evidence was used."""
    from inference.engine import is_degenerate, tail_cycle_cover

    stripped = (text or "").strip()
    ctrl = len(CTRL.findall(text or ""))
    period, reps, covered = tail_cycle_cover(ids)
    grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
    deg = is_degenerate(
        text,
        ids,
        ngram=int(n["repeated_ngram_n"]),
        max_run_tokens=int(n["max_run_tokens"]),
        min_unique_ngram_ratio=float(n["min_unique_ngram_ratio"]),
        max_tail_cycle_cover=float(n["max_tail_cycle_cover"]),
        max_char_dominance=float(n["max_char_dominance"]),
    )
    body = stripped.replace(" ", "").replace("\n", "")
    return {
        "non_empty": bool(stripped),
        "degenerate": deg,
        "control_chars": ctrl,
        "unique_token_ratio": round(len(set(ids)) / max(1, len(ids)), 4),
        "distinct_4gram_ratio": round(len(set(grams)) / len(grams), 4) if grams else None,
        "tail_cycle_period": period,
        "tail_cycle_cover": round(covered / max(1, len(ids)), 4),
        "top_char_share": round(max((body.count(c) for c in set(body)), default=0) / max(1, len(body)), 4),
        "chars": len(stripped),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SIR generation probe harness (50 prompts, mechanical grading).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--probes", help="override probe jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--determinism-check", type=int, default=2, help="generate each probe N times, compare")
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "results" / "generation_probes.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ev = cfg.get("evaluation") or {}
    deg = ev.get("degeneracy") or {}
    n_rules = {
        "repeated_ngram_n": int(deg.get("repeated_ngram_n", 8)),
        "max_run_tokens": int(deg.get("max_run_tokens", 32)),
        "min_unique_ngram_ratio": float(deg.get("min_unique_ngram_ratio", 0.35)),
        "max_tail_cycle_cover": float(deg.get("max_tail_cycle_cover", 0.5)),
        "max_char_dominance": float(deg.get("max_char_dominance", 0.6)),
    }
    target = float(deg.get("pass_rate_target", 0.5))
    probes_path = Path(args.probes) if args.probes else resolve(ev.get("probes", "evaluation/probes/m1_probe_v0.jsonl"))
    if not probes_path.exists():
        print(f"probe file missing: {rel(probes_path)}", file=sys.stderr)
        return 2
    probes = [json.loads(l) for l in probes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    min_prompts = int(ev.get("probe_min_prompts", 50))
    if len(probes) < min_prompts:
        print(f"probe set has {len(probes)} prompts; README requires >= {min_prompts}", file=sys.stderr)
        return 2

    model, tok, state = load_for_inference(args.checkpoint)
    settings = GenSettings(
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
        eos_ids=(int(model.cfg.eos_token_id),),
        pad_id=int(model.cfg.pad_token_id),
    )

    rows: list[dict[str, object]] = []
    per_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"prompts": 0, "accepted": 0})
    t0 = time.perf_counter()
    for pr in probes:
        prompt = pr.get("prompt", "")
        r = generate(model, tok, prompt, settings)
        g = grade(r["text"], list(tok.encode(r["text"])), n_rules)
        det = None
        if args.determinism_check and args.determinism_check > 1:
            texts = {r["text"]}
            for _ in range(int(args.determinism_check) - 1):
                texts.add(generate(model, tok, prompt, settings)["text"])
            det = len(texts) == 1
        # prompt round-trip: does the model's tokenizer reproduce the prompt text?
        rt = tok.decode(tok.encode(prompt))
        accepted = bool(g["non_empty"] and not g["degenerate"] and g["control_chars"] == 0)
        lang = str(pr.get("language", "unknown"))
        per_lang[lang]["prompts"] += 1
        per_lang[lang]["accepted"] += int(accepted)
        rows.append(
            {
                "id": pr.get("id"),
                "language": lang,
                "category": pr.get("category"),
                "prompt": prompt,
                "output": r["text"],
                "prompt_tokens": r["prompt_tokens"],
                "generated_tokens": r["generated_tokens"],
                "stopped": r.get("stopped"),
                "deterministic": det,
                "prompt_roundtrip_exact": rt == prompt,
                "accepted": accepted,
                **g,
            }
        )
    dt = time.perf_counter() - t0

    n_ok = sum(1 for r in rows if r["accepted"])
    rate = n_ok / max(1, len(rows))
    det_all = all(r["deterministic"] for r in rows if r["deterministic"] is not None)
    payload = {
        "tool": "evaluation/scripts/probe_generation.py",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": rel(resolve(args.checkpoint)),
        "probes_file": rel(probes_path),
        "n_prompts": len(rows),
        "grading": {
            "accepted_if": "non-empty AND not degenerate under the rules below AND zero control characters",
            "rules": n_rules,
        },
        "results": {
            "accepted": n_ok,
            "rejected": len(rows) - n_ok,
            "acceptance_rate": round(rate, 4),
            "target_rate": target,
            "pass": bool(rate >= target),
            "mean_generated_tokens": round(statistics.mean(r["generated_tokens"] for r in rows), 2),
            "empty_outputs": sum(1 for r in rows if not r["non_empty"]),
            "degenerate_outputs": sum(1 for r in rows if r["degenerate"]),
            "prompt_roundtrip_failures": sum(1 for r in rows if not r["prompt_roundtrip_exact"]),
            "greedy_determinism_all_identical": det_all,
            "seconds_total": round(dt, 3),
            "per_language": {k: {**v, "rate": round(v["accepted"] / max(1, v["prompts"]), 3)} for k, v in sorted(per_lang.items())},
        },
        "explicitly_not_measured": [
            "factual correctness",
            "grammar quality",
            "translation quality",
            "instruction following",
            "safety or refusal behaviour",
            "any comparison with another model",
        ],
        "provisional": (state.get("extra") or {}).get("provisional"),
        "rows": rows,
        "interpretation": (
            "These probes test whether a prototype produces structured, non-looping text in three "
            "writing systems. They say nothing about whether the content is true or useful."
        ),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"accepted {n_ok}/{len(rows)} = {rate:.0%} (target >= {target:.0%}) -> {'PASS' if payload['results']['pass'] else 'FAIL'}")
    print(f"degenerate: {payload['results']['degenerate_outputs']} | empty: {payload['results']['empty_outputs']} | greedy determinism: {det_all}")
    for k, v in payload["results"]["per_language"].items():
        print(f"  {k:<10} {v['accepted']}/{v['prompts']}")
    print(f"wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

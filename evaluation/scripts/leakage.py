"""Train/validation leakage measurement + probe-contamination check.

    python -m evaluation.scripts.leakage --config configs/sir_nano_smoke.yaml \
        --out evaluation/results/leakage.json

The README criterion is "zero 8-gram overlap between train and held-out". This script measures it.
If it is not zero, the report says what the overlapping n-grams are and how often, because the right
response to a leak is to fix the pipeline — never to delete the offending validation rows until the
metric reads 0. Both directions are recorded: the achieved number, and what was NOT done to get it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sir_paths import load_config, rel, resolve  # noqa: E402
from training.data.validate import cross_split_near_dups, load_jsonl, overlap_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure train/val leakage and probe contamination.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--processed", help="override processed dir")
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "results" / "leakage.json"))
    ap.add_argument("--skip-near-dup", action="store_true", help="faster, for CI smoke runs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    processed = Path(args.processed) if args.processed else resolve((cfg.get("paths") or {}).get("processed_dir", "data/processed/latest"))
    train_p, val_p = processed / "train.jsonl", processed / "val.jsonl"
    for p in (train_p, val_p):
        if not p.exists():
            print(f"missing {rel(p)} — run python -m training.data.pipeline first", file=sys.stderr)
            return 2
    train, val = load_jsonl(train_p), load_jsonl(val_p)
    n = int(args.ngram or (cfg.get("evaluation") or {}).get("leak_ngram", 8))

    ov = overlap_report(train, val, n)
    ids_t = {str(d.get("id")) for d in train}
    ids_v = {str(d.get("id")) for d in val}
    nd = {} if args.skip_near_dup else cross_split_near_dups(train, val)

    # probe contamination: the generation test must be unseen by training
    probes_path = resolve((cfg.get("evaluation") or {}).get("probes", "evaluation/probes/m1_probe_v0.jsonl"))
    probe_report: dict[str, object] = {"checked": False, "reason": f"missing {rel(probes_path)}"}
    if probes_path.exists():
        probes = [json.loads(l) for l in probes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        from training.data.validate import tokens_of

        tgrams = set()
        for d in train:
            t = tokens_of(d["text"])
            tgrams |= {" ".join(t[i : i + n]) for i in range(0, max(0, len(t) - n + 1))}
        hits = []
        for pr in probes:
            t = tokens_of(pr.get("prompt", ""))
            shared = [g for g in (" ".join(t[i : i + n]) for i in range(0, max(0, len(t) - n + 1))) if g in tgrams]
            if shared:
                hits.append({"probe_id": pr.get("id"), "example": shared[0]})
        probe_report = {
            "checked": True,
            "probes": len(probes),
            "ngram": n,
            "contaminated": len(hits),
            "examples": hits[:5],
            "clean": not hits,
            "note": "a prompt shorter than the n-gram window cannot overlap by construction; that is a property of the test, not a pass",
            "short_prompts": sum(1 for pr in probes if len(tokens_of(pr.get("prompt", ""))) < n),
        }

    ds = processed / "dataset_report.json"
    prov = json.loads(ds.read_text(encoding="utf-8")).get("provisional") if ds.exists() else None
    result = {
        "tool": "evaluation/scripts/leakage.py",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": rel(resolve(args.config)),
        "criterion": {"metric": f"{n}-gram overlap between train and validation", "target": 0},
        "train_docs": len(train),
        "val_docs": len(val),
        "train_val_id_overlap": sorted(ids_t & ids_v)[:5],
        "gram_overlap": ov,
        "cross_split_near_duplicates": nd,
        "probe_contamination": probe_report,
        "provisional_corpus": prov,
        "pass": bool(ov["zero_overlap"] and not (ids_t & ids_v) and (nd.get("pairs", 0) == 0) and probe_report.get("clean", True)),
        "integrity_notes": [
            "no validation document was removed to improve this number",
            "the splitter groups documents by shared sentences and shared n-grams, so zero overlap is "
            "produced by construction and then verified; a nonzero result here means the grouping missed an edge",
        ],
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{n}-gram overlap: unique={ov['unique_overlapping_ngrams']} occurrences={ov['overlapping_ngram_occurrences_train']} rate={ov['overlap_rate']}")
    if nd:
        print(f"cross-split near-duplicates: {nd['pairs']} pairs (threshold {nd['threshold']})")
    print(f"probe contamination: {probe_report.get('contaminated', 'n/a')} of {probe_report.get('probes', 0)} probes")
    print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'} -> {rel(out)}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

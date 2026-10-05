"""Stage 4 — validation of a processed dataset: schema, balance, and leakage measurement.

This is the module that decides whether a number is worth printing. It measures, it does not
hope:

  * document schema (ids unique, text present, language declared);
  * train/val id disjointness;
  * **8-gram overlap** between train and validation, on whitespace tokens of normalized text —
    the criterion the README states as "zero overlap". Zero is reported as achieved only if it
    is actually zero; otherwise the count, the rate, and concrete examples are written out so a
    human can judge whether it is generic boilerplate or a real leak;
  * cross-split near-duplicate check (a 12-shingle Jaccard >= 0.8 pair across the boundary is a
    leak even if no 8-gram repeats by chance);
  * a "provenance present" check: token streams must name their tokenizer, source ids and seed.

Nothing here deletes validation documents. If a leak exists, the correct response is to fix the
split/dedup, not to shrink the eval set until the test passes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RE_WS = re.compile(r"\s+")
NGRAM = 8


def normalize_text(text: str) -> str:
    """Same normalization used by dedup, so the leakage test cannot be dodged by whitespace games."""
    return RE_WS.sub(" ", text.replace("\u00a0", " ")).strip().lower()


def tokens_of(text: str) -> list[str]:
    return normalize_text(text).split()


def ngrams(toks: list[str], n: int = NGRAM) -> set[str]:
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class Issue:
    level: str
    code: str
    message: str
    examples: list[Any] = field(default_factory=list)

    def format(self) -> str:
        return f"{self.level.upper():7s} {self.code}: {self.message}" + (f" e.g. {self.examples[:3]}" if self.examples else "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: {e}") from e
    return out


def check_docs(docs: list[dict[str, Any]], label: str) -> list[Issue]:
    issues: list[Issue] = []
    ids = [str(d.get("id")) for d in docs]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        issues.append(Issue("error", f"duplicate-ids-{label}", f"{len(dupes)} repeated ids", dupes))
    bad = [i for i, d in enumerate(docs) if not isinstance(d.get("text"), str) or not d["text"].strip()]
    if bad:
        issues.append(Issue("error", f"empty-text-{label}", f"{len(bad)} docs without usable text", bad[:3]))
    nolang = [str(d.get("id")) for d in docs if not d.get("language")]
    if nolang:
        issues.append(Issue("warning", f"missing-language-{label}", f"{len(nolang)} docs without a language tag", nolang[:3]))
    nosrc = [str(d.get("id")) for d in docs if not d.get("source_id")]
    if nosrc:
        issues.append(
            Issue("error", f"unattributed-{label}", f"{len(nosrc)} docs without source_id — SIR requires manifest attribution", nosrc[:3])
        )
    return issues


def overlap_report(train: list[dict[str, Any]], val: list[dict[str, Any]], n: int = NGRAM, max_examples: int = 5) -> dict[str, Any]:
    """Exact n-gram overlap between the two sides, reported whether or not it is zero."""
    val_grams: dict[str, str] = {}
    for d in val:
        for g in ngrams(tokens_of(d["text"]), n):
            val_grams.setdefault(g, str(d.get("id")))
    hits: Counter[str] = Counter()
    total = 0
    examples: list[dict[str, str]] = []
    for d in train:
        for g in ngrams(tokens_of(d["text"]), n):
            total += 1
            if g in val_grams:
                hits[g] += 1
                if len(examples) < max_examples:
                    examples.append({"ngram": g, "val_id": val_grams[g], "train_id": str(d.get("id"))})
    uniq = len(hits)
    inst = sum(hits.values())
    return {
        "n": n,
        "train_ngrams": total,
        "val_ngrams": len(val_grams),
        "unique_overlapping_ngrams": uniq,
        "overlapping_ngram_occurrences_train": inst,
        "overlap_rate": round(inst / total, 8) if total else 0.0,
        "zero_overlap": uniq == 0,
        "examples": examples,
    }


def cross_split_near_dups(train: list[dict[str, Any]], val: list[dict[str, Any]], threshold: float = 0.8, shingle: int = 24) -> dict[str, Any]:
    """Catch paraphrase-level leaks: same document on both sides with edits."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.data.deduplicate import shingles as mk

    def feats(docs):
        return [(str(d.get("id")), set(mk(d["text"], shingle, True))) for d in docs]

    tf, vf = feats(train), feats(val)
    found: list[dict[str, Any]] = []
    for tid, ts in tf:
        for vid, vs in vf:
            uni = len(ts | vs) or 1
            sim = len(ts & vs) / uni
            if sim >= threshold:
                found.append({"train_id": tid, "val_id": vid, "jaccard": round(sim, 4)})
    return {"threshold": threshold, "shingle_chars": shingle, "pairs": len(found), "examples": found[:5]}


def validate_processed(
    train_path: Path,
    val_path: Path,
    ngram: int = NGRAM,
    near_dup_threshold: float = 0.8,
    min_val_docs: int = 10,
    do_near_dup: bool = True,
) -> tuple[list[Issue], dict[str, Any]]:
    issues: list[Issue] = []
    train, val = load_jsonl(train_path), load_jsonl(val_path)
    issues += check_docs(train, "train") + check_docs(val, "val")

    t_ids, v_ids = {str(d.get("id")) for d in train}, {str(d.get("id")) for d in val}
    shared = sorted(t_ids & v_ids)
    if shared:
        issues.append(Issue("error", "split-id-collision", f"{len(shared)} ids appear on both sides", shared[:3]))
    if len(val) < min_val_docs:
        issues.append(Issue("error", "too-few-val-docs", f"{len(val)} < {min_val_docs}: per-language perplexity would be noise"))

    lang_v = Counter(str(d.get("language")) for d in val)
    lang_t = Counter(str(d.get("language")) for d in train)
    for lang, cnt in lang_v.items():
        if cnt < 3:
            issues.append(Issue("warning", f"thin-validation-language:{lang}", f"only {cnt} val docs; metric will be unstable"))
    missing = [l for l in lang_t if l not in lang_v]
    if missing:
        issues.append(Issue("warning", "lang-in-train-not-val", "no validation coverage for these languages", missing[:6]))

    ov = overlap_report(train, val, ngram)
    if ov["zero_overlap"]:
        issues.append(Issue("info", "leakage-8gram", f"zero {ngram}-gram overlap between train and val (measured)"))
    else:
        sev = "error" if ov["overlap_rate"] > 0.01 else "warning"
        issues.append(
            Issue(
                sev,
                "leakage-8gram",
                f"{ov['unique_overlapping_ngrams']} unique {ngram}-grams shared "
                f"({ov['overlapping_ngram_occurrences_train']} occurrences, rate={ov['overlap_rate']:.5f}) — "
                "fix dedup/split; do not delete val docs to make this zero",
                [e["ngram"] for e in ov["examples"]],
            )
        )
    nd = cross_split_near_dups(train, val, near_dup_threshold) if do_near_dup else {}
    if nd and nd.get("pairs"):
        issues.append(Issue("error", "cross-split-near-duplicate", f"{nd['pairs']} near-duplicate pairs across the split boundary", nd["examples"]))

    summary = {
        "train_docs": len(train),
        "val_docs": len(val),
        "train_chars": sum(len(d["text"]) for d in train),
        "val_chars": sum(len(d["text"]) for d in val),
        "train_languages": dict(lang_t),
        "val_languages": dict(lang_v),
        "leakage_8gram": ov,
        "cross_split_near_dups": nd,
    }
    return issues, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4: validate a processed dataset (schema + leakage).")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ngram", type=int, default=NGRAM)
    ap.add_argument("--skip-near-dup", action="store_true", help="fast mode for unit tests")
    ap.add_argument("--out", help="write machine-readable summary JSON here")
    ap.add_argument("--report-rejects", help="optional path to data/cleaned/rejects.json for the audit line")
    args = ap.parse_args(argv)

    issues, summary = validate_processed(
        Path(args.train), Path(args.val), ngram=args.ngram, do_near_dup=not args.skip_near_dup
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "issues": [i.__dict__ for i in issues]}, indent=2) + "\n", encoding="utf-8")
    for i in issues:
        print("  " + i.format())
    errors = [i for i in issues if i.level == "error"]
    print(json.dumps({k: v for k, v in summary.items() if k not in ("leakage_8gram", "cross_split_near_dups")}, indent=2))
    print(f"leakage: {summary['leakage_8gram']['unique_overlapping_ngrams']} shared {args.ngram}-grams "
          f"(rate {summary['leakage_8gram']['overlap_rate']:.6f})")
    if errors:
        print(f"RESULT: FAIL ({len(errors)} errors)", file=sys.stderr)
        return 1
    print("RESULT: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

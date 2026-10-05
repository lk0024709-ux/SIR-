"""Stage 3 — train/validation split, leakage-safe by construction.

A naive "hash the document id and take 5%" split is fine only when documents are independent.
They are not, in any real corpus and especially not in a composed one: boilerplate sentences,
repeated notices, or template-generated documents put the *same sentence* on both sides, and then
validation perplexity measures memorisation. SIR's M1 acceptance criterion is validation-based,
so the split is part of the measurement integrity, not a housekeeping step.

What this module does
    1. Builds content components: documents that share a normalized sentence (and, in `ngram`
       mode, documents that share an n-gram of length `leak_ngram`) are unioned into one group.
    2. Assigns WHOLE components to train or val by a seeded hash, per language, so the split is
       stable, order-independent, and stratified.
    3. Reports the achieved fractions (including overshoot when a single component is bigger than
       the val budget) instead of quietly rounding, and never drops a validation document to make
       a leakage number look good.

Consequence worth stating plainly: with `group_by=ngram`, zero train/val n-gram overlap is
guaranteed by construction. validate.py still measures it, because a nonzero result would mean the
grouping missed an edge — the check is a test of the splitter, not a formality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BUCKETS = 10_000
SENT_SPLIT = re.compile(r"(?<=[.!?।\u0965])\s+|\n+")
WS = re.compile(r"\s+")


@dataclass
class SplitConfig:
    val_frac: float = 0.05
    test_frac: float = 0.0
    seed: int = 20261005
    group_by: str = "auto"  # auto | id | sentence | ngram
    leak_ngram: int = 8
    max_docs_for_ngram_graph: int = 40_000  # n-gram graph is O(corpus); above this, fall back to sentences

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitResult:
    train: list[dict[str, Any]] = field(default_factory=list)
    val: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)
    per_language: dict[str, dict[str, int]] = field(default_factory=dict)
    method: str = "id"
    notes: list[str] = field(default_factory=list)
    components: int = 0
    seed: int = 0
    val_frac_requested: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "components": self.components,
            "train_docs": len(self.train),
            "val_docs": len(self.val),
            "test_docs": len(self.test),
            "val_frac_requested": self.val_frac_requested,
            "val_frac_achieved": round(len(self.val) / max(1, len(self.train) + len(self.val)), 5),
            "train_chars": sum(len(d["text"]) for d in self.train),
            "val_chars": sum(len(d["text"]) for d in self.val),
            "per_language": self.per_language,
            "notes": self.notes,
        }


class _Union:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def norm_sentence(s: str) -> str:
    return WS.sub(" ", s.replace("\u00a0", " ")).strip().lower()


def sentences_of(text: str) -> list[str]:
    return [norm_sentence(s) for s in SENT_SPLIT.split(text) if s and len(norm_sentence(s)) > 3]


def word_ngrams(text: str, n: int) -> set[str]:
    toks = norm_sentence(text).split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_components(docs: list[dict[str, Any]], mode: str, leak_ngram: int, max_docs_ngram: int) -> tuple[list[int], str, list[str]]:
    """Return (root index per doc, method used, notes)."""
    notes: list[str] = []
    n = len(docs)
    u = _Union(n)

    if mode == "id":
        return list(range(n)), "id", ["group_by=id: each document is its own group (leakage risk accepted by config)"]

    # shared-sentence edges (cheap, catches template corpora)
    owner: dict[str, int] = {}
    for i, d in enumerate(docs):
        for s in sentences_of(d["text"]):
            h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
            j = owner.get(h, -1)
            if j >= 0:
                u.union(min(i, j), max(i, j))
            else:
                owner[h] = i
    method = "sentence"

    want_ngram = mode in ("auto", "ngram")
    if want_ngram and n <= max_docs_ngram:
        og: dict[str, int] = {}
        for i, d in enumerate(docs):
            for g in word_ngrams(d["text"], leak_ngram):
                h = hashlib.blake2b(g.encode("utf-8"), digest_size=16).digest()
                j = og.get(h, -1)
                if j >= 0:
                    u.union(min(i, j), max(i, j))
                else:
                    og[h] = i
        method = f"ngram{leak_ngram}+sentence"
    elif want_ngram:
        notes.append(
            f"ngram grouping skipped: {n} docs > max_docs_for_ngram_graph={max_docs_ngram}. "
            "Sentence-level grouping still applied; validate.py must be run to measure residual overlap."
        )
    elif mode == "ngram":
        raise ValueError("group_by=ngram is impossible above max_docs_for_ngram_graph")

    return [u.find(i) for i in range(n)], method, notes


def split(docs: list[dict[str, Any]], cfg: SplitConfig | None = None) -> SplitResult:
    cfg = cfg or SplitConfig()
    if not 0.0 <= cfg.val_frac < 0.5:
        raise ValueError("val_frac must be in [0, 0.5): a majority-validation split is a mistake, not a feature")
    if not 0.0 <= cfg.test_frac < 0.5:
        raise ValueError("test_frac must be in [0, 0.5)")
    roots, method, notes = build_components(docs, cfg.group_by, cfg.leak_ngram, cfg.max_docs_for_ngram_graph)

    res = SplitResult(method=method, seed=cfg.seed, val_frac_requested=cfg.val_frac, notes=list(notes))
    comps: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(roots):
        comps[r].append(i)
    res.components = len(comps)
    if len(comps) < 2 and method != "id":
        raise ValueError(
            f"content grouping merged the whole corpus into {len(comps)} component(s); a leakage-safe "
            "split is impossible on this data. Reduce duplication (dedup harder) or accept group_by=id "
            "and report the overlap honestly."
        )

    # stratify per language, then assign whole components by hashed order
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        by_lang[str(d.get("language", "unknown"))].append(i)

    for lang, idxs in sorted(by_lang.items()):
        total = len(idxs)
        lang_comps = defaultdict(list)
        for i in idxs:
            lang_comps[roots[i]].append(i)
        ordered = sorted(
            lang_comps.items(),
            key=lambda kv: (int.from_bytes(hashlib.blake2b(f"{cfg.seed}:{kv[0]}".encode(), digest_size=8).digest(), "big") % BUCKETS, kv[0]),
        )
        if len(ordered) < 2:
            raise ValueError(
                f"language {lang!r}: all {total} documents form ONE content component, so a "
                "leakage-safe split cannot also guarantee coverage on both sides. Either make the "
                "corpus less repetitive (distinct sentences per document group) or set "
                "group_by=id and report the measured n-gram overlap instead of claiming zero."
            )
        target_val = max(1, round(total * cfg.val_frac)) if total >= 2 else 0
        target_test = max(0, round(total * cfg.test_frac))
        v: list[int] = []
        t: list[int] = []
        tr: list[int] = []
        for root, members in ordered:
            if len(v) < target_val:
                v.extend(members)
            elif target_test and len(t) < target_test:
                t.extend(members)
            else:
                tr.extend(members)
        if v and len(v) > 2 * max(1, target_val):
            res.notes.append(
                f"{lang}: smallest content group is {len(v)} docs vs val target {target_val}; validation is "
                "larger than requested because splitting a group would leak. Not rounded away."
            )
        res.train.extend(sorted(tr))
        res.val.extend(sorted(v))
        res.test.extend(sorted(t))
        res.per_language[lang] = {"train": len(tr), "val": len(v), "test": len(t), "components": len(lang_comps)}

    def docs_of(idxs):
        return [docs[i] for i in sorted(set(idxs))]

    res.train = docs_of(res.train)
    res.val = docs_of(res.val)
    res.test = docs_of(res.test)
    return res


def write_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: leakage-safe, deterministic, stratified split.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--group-by", default="auto", choices=["auto", "id", "sentence", "ngram"])
    ap.add_argument("--leak-ngram", type=int, default=8)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.data.clean import load_documents

    docs = list(load_documents(Path(args.inp)))
    res = split(
        docs,
        SplitConfig(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed, group_by=args.group_by, leak_ngram=args.leak_ngram),
    )
    out = Path(args.outdir)
    write_jsonl(out / "train.jsonl", res.train)
    write_jsonl(out / "val.jsonl", res.val)
    if res.test:
        write_jsonl(out / "test.jsonl", res.test)
    payload = res.as_dict()
    payload["out_dir"] = str(out)
    (out / "split_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

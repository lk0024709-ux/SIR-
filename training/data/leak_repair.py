"""Close the residual train/validation leakage by *repairing the split*, never by shrinking the test.

The splitter groups documents that share a normalized sentence. That catches boilerplate and
copy-paste, but it cannot catch two documents that share an 8-gram without sharing a whole
sentence — repeated liturgical verse, a recurring legal phrase, a quoted stanza. So the pipeline
measures the 8-gram overlap after splitting (G4) and, if it is not zero, this module fixes the
cause:

    1. find the offending (train document, validation document) pairs;
    2. union the content components they belong to;
    3. move the merged component to TRAIN (the corpus loses a validation document, never a
       training document, and the validation set is never trimmed to make a number look good);
    4. re-measure; repeat up to `max_iters` times.

The loop is bounded, deterministic and logs every move. If it does not converge, the residual
overlap is reported and the G4 gate fails: an unconverged repair is a finding, not an error to hide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.dedup_stream import JsonlWriter, iter_jsonl  # noqa: E402


def _ngram_hashes(text: str, n: int, cache: dict[str, int]) -> np.ndarray:
    toks = str(text).lower().split()
    if len(toks) < n:
        return np.zeros(0, dtype=np.uint64)
    th = np.fromiter(
        (cache.setdefault(t, int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "big")) for t in toks),
        dtype=np.uint64,
        count=len(toks),
    )
    acc = th.copy()
    P = np.uint64(0x100000001B3)
    for j in range(1, n):
        acc = acc[: acc.size - 1] * P + th[j:]
    return acc


def find_leaking_pairs(train_path: Path, val_path: Path, n: int = 8, max_pairs: int = 20_000) -> dict[str, Any]:
    """Return the (train_key, val_key) document pairs that share at least one n-gram."""
    cache: dict[str, int] = {}
    val_hash_chunks: list[np.ndarray] = []
    val_doc_of_hash: list[np.ndarray] = []
    val_keys: list[tuple[str, str]] = []
    for idx, doc in enumerate(iter_jsonl(val_path)):
        h = _ngram_hashes(doc.get("text") or "", n, cache)
        if h.size:
            val_hash_chunks.append(h)
            val_doc_of_hash.append(np.full(h.size, idx, dtype=np.int32))
        val_keys.append((str(doc.get("source_id")), str(doc.get("doc_id"))))
    if not val_hash_chunks:
        return {"pairs": [], "truncated": False, "val_docs": len(val_keys)}
    val_hashes = np.concatenate(val_hash_chunks)
    val_doc_idx = np.concatenate(val_doc_of_hash)
    order = np.argsort(val_hashes, kind="stable")
    val_hashes = val_hashes[order]
    val_doc_idx = val_doc_idx[order]

    seen: set[tuple[str, str]] = set()
    truncated = False
    for doc in iter_jsonl(train_path):
        h = _ngram_hashes(doc.get("text") or "", n, cache)
        if not h.size:
            continue
        lo = np.searchsorted(val_hashes, h, side="left")
        hi = np.searchsorted(val_hashes, h, side="right")
        hit_mask = hi > lo
        if not hit_mask.any():
            continue
        train_key = (str(doc.get("source_id")), str(doc.get("doc_id")))
        for k in np.nonzero(hit_mask)[0][:50]:
            for vi in np.unique(val_doc_idx[lo[k] : hi[k]]):
                vsrc, vdoc = val_keys[int(vi)]
                seen.add((train_key[0], train_key[1], vsrc, vdoc))
                if len(seen) >= max_pairs:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break
    pairs = [
        {"train_source": a, "train_doc": b, "val_source": c, "val_doc": d}
        for (a, b, c, d) in sorted(seen)
    ]
    return {"pairs": pairs, "truncated": truncated, "val_docs": len(val_keys), "val_ngrams": int(val_hashes.size)}


def repair_split(
    out_dir: Path,
    *,
    n: int = 8,
    max_iters: int = 6,
    max_pairs: int = 20_000,
) -> dict[str, Any]:
    """Move offending validation components to train until the measured 8-gram overlap is zero."""
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    log: list[dict[str, Any]] = []
    moved_total = 0
    converged = False
    for it in range(1, max_iters + 1):
        found = find_leaking_pairs(train_path, val_path, n=n, max_pairs=max_pairs)
        pairs = found["pairs"]
        if not pairs:
            converged = True
            log.append({"iteration": it, "leaking_pairs": 0, "moved_documents": 0})
            break

        # which validation *components* are implicated by the leaking pairs?
        comp_of_val: dict[str, int] = {}
        for doc in iter_jsonl(val_path):
            comp_of_val[f"{doc.get('source_id')}\u0000{doc.get('doc_id')}"] = int(doc.get("component_id", -1))
        move_components: set[int] = set()
        for p in pairs:
            vcomp = comp_of_val.get(f"{p['val_source']}\u0000{p['val_doc']}")
            if vcomp is not None:
                move_components.add(vcomp)

        # rewrite both sides; every implicated component is forced to train
        tmp_train = out_dir / "train.jsonl.tmp"
        tmp_val = out_dir / "val.jsonl.tmp"
        moved = 0
        with JsonlWriter(tmp_train) as tw, JsonlWriter(tmp_val) as vw:
            for doc in iter_jsonl(train_path):
                tw.write(doc)
            for doc in iter_jsonl(val_path):
                if int(doc.get("component_id", -1)) in move_components:
                    moved += 1
                    tw.write({**doc, "split": "train", "moved_from_validation_by": f"leak-repair-{it}"})
                else:
                    vw.write(doc)
        tmp_train.replace(train_path)
        tmp_val.replace(val_path)
        moved_total += moved
        log.append(
            {
                "iteration": it,
                "leaking_pairs": len(pairs),
                "pairs_truncated": found["truncated"],
                "components_moved": len(move_components),
                "moved_documents": moved,
            }
        )
        if moved == 0:
            break
    result = {
        "tool": "training/data/leak_repair.py",
        "n": n,
        "max_iters": max_iters,
        "converged": converged,
        "moved_documents_total": moved_total,
        "log": log,
        "policy": "merged components move to TRAIN; validation is never trimmed to improve a number",
    }
    (out_dir / "leak_repair_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repair train/val leakage by merging offending components into train.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=6)
    args = ap.parse_args(argv)
    res = repair_split(Path(args.outdir), n=args.ngram, max_iters=args.max_iters)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=2))
    for entry in res["log"]:
        print("  ", json.dumps(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

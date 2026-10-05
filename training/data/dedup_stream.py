"""Streaming, deterministic deduplication for a corpus that does not fit in RAM.

Why this exists next to `training/data/deduplicate.py`
------------------------------------------------------
`deduplicate.py` is the in-memory reference implementation: it takes a list of documents and is
easy to reason about and to test. The M1 corpus cannot use it — 200M tokens is far more than this
box (≈3.9 GB RAM) can hold, and the honest alternative to "load everything and hope" is a
two-pass streaming design:

  pass 1  read the corpus once, write a *band index* to SQLite (signature bands + byte offsets)
  pairs   ask SQLite which documents collide on a band (bounded by `max_candidates`)
  pass 2  re-read only the candidate documents to compute exact Jaccard, then union-find clusters

Determinism: the input order is the tie-break. The **first** document of a cluster wins, whether
the cluster spans sources or not, and every removed document records which document replaced it.
Nothing depends on dict/hash iteration order or on file-system ordering.

Memory: signatures, shingles and candidate sets are bounded; the corpus itself is never held.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from training.data.deduplicate import DedupConfig, normalize_for_compare, shingles, signature

# --------------------------------------------------------------------------------------
# small shared helpers (also used by build_corpus / split_stream)
# --------------------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield documents from a JSONL file one at a time. Blank lines are skipped, not guessed at."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Append-only JSONL writer that measures what it wrote (bytes and sha256) as it goes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._hash = hashlib.sha256()
        self.bytes_written = 0
        self.records = 0

    def write(self, doc: dict[str, Any]) -> None:
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        self._fh.write(line)
        payload = line.encode("utf-8")
        self._hash.update(payload)
        self.bytes_written += len(payload)
        self.records += 1

    def write_text(self, line: str) -> None:
        self._fh.write(line)
        self._hash.update(line.encode("utf-8"))
        self.bytes_written += len(line.encode("utf-8"))

    def close(self) -> dict[str, Any]:
        self._fh.close()
        return {"path": str(self.path), "records": self.records, "bytes": self.bytes_written, "sha256": self._hash.hexdigest()}

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def doc_fingerprint(text: str) -> str:
    """Exact-duplicate key: whitespace-collapsed text, hashed. Case is preserved."""
    return hashlib.sha256(normalize_for_compare(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# exact duplicates
# --------------------------------------------------------------------------------------


@dataclass
class ExactReport:
    read: int = 0
    kept: int = 0
    removed: int = 0
    duplicate_groups: int = 0
    cross_source_removals: int = 0
    removed_by_reason: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def stream_exact_dedup(in_path: Path, keep_path: Path, removed_path: Path) -> tuple[ExactReport, dict[str, Any]]:
    """Drop exact duplicates, keeping the first occurrence in file order.

    `removed_path` receives one record per dropped document: doc_id, source_id, reason,
    kept_doc_id, kept_source_id. That record is what makes "documented earliest-source policy"
    checkable after the fact instead of being a claim in a docstring.
    """
    rep = ExactReport()
    seen: dict[str, tuple[str, str]] = {}  # fingerprint -> (doc_id, source_id)
    with JsonlWriter(keep_path) as keep, JsonlWriter(removed_path) as removed:
        for doc in iter_jsonl(in_path):
            rep.read += 1
            text = str(doc.get("text") or "")
            if not text:
                rep.removed += 1
                rep.removed_by_reason["empty_text"] = rep.removed_by_reason.get("empty_text", 0) + 1
                removed.write({"doc_id": doc.get("doc_id"), "source_id": doc.get("source_id"), "reason": "empty_text"})
                continue
            fp = doc_fingerprint(text)
            if fp in seen:
                winner_doc, winner_src = seen[fp]
                rep.removed += 1
                rep.duplicate_groups += 0  # group counted below
                rep.removed_by_reason["exact_duplicate"] = rep.removed_by_reason.get("exact_duplicate", 0) + 1
                if str(doc.get("source_id")) != winner_src:
                    rep.cross_source_removals += 1
                removed.write(
                    {
                        "doc_id": doc.get("doc_id"),
                        "source_id": doc.get("source_id"),
                        "reason": "exact_duplicate",
                        "kept_doc_id": winner_doc,
                        "kept_source_id": winner_src,
                        "normalized_sha256": fp,
                    }
                )
                continue
            seen[fp] = (str(doc.get("doc_id")), str(doc.get("source_id")))
            keep.write(doc)
            rep.kept += 1
    rep.duplicate_groups = rep.removed_by_reason.get("exact_duplicate", 0)
    return rep, {"read": rep.read, "kept": rep.kept}


# --------------------------------------------------------------------------------------
# near duplicates
# --------------------------------------------------------------------------------------


@dataclass
class NearReport:
    read: int = 0
    kept: int = 0
    removed: int = 0
    candidate_pairs: int = 0
    pairs_verified: int = 0
    clusters: int = 0
    skipped_docs: int = 0
    skipped_bands: int = 0
    max_band_size: int = 0
    jaccard_min: float = 1.0
    jaccard_max: float = 0.0
    cross_source_removals: int = 0
    winners_by_source: dict[str, int] = field(default_factory=dict)
    removed_by_source: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _band_keys(sig: tuple[int, ...], band_size: int) -> list[str]:
    out = []
    for i in range(0, len(sig), band_size):
        band = sig[i : i + band_size]
        if len(band) < band_size:
            break
        out.append(hashlib.blake2b(",".join(str(x) for x in band).encode("ascii"), digest_size=8).hexdigest())
    return out


class _Union:
    """Union-find over integer indices with deterministic (smallest-index) roots."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent
        while p.get(x, x) != x:
            p[x] = p.get(p[x], p[x])
            x = p[x]
        return p.get(x, x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def stream_near_dedup(
    in_path: Path,
    keep_path: Path,
    removed_path: Path,
    *,
    cfg: DedupConfig,
    workdir: Path,
    max_band_size: int = 200,
    max_candidates: int = 200_000,
) -> tuple[NearReport, dict[str, Any]]:
    """MinHash-LSH near-duplicate removal with a SQLite band index.

    A band shared by more than `max_band_size` documents is treated as boilerplate and its
    pairwise expansion is truncated (the truncation is counted in the report, never hidden).
    `max_candidates` bounds how many candidate pairs the verification pass will look at.
    """
    started = time.time()
    rep = NearReport(max_band_size=max_band_size)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "near_dup.sqlite"
    if db_path.exists():
        db_path.unlink()

    # ---- pass 1: signatures -> band index, plus an offset table for random access in pass 2
    offsets: list[tuple[int, int]] = []
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE bands (band TEXT NOT NULL, doc INTEGER NOT NULL)")
    conn.execute("CREATE TABLE docs (doc INTEGER PRIMARY KEY, offset INTEGER, length INTEGER)")
    row = 0
    with Path(in_path).open("rb") as fh:
        while True:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            doc = json.loads(raw.decode("utf-8"))
            text = str(doc.get("text") or "")
            rep.read += 1
            if not text:
                rep.skipped_docs += 1
                continue
            sig = signature(text, cfg)
            keys = _band_keys(sig, cfg.band_size)
            conn.executemany("INSERT INTO bands VALUES (?, ?)", [(k, row) for k in keys])
            conn.execute("INSERT INTO docs VALUES (?, ?, ?)", (row, offset, len(raw)))
            offsets.append((offset, len(raw)))
            row += 1
            if row % 5000 == 0:
                conn.commit()
    conn.commit()
    conn.execute("CREATE INDEX idx_bands ON bands (band, doc)")

    # ---- candidate pairs: bands with more than one document
    pairs: set[tuple[int, int]] = set()
    truncated_bands = 0
    for (band,) in conn.execute("SELECT DISTINCT band FROM bands GROUP BY band HAVING COUNT(*) > 1"):
        members = [d for (d,) in conn.execute("SELECT doc FROM bands WHERE band = ? ORDER BY doc", (band,))]
        if len(members) > max_band_size:
            truncated_bands += 1
            members = members[:max_band_size]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
                if len(pairs) > max_candidates:
                    break
            if len(pairs) > max_candidates:
                break
        if len(pairs) > max_candidates:
            break
    rep.skipped_bands = truncated_bands
    rep.candidate_pairs = len(pairs)

    # ---- pass 2: verify candidates by real Jaccard on shingles, union-find the survivors
    shingle_cache: dict[int, set[str]] = {}

    def shingles_at(fh, idx: int) -> set[str]:
        if idx not in shingle_cache:
            offset, length = offsets[idx]
            fh.seek(offset)
            doc = json.loads(fh.read(length).decode("utf-8"))
            shingle_cache[idx] = set(shingles(str(doc.get("text") or ""), cfg.shingle_chars))
        return shingle_cache[idx]

    union = _Union()
    verified = 0
    jmin, jmax = 1.0, 0.0
    with Path(in_path).open("rb") as fh:
        for a, b in sorted(pairs):
            ja = shingles_at(fh, a)
            jb = shingles_at(fh, b)
            if not ja or not jb:
                continue
            j = _jaccard(ja, jb)
            if j >= cfg.threshold:
                union.union(a, b)
                verified += 1
                jmin, jmax = min(jmin, j), max(jmax, j)
            if len(shingle_cache) > max_candidates // 4:
                shingle_cache.clear()
    rep.pairs_verified = verified
    if verified:
        rep.jaccard_min, rep.jaccard_max = round(jmin, 4), round(jmax, 4)

    # ---- random-access index for the emit pass (offset/length per document, deterministic order)
    doc_rows: list[tuple[int, int, int]] = [
        (d, o, l) for d, o, l in conn.execute("SELECT doc, offset, length FROM docs ORDER BY doc")
    ]
    conn.close()
    assert [d for d, _, _ in doc_rows] == list(range(len(doc_rows))), "document index must equal its row number"

    # ---- emit: the lowest document index of each cluster wins (deterministic, file order)
    with JsonlWriter(keep_path) as keep, JsonlWriter(removed_path) as removed, Path(in_path).open("rb") as fh:
        for idx, (_, offset, length) in enumerate(doc_rows):
            winner_idx = union.find(idx) if idx in union.parent else idx
            if winner_idx == idx:
                fh.seek(offset)
                keep.write(json.loads(fh.read(length).decode("utf-8")))
                rep.kept += 1
                continue
            # a removed document still records *which* document replaced it and from where
            fh.seek(offset)
            doc = json.loads(fh.read(length).decode("utf-8"))
            w_off, w_len = doc_rows[winner_idx][1], doc_rows[winner_idx][2]
            fh.seek(w_off)
            wdoc = json.loads(fh.read(w_len).decode("utf-8"))
            src, winner_src = str(doc.get("source_id")), str(wdoc.get("source_id"))
            rep.removed += 1
            rep.removed_by_source[src] = rep.removed_by_source.get(src, 0) + 1
            rep.winners_by_source[winner_src] = rep.winners_by_source.get(winner_src, 0) + 1
            if src != winner_src:
                rep.cross_source_removals += 1
            removed.write(
                {
                    "doc_id": doc.get("doc_id"),
                    "source_id": doc.get("source_id"),
                    "reason": "near_duplicate",
                    "kept_doc_id": wdoc.get("doc_id"),
                    "kept_source_id": winner_src,
                }
            )
    rep.clusters = len({union.find(i) for i in union.parent}) if union.parent else 0
    rep.elapsed_seconds = round(time.time() - started, 2)
    meta = {
        "candidate_pairs": rep.candidate_pairs,
        "bands_truncated": truncated_bands,
        "sqlite_index": str(db_path),
        "threshold": cfg.threshold,
        "num_perm": cfg.num_perm,
        "band_size": cfg.band_size,
        "shingle_chars": cfg.shingle_chars,
        "note": "cluster winner = lowest document index in file order (deterministic earliest-source policy)",
    }
    return rep, meta

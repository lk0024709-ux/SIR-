"""Stage 2 — deduplication: exact hash dedup, then MinHash/LSH near-duplicate removal.

Deterministic by design: no RNG, stable hashing, and each cluster is resolved to its earliest
document id, so two runs on the same input remove exactly the same documents.

Why it matters for SIR: duplicated boilerplate quietly inflates validation scores, and SIR's
acceptance criteria are validation-based. Dedup is therefore part of the *measurement* integrity,
not just data hygiene.

Scale note (kept honest): this is pure-Python MinHash + LSH. It is fine for the fixture and for
tens of thousands of documents; it is NOT the right tool for a web crawl. Before Phase 2 the
project should switch to a JVM/Go-grade dedup (e.g. spark-datasketches style) — recorded as a
known limitation in docs/dataset-policy.md rather than hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WS = re.compile(r"\s+")
MERSENNE_PRIME = (1 << 61) - 1


@dataclass
class DedupConfig:
    num_perm: int = 64
    band_size: int = 8
    threshold: float = 0.8
    shingle_chars: int = 24
    collapse_whitespace: bool = True


@dataclass
class DedupReport:
    read: int = 0
    kept: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    near_clusters: int = 0
    exact_groups: int = 0
    threshold: float = 0.8
    num_perm: int = 64
    band_size: int = 8
    shingle_chars: int = 24
    measured_pair_similarity_min: float | None = None
    exact_group_sizes: list[int] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("clusters",)}
        d["near_clusters_top"] = sorted(self.clusters, key=lambda c: -c["size"])[:10]
        return d


def normalize_for_compare(text: str, collapse: bool = True) -> str:
    t = text.replace("\u00a0", " ")
    return _WS.sub(" ", t).strip() if collapse else text.strip()


def shingles(text: str, size: int, collapse: bool = True) -> list[str]:
    t = normalize_for_compare(text, collapse)
    if len(t) <= size:
        return [t] if t else []
    step = max(1, size // 3)
    return [t[i : i + size] for i in range(0, len(t) - size + 1, step)]


def _hash64(payload: bytes, salt: int) -> int:
    d = hashlib.blake2b(payload, digest_size=8, key=salt.to_bytes(8, "little")).digest()
    return int.from_bytes(d, "big")


def signature(text: str, cfg: DedupConfig) -> tuple[int, ...]:
    sh = shingles(text, cfg.shingle_chars, cfg.collapse_whitespace)
    if not sh:
        return tuple([0] * cfg.num_perm)
    payloads = [s.encode("utf-8") for s in sh]
    return tuple(min(_hash64(p, k + 1) for p in payloads) for k in range(cfg.num_perm))


def exact_dedup(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    seen: dict[str, int] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    for idx, d in enumerate(docs):
        key = hashlib.sha256(normalize_for_compare(d["text"]).encode("utf-8")).hexdigest()
        groups[key].append(str(d.get("id", idx)))
        if key in seen:
            continue
        seen[key] = idx
        kept.append(d)
    return kept, {k: v for k, v in groups.items() if len(v) > 1}


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
        if ra == rb:
            return
        # smaller index is always the root -> earliest document survives
        if ra < rb:
            self.p[rb] = ra
        else:
            self.p[ra] = rb


def near_dedup(docs: list[dict[str, Any]], cfg: DedupConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    """LSH candidate generation + exact Jaccard verification. Returns (kept, clusters, sims)."""
    n_bands = max(1, cfg.num_perm // cfg.band_size)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    sets: list[set[str]] = []
    for i, d in enumerate(docs):
        sig = signature(d["text"], cfg)
        sets.append(set(shingles(d["text"], cfg.shingle_chars, cfg.collapse_whitespace)))
        for b in range(n_bands):
            band = sig[b * cfg.band_size : (b + 1) * cfg.band_size]
            buckets[(b, hash(band) & MERSENNE_PRIME)].append(i)

    u = _Union(len(docs))
    candidate_pairs = 0
    verified: dict[tuple[int, int], float] = {}
    for _, members in buckets.items():
        if len(members) < 2:
            continue
        members.sort()
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                candidate_pairs += 1
                if u.find(i) == u.find(j):
                    continue
                uni = len(sets[i] | sets[j]) or 1
                sim = len(sets[i] & sets[j]) / uni
                if sim >= cfg.threshold:
                    verified[(i, j)] = sim
                    u.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(docs)):
        clusters[u.find(i)].append(i)

    kept_idx: list[int] = []
    clusters_out: list[dict[str, Any]] = []
    sims: list[float] = []
    for root, members in sorted(clusters.items()):
        members = sorted(members)
        if len(members) > 1:
            ids = [str(docs[m].get("id", m)) for m in members]
            for m in members[1:]:
                i, j = members[0], m
                sim = verified.get((i, j)) or (len(sets[i] & sets[j]) / (len(sets[i] | sets[j]) or 1))
                sims.append(sim)
            clusters_out.append({"size": len(members), "kept": ids[0], "removed": ids[1:]})
        kept_idx.append(members[0])
    kept = [docs[i] for i in sorted(kept_idx)]
    return kept, clusters_out, sims


def deduplicate(docs: list[dict[str, Any]], cfg: DedupConfig | None = None) -> tuple[list[dict[str, Any]], DedupReport]:
    cfg = cfg or DedupConfig()
    rep = DedupReport(threshold=cfg.threshold, num_perm=cfg.num_perm, band_size=cfg.band_size, shingle_chars=cfg.shingle_chars)
    rep.read = len(docs)
    exact_kept, exact_groups = exact_dedup(docs)
    rep.exact_removed = rep.read - len(exact_kept)
    rep.exact_groups = len(exact_groups)
    near_kept, clusters, sims = near_dedup(exact_kept, cfg)
    rep.near_removed = len(exact_kept) - len(near_kept)
    rep.near_clusters = len(clusters)
    rep.kept = len(near_kept)
    rep.measured_pair_similarity_min = round(min(sims), 4) if sims else None
    rep.clusters = clusters
    rep.exact_group_sizes = sorted((len(v) for v in exact_groups.values()), reverse=True)
    return near_kept, rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: exact + near-duplicate removal (deterministic).")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=64)
    ap.add_argument("--band-size", type=int, default=8)
    ap.add_argument("--shingle-chars", type=int, default=24)
    ap.add_argument("--report", help="write JSON report here")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.data.clean import load_documents

    docs = list(load_documents(Path(args.inp)))
    cfg = DedupConfig(num_perm=args.num_perm, band_size=args.band_size, threshold=args.threshold, shingle_chars=args.shingle_chars)
    kept, rep = deduplicate(docs, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    report = rep.as_dict()
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "near_clusters_top"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

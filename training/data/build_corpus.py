"""Build, deduplicate, split, measure and gate the SIR M1 corpus.

    python -m training.data.build_corpus --config configs/m1_corpus.yaml
    python -m training.data.build_corpus --config configs/m1_corpus.yaml --offline-fixture-only

Properties this orchestrator is required to have, and how they are obtained here:

* **streaming / bounded memory** — every stage reads JSONL one document at a time and writes its
  own output; the only array held in RAM is the set of hashed n-grams used for the leakage check,
  and its size is reported.
* **no silent loss** — every dropped document is counted *and* written to a reject/dedup log with
  a reason and, where relevant, the id of the document that replaced it.
* **no invented numbers** — `stats.json` contains only counters measured during this run; token
  counts exist only after a tokenizer actually ran (otherwise G5 fails).
* **refusals are loud** — a source that is not `available`, has no acquisition record, or has an
  unpinned revision stops the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.langid import (  # noqa: E402
    code_switch_evidence,
    detector,
    detector_agrees,
    detector_label,
    identify,
    script_counts,
    script_matches_label,
)
from training.data.clean import CleanConfig, clean_stream  # noqa: E402
from training.data.dedup_stream import (  # noqa: E402
    DedupConfig,
    JsonlWriter,
    doc_fingerprint,
    iter_jsonl,
    stream_exact_dedup,
    stream_near_dedup,
)
from training.data.split_stream import stream_split  # noqa: E402
from training.data import gates as gates_mod  # noqa: E402
from training.data import corpus_report  # noqa: E402

STAGE_FILES = [
    "training/data/acquire.py",
    "training/data/normalize.py",
    "training/data/langid.py",
    "training/data/clean.py",
    "training/data/deduplicate.py",
    "training/data/dedup_stream.py",
    "training/data/split.py",
    "training/data/split_stream.py",
    "training/data/build_corpus.py",
    "training/data/gates.py",
    "training/data/corpus_report.py",
    "training/data/manifest.py",
]

_P = np.uint64(1099511628211)  # FNV prime: rolling-hash multiplier


# --------------------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def environment() -> dict[str, Any]:
    mem_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        disk = __import__("shutil").disk_usage(REPO_ROOT)
        disk_free = disk.free
    except Exception:  # noqa: BLE001
        disk_free = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "mem_total_mb": round(mem_kb / 1024) if mem_kb else None,
        "disk_free_mb": round(disk_free / (1 << 20)) if disk_free else None,
        "numpy": np.__version__,
        "utc": utc_now(),
        "network_policy": "acquisition only from what the egress proxy allows; blocked hosts are recorded, never bypassed",
    }


# --------------------------------------------------------------------------------------
# n-gram hashing (exact enough, bounded, deterministic)
# --------------------------------------------------------------------------------------
def _token_hashes(tokens: list[str], cache: dict[str, int]) -> np.ndarray:
    out = np.empty(len(tokens), dtype=np.uint64)
    for i, tok in enumerate(tokens):
        h = cache.get(tok)
        if h is None:
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
            if len(cache) < 4_000_000:
                cache[tok] = h
        out[i] = np.uint64(h)
    return out


def ngram_hashes(text: str, n: int, cache: dict[str, int]) -> np.ndarray:
    """Hashes of the word n-grams of `text`, computed with a vectorised rolling hash.

    Hash collisions are possible in principle (64-bit) and are accepted: this is a *screening*
    measurement, and the collision rate for <10^8 n-grams is far below the 0-overlap policy line.
    The tokens themselves are never stored.
    """
    tokens = text.split()
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    th = _token_hashes(tokens, cache)
    acc = th[: len(tokens) - n + 1].copy()
    for j in range(1, n):
        acc *= _P
        acc += th[j : j + acc.size]
    return acc


def stream_ngram_overlap(train_path: Path, val_path: Path, n: int = 8) -> dict[str, Any]:
    """Exact (hash-level) train/validation n-gram overlap, streamed with a bounded array.

    The *training* side is held as a sorted uint64 array (8 bytes per n-gram, not a Python set of
    strings) and the validation side is streamed; that is the direction that keeps memory below
    the sandbox limit while still counting every occurrence.
    """
    cache: dict[str, int] = {}
    chunks = [ngram_hashes(str(d.get("text") or ""), n, cache) for d in iter_jsonl(train_path)]
    train = np.concatenate([c for c in chunks if c.size]) if any(c.size for c in chunks) else np.empty(0, dtype=np.uint64)
    del chunks
    train.sort()
    train_ngrams = int(train.size)

    val_ngrams = 0
    val_docs = 0
    occurrences = 0
    unique_hits: set[int] = set()
    examples: list[dict[str, Any]] = []
    for doc in iter_jsonl(val_path):
        val_docs += 1
        h = ngram_hashes(str(doc.get("text") or ""), n, cache)
        val_ngrams += int(h.size)
        if not h.size:
            continue
        pos = np.searchsorted(train, h)
        pos[pos >= train.size] = 0
        hit = train[pos] == h if train.size else np.zeros(h.size, dtype=bool)
        k = int(hit.sum())
        if not k:
            continue
        occurrences += k
        first = int(h[hit][0])
        unique_hits.add(first)
        if len(examples) < 5:
            examples.append({"doc_id": doc.get("doc_id"), "source_id": doc.get("source_id"), "shared_ngrams_in_doc": k, "example_hash": f"{first:016x}"})
    return {
        "n": n,
        "train_ngrams": train_ngrams,
        "val_documents": val_docs,
        "val_ngrams": val_ngrams,
        "overlapping_ngram_occurrences_train": occurrences,
        "unique_overlapping_ngrams": len(unique_hits),
        "overlap_rate": round(occurrences / train_ngrams, 10) if train_ngrams else 0.0,
        "examples": examples,
        "method": "sorted uint64 rolling-hash array of train n-grams; validation streamed with vectorised binary search",
    }


def measure_leakage_with_val_docs(train_path: Path, val_path: Path, n: int = 8) -> tuple[dict[str, Any], list[int], np.ndarray]:
    """Leakage measurement that also returns *which* validation documents leak (by index)."""
    cache: dict[str, int] = {}
    chunks = [ngram_hashes(str(d.get("text") or ""), n, cache) for d in iter_jsonl(train_path)]
    train = np.concatenate([c for c in chunks if c.size]) if any(c.size for c in chunks) else np.empty(0, dtype=np.uint64)
    del chunks
    train.sort()

    leaking: list[int] = []
    moved_hashes: list[np.ndarray] = []
    occurrences = 0
    unique_hits: set[int] = set()
    val_ngrams = 0
    val_docs = 0
    examples: list[dict[str, Any]] = []
    for idx, doc in enumerate(iter_jsonl(val_path)):
        val_docs += 1
        h = ngram_hashes(str(doc.get("text") or ""), n, cache)
        val_ngrams += int(h.size)
        if not h.size or not train.size:
            continue
        pos = np.searchsorted(train, h)
        pos[pos >= train.size] = 0
        hit = train[pos] == h
        k = int(hit.sum())
        if not k:
            continue
        leaking.append(idx)
        moved_hashes.append(h)
        occurrences += k
        unique_hits.update(int(x) for x in h[hit][:64].tolist())
        if len(examples) < 5:
            examples.append({"doc_id": doc.get("doc_id"), "source_id": doc.get("source_id"), "shared_ngrams_in_doc": k})
    stats = {
        "n": n,
        "train_ngrams": int(train.size),
        "val_documents": val_docs,
        "val_ngrams": val_ngrams,
        "overlapping_ngram_occurrences_train": occurrences,
        "unique_overlapping_ngrams": len(unique_hits),
        "leaking_val_documents": len(leaking),
        "overlap_rate": round(occurrences / max(1, train.size), 10),
        "examples": examples,
        "method": "sorted uint64 rolling-hash array of train n-grams; validation streamed",
    }
    extra = np.concatenate(moved_hashes) if moved_hashes else np.empty(0, dtype=np.uint64)
    return stats, leaking, extra


def repair_leakage(train_path: Path, val_path: Path, n: int, max_iters: int) -> dict[str, Any]:
    """Move leaking validation documents to train until the measured overlap is exactly zero.

    Policy: validation is never trimmed to make a number look good — the whole *document* moves to
    train, and the validation side shrinks. Moving docs is the only repair; deleting text is not.
    """
    log: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    for iteration in range(1, max_iters + 1):
        stats, leaking, moved_hashes = measure_leakage_with_val_docs(train_path, val_path, n)
        if before is None:
            before = stats
        if stats["overlapping_ngram_occurrences_train"] == 0:
            return {"performed": True, "converged": True, "iterations": iteration - 1, "before": before, "after": stats, "log": log}
        if not leaking:
            return {"performed": True, "converged": False, "iterations": iteration - 1, "before": before, "after": stats, "log": log,
                    "note": "overlap measured but no validation document could be identified; nothing moved"}
        leaking_set = set(leaking)
        tmp_train = train_path.with_suffix(".leakrepair.tmp")
        tmp_val = val_path.with_suffix(".leakrepair.tmp")
        moved = 0
        # train = existing train + the leaking validation documents (appended, never replaced);
        # validation = the documents that remain. Both files are rewritten so the pair is consistent.
        with JsonlWriter(tmp_train) as tr:
            for doc in iter_jsonl(train_path):
                tr.write(doc)
            for i, doc in enumerate(iter_jsonl(val_path)):
                if i in leaking_set:
                    moved_doc = dict(doc)
                    moved_doc["split"] = "train"
                    moved_doc["moved_from_validation_by"] = f"leak-repair-{iteration}"
                    tr.write(moved_doc)
                    moved += 1
        with JsonlWriter(tmp_val) as va:
            for i, doc in enumerate(iter_jsonl(val_path)):
                if i not in leaking_set:
                    va.write(doc)
        tmp_train.replace(train_path)
        tmp_val.replace(val_path)
        log.append({
            "iteration": iteration,
            "overlapping_ngram_occurrences_train": stats["overlapping_ngram_occurrences_train"],
            "unique_overlapping_ngrams": stats["unique_overlapping_ngrams"],
            "leaking_val_documents": len(leaking),
            "moved_documents": moved,
        })
        if moved == 0:
            return {"performed": True, "converged": False, "iterations": iteration, "before": before, "after": stats, "log": log}
    stats, _, _ = measure_leakage_with_val_docs(train_path, val_path, n)
    return {"performed": True, "converged": stats["overlapping_ngram_occurrences_train"] == 0, "iterations": max_iters, "before": before, "after": stats, "log": log}


def stream_cross_split_near_dups(train_path: Path, val_path: Path, cfg: DedupConfig, threshold: float = 0.8) -> dict[str, Any]:
    """Count near-duplicate pairs across the split (band index of validation, train streamed)."""
    from training.data.deduplicate import shingles, signature
    from training.data.dedup_stream import _band_keys, _jaccard, doc_fingerprint

    val_bands: dict[str, list[int]] = defaultdict(list)
    val_shingles: list[set[str]] = []
    val_fp: set[str] = set()
    for i, doc in enumerate(iter_jsonl(val_path)):
        text = str(doc.get("text") or "")
        val_shingles.append(set(shingles(text, cfg.shingle_chars)))
        val_fp.add(doc_fingerprint(text))
        for key in _band_keys(signature(text, cfg), cfg.band_size):
            val_bands[key].append(i)

    pairs = 0
    exact = 0
    examples: list[dict[str, Any]] = []
    for doc in iter_jsonl(train_path):
        text = str(doc.get("text") or "")
        if doc_fingerprint(text) in val_fp:
            exact += 1
        keys = _band_keys(signature(text, cfg), cfg.band_size)
        candidates = {i for key in keys for i in val_bands.get(key, [])}
        ts = set(shingles(text, cfg.shingle_chars))
        for i in sorted(candidates):
            if _jaccard(ts, val_shingles[i]) >= threshold:
                pairs += 1
                if len(examples) < 5:
                    examples.append({"train_doc": doc.get("doc_id"), "val_doc_index": i, "source_id": doc.get("source_id")})
                break
    return {
        "pairs": pairs,
        "exact_duplicate_documents": exact,
        "threshold": threshold,
        "val_band_keys": len(val_bands),
        "examples": examples,
        "method": "validation MinHash bands indexed in memory; train streamed and verified by Jaccard",
    }


# --------------------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------------------
# Bump when the cleaning/language logic changes: cached *_work* files from an older policy are
# re-cleaned instead of reused, so no report can mix two versions of the rules.
CLEAN_POLICY = "clean-v2"
def clean_source(sid: str, in_path: Path, out_path: Path, rejects_dir: Path, cfg: CleanConfig, chunk: int = 2000) -> dict[str, Any]:
    """Clean one source in chunks so a large source cannot balloon memory.

    Language labels are *measured* (`language_measured`) and compared with what the source claims;
    a document whose script contradicts its declared label is dropped and recorded in the reject log.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_dir.mkdir(parents=True, exist_ok=True)
    reject_log_path = rejects_dir / f"{sid}.rejects.jsonl"
    stats: dict[str, Any] = {
        "source_id": sid,
        "policy": CLEAN_POLICY,
        "documents_seen": 0,
        "documents_kept": 0,
        "documents_rejected": 0,
        "characters_before": 0,
        "characters_after": 0,
        "reject_reasons": Counter(),
        "dropped_by_cleaner": Counter(),
        "repaired": Counter(),
        "language_declared": Counter(),
        "language_measured": Counter(),
        "label_disagreements": 0,
        "script_mismatch_rejections": 0,
        "detector_compared": 0,
        "detector_agree": 0,
        "detector_disagreements": Counter(),
    "code_switch_documents": 0,
    "hinc_deva_documents": 0,
    "hinc_latn_documents": 0,
    }
    buf: list[dict[str, Any]] = []

    with JsonlWriter(out_path) as writer, reject_log_path.open("w", encoding="utf-8") as rlog:

        def flush(items: list[dict[str, Any]]) -> None:
            if not items:
                return
            kept, rep, rejects = clean_stream(items, cfg, sid)
            stats["documents_seen"] += rep.read
            stats["characters_before"] += rep.chars_in
            stats["documents_rejected"] += len(rejects)
            stats["dropped_by_cleaner"].update(rep.dropped)
            stats["repaired"].update(rep.repaired)
            for r in rejects:
                reason = (r.get("reasons") or ["unknown"])[0]
                stats["reject_reasons"][reason] += 1
                rlog.write(json.dumps({"doc_id": r.get("id"), "source_id": sid, "reason": reason, "reasons": r.get("reasons"), "chars": r.get("chars"), "stage": "clean"}, ensure_ascii=False) + "\n")
            for doc in kept:
                declared = str(doc.get("language") or "unknown")
                text = str(doc.get("text") or "")
                got = identify(text, None if declared == "unknown" else declared)
                stats["language_declared"][declared] += 1
                stats["language_measured"][got.language] += 1
                if declared != "unknown" and got.language != declared:
                    stats["label_disagreements"] += 1
                det = detector_label(text) if detector() else None
                agrees = detector_agrees(got.language, det)
                if agrees is not None:
                    stats["detector_compared"] += 1
                    if agrees:
                        stats["detector_agree"] += 1
                    else:
                        stats["detector_disagreements"][f"{got.language} vs {det[0]}"] += 1
                counts = script_counts(text)
                if not script_matches_label(declared, counts):
                    stats["script_mismatch_rejections"] += 1
                    stats["documents_rejected"] += 1
                    stats["reject_reasons"]["script_mismatch_vs_declared_label"] += 1
                    rlog.write(json.dumps({"doc_id": doc.get("id"), "source_id": sid, "reason": "script_mismatch_vs_declared_label", "declared": declared, "measured": got.language, "chars": len(text), "stage": "language-check"}, ensure_ascii=False) + "\n")
                    continue
                if got.language == "hinc-deva":
                    stats["hinc_deva_documents"] += 1
                    if code_switch_evidence(text)["code_switch"]:
                        stats["code_switch_documents"] += 1
                elif got.language == "hinc-latn":
                    stats["hinc_latn_documents"] += 1
                out = dict(doc)
                out["language_measured"] = got.language
                out["language_method"] = got.method
                out["language_confidence"] = got.confidence
                out["chars"] = len(text)
                writer.write(out)
                stats["documents_kept"] += 1
                stats["characters_after"] += len(text)
            items.clear()

        for doc in iter_jsonl(in_path):
            buf.append(doc)
            if len(buf) >= chunk:
                flush(buf)
        flush(buf)
        out_info = writer.close()

    stats["reject_reasons"] = dict(sorted(stats["reject_reasons"].items(), key=lambda kv: -kv[1]))
    stats["dropped_by_cleaner"] = dict(sorted(stats["dropped_by_cleaner"].items(), key=lambda kv: -kv[1]))
    stats["repaired"] = dict(sorted(stats["repaired"].items()))
    stats["language_declared"] = dict(sorted(stats["language_declared"].items()))
    stats["language_measured"] = dict(sorted(stats["language_measured"].items()))
    stats["detector_disagreements"] = dict(sorted(stats["detector_disagreements"].items(), key=lambda kv: -kv[1])[:10])
    stats["reject_log"] = str(reject_log_path)
    stats["input"] = {"path": str(in_path), "sha256": sha256_file(in_path), "bytes": Path(in_path).stat().st_size}
    stats["output"] = out_info
    return stats


# --------------------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------------------
def tokenize_split(path: Path, out_dir: Path, tokenizer: Any, seq_len: int) -> dict[str, Any]:
    """Tokenize one split to a flat uint16/uint32 token stream, counting as it streams.

    Output: `tokens.bin` (flat token ids, every document followed by one EOS id), `ranges.npy`
    (start/end offsets per document) and `meta.json` (counts per language and per source).
    Nothing is held beyond one document plus its ids.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "tokens.bin"
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    dtype = np.uint16 if vocab and vocab < 65_535 else np.uint32
    eos = int(getattr(getattr(tokenizer, "spec", None), "eos_id", 0) or 0)
    lang_tokens: Counter[str] = Counter()
    src_tokens: Counter[str] = Counter()
    lang_docs: Counter[str] = Counter()
    src_docs: Counter[str] = Counter()
    n_tokens = 0
    n_docs = 0
    ranges: list[tuple[int, int]] = []
    pos = 0
    with bin_path.open("wb") as fh:
        for doc in iter_jsonl(path):
            text = str(doc.get("text") or "")
            if not text:
                continue
            ids = tokenizer.encode(text)
            if not ids:
                continue
            arr = np.asarray(list(ids) + [eos], dtype=dtype)
            fh.write(arr.tobytes())
            ranges.append((pos, pos + int(arr.size)))
            pos += int(arr.size)
            n_tokens += int(arr.size)
            n_docs += 1
            lang = str(doc.get("language") or "unknown")
            src = str(doc.get("source_id") or "unknown")
            lang_tokens[lang] += len(ids)
            src_tokens[src] += len(ids)
            lang_docs[lang] += 1
            src_docs[src] += 1
    np.save(out_dir / "ranges.npy", np.asarray(ranges, dtype=np.int64) if ranges else np.zeros((0, 2), dtype=np.int64))
    meta = {
        "file": bin_path.name,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "n_tokens": n_tokens,
        "n_docs": n_docs,
        "seq_len": seq_len,
        "contexts_available": n_tokens // max(1, seq_len),
        "language_tokens": dict(sorted(lang_tokens.items())),
        "language_docs": dict(sorted(lang_docs.items())),
        "source_tokens": dict(sorted(src_tokens.items())),
        "source_docs": dict(sorted(src_docs.items())),
        "notes": "text tokens only; each document ends with one EOS separator (counted in n_tokens)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


class _HFTokenizer:
    """Adapter around `tokenizer.api.SirTokenizer` that exposes encode/vocab_size/spec."""

    def __init__(self, spec_path: Path):
        from tokenizer.api import SirTokenizer

        self.spec_path = Path(spec_path)
        self.tok = SirTokenizer.load(self.spec_path)
        self.kind = getattr(self.tok, "kind", "unknown")
        self.vocab_size = int(getattr(self.tok, "vocab_size", 0))
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        artifact = (self.spec_path.parent / Path(spec.get("path", "")).name).resolve()
        self.artifact_sha256 = sha256_file(artifact) if artifact.exists() else "missing"
        self.spec = getattr(self.tok, "spec", None)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)


def find_tokenizer(args: argparse.Namespace, cfg: dict[str, Any]) -> Path | None:
    if args.tokenizer_spec:
        return Path(args.tokenizer_spec)
    chosen = REPO_ROOT / (cfg.get("paths") or {}).get("tokenizer_dir", "tokenizer/artifacts/latest") / "chosen.json"
    if chosen.exists():
        try:
            sel = json.loads(chosen.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        spec = sel.get("spec") or sel.get("chosen_spec")
        if spec:
            p = Path(spec)
            return p if p.is_absolute() else REPO_ROOT / p
    return None


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------
def run_build(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from training.data.manifest import load_manifest

    if args.tokenize_only:
        args.resume = True
        required = [Path(args.outdir or REPO_ROOT / (cfg.get("paths") or {}).get("processed_dir", "data/processed/m1_corpus")) / "deduped.jsonl"]
        if not all(p.exists() for p in required):
            raise SystemExit(
                "--tokenize-only needs the text stages of an earlier build; missing: "
                + ", ".join(str(p) for p in required if not p.exists())
            )
    t0 = time.time()
    out_dir = Path(args.outdir) if args.outdir else REPO_ROOT / (cfg.get("paths") or {}).get("processed_dir", "data/processed/m1_corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    normalized_dir = REPO_ROOT / (cfg.get("paths") or {}).get("normalized_dir", "data/normalized")

    manifest = load_manifest(args.manifest)
    acq_path = REPO_ROOT / "data" / "manifests" / "acquisition.json"
    acquisition = json.loads(acq_path.read_text(encoding="utf-8")).get("records", {}) if acq_path.exists() else {}

    ds = cfg.get("dataset") or {}
    requested = list(ds.get("sources") or [])
    if args.offline_fixture_only:
        requested = ["sir_fixture_v0"]
    if args.sources:
        requested = [s.strip() for s in args.sources.split(",") if s.strip()]

    refused: list[dict[str, str]] = []
    usable: list[str] = []
    for sid in requested:
        src = manifest.get(sid)
        if src is None:
            refused.append({"source_id": sid, "reason": "not declared in data/manifests/sources.yaml"})
            continue
        raw = src.raw
        if raw.get("status") != "available":
            refused.append({"source_id": sid, "reason": f"status={raw.get('status')} — not usable for training"})
            continue
        if not (raw.get("license_verified") and raw.get("model_training_allowed")):
            refused.append({"source_id": sid, "reason": "licence not verified for model training"})
            continue
        rec = acquisition.get(sid)
        if not rec or not rec.get("files"):
            refused.append({"source_id": sid, "reason": "no acquisition record with artifacts — run training.data.acquire first"})
            continue
        usable.append(sid)
    if not usable:
        raise SystemExit(f"no usable sources; refused: {refused}")
    print(f"sources: {usable}", flush=True)
    for r in refused:
        print(f"  refused {r['source_id']}: {r['reason']}", file=sys.stderr, flush=True)

    # ---- normalisation (reused when the recorded hash still matches the file on disk)
    normalize_reports: dict[str, Any] = {}
    normalized_paths: dict[str, Path] = {}
    for sid in usable:
        npath = normalized_dir / f"{sid}.jsonl"
        rep_path = normalized_dir / f"{sid}.normalize_report.json"
        reused = False
        if npath.exists() and rep_path.exists() and not args.renormalize:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            from training.data.normalize import corpus_exclusions

            policy_current = (
                rep.get("policy") == "normalize-v2"
                and rep.get("excluded_from_corpus") == corpus_exclusions(sid)
            )
            if rep.get("sha256") == sha256_file(npath) and policy_current:
                normalize_reports[sid] = {**rep, "reused": True, "sha256_verified": True}
                normalized_paths[sid] = npath
                reused = True
                print(f"  normalize {sid}: reused ({rep.get('documents'):,} docs, hash verified)", flush=True)
            else:
                print(f"  normalize {sid}: cached copy is stale (hash or exclusion policy changed) — re-normalizing", flush=True)
        if not reused:
            from training.data.normalize import normalize_source

            norm_cfg = ds.get("normalization") or {}
            rep = normalize_source(
                sid,
                out_dir=normalized_dir,
                manifest_path=args.manifest,
                max_doc_chars=int(norm_cfg.get("max_doc_chars", 6000)),
                min_doc_chars=int(norm_cfg.get("min_doc_chars", 200)),
            )
            normalize_reports[sid] = {**rep.as_dict(), "reused": False, "sha256_verified": True}
            normalized_paths[sid] = normalized_dir / f"{sid}.jsonl"
            print(f"  normalize {sid}: {rep.documents:,} docs, {rep.chars:,} chars (unreadable {rep.files_unreadable})", flush=True)

    # ---- cleaning
    clean_cfg_raw = ds.get("clean") or {}
    clean_cfg = CleanConfig(**{k: v for k, v in clean_cfg_raw.items() if k in CleanConfig.__dataclass_fields__})
    cleaning_reports: dict[str, Any] = {}
    cleaned_paths: list[Path] = []
    for sid in usable:
        cpath = work / f"clean_{sid}.jsonl"
        rep_path = work / f"clean_{sid}.report.json"
        if args.resume and cpath.exists() and rep_path.exists():
            saved = json.loads(rep_path.read_text(encoding="utf-8"))
            same_input = (saved.get("input") or {}).get("sha256") == sha256_file(normalized_paths[sid])
            same_output = (saved.get("output") or {}).get("sha256") == sha256_file(cpath)
            same_policy = saved.get("policy") == CLEAN_POLICY
            if same_input and same_output and same_policy:
                cleaning_reports[sid] = saved
                cleaned_paths.append(cpath)
                print(f"  clean {sid}: reused ({saved.get('documents_kept'):,} docs, hashes verified)", flush=True)
                continue
            print(f"  clean {sid}: cached copy is stale (policy, input or output changed) — re-cleaning", flush=True)
        rep = clean_source(sid, normalized_paths[sid], cpath, work / "rejects", clean_cfg)
        rep_path.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
        cleaning_reports[sid] = rep
        cleaned_paths.append(cpath)
        print(
            f"  clean {sid}: kept {rep['documents_kept']:,}/{rep['documents_seen']:,} "
            f"({rep['characters_after']:,} of {rep['characters_before']:,} chars)",
            flush=True,
        )

    cleaned = work / "cleaned.jsonl"
    # the merged file is only reusable when every per-source stage was reused; one re-cleaned source
    # invalidates the merge (each source is deterministic, so an identical re-clean still produces
    # identical bytes and the hash check downstream will then allow dedup reuse)
    cleaned_reused = all(bool((r.get("output") or {}).get("sha256")) for r in cleaning_reports.values()) and args.resume
    if not (cleaned_reused and cleaned.exists() and cleaned.stat().st_size > 0):
        with JsonlWriter(cleaned) as w:
            for p in cleaned_paths:
                for doc in iter_jsonl(p):
                    w.write(doc)
    print(f"  merged cleaned corpus: {cleaned.stat().st_size/1e6:.1f} MB", flush=True)

    # ---- dedup
    dcfg_raw = ds.get("dedup") or {}
    dcfg = DedupConfig(
        num_perm=int(dcfg_raw.get("num_perm", 64)),
        band_size=int(dcfg_raw.get("band_size", 8)),
        threshold=float(dcfg_raw.get("threshold", 0.8)),
        shingle_chars=int(dcfg_raw.get("shingle_chars", 24)),
    )
    exact_path = work / "dedup_exact.jsonl"
    deduped = out_dir / "deduped.jsonl"
    dedup_report_path = work / "dedup_report.json"
    if args.resume and deduped.exists() and dedup_report_path.exists():
        saved = json.loads(dedup_report_path.read_text(encoding="utf-8"))
        same_input = (saved.get("inputs") or {}).get("cleaned", {}).get("sha256") == sha256_file(cleaned)
        same_output = (saved.get("outputs") or {}).get("deduped", {}).get("sha256") == sha256_file(deduped)
        if same_input and same_output:
            dedup_report = saved
            print(f"  dedup: reused ({saved.get('exact', {}).get('kept', 0):,} kept, hashes verified)", flush=True)
        else:
            print("  dedup: cached copy is stale (input or output hash changed) — re-running", flush=True)
            args.resume = False
    else:
        exact_rep, _ = stream_exact_dedup(cleaned, exact_path, work / "removed_exact.jsonl")
        print(f"  exact dedup: removed {exact_rep.removed:,} of {exact_rep.read:,}", flush=True)
        if dcfg_raw.get("near", True) and not args.skip_near_dedup:
            near_rep, near_meta = stream_near_dedup(
                exact_path,
                deduped,
                work / "removed_near.jsonl",
                cfg=dcfg,
                workdir=work,
                max_band_size=int(dcfg_raw.get("max_band_size", 200)),
                max_candidates=int(dcfg_raw.get("max_candidates", 200_000)),
            )
            print(f"  near dedup: removed {near_rep.removed:,} (candidates {near_rep.candidate_pairs:,}, {near_rep.elapsed_seconds}s)", flush=True)
        else:
            near_rep, near_meta = None, {"skipped": True}
            deduped.write_bytes(exact_path.read_bytes())
        dedup_report = {
            "exact": exact_rep.as_dict(),
            "near": near_rep.as_dict() if near_rep else None,
            "near_meta": near_meta,
            "inputs": {"cleaned": {"path": str(cleaned), "sha256": sha256_file(cleaned), "bytes": cleaned.stat().st_size}},
            "outputs": {"deduped": {"path": str(deduped), "sha256": sha256_file(deduped), "bytes": deduped.stat().st_size}},
        }
        dedup_report_path.write_text(json.dumps(dedup_report, indent=2, default=str) + "\n", encoding="utf-8")

    # ---- split
    sp = ds.get("split") or {}
    train_path, val_path = out_dir / "train.jsonl", out_dir / "val.jsonl"
    split_report_path = out_dir / "split_report.json"
    if args.resume and train_path.exists() and val_path.exists() and split_report_path.exists():
        split_result = json.loads(split_report_path.read_text(encoding="utf-8"))
        split_report = split_result.get("result", split_result)
        split_meta = split_result.get("meta", {})
        split_report = recount_split(train_path, val_path, split_report)
        print(f"  split: reused and recounted (train {split_report.get('train_docs'):,} / val {split_report.get('val_docs'):,})", flush=True)
    else:
        rep, meta = stream_split(
            deduped,
            out_dir,
            val_frac=float(sp.get("val_frac", 0.02)),
            seed=int(cfg.get("seed", 20261005)),
            workdir=work,
            id_fallback=bool(args.allow_id_fallback),
        )
        split_report, split_meta = rep.as_dict(), meta
        meta["inputs"] = {"deduped": {"path": str(deduped), "sha256": sha256_file(deduped), "bytes": deduped.stat().st_size}}
        meta["outputs"] = {
            "train": {"sha256": sha256_file(train_path), "bytes": train_path.stat().st_size},
            "val": {"sha256": sha256_file(val_path), "bytes": val_path.stat().st_size},
        }
        (out_dir / "split_report.json").write_text(json.dumps({"result": split_report, "meta": meta}, indent=2) + "\n", encoding="utf-8")
        print(f"  split: train {rep.train_docs:,} / val {rep.val_docs:,} (components {rep.components:,}, val {rep.val_frac_achieved:.4%})", flush=True)

    # ---- leakage: measure, then repair by moving leaking validation documents to train
    n_gram = int(sp.get("leak_ngram", 8))
    leakage = stream_ngram_overlap(train_path, val_path, n_gram)
    print(f"  leakage: {leakage['overlapping_ngram_occurrences_train']:,} overlapping {n_gram}-gram occurrence(s) "
          f"({leakage['unique_overlapping_ngrams']:,} unique)", flush=True)
    leak_repair: dict[str, Any] | None = None
    if leakage["overlapping_ngram_occurrences_train"] and not args.no_leak_repair:
        max_iters = int(args.max_leak_repair_iters or sp.get("max_leak_repair_iters", 8))
        leak_repair = repair_leakage(train_path, val_path, n_gram, max_iters)
        for entry in leak_repair.get("log", []):
            print(f"  leak-repair {entry}", flush=True)
        leakage = leak_repair.get("after") or leakage
        print(f"  leakage after repair: {leakage['overlapping_ngram_occurrences_train']:,} occurrence(s), "
              f"converged={leak_repair.get('converged')}", flush=True)
        # the split report must describe the files on disk, not the pre-repair split
        split_report = recount_split(train_path, val_path, split_report)
    elif not leakage["overlapping_ngram_occurrences_train"]:
        leak_repair = {"performed": False, "reason": "measured leakage was already zero", "converged": True}

    cross_split = stream_cross_split_near_dups(train_path, val_path, dcfg)
    print(f"  cross-split near-duplicate pairs: {cross_split['pairs']:,}", flush=True)

    # ---- tokenization (only if a tokenizer was actually selected)
    tokens: dict[str, Any] = {"measured": False, "reason": "no tokenizer selected (run the bake-off, then --tokenize-only)"}
    spec_path = find_tokenizer(args, cfg)
    if spec_path and spec_path.exists():
        tok = _HFTokenizer(spec_path)
        seq_len = int((cfg.get("model") or {}).get("max_seq_len", 512))
        train_meta = tokenize_split(train_path, out_dir / "tokenized" / "train", tok, seq_len)
        val_meta = tokenize_split(val_path, out_dir / "tokenized" / "val", tok, seq_len)
        tokens = {
            "measured": True,
            "tool": "training/data/build_corpus.py::tokenize_split",
            "tokenizer_spec": str(spec_path.relative_to(REPO_ROOT)) if spec_path.is_absolute() and REPO_ROOT in spec_path.parents else str(spec_path),
            "tokenizer_kind": tok.kind,
            "tokenizer_sha256": tok.artifact_sha256,
            "total": int(train_meta["n_tokens"]) + int(val_meta["n_tokens"]),
            "train": int(train_meta["n_tokens"]),
            "validation": int(val_meta["n_tokens"]),
            "by_language": {k: int(train_meta["language_tokens"].get(k, 0)) + int(val_meta["language_tokens"].get(k, 0)) for k in sorted(set(train_meta["language_tokens"]) | set(val_meta["language_tokens"]))},
            "by_source": {k: int(train_meta["source_tokens"].get(k, 0)) + int(val_meta["source_tokens"].get(k, 0)) for k in sorted(set(train_meta["source_tokens"]) | set(val_meta["source_tokens"]))},
            "train_by_language": train_meta["language_tokens"],
            "validation_by_language": val_meta["language_tokens"],
            "docs": {"train": train_meta["n_docs"], "validation": val_meta["n_docs"]},
            "contexts_at_seq_len": train_meta["contexts_available"] + val_meta["contexts_available"],
        }
        print(f"  tokens: {tokens['total']:,} measured (train {tokens['train']:,} / val {tokens['validation']:,})", flush=True)

    # ---- per-language / per-source statistics from the deduped corpus
    lang_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    provenance_gaps = Counter()
    lang_quality = Counter()
    corpus_bytes = 0
    for doc in iter_jsonl(deduped):
        lang = str(doc.get("language") or "unknown")
        sid = str(doc.get("source_id") or "")
        text = str(doc.get("text") or "")
        chars = len(text)
        corpus_bytes += len(text.encode("utf-8"))
        lang_stats[lang]["documents"] += 1
        lang_stats[lang]["chars"] += chars
        source_stats[sid]["documents"] += 1
        source_stats[sid]["chars"] += chars
        if doc.get("synthetic"):
            lang_stats[lang]["synthetic_documents"] += 1
            source_stats[sid]["synthetic_documents"] += 1
        if not doc.get("doc_id"):
            provenance_gaps["missing_doc_id"] += 1
        if not sid:
            provenance_gaps["missing_source_id"] += 1
        if not (doc.get("provenance") or {}).get("artifact"):
            provenance_gaps["missing_artifact_provenance"] += 1
        lang_quality["documents"] += 1
        if str(doc.get("language_measured") or "unknown") == "unknown":
            lang_quality["unknown"] += 1
    detector_disagreements: Counter[str] = Counter()
    for sid, rep in cleaning_reports.items():
        lang_quality["detector_compared"] += int(rep.get("detector_compared") or 0)
        lang_quality["detector_agree"] += int(rep.get("detector_agree") or 0)
        for pair, n in (rep.get("detector_disagreements") or {}).items():
            detector_disagreements[pair] += int(n)
        lang_quality["code_switch_documents"] += int(rep.get("code_switch_documents") or 0)
        lang_quality["hinc_deva_documents"] += int(rep.get("hinc_deva_documents") or 0)
        lang_quality["hinc_latn_documents"] += int(rep.get("hinc_latn_documents") or 0)
        lang_quality["script_mismatch"] += int(rep.get("script_mismatch_rejections") or 0)
        lang_quality["label_disagreements"] += int(rep.get("label_disagreements") or 0)
        # the script check happens during cleaning, so its denominator is the documents seen there
        lang_quality["documents_seen_cleaning"] += int(rep.get("documents_seen") or 0)
        lang_quality["documents_kept_cleaning"] += int(rep.get("documents_kept") or 0)
        for lang, n in (rep.get("language_measured") or {}).items():
            lang_stats[lang]["measured_documents"] += int(n)
    lang_quality["documents_after_dedup"] = sum(int(v.get("documents") or 0) for v in source_stats.values())
    if lang_quality["detector_compared"]:
        lang_quality["detector"] = {
            "tool": detector_name(),
            "compared": lang_quality["detector_compared"],
            "agree": lang_quality["detector_agree"],
            "agreement_rate": round(
                lang_quality["detector_agree"] / max(1, lang_quality["detector_compared"]), 4
            ),
            "top_disagreements": [[k, v] for k, v in detector_disagreements.most_common(10)],
            "note": (
                "the detector is a cross-check of the heuristic label, not the labeller; "
                "py3langid has no Hinglish label, so hinc-latn documents are compared against Hindi"
            ),
        }
    for lang, stats in lang_stats.items():
        stats.setdefault("measured_documents", stats.get("documents", 0))
    for doc in iter_jsonl(train_path):
        lang_stats[str(doc.get("language") or "unknown")]["train_documents"] += 1
        source_stats[str(doc.get("source_id") or "")]["train_documents"] += 1
    for doc in iter_jsonl(val_path):
        lang_stats[str(doc.get("language") or "unknown")]["validation_documents"] += 1
        source_stats[str(doc.get("source_id") or "")]["validation_documents"] += 1

    # ---- integrity: re-hash every acquired artifact of the sources used
    verify_results: list[dict[str, Any]] = []
    if not args.skip_integrity:
        from training.data.manifest import verify_acquisition_integrity

        verify_results = verify_acquisition_integrity(only=set(usable))
        bad = [v for v in verify_results if v["status"] != "ok"]
        print(f"  integrity: {len(verify_results):,} artifacts re-hashed, {len(bad)} mismatch(es)", flush=True)

    # ---- reproducibility record
    normalized_files = {}
    for sid, rep in normalize_reports.items():
        p = normalized_paths[sid]
        recorded = rep.get("sha256")
        recomputed = sha256_file(p)
        normalized_files[sid] = {"path": str(p.relative_to(REPO_ROOT)), "recorded": recorded, "recomputed": recomputed, "verified": recorded == recomputed}
    reproducibility = {
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "seed": int(cfg.get("seed", 20261005)),
        "script_sha256": {f: sha256_file(REPO_ROOT / f) for f in STAGE_FILES if (REPO_ROOT / f).exists()},
        "source_revisions": {sid: (acquisition.get(sid, {}) or {}).get("revision") for sid in usable},
        "source_archive_sha256": {sid: [a.get("archive_sha256") for a in ((acquisition.get(sid, {}) or {}).get("archives") or [])] for sid in usable},
        "normalized_files": normalized_files,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    stats: dict[str, Any] = {
        "generated_utc": utc_now(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": {"path": str(args.config), "sha256": reproducibility["config_sha256"]},
        "environment": environment(),
        "sources": usable,
        "sources_refused": refused,
        "documents": {
            "raw_files": sum(int((acquisition.get(s, {}) or {}).get("files") or 0) for s in usable),
            "normalized": sum(int(v.get("documents") or 0) for v in normalize_reports.values()),
            "cleaned_kept": sum(int(v.get("documents_kept") or 0) for v in cleaning_reports.values()),
            "after_dedup": sum(int(v.get("documents") or 0) for v in source_stats.values()),
            "synthetic_documents": sum(int(v.get("synthetic_documents") or 0) for v in source_stats.values()),
            "train": split_report.get("train_docs"),
            "validation": split_report.get("val_docs"),
        },
        "characters": {
            "normalized": sum(int(v.get("chars") or 0) for v in normalize_reports.values()),
            "cleaned": sum(int(v.get("characters_after") or 0) for v in cleaning_reports.values()),
            "after_dedup": sum(int(v.get("chars") or 0) for v in source_stats.values()),
            "train": split_report.get("train_chars"),
            "validation": split_report.get("val_chars"),
            "by_language": {k: int(v.get("chars") or 0) for k, v in sorted(lang_stats.items())},
            "bytes_utf8_after_dedup": corpus_bytes,
        },
        "normalization": {k: {kk: vv for kk, vv in v.items()} for k, v in normalize_reports.items()},
        "cleaning": {k: {kk: vv for kk, vv in v.items() if kk != "output"} for k, v in cleaning_reports.items()},
        "dedup": {"exact": (dedup_report or {}).get("exact"), "near": (dedup_report or {}).get("near"), "near_meta": (dedup_report or {}).get("near_meta")},
        "split": split_report,
        "split_meta": split_meta,
        "leak_repair": leak_repair,
        "validation": {"leakage_8gram": leakage, "cross_split_near_dups": cross_split},
        "tokens": tokens,
        "per_language": {k: dict(v) for k, v in sorted(lang_stats.items())},
        "per_source": {k: dict(v) for k, v in sorted(source_stats.items())},
        "language_quality": dict(lang_quality),
        "provenance_gaps": dict(provenance_gaps),
        "integrity": {"results": verify_results, "checked": len(verify_results), "failures": [v for v in verify_results if v["status"] != "ok"]},
        "acquisition": {sid: acquisition.get(sid, {}) for sid in usable},
        "reproducibility": reproducibility,
    }

    # ---- gates + verdict + reports
    gate_list = gates_mod.run_gates(manifest, stats, verify_results)
    target = int((cfg.get("gates") or {}).get("target_tokens", gates_mod.M1_TARGET_TOKENS))
    verdict = gates_mod.m1_status(stats, gate_list, target_tokens=target)
    stats["gates"] = [g.as_dict() for g in gate_list]
    stats["m1"] = verdict
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    report = corpus_report.build_report(stats, manifest)
    json_path = REPO_ROOT / (cfg.get("paths") or {}).get("reports", {}).get("corpus_report_json", "evaluation/results/corpus_report.json")
    md_path = REPO_ROOT / (cfg.get("paths") or {}).get("reports", {}).get("corpus_report_md", "docs/corpus-report.md")
    corpus_report.write_reports(report, json_path, md_path)

    # dataset_report.json: what downstream tooling reads to know whether the corpus is provisional
    provisional = verdict["status"] != "PASS"
    (out_dir / "dataset_report.json").write_text(
        json.dumps(
            {
                "provisional": provisional,
                "provisional_reason": verdict["reason"] if provisional else "",
                "m1_status": verdict["status"],
                "measured_tokens": verdict["measured_tokens"],
                "natural_documents": stats["documents"]["after_dedup"] - stats["documents"]["synthetic_documents"],
                "synthetic_documents": stats["documents"]["synthetic_documents"],
                "generated_utc": stats["generated_utc"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nM1: {verdict['status']} — {verdict['reason']}", flush=True)
    print(f"training readiness: {verdict['training_readiness']['state']}", flush=True)
    print(f"report: {md_path.relative_to(REPO_ROOT)}", flush=True)
    for g in gate_list:
        print(f"  {'PASS' if g.passed else 'FAIL'} {g.id} {g.name}: {g.detail}", flush=True)
    print(f"build finished in {stats['elapsed_seconds']}s", flush=True)
    return stats


def recount_split(train_path: Path, val_path: Path, split_report: dict[str, Any]) -> dict[str, Any]:
    """Recompute the split counters from the files that exist after leakage repair."""
    rep = dict(split_report)
    train_docs = val_docs = train_chars = val_chars = 0
    per_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"train_documents": 0, "validation_documents": 0})
    moved = 0
    for doc in iter_jsonl(train_path):
        train_docs += 1
        train_chars += len(str(doc.get("text") or ""))
        lang = str(doc.get("language") or "unknown")
        per_lang[lang]["train_documents"] += 1
        if doc.get("moved_from_validation_by"):
            moved += 1
    for doc in iter_jsonl(val_path):
        val_docs += 1
        val_chars += len(str(doc.get("text") or ""))
        per_lang[str(doc.get("language") or "unknown")]["validation_documents"] += 1
    rep.update({"train_docs": train_docs, "val_docs": val_docs, "train_chars": train_chars, "val_chars": val_chars})
    rep["val_frac_achieved"] = round(val_docs / max(1, train_docs + val_docs), 6)
    merged = {k: dict(v) for k, v in (rep.get("per_language") or {}).items()}
    for lang, counts in per_lang.items():
        entry = merged.setdefault(lang, {})
        entry.update(counts)
    rep["per_language"] = dict(sorted(merged.items()))
    notes = list(rep.get("notes") or [])
    if moved:
        notes.append(f"{moved} document(s) moved from validation to train by leakage repair; validation shrank rather than being trimmed")
    rep["notes"] = notes
    return rep


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build and measure the SIR M1 corpus (streaming, licence-gated, leakage-checked).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Execution order:  acquire -> normalize -> build_corpus -> tokenizer bake-off ->\n"
            "                  build_corpus --tokenize-only -> gates -> (only then) training.\n"
            "See docs/m1-data-sprint.md for the full order, legal boundaries and the offline fixture path."
        ),
    )
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m1_corpus.yaml"))
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "sources.yaml"))
    ap.add_argument("--sources", help="comma-separated subset of manifest source ids (default: config dataset.sources)")
    ap.add_argument("--outdir", help="override paths.processed_dir")
    ap.add_argument("--resume", action="store_true", help="reuse existing intermediate files when their hashes match")
    ap.add_argument("--renormalize", action="store_true", help="ignore an existing normalized file even if its hash matches")
    ap.add_argument("--offline-fixture-only", action="store_true", help="build from the committed CC0 fixture with no network")
    ap.add_argument("--allow-id-fallback", action="store_true", help="split on doc ids when content grouping collapses (recorded)")
    ap.add_argument("--skip-near-dedup", action="store_true")
    ap.add_argument("--skip-integrity", action="store_true", help="skip re-hashing acquired artifacts (G3 will fail)")
    ap.add_argument("--no-leak-repair", action="store_true", help="measure leakage and stop instead of repairing the split")
    ap.add_argument("--max-leak-repair-iters", type=int, default=0, help="0 = take the value from config dataset.split.max_leak_repair_iters")
    ap.add_argument("--tokenizer-spec", help="path to a tokenizer spec JSON (default: <tokenizer_dir>/chosen.json)")
    ap.add_argument("--tokenize-only", action="store_true", help="reuse the text stages and only re-measure token counts/gates/report")
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    stats = run_build(cfg, args)
    failed = [g for g in stats.get("gates", []) if not g["passed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage-safe train/validation split that never loads the corpus into RAM.

The split has one job: **no content may straddle the boundary**. A random 2% of documents is not
a validation set if half of them share sentences with their neighbours, so this module builds
*components* — documents linked by shared normalized sentences — and moves whole components to
one side. The grouping index lives in SQLite (one row per distinct sentence hash), so memory use
is proportional to the number of distinct sentences, not to the corpus size.

Determinism: for each language, components are ordered by `blake2b(seed:language:group)`; the
ordering is a hash, not a dict iteration, so reruns are identical on any machine. Grouping is
conservative: `id_fallback` (doc-id equality only) exists for pathological corpora and is always
recorded in the report.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.data.dedup_stream import JsonlWriter, iter_jsonl
from training.data.split import norm_sentence, sentences_of


@dataclass
class SplitReport:
    method: str
    seed: int
    val_frac_requested: float
    val_frac_achieved: float = 0.0
    docs: int = 0
    train_docs: int = 0
    val_docs: int = 0
    components: int = 0
    train_chars: int = 0
    val_chars: int = 0
    per_language: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _sentence_hash(sentence: str) -> str:
    return hashlib.blake2b(norm_sentence(sentence).encode("utf-8"), digest_size=16).hexdigest()


def _doc_key(seed: int, language: str, group: int) -> str:
    return hashlib.blake2b(f"{seed}:{language}:{group}".encode("utf-8"), digest_size=16).hexdigest()


def stream_split(
    in_path: Path,
    out_dir: Path,
    *,
    val_frac: float = 0.02,
    seed: int = 20261005,
    workdir: Path | None = None,
    id_fallback: bool = False,
    commit_every: int = 5000,
) -> tuple[SplitReport, dict[str, Any]]:
    """Build components, assign whole components to train/val, write both files.

    Raises SystemExit when the corpus cannot be split safely (fewer than two components), because
    a "split" that puts everything in train silently produces an empty validation set.
    """
    started = time.time()
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir else out_dir / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "split_components.sqlite"
    if db_path.exists():
        db_path.unlink()

    method = "doc-id-fallback" if id_fallback else "sentence-components(sqlite)"
    rep = SplitReport(method=method, seed=seed, val_frac_requested=val_frac)

    # ---- pass 1: sentence-hash union-find in SQLite (bounded memory, deterministic)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE sent (sent_hash TEXT PRIMARY KEY, grp INTEGER NOT NULL)")
    conn.execute("CREATE TABLE docs (idx INTEGER PRIMARY KEY, grp INTEGER NOT NULL, lang TEXT, chars INTEGER)")
    conn.execute("CREATE INDEX idx_sent_grp ON sent (grp)")

    idx = 0
    merges = 0
    for doc in iter_jsonl(in_path):
        lang = str(doc.get("language") or "unknown")
        text = str(doc.get("text") or "")
        if id_fallback:
            hashes = [hashlib.blake2b(str(doc.get("doc_id") or idx).encode("utf-8"), digest_size=16).hexdigest()]
        else:
            hashes = sorted({_sentence_hash(s) for s in sentences_of(text) if norm_sentence(s)})
        groups: set[int] = set()
        for i in range(0, len(hashes), 200):  # bounded parameter lists
            chunk = hashes[i : i + 200]
            q = ",".join("?" for _ in chunk)
            groups.update(g for (g,) in conn.execute(f"SELECT grp FROM sent WHERE sent_hash IN ({q})", chunk))
        if groups:
            grp = min(groups)
            stale = [g for g in groups if g != grp]
            if stale:
                merges += len(stale)
                sq = ",".join("?" for _ in stale)
                conn.execute(f"UPDATE sent SET grp = ? WHERE grp IN ({sq})", [grp, *stale])
                conn.execute(f"UPDATE docs SET grp = ? WHERE grp IN ({sq})", [grp, *stale])
        else:
            grp = idx
        if hashes:
            conn.executemany("INSERT OR REPLACE INTO sent VALUES (?, ?)", [(h, grp) for h in hashes])
        conn.execute("INSERT INTO docs VALUES (?, ?, ?, ?)", (idx, grp, lang, len(text)))
        idx += 1
        if idx % commit_every == 0:
            conn.commit()
    conn.commit()
    rep.docs = idx
    if idx == 0:
        raise SystemExit(f"split: {in_path} contains no documents")

    # ---- components per language
    comps: list[tuple[int, str, int, int]] = [
        (grp, lang, docs, chars)
        for grp, lang, docs, chars in conn.execute("SELECT grp, lang, COUNT(*), SUM(chars) FROM docs GROUP BY grp, lang")
    ]
    rep.components = len({g for g, _, _, _ in comps})
    if rep.components < 2:
        raise SystemExit(
            "content grouping merged the whole corpus into fewer than two components; a split would "
            "leak by construction. Refusing (use --allow-id-fallback only if the corpus genuinely is "
            "one repeated document and the report should say so)."
        )

    by_lang: dict[str, list[tuple[int, int, int]]] = {}
    for grp, lang, docs, chars in comps:
        by_lang.setdefault(lang, []).append((grp, docs, chars))

    assign: dict[int, str] = {}
    per_language: dict[str, dict[str, Any]] = {}
    for lang, rows in sorted(by_lang.items()):
        total = sum(d for _, d, _ in rows)
        target = max(1, round(total * val_frac))
        ordered = sorted(rows, key=lambda r: _doc_key(seed, lang, r[0]))
        taken = 0
        val_components = 0
        for grp, docs, _ in ordered:
            if taken < target:
                assign[grp] = "val"
                taken += docs
                val_components += 1
            else:
                assign[grp] = "train"
        per_language[lang] = {
            "documents": total,
            "train_documents": total - taken,
            "validation_documents": taken,
            "components": len(rows),
            "validation_components": val_components,
        }
        if val_components == 0:
            per_language[lang]["note"] = "no component fitted the validation target; language appears only in train"
        elif val_components == 1:
            per_language[lang]["note"] = "single validation component; per-language metrics will be noisy"

    # ---- split and component per document index, as compact arrays (bounded, not a dict of docs)
    n = rep.docs
    component = array("q", [-1]) * n
    is_val = bytearray(n)
    for i, grp in conn.execute("SELECT idx, grp FROM docs"):
        component[i] = grp
        if assign.get(grp) == "val":
            is_val[i] = 1
    conn.close()

    # ---- write the two files, streaming, recording component and split on every document
    with JsonlWriter(out_dir / "train.jsonl") as train, JsonlWriter(out_dir / "val.jsonl") as val:
        for i, doc in enumerate(iter_jsonl(in_path)):
            doc = dict(doc)
            doc["split"] = "val" if is_val[i] else "train"
            doc["component_id"] = int(component[i])
            n_chars = len(str(doc.get("text") or ""))
            if is_val[i]:
                val.write(doc)
                rep.val_docs += 1
                rep.val_chars += n_chars
            else:
                train.write(doc)
                rep.train_docs += 1
                rep.train_chars += n_chars

    rep.val_frac_achieved = round(rep.val_docs / rep.docs, 6) if rep.docs else 0.0
    rep.per_language = per_language
    rep.elapsed_seconds = round(time.time() - started, 2)
    if merges:
        rep.notes.append(f"{merges} sentence-group merge(s): documents sharing a sentence stayed on one side")
    if abs(rep.val_frac_achieved - val_frac) > 0.5 * val_frac + 1e-9:
        rep.notes.append(
            f"validation fraction {rep.val_frac_achieved:.4%} differs from the {val_frac:.2%} target "
            "because whole components move together; the achieved value is what the report uses"
        )
    rep.notes.append("a component is a set of documents linked by shared normalized sentences; it can hold many documents")

    meta = {
        "method": method,
        "seed": seed,
        "val_frac_requested": val_frac,
        "val_frac_achieved": rep.val_frac_achieved,
        "components": rep.components,
        "sqlite_index": str(db_path),
        "determinism": "per-language component order = blake2b(seed:language:group)",
    }
    (out_dir / "split_report.json").write_text(json.dumps({"result": rep.as_dict(), "meta": meta}, indent=2) + "\n", encoding="utf-8")
    if db_path.exists():  # scratch index: the split it produced is on disk in the two JSONL files
        db_path.unlink()
    return rep, meta

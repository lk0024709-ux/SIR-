"""Tests for the corpus build pipeline.

Everything here runs on a handful of tiny documents in tmp dirs. The point is not to test the
model or the real corpus: it is to pin the invariants that make the real corpus trustworthy.

  * normalisation produces the documented schema and keeps provenance;
  * language labels are heuristic and say so, and per-record labels beat filename guesses;
  * synthetic text stays separable from natural text at every stage;
  * cleaning records every rejection (count *and* per-document record);
  * dedup is deterministic, keeps the earliest document, and names the winner;
  * the split never lets one content component straddle the boundary;
  * leakage is measured, then repaired by moving whole validation documents to train;
  * gates fail when they should, and the M1 verdict never rounds up;
  * the report and the token counts are generated, never hand-written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

HI = "नगर निगम ने जल भराव वाली सड़कों पर पंप लगाने का आदेश दिया है और सफाई अभियान भी चलाया।"
EN = "The laboratory recorded sensor drift over six weeks before reporting the final measurement."
ROMAN = "aaj office nahi jaunga, meeting cancel ho gayi thi subah hi, kal dekhenge phir se"
MIXED = "मैं अभी report finalize कर रहा हूँ, दस मिनट बाद भेजता हूँ और फिर call करता हूँ।"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _doc(i: int, text: str, *, lang="hi", source="src_a", synthetic=False, **extra) -> dict:
    return {
        "doc_id": f"{source}:doc{i}",
        "id": f"{source}:doc{i}",
        "source_id": source,
        "language": lang,
        "text": text,
        "synthetic": synthetic,
        "natural": not synthetic,
        "license": "CC0-1.0",
        "license_verified": True,
        "provenance": {"artifact": f"data/raw/{source}/x.txt", "artifact_sha256": "0" * 64},
        **extra,
    }


def _unit_manifest(tmp_path: Path, **over) -> Path:
    src = {
        "id": "unit_source", "name": "Unit source", "provider": "tests", "kind": "web-corpus",
        "url": "https://example.invalid/data", "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_verified": True, "license_evidence": "read the licence text on 2026-10-05: CC0 1.0",
        "languages": ["en"], "domain": "test", "natural": True, "synthetic": False,
        "acquisition": {"method": "local_fixture"}, "download_method": "GET",
        "redistribution_allowed": True, "commercial_use": "allowed", "derivatives": "allowed",
        "model_training_allowed": True, "attribution_required": False, "share_alike": False,
        "provenance": "unit test", "size_tokens": None, "status": "available", "notes": "unit test",
    }
    src.update(over)
    p = tmp_path / "sources.yaml"
    p.write_text(
        json.dumps({"schema_version": 2, "updated": "2026-10-05", "policy": {"require_training_permission": True}, "sources": [src]}),
        encoding="utf-8",
    )
    return p


# --------------------------------------------------------------------------------------
# language identification (heuristic, and honest about it)
# --------------------------------------------------------------------------------------
def test_langid_labels_the_required_space():
    from training.data.langid import LABELS, identify

    for text, expected in ((HI, "hi"), (EN, "en"), (ROMAN, "hinc-latn"), (MIXED, "hinc-deva")):
        got = identify(text)
        assert got.language == expected, f"{text[:30]}… -> {got.language} (expected {expected})"
        assert got.heuristic is True and got.method
    assert set(LABELS) >= {"hi", "en", "hinc-latn", "hinc-deva", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa", "or", "as", "ur", "sa", "unknown"}


def test_langid_distinguishes_devanagari_languages_and_rejects_symbol_noise():
    from training.data.langid import identify

    marathi = "आहे आणि नाही असे म्हणून त्यांनी सर्व काही केले होते मध्ये या त्या च्या"
    sanskrit = "अथ श्लोक इति एव च तथा अपि न सर्वे धर्म कर्म ब्रह्म मोक्ष आत्मा परम"
    assert identify(marathi).language == "mr"
    assert identify(sanskrit).language == "sa"
    assert identify("0123 4567 !@#$ %^&*").language == "unknown"


def test_langid_combining_marks_are_kept_with_their_word():
    """A plain `\\w` splits Devanagari at every vowel sign and wrecks the marker lexicons."""
    from training.data.langid import iter_words

    words = list(iter_words("नगर निगम ने जल भराव वाली सड़कों पर पंप लगाने का आदेश"))
    assert "नगर" in words and "सड़कों" in words and "आदेश" in words
    assert not any(len(w) == 1 and w in {"न", "ग", "म"} for w in words)


def test_script_check_catches_a_label_that_contradicts_the_text():
    from training.data.langid import script_counts, script_matches_label

    assert script_matches_label("hi", script_counts(HI))
    assert not script_matches_label("ta", script_counts(HI))
    assert script_matches_label("unknown", script_counts("1234"))


# --------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------
def test_normalize_strips_markup_and_flattens_json(tmp_path):
    from training.data.normalize import html_to_text, read_blocks

    p = tmp_path / "ch.xhtml"
    p.write_text(
        "<html><head><style>p{color:red}</style></head><body><p>Chapter one</p><p>  Second   line </p><script>var x=1;</script></body></html>",
        encoding="utf-8",
    )
    blocks, fmt = read_blocks(p)
    joined = " ".join(blocks)
    assert fmt == "markup" and "Chapter one" in joined and "Second" in joined
    assert "color:red" not in joined and "var x=1" not in joined

    j = tmp_path / "x.json"
    j.write_text(json.dumps({"language": "bengali", "text": {"0": {"0": "প্রথম", "1": "দ্বিতীয়"}}}), encoding="utf-8")
    blocks, fmt = read_blocks(j)
    assert fmt == "json" and any("প্রথম" in b for b in blocks)


def test_normalize_uses_per_record_language_when_the_artifact_has_it(tmp_path):
    from training.data.normalize import read_blocks_with_language

    p = tmp_path / "docs.json"
    p.write_text(json.dumps([{"id": "a", "language": "hi", "text": HI}, {"id": "b", "language": "en", "text": EN}]), encoding="utf-8")
    pairs, fmt = read_blocks_with_language(p)
    langs = {lang for _, lang in pairs}
    assert fmt == "json-records" and langs == {"hi", "en"}


def test_normalize_produces_the_documented_schema(tmp_path, monkeypatch):
    from training.data import normalize as nz

    raw = tmp_path / "data" / "raw" / "unit_source"
    raw.mkdir(parents=True)
    (raw / "book.txt").write_text("एक अनुच्छेद जो पर्याप्त लंबा है। " * 60, encoding="utf-8")
    ledger = {
        "data/raw/unit_source/book.txt": {
            "path": "data/raw/unit_source/book.txt",
            "sha256": "b" * 64,
            "bytes": (raw / "book.txt").stat().st_size,
            "url": "https://example.invalid/book.txt",
            "origin_repo": "example/repo",
            "origin_revision": "a" * 40,
        }
    }
    monkeypatch.setattr(nz, "load_ledger", lambda sid: ledger)
    monkeypatch.setattr(nz, "REPO_ROOT", tmp_path)
    rep = nz.normalize_source("unit_source", out_dir=tmp_path / "norm", manifest_path=_unit_manifest(tmp_path))
    assert rep.documents >= 1
    doc = _read_jsonl(tmp_path / "norm" / "unit_source.jsonl")[0]
    for field in ("doc_id", "source_id", "language", "text", "license", "license_verified", "provenance"):
        assert field in doc, f"normalized documents must carry {field}"
    assert doc["provenance"]["artifact_sha256"] == "b" * 64
    assert doc["provenance"]["origin_revision"] == "a" * 40
    # the recorded hash must be the file's hash, so a later gate can re-verify it
    assert rep.sha256 == hashlib.sha256((tmp_path / "norm" / "unit_source.jsonl").read_bytes()).hexdigest()


def test_normalize_marks_documentation_and_unpacked_archives(tmp_path, monkeypatch):
    from training.data import normalize as nz

    (tmp_path / "data" / "raw" / "unit_source").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "unit_source" / "book.txt").write_text("पाठ " * 300, encoding="utf-8")
    ledger = {
        "data/raw/unit_source/archive.zip": {"path": "data/raw/unit_source/archive.zip", "sha256": "c" * 64, "bytes": 10},
        "data/raw/unit_source/CARD.md": {"path": "data/raw/unit_source/CARD.md", "sha256": "d" * 64, "bytes": 10, "role": "documentation", "excluded_from_corpus": True},
        "data/raw/unit_source/book.txt": {"path": "data/raw/unit_source/book.txt", "sha256": "e" * 64, "bytes": 500, "extracted_from": "archive.zip"},
    }
    monkeypatch.setattr(nz, "load_ledger", lambda sid: ledger)
    monkeypatch.setattr(nz, "REPO_ROOT", tmp_path)
    rep = nz.normalize_source("unit_source", out_dir=tmp_path / "norm", manifest_path=_unit_manifest(tmp_path))
    assert rep.files_unreadable == 0
    assert rep.files_excluded_from_corpus == 1
    assert rep.files_skipped_archive_container == 1
    assert rep.excluded_from_corpus == ["data/raw/unit_source/CARD.md"]
    assert rep.policy == "normalize-v2"
    assert rep.documents >= 1


# --------------------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------------------
def test_cleaning_records_every_rejection_and_keeps_metadata(tmp_path):
    from training.data.build_corpus import clean_source
    from training.data.clean import CleanConfig

    docs = [_doc(i, f"यह एक वैध अनुच्छेद है जो पर्याप्त लंबा है और इसमें कोई दोष नहीं है। संख्या {i}।" * 4, lang="hi") for i in range(5)]
    docs.append(_doc(100, "छोटा।", lang="hi"))
    docs.append(_doc(101, "x" * 400, lang="hi"))
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep = clean_source("src_a", src, tmp_path / "kept.jsonl", tmp_path / "rejects", CleanConfig(min_chars=120))
    assert rep["documents_seen"] == 7 and rep["documents_kept"] == 5 and rep["documents_rejected"] == 2
    log = [json.loads(l) for l in (tmp_path / "rejects" / "src_a.rejects.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(log) == rep["documents_rejected"], "one record per rejected document, no silent drops"
    assert all(r["doc_id"] and r["reason"] for r in log)
    assert not any("text" in r for r in log), "the reject log must not contain the rejected text"
    kept = _read_jsonl(tmp_path / "kept.jsonl")
    assert all(d["source_id"] == "src_a" and d["license"] == "CC0-1.0" and d["provenance"]["artifact"] for d in kept)
    assert all(d["language_measured"] in {"hi", "hinc-deva"} for d in kept)


def test_cleaning_drops_a_document_whose_script_contradicts_its_label(tmp_path):
    from training.data.build_corpus import clean_source
    from training.data.clean import CleanConfig

    docs = [_doc(0, "यह एक वैध देवनागरी अनुच्छेद है जो पर्याप्त लंबा है और कोई दोष नहीं है।" * 4, lang="hi"),
            _doc(1, EN * 4, lang="hi")]  # declared Hindi, written in Latin script
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep = clean_source("src_a", src, tmp_path / "kept.jsonl", tmp_path / "rejects", CleanConfig(min_chars=60))
    assert rep["script_mismatch_rejections"] >= 1
    log = [json.loads(l) for l in (tmp_path / "rejects" / "src_a.rejects.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any(r["reason"] == "script_mismatch_vs_declared_label" for r in log)


# --------------------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------------------
def test_exact_dedup_keeps_earliest_and_names_the_winner(tmp_path):
    from training.data.dedup_stream import stream_exact_dedup

    src = _write_jsonl(
        tmp_path / "in.jsonl",
        [_doc(0, "साझा पाठ " * 40, source="src_a"), _doc(1, "साझा पाठ " * 40, source="src_b"), _doc(2, EN * 3, lang="en", source="src_c")],
    )
    rep, _ = stream_exact_dedup(src, tmp_path / "keep.jsonl", tmp_path / "removed.jsonl")
    assert (rep.read, rep.kept, rep.removed) == (3, 2, 1)
    kept = _read_jsonl(tmp_path / "keep.jsonl")
    assert kept[0]["source_id"] == "src_a", "the earliest document must win"
    removed = _read_jsonl(tmp_path / "removed.jsonl")[0]
    assert removed["kept_source_id"] == "src_a" and removed["kept_doc_id"].startswith("src_a")
    assert removed["reason"] == "exact_duplicate"


def test_near_dedup_is_deterministic_and_reports_the_winner(tmp_path):
    from training.data.dedup_stream import stream_near_dedup
    from training.data.deduplicate import DedupConfig

    base = "The municipal corporation published the tender documents in a machine readable format. " * 6
    docs = [_doc(0, base, lang="en", source="a"), _doc(1, base + " One extra sentence appended.", lang="en", source="b"), _doc(2, EN * 5, lang="en", source="c")]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    cfg = DedupConfig(num_perm=64, band_size=8, threshold=0.8, shingle_chars=24)
    r1, m1 = stream_near_dedup(src, tmp_path / "k1.jsonl", tmp_path / "r1.jsonl", cfg=cfg, workdir=tmp_path / "w1")
    r2, m2 = stream_near_dedup(src, tmp_path / "k2.jsonl", tmp_path / "r2.jsonl", cfg=cfg, workdir=tmp_path / "w2")
    assert (r1.kept, r1.removed) == (2, 1)
    assert (r2.kept, r2.removed) == (r1.kept, r1.removed)
    assert (tmp_path / "k1.jsonl").read_bytes() == (tmp_path / "k2.jsonl").read_bytes()
    rec = _read_jsonl(tmp_path / "r1.jsonl")[0]
    assert rec["reason"] == "near_duplicate" and rec["kept_source_id"] == "a" and rec["source_id"] == "b"
    assert m1["note"].startswith("cluster winner")


def test_dedup_does_not_hold_the_corpus_in_memory(tmp_path):
    import tracemalloc

    from training.data.dedup_stream import stream_exact_dedup

    docs = [_doc(i, f"{EN} unique sentence {i} {'word ' * 400}", lang="en") for i in range(300)]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    tracemalloc.start()
    stream_exact_dedup(src, tmp_path / "keep.jsonl", tmp_path / "removed.jsonl")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < (tmp_path / "in.jsonl").stat().st_size, "the stage must stream, not accumulate the corpus"


# --------------------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------------------
def test_split_keeps_components_together_and_records_component_ids(tmp_path):
    from training.data.split_stream import stream_split

    docs = [_doc(i, f"अनोखा अनुच्छेद संख्या {i} जिसमें भिन्न शब्द हैं और कोई दोहराव नहीं है। {chr(0x0900 + i)}", lang="hi") for i in range(12)]
    shared = "यह वाक्य दो दस्तावेज़ों में एक जैसा है और इसे अलग नहीं किया जाना चाहिए।"
    docs += [_doc(100, f"{shared} पहला दस्तावेज़।", lang="hi"), _doc(101, f"{shared} दूसरा दस्तावेज़।", lang="hi")]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep, meta = stream_split(src, tmp_path / "out", val_frac=0.25, seed=7, workdir=tmp_path / "work")
    train, val = _read_jsonl(tmp_path / "out" / "train.jsonl"), _read_jsonl(tmp_path / "out" / "val.jsonl")
    assert train and val
    assert not ({d["component_id"] for d in train} & {d["component_id"] for d in val}), "a component must not straddle the split"
    by_id = {d["doc_id"]: d["component_id"] for d in train + val}
    assert by_id["src_a:doc100"] == by_id["src_a:doc101"], "documents sharing a sentence belong to one component"
    assert all(d["split"] in {"train", "val"} for d in train + val)
    assert rep.components >= 2 and meta["seed"] == 7


def test_split_is_deterministic_and_refuses_unsplittable_input(tmp_path):
    from training.data.split_stream import stream_split

    docs = [_doc(i, f"वाक्य संख्या {i} बिल्कुल अलग है और दोहराया नहीं गया।", lang="hi") for i in range(10)]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    a, _ = stream_split(src, tmp_path / "a", val_frac=0.3, seed=11, workdir=tmp_path / "wa")
    b, _ = stream_split(src, tmp_path / "b", val_frac=0.3, seed=11, workdir=tmp_path / "wb")
    assert a.train_docs == b.train_docs and (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()

    same = _write_jsonl(tmp_path / "same.jsonl", [_doc(i, "यही वाक्य हर दस्तावेज़ में दोहराया गया है इसलिए पूरा संग्रह एक ही घटक है।", lang="hi") for i in range(6)])
    with pytest.raises(SystemExit):
        stream_split(same, tmp_path / "out2", val_frac=0.2, seed=1, workdir=tmp_path / "w2")


# --------------------------------------------------------------------------------------
# leakage: measure, then repair
# --------------------------------------------------------------------------------------
def test_ngram_overlap_is_measured_and_repaired_by_moving_validation_documents(tmp_path):
    from training.data.build_corpus import repair_leakage, stream_ngram_overlap

    shared = " ".join(f"tok{i}" for i in range(20))
    train = _write_jsonl(tmp_path / "train.jsonl", [_doc(0, f"prefix words {shared} suffix words", lang="en")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_doc(1, f"other prefix {shared} other suffix", lang="en"), _doc(2, "totally different content with no shared n-grams at all", lang="en")])
    leak = stream_ngram_overlap(train, val, 8)
    assert leak["unique_overlapping_ngrams"] > 0 and leak["val_documents"] == 2

    repair = repair_leakage(train, val, 8, 3)
    assert repair["converged"] is True and repair["log"][0]["moved_documents"] == 1
    assert stream_ngram_overlap(train, val, 8)["unique_overlapping_ngrams"] == 0
    train_after = _read_jsonl(train)
    moved = [d for d in train_after if d.get("moved_from_validation_by")]
    assert moved and moved[0]["split"] == "train", "the leaking document moves to train, it is not deleted"
    assert len(train_after) == 2, "repair appends to train; it must never replace the training documents"
    assert len(_read_jsonl(val)) == 1


# --------------------------------------------------------------------------------------
# gates + M1 verdict
# --------------------------------------------------------------------------------------
def _fake_stats(**over) -> dict:
    stats = {
        "per_source": {"unit_source": {"documents": 10, "chars": 1000}},
        "per_language": {"en": {"documents": 10, "chars": 1000}},
        "documents": {"raw_files": 1, "normalized": 10, "cleaned_kept": 10, "after_dedup": 10, "synthetic_documents": 0},
        "characters": {"cleaned": 1000, "after_dedup": 1000, "train": 900, "validation": 100},
        "validation": {"leakage_8gram": {"unique_overlapping_ngrams": 0, "overlapping_ngram_occurrences_train": 0}, "cross_split_near_dups": {"pairs": 0}},
        "tokens": {"measured": True, "tool": "tests", "tokenizer_sha256": "a" * 64, "total": 12_000_000, "train": 11_000_000, "validation": 1_000_000, "by_language": {"en": 12_000_000}, "by_source": {"unit_source": 12_000_000}},
        "language_quality": {"documents": 10, "unknown": 0, "script_mismatch": 0, "documents_seen_cleaning": 10, "label_disagreements": 0},
        "split": {"method": "test", "seed": 1, "val_frac_requested": 0.1, "val_frac_achieved": 0.1, "train_docs": 9, "val_docs": 1, "components": 10, "notes": []},
        "dedup": {"exact": {"removed": 1}, "near": {"removed": 0, "cross_source_removals": 0}},
        "provenance_gaps": {},
        "acquisition": {"unit_source": {"files": 1, "revision": "a" * 40, "archives": [{"repo": "x", "resolved_commit": "a" * 40, "archive_sha256": "z"}]}},
        "reproducibility": {
            "git_commit": "deadbeef", "config_sha256": "f" * 64, "seed": 1,
            "script_sha256": {"x.py": "b" * 64},
            "source_revisions": {"unit_source": "a" * 40},
            "normalized_files": {"unit_source": {"path": "data/normalized/unit_source.jsonl", "recorded": "c" * 64, "recomputed": "c" * 64}},
        },
    }
    stats.update(over)
    return stats


def test_gates_pass_on_a_clean_stats_file(tmp_path):
    from training.data.gates import run_gates
    from training.data.manifest import load_manifest

    gates = run_gates(load_manifest(_unit_manifest(tmp_path)), _fake_stats(), [{"source_id": "unit_source", "path": "p", "status": "ok"}])
    assert [g.id for g in gates] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    assert all(g.passed for g in gates), [g.detail for g in gates if not g.passed]


def test_gates_fail_when_they_should(tmp_path):
    from training.data.gates import run_gates
    from training.data.manifest import load_manifest

    stats = _fake_stats(
        validation={"leakage_8gram": {"unique_overlapping_ngrams": 3, "overlapping_ngram_occurrences_train": 9}, "cross_split_near_dups": {"pairs": 1}},
        tokens={"measured": False},
        language_quality={"documents": 10, "unknown": 1, "script_mismatch": 1, "documents_seen_cleaning": 10},
    )
    gates = {g.id: g for g in run_gates(load_manifest(_unit_manifest(tmp_path)), stats, [{"source_id": "unit_source", "path": "p", "status": "sha256-mismatch"}])}
    assert not gates["G3"].passed and not gates["G4"].passed and not gates["G5"].passed and not gates["G6"].passed


def test_gate_license_refuses_a_source_without_a_verified_licence(tmp_path):
    from training.data.gates import gate_license
    from training.data.manifest import load_manifest

    manifest = load_manifest(REPO_ROOT / "data" / "manifests" / "sources.yaml")
    res = gate_license(manifest, _fake_stats(per_source={"sir_fixture_v0": {"documents": 1}, "definitely_not_declared": {"documents": 5}}))
    assert not res.passed and "definitely_not_declared" in res.measured["offenders"]


def test_m1_verdict_never_rounds_up(tmp_path):
    from training.data.gates import m1_status, run_gates
    from training.data.manifest import load_manifest

    manifest = load_manifest(_unit_manifest(tmp_path))
    verify = [{"source_id": "unit_source", "path": "p", "status": "ok"}]

    blocked = m1_status(_fake_stats(tokens={"measured": False}), run_gates(manifest, _fake_stats(tokens={"measured": False}), verify))
    assert blocked["status"] == "BLOCKED" and "NOT READY" in blocked["training_readiness"]["state"]

    partial = m1_status(_fake_stats(), run_gates(manifest, _fake_stats(), verify))
    assert partial["status"] == "PARTIAL" and partial["measured_tokens"] == 12_000_000
    assert partial["training_readiness"]["state"].startswith("READY FOR A DATA-SCALED")

    big = _fake_stats(tokens={"measured": True, "tool": "t", "tokenizer_sha256": "a" * 64, "total": 250_000_000, "train": 245_000_000, "validation": 5_000_000})
    passed = m1_status(big, run_gates(manifest, big, verify))
    assert passed["status"] == "PASS" and passed["training_readiness"]["state"] == "READY FOR M1 TRAINING"


def test_synthetic_documents_are_counted_separately(tmp_path):
    from training.data.corpus_report import build_report
    from training.data.manifest import load_manifest

    stats = _fake_stats(documents={"raw_files": 1, "normalized": 10, "cleaned_kept": 10, "after_dedup": 10, "synthetic_documents": 4})
    report = build_report(stats, load_manifest(_unit_manifest(tmp_path)))
    assert report["corpus"]["documents"]["synthetic"] == 4
    assert report["corpus"]["documents"]["natural"] == 6, "synthetic text must never inflate the natural count"


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def test_corpus_report_json_and_markdown_are_generated(tmp_path):
    from training.data.corpus_report import build_report, render_markdown, write_reports
    from training.data.gates import m1_status, run_gates
    from training.data.manifest import load_manifest

    manifest = load_manifest(_unit_manifest(tmp_path))
    stats = _fake_stats()
    stats["gates"] = [g.as_dict() for g in run_gates(manifest, stats, [{"source_id": "unit_source", "path": "p", "status": "ok"}])]
    stats["m1"] = m1_status(stats, run_gates(manifest, stats, [{"source_id": "unit_source", "path": "p", "status": "ok"}]))
    report = build_report(stats, manifest)
    md = render_markdown(report)
    for section in ("## Languages", "## Sources", "## Split", "## Quality", "## Gates", "## What this corpus is not", "## Reproducibility"):
        assert section in md, f"the report must contain {section}"
    assert "12,000,000" in md, "the measured token count must appear in the report"
    write_reports(report, tmp_path / "r.json", tmp_path / "r.md")
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["m1"]["status"] == "PARTIAL"
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# SIR — M1 corpus report")


# --------------------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------------------
def test_token_counts_come_from_running_a_tokenizer(tmp_path):
    from types import SimpleNamespace

    from training.data.build_corpus import tokenize_split

    docs = [_doc(0, "one two three four five", lang="en"), _doc(1, "six seven eight", lang="en", source="src_b")]
    src = _write_jsonl(tmp_path / "train.jsonl", docs)

    class StubTokenizer:
        vocab_size = 64
        kind = "stub"

        def encode(self, text: str) -> list[int]:
            return [len(t) for t in text.split()]

        spec = SimpleNamespace(eos_id=63)

    meta = tokenize_split(src, tmp_path / "tok", StubTokenizer(), seq_len=4)
    assert meta["n_tokens"] == sum(len(d["text"].split()) for d in docs) + len(docs), "one EOS per document"
    assert meta["n_docs"] == 2 and meta["source_tokens"] == {"src_a": 5, "src_b": 3}
    assert meta["contexts_available"] == meta["n_tokens"] // 4
    assert (tmp_path / "tok" / "tokens.bin").stat().st_size == meta["n_tokens"] * 2


# --------------------------------------------------------------------------------------
# offline fixture path (no network)
# --------------------------------------------------------------------------------------
def test_offline_fixture_build_refuses_the_network(tmp_path, monkeypatch):
    """The committed CC0 fixture must produce a corpus with the network closed."""
    import socket

    from training.data import build_corpus as bc

    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("the offline fixture build attempted a network call")

    monkeypatch.setattr(socket, "create_connection", explode, raising=False)
    monkeypatch.setattr(socket, "socket", explode, raising=False)
    from training.data.clean import CleanConfig

    docs = [_doc(i, f"वाक्य संख्या {i} जो अलग है और दोहराया नहीं गया है " * 4, lang="hi", synthetic=True, source="sir_fixture_v0") for i in range(20)]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep = bc.clean_source("sir_fixture_v0", src, tmp_path / "cleaned.jsonl", tmp_path / "rejects", CleanConfig(min_chars=60))
    assert rep["documents_seen"] == 20 and rep["documents_kept"] >= 1
    kept = _read_jsonl(tmp_path / "cleaned.jsonl")
    assert all(d["synthetic"] is True for d in kept), "synthetic provenance must survive cleaning"
    assert all(d["source_id"] == "sir_fixture_v0" for d in kept)

def test_detector_cross_check_never_overrules_the_heuristic():
    """py3langid is optional; when present it must be reported as a check, not as the labeller."""
    from training.data.langid import LABELS, detector, detector_agrees, detector_label, detector_name

    assert set(LABELS) >= {"hi", "en", "hinc-latn", "hinc-deva", "unknown"}
    if detector() is None:
        assert detector_name() is None and detector_label(HI) is None
        assert detector_agrees("hi", None) is None
        return
    got = detector_label(HI)
    assert got is not None and got[0] == "hi" and 0.0 <= got[1] <= 1.0
    assert detector_agrees("hi", got) is True
    assert detector_agrees("hi", ("ta", 0.9)) is False
    # Hinglish has no detector label of its own: a Hindi answer counts as agreement
    assert detector_agrees("hinc-latn", ("hi", 0.6)) is True
    assert detector_agrees("unknown", ("en", 0.9)) is None


def test_cleaning_records_the_detector_cross_check(tmp_path):
    from training.data.build_corpus import clean_source
    from training.data.clean import CleanConfig
    from training.data.langid import detector

    docs = [_doc(i, HI * 3, lang="hi") for i in range(3)] + [_doc(9, EN * 3, lang="en")]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep = clean_source("src_a", src, tmp_path / "kept.jsonl", tmp_path / "rejects", CleanConfig(min_chars=60))
    if detector() is None:
        assert rep["detector_compared"] == 0
        return
    assert rep["detector_compared"] == rep["documents_kept"] == 4
    assert rep["detector_agree"] >= 3  # the romanized-Hindi fixture is the known hard case
    assert isinstance(rep["detector_disagreements"], dict)


def test_report_exposes_the_required_measurement_keys(tmp_path):
    from training.data.corpus_report import build_report
    from training.data.manifest import load_manifest

    report = build_report(_fake_stats(), load_manifest(_unit_manifest(tmp_path)))
    m = report["measurements"]
    for key in ("raw_documents", "clean_documents", "characters", "bytes_utf8_after_dedup", "tokens",
                "tokens_train", "tokens_validation", "tokens_by_language", "tokens_by_source"):
        assert key in m, f"requirement 15 asks for {key}"
    assert report["sources_by_status"]["counts"]["available"] == 1
    assert "unit_source" in report["sources_by_status"]["ids"]["available"]

def test_code_switching_needs_real_words_from_both_scripts():
    from training.data.langid import code_switch_evidence

    mixed = code_switch_evidence("मैं अभी report finalize कर रहा हूँ और दस मिनट में भेजता हूँ")
    assert mixed["code_switch"] is True and mixed["latin_tokens"] >= 2 and mixed["devanagari_tokens"] >= 2
    assert code_switch_evidence("यह पूरा वाक्य केवल देवनागरी में लिखा गया है")["code_switch"] is False
    assert code_switch_evidence("this sentence is entirely english text")["code_switch"] is False
    # a single stray Latin character is not code-switching
    assert code_switch_evidence("यह वाक्य A के साथ है पर पूरा देवनागरी में लिखा गया है और लंबा भी है")["code_switch"] is False


def test_cleaning_separates_hinc_latn_and_hinc_deva(tmp_path):
    from training.data.build_corpus import clean_source
    from training.data.clean import CleanConfig

    docs = [
        _doc(0, " ".join(["aaj", "office", "nahi", "jaunga", "meeting", "cancel", "ho", "gayi", "thi", "subah", "hi"] * 3), lang="hinc-latn"),
        _doc(1, " ".join(["मैं", "अभी", "report", "finalize", "कर", "रहा", "हूँ", "और", "दस", "मिनट", "में", "भेजता", "हूँ"] * 4), lang="hinc-deva"),
    ]
    src = _write_jsonl(tmp_path / "in.jsonl", docs)
    rep = clean_source("src_a", src, tmp_path / "kept.jsonl", tmp_path / "rejects", CleanConfig(min_chars=60))
    assert rep["hinc_latn_documents"] >= 1
    assert rep["hinc_deva_documents"] >= 1
    assert rep["code_switch_documents"] >= 1, "hinc-deva with whole Latin words counts as real code-switching"

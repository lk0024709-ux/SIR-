"""Data layer: manifest gates, cleaning, dedup, leakage-safe split, validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data.clean import (
    CleanConfig,
    CleanReport,
    clean_stream,
    lang_guess,
    load_documents,
    normalize_text,
    quality_reject_reasons,
    romanized_hindi_score,
    try_recover_mojibake,
)
from training.data.deduplicate import DedupConfig, deduplicate, shingles
from training.data.manifest import ManifestError, load_manifest, record_acquisition, validate_manifest
from training.data.split import SplitConfig, build_components, split
from training.data.validate import check_docs, cross_split_near_dups, ngrams, overlap_report, tokens_of


# --------------------------------------------------------------------------------------
# manifest gates
# --------------------------------------------------------------------------------------
def _write_manifest(tmp: Path, sources: list[dict], policy: dict | None = None) -> Path:
    body = {"schema_version": 1, "updated": "2026-10-05", "policy": policy or {}, "sources": sources}
    p = tmp / "sources.yaml"
    import yaml

    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return p


def _good_source(**over) -> dict:
    base = {
        "id": "unit_source",
        "name": "unit",
        "kind": "fixture",
        "url": None,
        "license": "CC0-1.0",
        "languages": ["hi"],
        "natural": False,
        "size_tokens": None,
        "redistribution_allowed": True,
        "status": "available",
        "notes": "unit test source",
    }
    base.update(over)
    return base


def test_manifest_loads_repo_file():
    m = load_manifest(Path(__file__).resolve().parents[1] / "data" / "manifests" / "sources.yaml")
    assert "sir_fixture_v0" in m.sources
    assert m.sources["codemix_325k"].raw["status"] == "blocked"
    # a blocked source must explain itself
    assert m.sources["codemix_325k"].raw["blocked_reason"].strip()


def test_unverified_license_is_not_usable(tmp_path):
    p = _write_manifest(tmp_path, [_good_source(license="UNVERIFIED")])
    issues, usable = validate_manifest(load_manifest(p), {"records": {}})
    assert "unit_source" not in usable
    assert any(i.code.startswith("G2") for i in issues if i.level == "error")


def test_hand_typed_size_is_rejected(tmp_path):
    p = _write_manifest(tmp_path, [_good_source(size_tokens=12_345_678)])
    issues, _ = validate_manifest(load_manifest(p), {"records": {}})
    codes = {i.code for i in issues}
    assert "G3-hand-typed-size" in codes, "a human-typed corpus size must be treated as fabricated"


def test_available_requires_acquisition_record(tmp_path):
    p = _write_manifest(tmp_path, [_good_source()])
    issues, usable = validate_manifest(load_manifest(p), {"records": {}})
    assert not usable
    assert any(i.code == "G3-missing-acquisition" for i in issues)


def test_record_acquisition_measures_instead_of_trusting(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps([{"id": "a", "text": "एक"}, {"id": "b", "text": "दो"}]), encoding="utf-8")
    rec = record_acquisition("unit_source", [f], tmp_path / "acq.json")
    assert rec["bytes"] == f.stat().st_size > 0
    assert rec["files"] == 1 and len(rec["sha256"]) == 64
    acq = json.loads((tmp_path / "acq.json").read_text(encoding="utf-8"))
    assert acq["records"]["unit_source"]["sha256"] == rec["sha256"]


def test_blocked_without_reason_is_error(tmp_path):
    p = _write_manifest(tmp_path, [_good_source(status="blocked", blocked_reason="")])
    issues, _ = validate_manifest(load_manifest(p), {"records": {}})
    assert any(i.code == "G4-blocked-needs-reason" for i in issues)


def test_missing_field_and_bad_status(tmp_path):
    bad = _good_source()
    bad.pop("license")
    p = _write_manifest(tmp_path, [bad, _good_source(id="other", status="maybe")])
    issues, _ = validate_manifest(load_manifest(p), {"records": {}})
    assert any(i.code == "G0-missing-field" for i in issues)
    assert any(i.code == "G4-bad-status" for i in issues)


def test_unknown_manifest_path_raises(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.yaml")


def test_duplicate_ids_rejected(tmp_path):
    p = _write_manifest(tmp_path, [_good_source(), _good_source()])
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_no_natural_corpus_warning(tmp_path):
    p = _write_manifest(tmp_path, [_good_source(natural=False)])
    acq = tmp_path / "acq.json"
    f = tmp_path / "x.json"
    f.write_text(json.dumps([{"id": "a", "text": "b"}]), encoding="utf-8")
    record_acquisition("unit_source", [f], acq)
    issues, usable = validate_manifest(load_manifest(p), json.loads(acq.read_text()))
    assert "unit_source" in usable
    assert any(i.code == "Q0-no-natural-corpus" for i in issues)


# --------------------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------------------
def test_normalize_strips_zero_width_and_markup():
    rep = CleanReport()
    raw = "<p>नगर&nbsp;निगम​ ने आदेश​ दिया</p> {{update}} [citation needed] see https://x.gov/a for more"
    out = normalize_text(raw, CleanConfig(), rep)
    assert "\u200b" not in out and "\u00a0" not in out
    assert "<p>" not in out and "{{update}}" not in out and "https://" not in out
    assert "निगम" in out
    assert rep.repaired.get("urls_removed") == 1


def test_unicode_nfc_normalisation_is_applied():
    decomposed = "कैं\u0310"  # combining marks
    out = normalize_text(decomposed, CleanConfig(), CleanReport())
    import unicodedata

    assert out == unicodedata.normalize("NFC", out)


def test_mojibake_recovery_roundtrip():
    original = "मेघालय में बारिश होगी"
    broken = original.encode("utf-8").decode("latin-1")
    assert try_recover_mojibake(broken) == original


def test_mojibake_recovery_refuses_to_garble_english():
    assert try_recover_mojibake("plain english text, nothing broken here") is None


def test_lang_guess_per_script_and_code_switching():
    assert lang_guess("भारत का मौसम अच्छा है और यहाँ बारिश हुई है कल से")[0] == "hi"
    assert lang_guess("The weather in the city was quite pleasant all week")[0] == "en"
    assert lang_guess("aaj office nahi jaunga, meeting cancel ho gayi thi")[0] == "hinc-latn"
    assert lang_guess("मैं अभी report finalize कर रहा हूँ, दस मिनट बाद भेजता हूँ")[0] == "hinc-deva"
    assert lang_guess("0123 5678 !@#$ %%^&*")[0] == "unknown"


def test_romanized_hindi_score_is_lexicon_based():
    assert romanized_hindi_score("aaj main office ja raha hoon, kal dekhenge") > romanized_hindi_score("the quarterly report is attached")


def test_quality_filters_reject_junk():
    cfg = CleanConfig(min_chars=40)
    assert "too_short" in quality_reject_reasons("छोटा", cfg)
    assert "line_repetition" in quality_reject_reasons("बोलो।\n" * 40, cfg)
    # punctuation-heavy strings are rejected as low_alnum_ratio (category Po, not Symbol So)
    # either reason shows the filter is working; we assert the document is rejected at all
    reasons = quality_reject_reasons("!@#$%^&*()" * 10, cfg)
    assert any(r in reasons for r in ("symbol_soup", "low_alnum_ratio"))
    # a true Symbol-category string should trigger symbol_soup
    assert "symbol_soup" in quality_reject_reasons("★" * 40 + " test " + "★" * 40, CleanConfig(min_chars=40, max_symbol_ratio=0.1))


def test_clean_stream_logs_rejections_and_keeps_good_docs(mini_docs):
    docs = mini_docs + [{"id": "junk", "text": "ok", "language": "en"}, {"id": "notext"}]
    kept, rep, rejects = clean_stream(docs, CleanConfig(min_chars=40), "test_source")
    assert rep.read == len(docs)
    assert any(r["reasons"] for r in rejects)
    assert all("language" in d and "source_id" in d for d in kept)
    assert "missing_text_field" in rep.dropped


def test_load_documents_supports_json_jsonl_and_txt(tmp_path):
    docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
    (tmp_path / "a.json").write_text(json.dumps(docs), encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    (tmp_path / "c.txt").write_text("first para\n\nsecond para\n", encoding="utf-8")
    assert len(list(load_documents(tmp_path / "a.json"))) == 2
    assert len(list(load_documents(tmp_path / "b.jsonl"))) == 2
    assert len(list(load_documents(tmp_path / "c.txt"))) == 2


# --------------------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------------------
def test_shingles_step_forward_and_handle_short_text():
    assert shingles("short", 24, True) == ["short"]
    long = "क" * 60
    assert len(shingles(long, 24)) == (60 - 24) // 8 + 1


def test_exact_duplicates_removed(mini_docs):
    docs = list(mini_docs) + [{**mini_docs[0], "id": "clone-1"}]
    kept, rep = deduplicate(docs, DedupConfig())
    assert rep.exact_removed == 1
    assert sum(1 for d in kept if d["id"] == "clone-1") == 0


def test_near_duplicates_are_caught():
    # construct a longer base so that a small edit keeps Jaccard high
    base_text = "नगर निगम ने जल भराव वाली सड़कों पर पंप लगाने का आदेश दिया है। " * 3
    base = {"id": "base", "source_id": "test_source", "language": "hi", "text": base_text.strip()}
    # near-duplicate: same text with a single-word suffix (high overlap)
    edited = {"id": "neardup", "source_id": "test_source", "language": "hi", "text": base_text.strip() + " अतिरिक्त"}
    kept, rep = deduplicate([base, edited], DedupConfig(threshold=0.7))
    assert rep.near_removed >= 1, f"expected near-duplicate removal, got {rep.as_dict()}"
    assert any(c["size"] >= 2 for c in rep.clusters)


def test_dedup_is_order_stable_in_counts(mini_docs):
    docs = list(mini_docs) + [{**mini_docs[0], "id": "dup"}]
    _, r1 = deduplicate(docs, DedupConfig())
    _, r2 = deduplicate(docs[::-1], DedupConfig())
    assert r1.read == r2.read and r1.kept == r2.kept and r1.exact_removed == r2.exact_removed


# --------------------------------------------------------------------------------------
# split + leakage
# --------------------------------------------------------------------------------------
def test_content_components_group_documents_sharing_sentences(mini_docs):
    docs = list(mini_docs) + [{**mini_docs[0], "id": "twin"}]
    roots, method, _ = build_components(docs, "auto", 8, 10_000)
    assert "ngram" in method
    i = next(k for k, d in enumerate(docs) if d["id"] == "doc0000")
    j = next(k for k, d in enumerate(docs) if d["id"] == "twin")
    assert roots[i] == roots[j]


def test_split_is_deterministic_and_disjoint(mini_docs):
    a = split(list(mini_docs), SplitConfig(val_frac=0.25, seed=11))
    b = split(list(mini_docs), SplitConfig(val_frac=0.25, seed=11))
    assert [d["id"] for d in a.train] == [d["id"] for d in b.train]
    assert [d["id"] for d in a.val] == [d["id"] for d in b.val]
    assert not ({d["id"] for d in a.train} & {d["id"] for d in a.val})
    assert len(a.train) + len(a.val) == len(mini_docs)


def test_split_achieves_zero_ngram_overlap_on_repetitive_corpus(mini_docs):
    """The fixture is template-composed, i.e. deliberately leaky. A naive split would overlap."""
    res = split(list(mini_docs), SplitConfig(val_frac=0.3, seed=3))
    ov = overlap_report(res.train, res.val, 4)
    assert ov["zero_overlap"], ov["examples"][:2]


def test_naive_id_split_would_leak():
    """Documents here share sentences by construction, so id-based splitting must be shown to leak."""
    # two clusters: docs 0-5 share sentence A, docs 6-11 share sentence B.
    # group_by=id can split within a cluster -> leaks. group_by=auto keeps clusters intact -> no leak.
    sent_a = "भारत का मौसम अच्छा है और यहाँ बारिश हुई है कल से"
    sent_b = "प्रयोगशाला में तापमान नियंत्रण बिगड़ने से पूरा नमूना नष्ट हो गया"
    docs = []
    for i in range(12):
        sent = sent_a if i < 6 else sent_b
        docs.append({"id": f"doc{i:03d}", "source_id": "test_source", "language": "hi", "text": f"{sent} अतिरिक्त शब्द {i}"})
    res = split(list(docs), SplitConfig(val_frac=0.3, seed=3, group_by="id"))
    tr = " ".join(d["text"] for d in res.train)
    va = " ".join(d["text"] for d in res.val)
    shared = ngrams(tokens_of(tr), 4) & ngrams(tokens_of(va), 4)
    assert shared, "expected group_by=id to leak on a repetitive corpus"
    # leakage-safe split on same data should NOT leak (clusters stay intact)
    res2 = split(list(docs), SplitConfig(val_frac=0.3, seed=3, group_by="auto"))
    tr2 = " ".join(d["text"] for d in res2.train)
    va2 = " ".join(d["text"] for d in res2.val)
    shared2 = ngrams(tokens_of(tr2), 4) & ngrams(tokens_of(va2), 4)
    assert not shared2, "leakage-safe split should have zero 4-gram overlap"


def test_split_refuses_language_with_single_component():
    # all docs share the exact same sentence -> ONE content component, cannot split leakage-free
    docs = [
        {"id": f"doc{i:03d}", "source_id": "test_source", "language": "hinc-deva", "text": "मैं अभी report finalize कर रहा हूँ"}
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="ONE content component|content grouping merged"):
        split(docs, SplitConfig(val_frac=0.3, seed=1))


def test_split_rejects_silly_val_frac(mini_docs):
    with pytest.raises(ValueError):
        split(list(mini_docs), SplitConfig(val_frac=0.9))


def test_check_docs_flags_unattributed_and_duplicate_ids(mini_docs):
    docs = [{**mini_docs[0]}, {**mini_docs[0]}]
    issues = check_docs(docs, "train")
    assert any(i.code == "duplicate-ids-train" for i in issues)
    issues2 = check_docs([{**mini_docs[0], "source_id": None}], "val")
    assert any(i.code == "unattributed-val" for i in issues2)


def test_overlap_report_reports_real_numbers(mini_docs):
    shared = {"id": "x1", "text": "यह वाक्य दोनों तरफ़ मौजूद है और यह लंबा पर्याप्त है ताकि अट्ठारह शब्द बनें ठीक है"}
    train = [shared, *mini_docs[:4]]
    val = [{**shared, "id": "x2"}, mini_docs[5]]
    ov = overlap_report(train, val, 8)
    assert not ov["zero_overlap"]
    assert ov["unique_overlapping_ngrams"] >= 1
    assert ov["examples"] and "ngram" in ov["examples"][0]


def test_cross_split_near_duplicate_detection(mini_docs):
    base = mini_docs[0]
    tweak = {**base, "id": "tweaked", "text": base["text"][:-6] + " कुछ और।"}
    rep = cross_split_near_dups([base], [tweak], threshold=0.6)
    assert rep["pairs"] >= 1

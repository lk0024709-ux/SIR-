"""Dataset manifest and schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_manifest_schema_validation():
    from training.data.manifest import load_manifest, validate_manifest

    m = load_manifest(Path("data/manifests/sources.yaml"))
    # must have at least one blocked and one available
    statuses = [s.raw["status"] for s in m.sources.values()]
    assert "blocked" in statuses
    assert "available" in statuses
    # every source must have required fields
    for src in m.sources.values():
        for fld in ("id", "name", "license", "languages", "status"):
            assert fld in src.raw, f"missing {fld} in {src.id}"


def test_dataset_schema_validation(tmp_path):
    """Check that processed dataset files have required schema."""
    # create a minimal valid doc
    doc = {"id": "test001", "source_id": "test_source", "language": "hi", "text": "भारत का मौसम अच्छा है", "chars": 20}
    # check required fields
    for field in ("id", "source_id", "language", "text"):
        assert field in doc
    assert isinstance(doc["text"], str) and len(doc["text"]) > 0
    assert doc["language"] in ("hi", "en", "hinc-latn", "hinc-deva", "unknown")

    # check that load_documents accepts jsonl
    from training.data.clean import load_documents

    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = list(load_documents(p))
    assert len(loaded) == 1
    assert loaded[0]["id"] == "test001"


def test_processed_dataset_exists_and_has_no_leak():
    # smoke dataset should exist and pass leakage check
    p = Path("data/processed/smoke/train.jsonl")
    if not p.exists():
        pytest.skip("smoke dataset not built yet")
    from training.data.validate import load_jsonl, overlap_report

    train = load_jsonl(Path("data/processed/smoke/train.jsonl"))
    val = load_jsonl(Path("data/processed/smoke/val.jsonl"))
    assert len(train) > 0 and len(val) > 0
    ov = overlap_report(train, val, 8)
    assert ov["zero_overlap"], f"leakage detected: {ov['examples'][:2]}"


def test_tokenizer_artifacts_are_provisional():
    import json as js

    chosen = Path("tokenizer/artifacts/smoke/chosen.json")
    if not chosen.exists():
        pytest.skip("tokenizer not trained")
    sel = js.loads(chosen.read_text())
    assert "eligible_for_release" in sel
    assert sel["eligible_for_release"] is False, "smoke tokenizer must be marked not eligible for release"
    assert "why_not_release_grade" in sel


def test_provenance_exists_and_is_complete():
    prov = Path("data/processed/smoke/provenance.json")
    if not prov.exists():
        pytest.skip("provenance not generated")
    data = json.loads(prov.read_text())
    assert "git_commit" in data
    assert "script_sha256" in data or "provenance" in str(data).lower()
    assert "sources" in data

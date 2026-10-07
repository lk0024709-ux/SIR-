"""Training records: schema, provenance policy, dedup and leakage-safe splitting.

The tests concentrate on the policy, because that is where the schema earns its keep: a synthetic
record with no named teacher, or an uncertain licence, must not reach a training split.
"""

from __future__ import annotations

import pytest

from curriculum.graph import load_default_graph
from training.data.records import (
    DIFFICULTIES,
    LANGUAGES,
    PROVENANCE_CLASSES,
    RECORD_KINDS,
    TrainingRecord,
    assert_trainable,
    check_against_graph,
    dedupe,
    find_duplicates,
    load_records,
    partition_records,
    split_records,
    stats,
    validate_records,
    write_records,
)


def _record(rid: str = "r001", **overrides) -> TrainingRecord:
    base = dict(
        record_id=rid,
        kind="practice",
        task="What is 17% of 240?",
        response="40.8",
        curriculum_node="math.g05",
        domain="foundation",
        subject="mathematics",
        language="en",
        difficulty="intermediate",
        provenance="synthetic",
        verification_status="multi_teacher_agreement",
        license_status="cleared",
        source="teacher-pipeline-v0:teacher_a/teacher_b",
        teacher_models=["teacher_a", "teacher_b"],
    )
    base.update(overrides)
    return TrainingRecord(**base)


def test_learning_loop_is_the_record_kind_vocabulary():
    assert RECORD_KINDS == (
        "learn",
        "practice",
        "error_diagnosis",
        "correction",
        "application",
        "explanation",
        "verification",
        "transfer",
        "review",
        "progression",
    )


def test_valid_record_passes_validation():
    assert _record().validate() == []


def test_synthetic_record_must_name_its_teachers():
    errors = _record(teacher_models=[]).validate()
    assert any("teacher_models" in e for e in errors)


def test_error_kinds_require_error_tags():
    errors = _record(kind="error_diagnosis", error_tags=[]).validate()
    assert any("error_tag" in e for e in errors)
    assert _record(kind="correction", error_tags=["CALCULATION_ERROR"]).validate() == []


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("kind", "vibes", "kind"),
        ("difficulty", "impossible", "difficulty"),
        ("language", "klingon", "language"),
        ("provenance", "found", "provenance"),
        ("license_status", "probably_fine", "license_status"),
        ("verification_status", "trust_me", "verification_status"),
        ("source", "", "source"),
        ("curriculum_node", "", "curriculum_node"),
    ],
)
def test_invalid_vocabulary_is_rejected(field, value, needle):
    errors = _record(**{field: value}).validate()
    assert any(needle in e for e in errors), errors


def test_rationale_style_must_match_content():
    assert any("rationale is empty" in e for e in _record(rationale_style="concise", rationale="").validate())
    assert _record(rationale_style="concise", rationale="Divide by the denominator.").validate() == []


def test_content_hash_ignores_id_and_timestamp_but_tracks_content():
    a = _record("r001", created_utc="2026-01-01T00:00:00Z")
    b = _record("r999", created_utc="2027-01-01T00:00:00Z")
    assert a.content_hash == b.content_hash
    c = _record("r001", response="40.9")
    assert c.content_hash != a.content_hash


def test_partition_quarantines_uncertain_and_rejects_rejected():
    records = [
        _record("r001"),
        _record("r002", provenance="uncertain"),
        _record("r003", license_status="unknown_quarantine"),
        _record("r004", provenance="rejected"),
        _record("r005", license_status="rejected"),
    ]
    parts = partition_records(records)
    assert [r.record_id for r in parts["trainable"]] == ["r001"]
    assert [r.record_id for r in parts["quarantine"]] == ["r002", "r003"]
    assert [r.record_id for r in parts["rejected"]] == ["r004", "r005"]


def test_assert_trainable_blocks_a_bad_split():
    with pytest.raises(ValueError):
        assert_trainable([_record("r001"), _record("r002", provenance="uncertain")])
    assert_trainable([_record("r001")])


def test_unverified_content_is_quarantined_not_trainable():
    unverified = _record("r001", verification_status="unverified")
    assert unverified.trainable is False
    parts = partition_records([unverified])
    assert [r.record_id for r in parts["quarantine"]] == ["r001"]
    with pytest.raises(ValueError):
        assert_trainable([unverified])


def test_duplicate_detection_and_dedupe():
    records = [_record("r001"), _record("r002"), _record("r003", response="different")]
    dupes = find_duplicates(records)
    assert len(dupes) == 1
    assert len(dedupe(records)) == 2


def test_split_is_deterministic_and_keeps_identical_content_together():
    records = [_record(f"r{i:03d}", task=f"task {i}") for i in range(60)]
    records.append(_record("rdup", task="task 0"))  # same content as r000
    first = split_records(records, val_fraction=0.25, seed=1)
    second = split_records(list(reversed(records)), val_fraction=0.25, seed=1)
    assert {r.record_id for r in first["train"]} == {r.record_id for r in second["train"]}
    side = {r.record_id: name for name, group in first.items() for r in group}
    assert side["r000"] == side["rdup"], "identical content must never straddle the split"
    assert first["val"], "a 25% validation split on 60 records should not be empty"


def test_split_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        split_records([_record()], val_fraction=0.0)


def test_graph_cross_check_flags_unknown_node_and_subject_drift():
    graph = load_default_graph()
    ok = _record("r001")
    drift = _record("r002", subject="science")
    unknown = _record("r003", curriculum_node="nope.g01")
    errors = check_against_graph([ok, drift, unknown], graph)
    assert any("not in the graph" in e for e in errors)
    assert any("disagrees with node" in e for e in errors)
    assert not check_against_graph([ok], graph)


def test_stats_reports_measured_counts_only():
    records = [
        _record("r001"),
        # same teaching content (language differs): this is separate material, not a duplicate
        _record("r002", language="hi"),
        # same content hash as r001 but different provenance/source: metadata is not part of the
        # content hash, so this still counts as one unique item (and is reported as a duplicate)
        _record("r003", provenance="uncertain", source="another-source"),
    ]
    payload = stats(records)
    assert payload["records"] == 3
    assert payload["unique_content"] == 2
    assert payload["duplicate_content_groups"] == 1
    assert payload["by_language"]["hi"] == 1
    assert payload["by_provenance"]["uncertain"] == 1
    assert payload["policy"] == {"trainable": 2, "quarantine": 1, "rejected": 0}
    assert payload["characters"]["total"] > 0


def test_jsonl_round_trip(tmp_path):
    records = [_record("r001"), _record("r002", language="hi")]
    path = write_records(tmp_path / "records.jsonl", records)
    loaded = load_records(path)
    assert [r.record_id for r in loaded] == ["r001", "r002"]
    assert validate_records(loaded) == []
    assert loaded[0].teacher_models == ["teacher_a", "teacher_b"]


def test_vocabulary_sets_are_not_empty():
    assert len(DIFFICULTIES) == 6 and "olympiad" in DIFFICULTIES
    assert "hinc-latn" in LANGUAGES
    assert set(PROVENANCE_CLASSES) == {
        "self_authored",
        "licensed",
        "synthetic",
        "public_domain",
        "uncertain",
        "rejected",
    }

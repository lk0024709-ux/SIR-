"""Coverage states must be driven by assessment evidence only.

The central property under test: generating material does not move a node, assessing it does — and
MASTERED requires volume, accuracy, difficulty, transfer and verification, not a good average.
"""

from __future__ import annotations

from curriculum.coverage import (
    AssessmentRecord,
    CoverageState,
    CoverageThresholds,
    build_report,
    compute_node_coverage,
    load_assessments,
    validate_assessments,
)
from curriculum.graph import load_default_graph

TH = CoverageThresholds(
    practice_min_attempts=3,
    mastery_min_attempts=8,
    mastery_accuracy=0.8,
    min_difficulty="intermediate",
    require_transfer=True,
    require_verification=True,
    weak_accuracy=0.6,
)


def _rec(kind: str, correct: int, total: int, difficulty: str = "intermediate", verified: bool = True) -> AssessmentRecord:
    return AssessmentRecord(
        node_id="math.g05",
        kind=kind,
        correct=correct,
        total=total,
        difficulty=difficulty,
        verified=verified,
        source="test-suite",
    )


def _mastery_records(n: int = 8) -> list[AssessmentRecord]:
    recs = [_rec("practice", 5, 5) for _ in range(n - 2)]
    recs.append(_rec("transfer", 4, 5))
    recs.append(_rec("verification", 5, 5))
    return recs


def test_no_evidence_is_not_started_even_with_blockers_explained():
    cov = compute_node_coverage("math.g05", [], TH)
    assert cov.state is CoverageState.NOT_STARTED
    assert "no assessment evidence" in cov.blockers[0]


def test_one_attempt_is_learning():
    cov = compute_node_coverage("math.g05", [_rec("practice", 1, 1)], TH)
    assert cov.state is CoverageState.LEARNING
    assert any("practice_min_attempts" in b for b in cov.blockers)


def test_low_accuracy_stays_practice():
    cov = compute_node_coverage("math.g05", [_rec("practice", 3, 10) for _ in range(10)], TH)
    assert cov.state is CoverageState.PRACTICE
    assert any("accuracy" in b for b in cov.blockers)


def test_enough_volume_and_accuracy_without_transfer_is_assessed_not_mastered():
    recs = [_rec("practice", 9, 10) for _ in range(10)]
    cov = compute_node_coverage("math.g05", recs, TH)
    assert cov.state is CoverageState.ASSESSED
    assert any("transfer" in b for b in cov.blockers)


def test_full_evidence_mastery_requires_transfer_and_verification():
    cov = compute_node_coverage("math.g05", _mastery_records(), TH)
    assert cov.state is CoverageState.MASTERED
    assert cov.blockers == []
    assert cov.has_transfer and cov.has_verification


def test_difficulty_floor_blocks_mastery():
    recs = _mastery_records()
    recs = [
        AssessmentRecord(**{**r.to_dict(), "difficulty": "elementary"})
        for r in recs
    ]
    cov = compute_node_coverage("math.g05", recs, TH)
    assert cov.state is CoverageState.ASSESSED
    assert any("difficulty_reached" in b for b in cov.blockers)


def test_unverified_evidence_blocks_mastery():
    recs = [AssessmentRecord(**{**r.to_dict(), "verified": False}) for r in _mastery_records()]
    cov = compute_node_coverage("math.g05", recs, TH)
    assert cov.state is CoverageState.ASSESSED
    assert any("independently verified" in b for b in cov.blockers)


def test_review_due_downgrades_mastery_to_review_required():
    cov = compute_node_coverage("math.g05", _mastery_records(), TH, review_due=True)
    assert cov.state is CoverageState.REVIEW_REQUIRED
    assert any("spaced-review" in b for b in cov.blockers)


def test_threshold_validation_rejects_nonsense():
    for bad in (
        CoverageThresholds(mastery_accuracy=1.5),
        CoverageThresholds(min_difficulty="impossible"),
        CoverageThresholds(practice_min_attempts=10, mastery_min_attempts=2),
    ):
        try:
            bad.check()
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not validate")


def test_assessment_validation_catches_zero_denominator_and_unknown_nodes():
    graph = load_default_graph()
    recs = [
        AssessmentRecord(node_id="nope.g01", kind="practice", correct=1, total=1, source="s"),
        AssessmentRecord(node_id="math.g05", kind="practice", correct=1, total=0, source=""),
    ]
    errors = validate_assessments(recs, graph)
    assert any("not in the graph" in e for e in errors)
    assert any("total must be > 0" in e for e in errors)
    assert any("needs a source" in e for e in errors)


def test_report_counts_and_ready_to_learn():
    graph = load_default_graph()
    report = build_report(graph, _mastery_records(), TH)
    counts = report.counts
    assert counts["MASTERED"] == 1
    assert counts["NOT_STARTED"] == len(graph) - 1
    # children of a mastered node become candidates to learn next
    assert "math.g06" in report.ready_to_learn
    payload = report.to_dict(graph)
    assert payload["counts"]["MASTERED"] == 1
    assert "by_subject" in payload


def test_load_assessments_round_trip(tmp_path):
    path = tmp_path / "assessments.jsonl"
    recs = _mastery_records()
    path.write_text("\n".join(AssessmentRecord(**r.to_dict()).to_dict().__str__().replace("'", '"') for r in recs), encoding="utf-8")
    import json

    path.write_text("\n".join(json.dumps(r.to_dict()) for r in recs) + "\n", encoding="utf-8")
    loaded = load_assessments(path)
    assert len(loaded) == len(recs)
    assert loaded[0].node_id == "math.g05"
    assert validate_assessments(loaded) == []

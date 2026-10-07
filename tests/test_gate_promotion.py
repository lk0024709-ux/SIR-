"""Promotion gate: PROMOTE is only reachable through measured, verified, regression-checked evidence.

Every test here corresponds to a way a weaker system fakes a promotion: missing evidence, too few
cases, an unverified suite, skipped leakage/reproducibility checks, or a regression that was noticed
and then ignored.
"""

from __future__ import annotations

import json

from development.generations import load_contracts
from development.promotion import AxisEvidence, evaluate_promotion, load_axis_evidence
from development.regression import compare_runs
from sir_paths import REPO_ROOT

CONTRACTS, AXES = load_contracts()
NANO = CONTRACTS["nano"]
REQUIRED = list(NANO.required_evaluations)


def _evidence(*, score: float = 0.99, cases: int = 10, verified: bool = True) -> dict[str, AxisEvidence]:
    return {
        axis: AxisEvidence(
            axis=axis,
            score=score,
            cases=cases,
            source="evaluation/results/unit.json",
            suite_hash="h1",
            verified=verified,
        )
        for axis in REQUIRED
    }


def _regression(verdict_hash: str = "h1"):
    axes = {axis: {"score": 0.9} for axis in REQUIRED}
    return compare_runs({"axes": axes, "suite_hash": verdict_hash}, {"axes": axes, "suite_hash": verdict_hash})


def test_full_verified_evidence_promotes():
    decision = evaluate_promotion(
        NANO,
        _evidence(),
        evidence_meta={"verified_evidence": True, "suite_hash": "h1"},
        regression=_regression(),
        leakage={"present": True, "pass": True, "path": "l.json"},
        reproducibility={"present": True, "pass": True, "path": "r.json"},
    )
    assert decision.decision == "PROMOTE"
    assert decision.promoted is True
    assert decision.blocking == []
    assert all(c.status == "MET" for c in decision.criteria)


def test_missing_axis_evidence_holds_and_is_named():
    evidence = _evidence()
    del evidence["mathematics"]
    decision = evaluate_promotion(NANO, evidence, evidence_meta={"verified_evidence": True})
    assert decision.decision == "HOLD"
    assert any(b.startswith("axis:mathematics:") for b in decision.blocking)
    assert any(c.name == "axis:mathematics" and c.status == "MISSING" for c in decision.criteria)


def test_score_below_threshold_holds():
    decision = evaluate_promotion(NANO, _evidence(score=0.01), evidence_meta={"verified_evidence": True})
    assert decision.decision == "HOLD"
    assert all(c.status == "NOT_MET" for c in decision.criteria if c.name.startswith("axis:"))


def test_too_few_cases_is_missing_evidence_not_a_pass():
    decision = evaluate_promotion(NANO, _evidence(cases=1), evidence_meta={"verified_evidence": True})
    assert decision.decision == "HOLD"
    assert any("min_cases_per_axis" in b for b in decision.blocking)


def test_unverified_evidence_cannot_promote():
    decision = evaluate_promotion(
        NANO,
        _evidence(verified=False),
        evidence_meta={"verified_evidence": False},
        regression=_regression(),
    )
    assert decision.decision == "HOLD"
    assert any(c.name == "verified_evidence" and c.status == "MISSING" for c in decision.criteria)


def test_missing_leakage_or_reproducibility_report_holds():
    decision = evaluate_promotion(NANO, _evidence(), evidence_meta={"verified_evidence": True}, regression=_regression())
    assert decision.decision == "HOLD"
    names = {c.name: c.status for c in decision.criteria}
    assert names["no_leakage"] == "MISSING"
    assert names["reproducibility"] == "MISSING"


def test_failed_leakage_report_holds_even_with_perfect_scores():
    decision = evaluate_promotion(
        NANO,
        _evidence(),
        evidence_meta={"verified_evidence": True},
        regression=_regression(),
        leakage={"present": True, "pass": False, "path": "l.json"},
        reproducibility={"present": True, "pass": True},
    )
    assert decision.decision == "HOLD"
    assert any(c.name == "no_leakage" and c.status == "NOT_MET" for c in decision.criteria)


def test_missing_baseline_comparison_holds():
    decision = evaluate_promotion(NANO, _evidence(), evidence_meta={"verified_evidence": True}, regression=None)
    assert any(c.name == "no_axis_regression" and c.status == "MISSING" for c in decision.criteria)


def test_regression_failure_holds_and_invalid_baseline_warns():
    bad = compare_runs({"axes": {a: {"score": 0.9} for a in REQUIRED}, "suite_hash": "h1"},
                       {"axes": {a: {"score": 0.1} for a in REQUIRED}, "suite_hash": "h1"})
    decision = evaluate_promotion(NANO, _evidence(), evidence_meta={"verified_evidence": True}, regression=bad)
    assert decision.decision == "HOLD"
    assert any(c.name == "no_axis_regression" and c.status == "NOT_MET" for c in decision.criteria)

    invalid = compare_runs({"axes": {a: {"score": 0.9} for a in REQUIRED}, "suite_hash": "h1"},
                           {"axes": {a: {"score": 0.9} for a in REQUIRED}, "suite_hash": "h2"})
    decision = evaluate_promotion(NANO, _evidence(), evidence_meta={"verified_evidence": True}, regression=invalid)
    assert decision.decision == "HOLD"
    assert any("INVALID" in w for w in decision.warnings)


def test_no_evidence_at_all_holds_every_axis():
    decision = evaluate_promotion(NANO, {})
    assert decision.decision == "HOLD"
    assert len([c for c in decision.criteria if c.name.startswith("axis:")]) == len(REQUIRED)
    assert "parameter count" in decision.to_dict()["note"]


def test_load_axis_evidence_reads_a_real_suite_result(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "suite_hash": "abc123",
                "checkpoint": "runs/x/best.pt",
                "verified_evidence": True,
                "created_utc": "2026-10-07T00:00:00Z",
                "axes": {
                    "mathematics": {"cases": 6, "scored": 5, "score": 0.8},
                    "language": {"cases": 4, "scored": 0, "score": None},
                },
            }
        ),
        encoding="utf-8",
    )
    evidence, meta = load_axis_evidence(path)
    assert evidence["mathematics"].score == 0.8 and evidence["mathematics"].cases == 5
    assert evidence["language"].score is None
    assert meta["verified_evidence"] is True and meta["checkpoint"] == "runs/x/best.pt"


def test_committed_nano_contract_is_internally_consistent():
    assert NANO.target_params == 25_000_000
    assert NANO.training_status == "UNTRAINED"
    assert NANO.verification_status == "UNVERIFIED"
    assert NANO.validate(AXES) == []
    assert (REPO_ROOT / "configs" / "generation_contracts.yaml").exists()

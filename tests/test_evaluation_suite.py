"""The curated evaluation layer: deterministic scoring, honest counts, and no silent passes.

The tests pin the three rules that make a suite result usable as promotion evidence at all: manual
cases are counted but never scored, a crashing answer is an `error` rather than a wrong answer, and
`verified_evidence` is true only when every *scored* case was checked by a deterministic mode.
"""

from __future__ import annotations

import json

import pytest

from evaluation.suite import (
    AXES,
    EvalCase,
    StaticAnswerer,
    aggregate,
    load_suite,
    run_suite,
    score_case,
    suite_hash,
    summarise,
    write_result,
)
from sir_paths import REPO_ROOT

DEV_SUITE = REPO_ROOT / "evaluation" / "suites" / "dev_smoke_suite.jsonl"


def _case(case_id: str, **overrides) -> EvalCase:
    base = dict(
        case_id=case_id,
        axis="knowledge",
        prompt="Name the capital of Rajasthan.",
        expected="Jaipur",
        verification="exact",
        source="tests/fixtures/suite_v0",
    )
    base.update(overrides)
    return EvalCase(**base)


def test_case_validation_catches_missing_expected_answer_for_scored_modes():
    assert _case("c1").validate() == []
    errors = _case("c2", expected=None).validate()
    assert any("needs an expected answer" in e for e in errors)
    assert any("axis" in e for e in _case("c3", axis="vibes").validate())
    assert any("source" in e for e in _case("c4", source="").validate())


def test_manual_cases_may_not_carry_machine_checked_alternatives():
    errors = _case("c5", verification="manual", expected=None, accepted=["Jaipur"]).validate()
    assert any("accepted" in e for e in errors)


def test_exact_scoring_accepts_case_and_punctuation_variants():
    case = _case("c1")
    assert score_case(case, "jaipur").status == "pass"
    assert score_case(case, "Jaipur.").status == "pass"
    assert score_case(case, "Jodhpur").status == "fail"
    assert score_case(case, "").status == "fail"


def test_numeric_tolerance_and_accepted_alternatives():
    case = _case("c2", axis="mathematics", verification="numeric", expected="40.8", tolerance=0.0)
    assert score_case(case, "40.80").status == "pass"
    assert score_case(case, "41").status == "fail"

    tolerant = _case("c3", axis="mathematics", verification="numeric", expected="40.8", tolerance=0.5)
    assert score_case(tolerant, "41").status == "pass"

    alternatives = _case("c4", accepted=["Jaipur", "the Pink City"], expected="Jaipur")
    assert score_case(alternatives, "the Pink City").status == "pass"


def test_arithmetic_claims_mode_verifies_the_claim_not_the_format():
    case = _case(
        "c5",
        axis="mathematics",
        prompt="Compute 17% of 240.",
        verification="arithmetic_claims",
        expected="40.8",
    )
    assert score_case(case, "17% of 240 = 40.8").status == "pass"
    assert score_case(case, "17% of 240 = 39.8").status == "fail"


def test_manual_case_is_unscored_and_never_a_pass():
    case = _case("c6", verification="manual", expected=None)
    result = score_case(case, "a free-form answer a reviewer must grade")
    assert result.status == "unscored"
    assert "counted, not counted as a pass" in result.detail


def test_integrity_only_checks_presence_only():
    case = _case("c7", verification="integrity_only", expected=None)
    assert score_case(case, "something").status == "pass"
    result = score_case(case, "   ")
    assert result.status == "fail"
    assert result.checks[0]["evidence"]["note"] == "integrity_only checks presence, not correctness"
    passed = score_case(case, "something")
    assert "content was NOT verified" in passed.detail


def test_run_suite_counts_unscored_and_errors_separately():
    cases = [
        _case("c1"),
        _case("c2", axis="mathematics", verification="numeric", expected="40.8"),
        _case("c3", verification="manual", expected=None),
        _case("c4"),
    ]

    def boom(case: EvalCase) -> str:
        raise RuntimeError("answerer exploded")

    answers = {"c1": "Jaipur", "c2": "40.8", "c3": "reviewed later", "c4": boom}
    payload = run_suite(cases, StaticAnswerer(answers), suite_paths=[DEV_SUITE])
    assert payload["cases_total"] == 4
    assert payload["cases_scored"] == 2
    assert payload["cases_unscored"] == 1
    assert payload["cases_error"] == 1, "a crashing answerer must be visible, not a wrong answer"
    assert payload["verified_evidence"] is True
    assert payload["axes"]["knowledge"]["passed"] == 1
    assert payload["axes"]["knowledge"]["error"] == 1
    assert payload["suites"] == ["evaluation/suites/dev_smoke_suite.jsonl"]


def test_suite_of_only_manual_cases_is_not_verified_evidence():
    payload = run_suite([_case("c1", verification="manual", expected=None)], StaticAnswerer({"c1": "x"}))
    assert payload["cases_scored"] == 0
    assert payload["verified_evidence"] is False
    assert payload["axes"]["knowledge"]["score"] is None


def test_aggregate_scores_only_scored_cases():
    results = [
        score_case(_case("c1"), "Jaipur"),
        score_case(_case("c2"), "Jodhpur"),
        score_case(_case("c3", verification="manual", expected=None), "x"),
    ]
    axes = aggregate(results)
    knowledge = axes["knowledge"]
    assert knowledge["cases"] == 3 and knowledge["scored"] == 2
    assert knowledge["passed"] == 1 and knowledge["failed"] == 1
    assert knowledge["score"] == 0.5
    assert knowledge["unscored"] == 1


def test_suite_hash_is_stable_and_order_independent():
    cases = [_case("c1"), _case("c2")]
    assert suite_hash(cases) == suite_hash(list(reversed(cases)))
    assert suite_hash(cases) != suite_hash([_case("c1"), _case("c2", expected="Udaipur")])


def test_write_result_and_summarise(tmp_path):
    payload = run_suite([_case("c1")], StaticAnswerer({"c1": "Jaipur"}))
    path = write_result(payload, tmp_path / "result.json")
    assert path.exists()
    text = summarise(json.loads(path.read_text(encoding="utf-8")))
    assert "suite hash" in text and "knowledge" in text


def test_committed_dev_suite_is_valid_and_covers_every_axis():
    if not DEV_SUITE.exists():
        pytest.skip("dev suite not present")
    cases = load_suite(DEV_SUITE)
    assert len(cases) >= 20
    errors = {c.case_id: c.validate() for c in cases}
    assert all(not e for e in errors.values()), {k: v for k, v in errors.items() if v}
    covered = {c.axis for c in cases}
    assert covered == set(AXES), f"uncovered axes: {set(AXES) - covered}"
    manual = [c for c in cases if c.verification in ("manual", "none")]
    assert manual, "the suite must contain deliberately unscored cases so the runner's honesty is testable"
    assert all(c.source for c in cases)


def test_load_suite_refuses_an_invalid_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"case_id": "x", "axis": "nonsense", "prompt": "p", "source": "s"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(path)

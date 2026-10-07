"""Error-based learning: the twelve-category taxonomy and the wrong→correct→transfer chain.

The tests insist on two things the taxonomy is for: a case must be attributable (classified_by +
source), and it must be *complete* — a skeleton with no explanation is not training data and must not
be convertible into records.
"""

from __future__ import annotations

import json

import pytest

from development.errors import (
    CATEGORY_BY_CODE,
    LEARNING_SEQUENCE,
    ErrorCase,
    ErrorCategory,
    cases_from_evaluation,
    categories_for_codes,
    load_cases,
)

REQUIRED = {
    "FACTUAL_ERROR",
    "LOGICAL_ERROR",
    "CALCULATION_ERROR",
    "LANGUAGE_ERROR",
    "MISINTERPRETATION",
    "HALLUCINATION",
    "MISSING_CONTEXT",
    "BAD_ASSUMPTION",
    "INCOMPLETE_REASONING",
    "OVERGENERALIZATION",
    "SOURCE_ERROR",
    "CODE_ERROR",
}


def _case(**overrides) -> ErrorCase:
    base = dict(
        case_id="err.math.g05.001",
        curriculum_node="math.g05",
        category="CALCULATION_ERROR",
        wrong_answer="17% of 240 = 39.8",
        why_wrong="The percentage was applied as 0.17 * 234 instead of 0.17 * 240.",
        correct_principle="A percentage of a quantity is rate x quantity.",
        corrected_answer="17% of 240 = 40.8",
        new_similar_problem="What is 23% of 150?",
        transfer_problem="A shirt costs 800 rupees with a 25% discount; what is the price?",
        subject="mathematics",
        classified_by="teachers/verify.py:arithmetic_claims",
        verification_status="deterministically_verified",
        provenance="self_authored",
        source="evaluation/results/suite_v0.json",
    )
    base.update(overrides)
    return ErrorCase(**base)


def test_taxonomy_has_exactly_the_twelve_contract_categories():
    assert {c.value for c in ErrorCategory} == REQUIRED
    assert LEARNING_SEQUENCE == (
        "wrong_answer",
        "category",
        "why_wrong",
        "correct_principle",
        "corrected_answer",
        "new_similar_problem",
        "transfer_problem",
    )


def test_deterministic_codes_map_onto_categories_and_non_errors_do_not():
    assert categories_for_codes(["CALCULATION_ERROR"]) == ["CALCULATION_ERROR"]
    assert categories_for_codes(["ANSWER_MISMATCH"]) == ["FACTUAL_ERROR"]
    assert categories_for_codes(["MISSING_CITATION", "CONTRADICTS_SOURCE"]) == ["SOURCE_ERROR"]
    # "nobody checked yet" is not an error category
    assert categories_for_codes(["REQUIRES_HUMAN_REVIEW", "NO_VERIFIABLE_CLAIM"]) == []
    assert "REQUIRES_HUMAN_REVIEW" not in CATEGORY_BY_CODE
    assert len(CATEGORY_BY_CODE) >= 12


def test_a_complete_case_validates_and_renders_the_learning_sequence():
    case = _case()
    assert case.validate() == []
    pairs = case.learning_sequence()
    assert [name for name, _ in pairs] == [
        "wrong answer",
        "error classification",
        "why it is wrong",
        "correct principle",
        "corrected answer",
        "new similar problem",
        "transfer problem",
    ]
    rendered = case.render()
    assert "CALCULATION_ERROR" in rendered and "40.8" in rendered


def test_unnamed_category_and_attribution_are_refused():
    assert any("taxonomy" in e for e in _case(category="VIBES_ERROR").validate())
    assert any("classified_by" in e for e in _case(classified_by="").validate())
    assert any("source" in e for e in _case(source="").validate())


def test_incomplete_case_cannot_become_training_data():
    incomplete = _case(why_wrong="", corrected_answer="", transfer_problem="")
    with pytest.raises(ValueError):
        incomplete.to_training_records()


def test_conversion_emits_diagnosis_correction_and_transfer_records():
    records = _case().to_training_records()
    assert [r.kind for r in records] == ["error_diagnosis", "correction", "transfer"]
    for record in records:
        assert record.validate() == []
        assert record.error_tags == ["CALCULATION_ERROR"]
        assert record.curriculum_node == "math.g05"
        assert record.verification_status == "deterministically_verified"
    assert records[1].response == "17% of 240 = 40.8"


def test_last_two_learning_loop_stages_are_present_in_the_transfer_record():
    transfer = _case().to_training_records()[-1]
    assert "discount" in transfer.task
    assert "17% of 240 = 40.8" in transfer.response


def test_cases_from_evaluation_turns_failures_into_self_invalidating_skeletons(tmp_path):
    results = tmp_path / "suite.json"
    results.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "m1",
                        "status": "fail",
                        "axis": "mathematics",
                        "answer": "39.8",
                        "error_codes": ["CALCULATION_ERROR"],
                        "curriculum_node": "math.g05",
                        "language": "en",
                        "difficulty": "intermediate",
                    },
                    {"case_id": "m2", "status": "pass", "axis": "mathematics", "answer": "40.8"},
                ]
            }
        ),
        encoding="utf-8",
    )
    drafts = cases_from_evaluation(results)
    assert len(drafts) == 1, "only failures become error cases"
    draft = drafts[0]
    assert draft["category"] == "CALCULATION_ERROR"
    assert draft["wrong_answer"] == "39.8"
    assert draft["why_wrong"] == "" and draft["corrected_answer"] == ""
    case = ErrorCase.from_dict(draft)
    assert case.validate(), "an unfilled skeleton must not validate"
    with pytest.raises(ValueError):
        case.to_training_records()


def test_committed_error_case_file_is_valid():
    import pathlib

    path = pathlib.Path("data/errors/error_cases.jsonl")
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        pytest.skip("no curated error cases committed yet")
    cases = load_cases(path)
    assert cases
    for case in cases:
        assert case.validate() == [], f"{case.case_id}: {case.validate()}"

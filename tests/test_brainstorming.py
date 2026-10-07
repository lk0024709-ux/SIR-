"""Brainstorming rubric: a mechanical, offline, deterministic score — and its stated limits.

The rubric is allowed to be a lexical proxy, but it is not allowed to *pretend*: novelty without a
reference set, or any comparison dimension with fewer than two ideas, must come back UNSCORED rather
than as a silent zero or a silent pass.
"""

from __future__ import annotations

import pytest

from development.brainstorming import (
    WORKFLOW,
    BrainstormAttempt,
    distinct_idea_ratio,
    score_attempt,
    summarise,
    workflow_coverage,
)
from sir_paths import REPO_ROOT, load_config

RUBRIC = load_config(REPO_ROOT / "evaluation" / "rubrics" / "brainstorming_v0.yaml")

RICH = BrainstormAttempt(
    attempt_id="bs.001",
    problem="Reduce water use in a village school with a limited budget.",
    constraints=["budget under 50000 rupees", "no electricity during the day"],
    ideas=[
        "Rainwater harvesting tanks because the roof area collects enough monsoon water; a 20000 litre tank costs about 45000 rupees.",
        "Low-flow taps plus a repair schedule so that leaking pipes are fixed within 2 days and waste drops.",
        "A student water committee with weekly checks because behaviour change reduces use by roughly 15 percent.",
    ],
    alternatives=["Borewell recharge pits", "Tanker delivery contract"],
    tradeoffs=["Option 1 is cheap but depends on rainfall, whereas option 2 is reliable but costs more per litre."],
    challenges=["We assume the roof area is large enough — measure it first.", "Risk: the committee may stop meeting after a month."],
    verification_steps=["measure the roof area", "compare tank quotes from three suppliers"],
    cross_domain_links=["urban drainage design uses the same runoff calculation"],
    synthesis="Compare option 1 and option 2: recommend rainwater harvesting first because cost per litre is lowest, verify rainfall data before scaling.",
    actionable_answer="Install a 20000 litre tank within 45000 rupees and review performance after one monsoon.",
    references=["Standard conservation manuals list rainwater harvesting as a common first measure for schools."],
    subject="science",
    source="tests/fixtures/brainstorming_v0",
)


def test_rubric_weights_are_explicit_and_sum_to_one():
    dims = RUBRIC["dimensions"]
    assert set(dims) == {
        "diversity",
        "relevance",
        "novelty",
        "feasibility",
        "reasoning_quality",
        "constraint_adherence",
        "verification",
        "synthesis",
    }
    assert abs(sum(d["weight"] for d in dims.values()) - 1.0) < 1e-9
    assert RUBRIC["reporting"]["unscored_dimensions_must_be_stated"] is True


def test_rich_attempt_scores_every_dimension():
    score = score_attempt(RICH, RUBRIC)
    assert score.unscored_dimensions == []
    assert score.total is not None and 0.0 <= score.total <= 1.0
    assert len(score.dimensions) == 8
    assert all(d.evidence for d in score.dimensions), "every dimension must show its evidence"
    assert any("mechanical rubric only" in n for n in score.notes)


def test_novelty_is_unscored_without_a_reference_set():
    attempt = BrainstormAttempt(**{**RICH.to_dict(), "references": []})
    score = score_attempt(attempt, RUBRIC)
    assert "novelty" in score.unscored_dimensions
    novelty = next(d for d in score.dimensions if d.name == "novelty")
    assert novelty.score is None
    assert "UNSCORED" in novelty.notes


def test_one_idea_cannot_be_compared_so_comparison_dimensions_are_unscored():
    attempt = BrainstormAttempt(attempt_id="bs.002", problem="Save water", ideas=["Store rainwater in a tank."], references=["ref"])
    score = score_attempt(attempt, RUBRIC)
    assert {"diversity", "relevance", "novelty", "feasibility"} <= set(score.unscored_dimensions)
    assert any("UNSCORED" in n for n in score.notes)


def test_no_constraints_declared_means_constraint_adherence_is_unscored():
    attempt = BrainstormAttempt(**{**RICH.to_dict(), "constraints": []})
    score = score_attempt(attempt, RUBRIC)
    assert "constraint_adherence" in score.unscored_dimensions


def test_constraint_violation_phrases_reduce_the_score():
    messy = BrainstormAttempt(
        **{**RICH.to_dict(), "ideas": RICH.ideas + ["Ignore the budget and buy the biggest system; no limit on cost."]}
    )
    clean_score = next(d for d in score_attempt(RICH, RUBRIC).dimensions if d.name == "constraint_adherence").score
    messy_score = next(d for d in score_attempt(messy, RUBRIC).dimensions if d.name == "constraint_adherence").score
    assert messy_score < clean_score


def test_scoring_is_deterministic():
    assert score_attempt(RICH, RUBRIC).to_dict() == score_attempt(RICH, RUBRIC).to_dict()


def test_distinct_idea_ratio_detects_near_duplicates():
    ratio, distinct = distinct_idea_ratio(["Store rainwater in a tank", "Store rainwater in a tank"])
    assert ratio == 0.5 and distinct == ["Store rainwater in a tank"]
    ratio2, distinct2 = distinct_idea_ratio(["Store rainwater in a tank", "Fix leaking taps quickly"])
    assert ratio2 == 1.0 and len(distinct2) == 2


def test_workflow_coverage_uses_the_declared_stage_names():
    coverage = workflow_coverage(RICH)
    assert set(coverage) == set(WORKFLOW)
    assert coverage["problem"] is True
    assert coverage["identify_constraints"] is True
    assert coverage["generate_hypotheses"] is True
    assert coverage["challenge_assumptions"] is True
    assert coverage["verify"] is True
    assert coverage["synthesize"] is True


def test_summary_reports_coverage_and_unscored_dimensions():
    scores = [score_attempt(RICH, RUBRIC), score_attempt(BrainstormAttempt(attempt_id="bs.003", problem="", ideas=[]), RUBRIC)]
    summary = summarise(scores)
    assert summary["attempts"] == 2
    assert isinstance(summary["mean_total"], float)
    assert summary["unscored_dimensions"]
    assert set(summary["workflow_stage_coverage"]) == set(WORKFLOW)
    assert "not a quality score" in summary["note"]


def test_an_empty_attempt_can_never_look_good():
    score = score_attempt(BrainstormAttempt(attempt_id="x", problem="", ideas=[]), RUBRIC)
    assert score.total != 1.0
    assert score.unscored_dimensions
    assert all(d.score is None or d.score <= 0.0 for d in score.dimensions)

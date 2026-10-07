# Brainstorming Evaluation

<!-- Status: implemented and tested. No model attempt has been scored yet; the rubric is v0. -->

## Why it is scored at all

Brainstorming is one of SIR's long-term target abilities, which makes it exactly the kind of ability
that gets claimed loosely. The rubric writes down what "good brainstorming" means *before* measuring
it, so a score can be argued with.

## Workflow (`evaluation/rubrics/brainstorming_v0.yaml`)

problem → clarify objective → identify constraints → decompose → retrieve knowledge → generate
hypotheses → cross-domain connect → alternative approaches → compare trade-offs → challenge
assumptions → adversarial critique → verify → synthesise → actionable answer.

`workflow_coverage()` reports which stages are visible in an attempt's text. It is a coverage count,
not a quality score, and says so.

## Dimensions and weights

| Dimension | Weight | Mechanical signal |
|-----------|--------|-------------------|
| diversity | 0.15 | distinct-idea ratio vs. near-duplicates |
| relevance | 0.15 | vocabulary overlap with the problem, per idea |
| novelty | 0.10 | max similarity to a supplied reference set |
| feasibility | 0.15 | quantities, resources, explicit steps |
| reasoning_quality | 0.15 | causal connectives, assumptions challenged |
| constraint_adherence | 0.10 | declared constraints referenced; violation phrases |
| verification | 0.10 | verification verbs, explicit checks, units |
| synthesis | 0.10 | recommendation + comparison + option references |

Total = weighted mean over **scored dimensions only**.

## The honesty rules

- **Novelty is UNSCORED without a reference set** — "new relative to what" is unanswerable otherwise.
- With fewer than two ideas, diversity/relevance/novelty/feasibility are UNSCORED rather than zero:
  comparison dimensions need something to compare.
- No constraints declared → constraint adherence is UNSCORED.
- Every dimension reports its evidence, and the result states that this is a **mechanical rubric**:
  novelty and feasibility in particular still require human or model review before any capability
  claim.
- An empty attempt can never look good (all dimensions are None or 0.0).

```python
from development.brainstorming import BrainstormAttempt, score_attempt, summarise
score = score_attempt(BrainstormAttempt.from_dict(payload), rubric)
print(score.total, score.unscored_dimensions)
```

## Status

| Component | Status |
|-----------|--------|
| Rubric + scorer + workflow coverage | ✅ Implemented, ✅ Verified (`tests/test_brainstorming.py`) |
| Scoring of real model attempts | ⏳ Not started |
| Rubric version | v0 — expected to change once real attempts are scored |

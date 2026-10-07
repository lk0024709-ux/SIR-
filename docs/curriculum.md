# SIR Curriculum

<!-- Status: graph + requirements + coverage + review + sampler are implemented (verified by tests). No trained model has been measured against the graph yet. -->

## Purpose

SIR is taught from a machine-readable curriculum instead of an ad-hoc corpus. The curriculum says
what is learned, in what order, and — the part that is usually missing — what counts as *evidence*
that it was learned.

## 1. The curriculum graph

`data/curriculum/curriculum_graph.yaml` (`sir_curriculum_v0`) is the source of truth.
Measured at this commit: **178 nodes, 221 dependency edges, 254 topics, 16 subjects**, split into
tracks `foundation` (114) / `bridge` (2) / `advanced` (55) / `meta` (7).

- **Foundation (Class 1–10)** covers mathematics, science, social science, Hindi, English, Hinglish,
  computer science, reasoning, general knowledge, scientific thinking and Indian knowledge.
- **Advanced** covers calculus, linear algebra, probability/statistics, discrete mathematics,
  optimisation, numerical methods, physics (including quantum, relativity, statistical mechanics),
  chemistry, biology, computer science (OS, networks, compilers, security), AI/ML, engineering and
  research methodology.

Node schema (`curriculum/schema.py`): dotted lowercase ids, `track`, `band`, `grade` (foundation
only), `topics`, `capabilities`, `depends_on`, `provenance`. Indian-knowledge nodes must carry an
`epistemic_note` (≥ 40 characters) separating evidence-based content from tradition, so mythology is
never presented as history.

The graph must be acyclic, Class *n+1* must depend on Class *n*, and advanced/bridge nodes must have
prerequisites — enforced by `curriculum/graph.py` and by the test suite.

## 2. Requirement contract

`configs/curriculum_requirements.yaml` states what the project promises. `curriculum/requirements.py`
checks the graph against it: required topics per subject, grade coverage, learning-loop stages present
as record kinds, required prerequisite chains, and that the generation order matches
`configs/sir_human_development.yaml`. The check currently **passes with 0 violations**; the tests also
prove it can fail (removing a topic, reversing a chain and breaking grade coverage are all detected).

## 3. Coverage states (assessment-driven)

`curriculum/coverage.py` derives a state per node from **assessment records only**:

`NOT_STARTED → LEARNING → PRACTICE → ASSESSED → MASTERED`, plus `REVIEW_REQUIRED`.

Material existing is not progress. `MASTERED` requires all of: enough attempts
(`mastery_min_attempts = 8`), accuracy ≥ 0.80, difficulty ≥ intermediate, at least one transfer item,
and independently verified evidence. Each unmet condition is reported in `blockers`, so a state can be
argued with. `REVIEW_REQUIRED` overrides a mastered node whose spaced-review date has passed.

Current measured state: **no committed assessment evidence, so every node is `NOT_STARTED`**. That is
the honest number, not a placeholder.

```bash
python -m curriculum.coverage --assessments <file.jsonl> --out evaluation/results/curriculum_coverage.json
```

## 4. Spaced review

`curriculum/review.py` schedules reviews on intervals `1, 3, 7, 16, 35, 75, 160` days, tracking
`last_seen_day`, `performance`, `reviews` and a `forgetting_risk` heuristic. The heuristic is monotone
in time and stated to be a heuristic — it is not a fitted memory model, and the module says so in its
summary output. State lives in `data/curriculum/review_state.jsonl` (generated, not committed).

## 5. Curriculum-aware sampler

`curriculum/sampler.py` builds a deterministic training plan from records + policy
(`configs/curriculum_sampling.yaml`):

| Bucket | Meaning |
|--------|---------|
| `current` | active frontier node |
| `review` | node whose spaced review is due |
| `weak_areas` | measured accuracy below `weak_accuracy` |
| `transfer` | cross-context items |
| `error_correction` | records carrying error tags (highest precedence) |

The mixture is configuration, not code (`current 0.65 / review 0.15 / weak_areas 0.10 / transfer 0.05
/ error_correction 0.05` at this commit; the documented example 70/15/10/5 is a starting point, not a
law). Allocation uses largest-remainder rounding, ordering is seeded and deterministic, and a bucket
that cannot be filled produces an explicit `shortfall` note plus a redistribution record — never
silent padding. Language mix is a soft target reported as drift.

```bash
python -m curriculum.sampler --records <records.jsonl> [--coverage coverage.json] --out plan.json
```

## Status

| Component | Status | Evidence |
|-----------|--------|----------|
| Graph + validator | ✅ Implemented, ✅ Verified | `tests/test_curriculum_graph.py` |
| Requirement contract | ✅ Implemented, ✅ Verified | `tests/test_curriculum_requirements.py`, 0 violations |
| Coverage states | ✅ Implemented, ✅ Verified | `tests/test_curriculum_coverage.py` |
| Spaced review | ✅ Implemented, ✅ Verified | `tests/test_curriculum_review.py` |
| Sampler | ✅ Implemented, ✅ Verified | `tests/test_curriculum_sampler.py` |
| Coverage of a *trained* model | ⏳ Not started | no model has been assessed against the graph |

## What this document is not

It is not a claim that SIR has learned the curriculum. It is the machinery that will decide — from
assessment evidence only — when that claim becomes permissible.

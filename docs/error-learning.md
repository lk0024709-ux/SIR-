# Error-Based Learning

<!-- Status: implemented and tested. 8 self-authored cases committed and unverified by design; 0 records have been trained on. -->

## Principle

A mistake is more instructive when it is classified, explained and then re-used — as a diagnosis
problem, a correction problem and a transfer problem. The loop is:

```
wrong answer → classification → why it is wrong → correct principle → corrected answer
             → new similar problem → transfer problem
```

## Taxonomy (`development/errors.py`, 12 categories)

`FACTUAL_ERROR`, `LOGICAL_ERROR`, `CALCULATION_ERROR`, `LANGUAGE_ERROR`, `MISINTERPRETATION`,
`HALLUCINATION`, `MISSING_CONTEXT`, `BAD_ASSUMPTION`, `INCOMPLETE_REASONING`, `OVERGENERALIZATION`,
`SOURCE_ERROR`, `CODE_ERROR`.

Deterministic verifier/critic codes map onto these categories (`ANSWER_MISMATCH → FACTUAL_ERROR`,
`LANGUAGE_SCRIPT_MISMATCH → LANGUAGE_ERROR`, `MISSING_CITATION → SOURCE_ERROR`, `CODE_TEST_FAILURE →
CODE_ERROR`, …). Codes that mean *"nobody checked yet"* (`REQUIRES_HUMAN_REVIEW`,
`NO_VERIFIABLE_CLAIM`) are deliberately **not** error categories: "unchecked" and "wrong" are
different things.

## Authoring rules

1. Every case must be complete — all seven stages non-empty — or it cannot be converted into records.
2. Every case names `classified_by` (which verifier, critic or reviewer assigned the category) and
   `source` (which evaluation run produced the mistake). A category nobody signs is not a category.
3. Failed evaluation cases can be turned into **skeletons** (`cases_from_evaluation`) containing the
   wrong answer and the mapped error code but *no* fabricated explanation. A skeleton stays invalid
   until a reviewer fills the prose, so an unfinished case can never leak into training.
4. Conversion never upgrades verification. Committed cases are `self_authored` + `unverified`, and the
   records built from them are therefore quarantined, not trainable.

```bash
python -m development.errors --cases data/errors/error_cases.jsonl --out-records /tmp/error_records.jsonl
python -m development.errors --from-suite evaluation/results/suite_v0.json --skeletons-out /tmp/skeletons.jsonl
```

## Committed inventory (`data/errors/error_cases.jsonl`, card in `data/errors/CARD.md`)

8 self-authored cases: arithmetic miscalculation, average-speed bad assumption, invalid syllogism,
acid overgeneralisation, Hindi script/register error, Python off-by-one, a geography fact, and an
unsourced "study proves" claim. Measured conversion: **24 records** (`error_diagnosis`, `correction`,
`transfer` per case), all `unverified` → **0 trainable, 24 quarantined**.

These numbers are small on purpose: the point of this phase is that the machinery refuses to inflate
them.

## Status

| Component | Status |
|-----------|--------|
| Taxonomy + mapping | ✅ Implemented, ✅ Verified (`tests/test_error_learning.py`) |
| Case schema + seven-stage chain | ✅ Implemented, ✅ Verified |
| Skeleton flow from evaluation failures | ✅ Implemented, ✅ Verified |
| Conversion to training records | ✅ Implemented, ✅ Verified |
| Independently verified cases | ⏳ 0 of 8 |

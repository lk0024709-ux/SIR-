# Generation Contracts and Promotion Gates

<!-- Status: implemented and tested. All nine generations are UNVERIFIED; SIR-Nano is UNTRAINED. No promotion has ever been granted. -->

## Why a contract per generation

A generation name is a promise. `configs/generation_contracts.yaml` makes each promise machine-readable
so it can be checked instead of asserted. Every generation carries:

`generation_id`, `name`, `training_stage`, `curriculum_scope`, `target_capabilities`,
`required_evaluations`, `promotion_requirements`, `known_limitations`, `training_status`,
`verification_status`, plus `model_config_ref`, `target_params` and `evidence` where they exist.

## The ladder

| id | name | training stage | required axes | target params | training status |
|----|------|----------------|---------------|---------------|-----------------|
| `nano` | SIR-Nano | foundation | 6 | 25,000,000 | UNTRAINED |
| `lite` | SIR-Lite | school mastery | 9 | — | PLANNED |
| `flash_lite` | SIR-Flash Lite | higher education | 10 | — | PLANNED |
| `flash` | SIR-Flash | advanced general | 11 | — | PLANNED |
| `pro` | SIR-Pro | professional | 11 | — | PLANNED |
| `pro_plus` | SIR-Pro+ | advanced professional | 11 | — | PLANNED |
| `pro_max` | SIR-Pro Max | deep problem solving | 11 | — | PLANNED |
| `ultra` | SIR-Ultra | research oriented | 11 | — | PLANNED |
| `expert` | SIR-Expert | specialist research | 11 | — | PLANNED |

Names and order are pinned to `configs/sir_human_development.yaml`; a test fails if they drift.
Required axes only ever grow along the ladder, and minimum scores never decrease.

Honesty rules enforced by `GenerationContract.validate()`:

- `TRAINED_*` requires evidence paths; `TRAINED_VERIFIED` additionally requires
  `verification_status: VERIFIED`;
- `PLANNED`/`ARCHITECTURE_ONLY` requires `known_limitations`;
- promotion thresholds must exist for **every** required axis and lie in `[0, 1]`;
- unknown axes are rejected.

## Promotion gate (`development/promotion.py`)

`PROMOTE` requires all of: every required axis present with ≥ `min_cases_per_axis` scored cases,
every score at or above its threshold, independently verified evidence, no leakage-gate failure,
a reproducibility report, and a regression comparison that is not `FAIL` or `INVALID`.

`MISSING` never passes. A missing leakage or reproducibility report is `MISSING`, not a warning. When
any criterion is unmet the decision is `HOLD` and each blocker is printed with its detail. The decision
note states plainly that promotion is not a statement about parameter count and not a capability claim
beyond the listed axes.

## Regression gate (`development/regression.py`)

Candidates are compared per axis against a recorded baseline with a tolerance (default 0.02,
overridable per axis). Two runs are only comparable when their `suite_hash` matches; otherwise the
verdict is `INVALID` and no score difference is reported. A missing axis counts as a regression, and
improvements elsewhere never compensate for it.

## Readiness gate (`development/readiness.py`)

Answers "may we start a serious pretraining run?", levels
`NOT_READY < PILOT_ONLY < READY_FOR_PARTIAL < READY_FOR_SERIOUS`, with thresholds
`50,000,000` train tokens (partial) and `200,000,000` (serious) from
`configs/curriculum_sampling.yaml`. Checks cover the curriculum contract, generation contracts,
dataset provenance gates, measured tokens, tokenizer availability, evaluation suites, measured
coverage and experiment tracking. An unknown check is never a pass, and the level is the weakest link.

Measured today: **`NOT_READY`** — the corpus is blocked at 0 measured tokens by provenance gate G5, no
coverage has been measured, and no experiment record exists. The gate therefore refuses serious
pretraining, which is the correct answer.

```bash
python -m development.generations            # contract table + status summary
python -m development.readiness              # writes evaluation/results/training_readiness.json
python -m development.promotion --generation nano --evidence evaluation/results/suite_v0.json
```

## Status

| Component | Status |
|-----------|--------|
| Contract file + validator | ✅ Implemented, ✅ Verified (`tests/test_generation_contracts.py`) |
| Promotion gate | ✅ Implemented, ✅ Verified (`tests/test_gate_promotion.py`) |
| Regression gate | ✅ Implemented, ✅ Verified (`tests/test_gate_regression.py`) |
| Readiness gate | ✅ Implemented, ✅ Verified (`tests/test_readiness.py`) |
| Any promotion | ❌ Never granted; nothing to promote |

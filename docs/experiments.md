# Experiment Tracking

<!-- Status: implemented and tested. Zero experiment records exist at this commit; the first run that should be recorded is the 0.9M smoke run, whose train_log.json is not committed. -->

## Why records, not a leaderboard

An experiment record exists so that a run can be re-identified and audited later — by a person who
does not trust the summary. `development/experiments.py` therefore refuses to write incomplete
records.

## Record schema

`experiment_id`, `created_utc`, `status`, and:

| Section | Required content |
|---------|------------------|
| `git` | commit (dirty flag + count of dirty files) |
| `dataset` | processed-dir fingerprint: file list, bytes, `tree_sha256` |
| `tokenizer` | spec path and/or SHA-256 |
| `model` | measured `parameter_count` (not the target) + config + config hash |
| `training` | `steps`, `tokens_seen`, `seed` (+ batch, optimizer, wall-clock, tokens/sec where available) |
| `evaluation` | trainer val loss/perplexity, tokens scored, external suite results |
| `known_failures` | required when status is `failed`/`aborted` |

Validation rules: a `completed` run **without evaluation results is refused**, and a failed/aborted
run without `known_failures` is refused. Records are written to `experiments/records/<id>.json` and
`load_records()` + `index()` give the auditable table.

```bash
python -m development.experiments --from-train-log runs/<run>/train_log.json \
    --processed-dir data/processed/smoke --tokenizer-spec <spec.json>
python -m development.experiments --list
```

`from_train_log` reads a real run's log: steps come from the maximum logged step, parameter count from
the run's counted parameters, and the dataset from the directory's `meta.json` + token stream hash.

## Relation to the gates

The readiness gate counts experiment records as a blocking requirement: without one, no serious
pretraining run is cleared. Promotion evidence (suite results) carries `suite_hash` and checkpoint
identity so a decision can be traced back to the record that produced it.

## Status

| Component | Status |
|-----------|--------|
| Record schema + validation | ✅ Implemented, ✅ Verified (`tests/test_experiments.py`) |
| `from_train_log` extraction | ✅ Implemented, ✅ Verified |
| Index/table + CLI | ✅ Implemented, ✅ Verified |
| Committed experiment records | ⏳ 0 — the smoke run predates this schema |

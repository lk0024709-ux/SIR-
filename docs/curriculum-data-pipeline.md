# Curriculum Data-Generation Pipeline

<!-- Status: implemented and tested. No external teacher transcripts are committed yet, so the measured output of this pipeline is currently zero accepted records (see "Measured output"). -->

## Purpose

Turn curriculum nodes into training material **without inventing facts and without losing
provenance**. The pipeline is offline by construction: external models are teachers whose answers are
recorded as transcripts, never runtime dependencies of SIR.

## Flow

```
curriculum node ──► request plan (kind x language x difficulty, seeded)
        │
        ├─► providers: recorded teacher transcripts (JSONL) · deterministic template (no facts)
        │
        ├─► critics  (field completeness, placeholders, degeneracy, language/script, arithmetic)
        │
        ├─► consensus + verification  (agreement · deterministic resolution · flag · reject)
        │
        └─► TrainingRecord  →  provenance partition: trainable / quarantine / rejected
                                    │
                                    └─► dedupe → train/val split (identical content stays together)
```

## Teachers and providers (`teachers/providers.py`)

`TeacherSpec` records `model_id`, `vendor`, `version`, `terms_note` and `output_reuse_permitted`.
There is deliberately **no network client in this repository**: a teacher's output enters the pipeline
as a recorded transcript (`RecordedProvider.from_jsonl`), and `unset/unknown` reuse terms force
quarantine. The `TemplateProvider` is deterministic scaffolding used to exercise the pipeline; it
carries no factual authority and is marked as such.

```bash
# exercise the machinery with the offline template (expected: drafts but no accepted records)
python -m teachers.pipeline --template --nodes math.g05 --out data/generated/curriculum_v0

# use real recorded teacher output
python -m teachers.pipeline --transcript data/generated/transcripts/teacher_a.jsonl \
    --transcript data/generated/transcripts/teacher_b.jsonl --out data/generated/curriculum_v0
```

## Critics (`teachers/critics.py`)

Deterministic, offline checks producing codes such as `MISSING_FIELD`, `PLACEHOLDER_TEXT`,
`TOO_SHORT`, `DEGENERATE_REPETITION`, `ANSWER_ECHOES_PROMPT`, `LANGUAGE_SCRIPT_MISMATCH`,
`MISSING_CITATION`, `MISSING_ASSUMPTION`. Reject verdicts remove a draft; warnings go to review.
Arithmetic found in a draft is verified exactly (`Fraction`, not floats).

## Consensus and verification (`teachers/consensus.py`, `teachers/verify.py`)

| Situation | Outcome | Record status |
|-----------|---------|---------------|
| ≥ 2 teachers, identical answer | `AGREED` | `multi_teacher_agreement` |
| Disagreement, exactly one answer passes deterministic verification | `RESOLVED_BY_VERIFICATION` | `deterministically_verified` |
| Disagreement with no mechanical way to decide | `DISAGREEMENT` | `critic_reviewed`, flagged |
| No answer passes verification | `REJECTED` | not used |
| One teacher (default) | `INSUFFICIENT_EVIDENCE` | quarantined |

Disagreement is never silently resolved by preference; dozens of answers that both "pass" integrity
checks escalate to review rather than a coin toss. Record kinds listed in
`require_verification_for` (transfer, verification, …) must reach `deterministically_verified` or
`human_reviewed` or they stay out of training.

## Record schema (`training/data/records.py`)

Every record carries: `record_id`, `kind` (one of the ten learning-loop stages), `task`, `response`,
`rationale` (+ `rationale_style`), `curriculum_node`, `domain`, `subject`, `grade_level`,
`difficulty`, `language`, `source`, `provenance`, `teacher_models`, `verification_status`,
`error_tags`, `license_status`, `created_utc`.

Policy, enforced in code and tests:

1. **Nothing reaches `trainable` by default.** Provenance must be `self_authored`, `licensed`,
   `synthetic` or `public_domain`; licence must be `cleared` or `restricted_no_redistribution`;
   verification must be *something* (`unverified` content is quarantined — content nobody checked is
   a hypothesis, not training material).
2. Synthetic records must name their `teacher_models`.
3. Error-diagnosis/correction records must carry `error_tags`.
4. Identical content never splits across train/val; dedupe is by content hash, not by id or timestamp.
5. `check_against_graph` refuses records whose node/subject contradict the curriculum graph.

## Measured output

Template run over two nodes at this commit: **120 requests, 120 drafts, 0 accepted, 120 quarantined,
0 rejected** (`data/generated/curriculum_v0/qc_report.json`). The pipeline refused to promote
single-teacher scaffolding into training data, which is the intended behaviour and the honest number
for a repository that has not yet run external teachers.

## Status

| Component | Status |
|-----------|--------|
| Providers, critics, consensus, verifier | ✅ Implemented, ✅ Verified (`tests/test_teachers_pipeline.py`) |
| Record schema + provenance policy | ✅ Implemented, ✅ Verified (`tests/test_records.py`) |
| End-to-end CLI | ✅ Implemented, ✅ Run (template, 0 accepted) |
| Recorded external teacher transcripts | ⏳ None committed |
| Accepted training records | 0 — none yet |

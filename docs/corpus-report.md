# SIR — M1 corpus report

*Generated 2026-10-05T13:41:44Z by `training/data/build_corpus.py` from `stats.json`. Every number below was measured by the pipeline; nothing here is estimated or hand-entered.*

## Headline

- **M1 status: BLOCKED** — 0 measured tokens is below the 10,000,000 floor; no training run may be started or reported as a model
- **Training readiness:** NOT READY — 0 measured tokens is below the 10M floor
- **Measured tokens:** not measured
- **Documents:** 35 after dedup (0 natural, 35 synthetic)
- **Characters:** 189,339 after dedup
- **Tokenizer:** not selected

## Languages

| Language | Documents | Characters | Tokens | Share of docs | Share of tokens | Synthetic docs | Train docs | Val docs |
|---|---|---|---|---|---|---|---|---|
| en | 10 | 55,953 | not measured | 28.57% | — | 10 | 1 | 0 |
| hi | 12 | 64,365 | not measured | 34.29% | — | 12 | 1 | 0 |
| hinc-deva | 5 | 24,952 | not measured | 14.29% | — | 5 | 1 | 0 |
| hinc-latn | 8 | 44,069 | not measured | 22.86% | — | 8 | 1 | 0 |

## Sources

| Source | Licence | Documents | Tokens | Licence verified | Training allowed | Attribution | Share-alike | Status |
|---|---|---|---|---|---|---|---|---|
| sir_fixture_v0 | CC0-1.0 | 35 | not measured | yes | yes | no | no | available |

## Split

- Method: `doc-id-fallback`, seed `20261005`
- Requested validation fraction: 0.0200; achieved: 0.0000
- Components (shared-sentence groups that move together): 35
- Train: 4 documents / 23,628 characters; validation: 0 documents / 0 characters
- note: validation fraction 11.4286% differs from the 2.00% target because whole components move together; the achieved value is what the report uses
- note: a component is a set of documents linked by shared normalized sentences; it can hold many documents
- note: 4 document(s) moved from validation to train by leakage repair; validation shrank rather than being trimmed
- leakage repair: None document(s) moved from validation to train over 1 iteration(s); converged: True

## Quality

- Exact duplicates removed: 0
- Near duplicates removed: 0 (cross-source: 0)
- Train/validation 8-gram leakage: 0 unique (0 occurrences in train), cross-split near-duplicate pairs: 0
- Language quality: not measured unknown-language and 0 script-mismatch of 35 documents; 2 label disagreements

## Gates

| Gate | Result | Name | Detail |
|---|---|---|---|
| G1 | PASS | licence | all sources carry a verified training permission |
| G2 | PASS | provenance | every document traces to a pinned artifact |
| G3 | PASS | integrity | 3 artifacts re-hashed, 0 mismatch(es) |
| G4 | PASS | leakage | shared 8-grams: 0 unique / 0 occurrences; cross-split near-duplicate pairs: 0 |
| G5 | **FAIL** | measured size | token counts are missing or were not produced by tokenization |
| G6 | PASS | language integrity | unknown 0.0000% (0/35), script mismatch 0.0000% (0/35 documents the cleaner examined) |
| G7 | PASS | reproducibility | reproducibility record complete and re-verified |

## What this corpus is not

- It is not a claim about language quality: the language labels come from a **script + marker-lexicon heuristic** (`training/data/langid.py`), not a statistical language identifier.
- It is not 200M tokens unless the measured token count above says so; the M1 target is a floor, not a goal to be rounded to.
- It is not a distribution target: the language mix is whatever the legally usable sources contain.
- Synthetic documents (if any) are labelled `synthetic: true` and counted separately; they never stand in for natural text.

## Reproducibility

- Git commit: `c437e46f1a59926a9f3d0ed17e582f7bb83232cf`
- Config: `None` sha256 `None…`
- Seed: `20261005`
- Source revisions pinned: 1; archive hashes recorded: 0
- Normalized files re-hashed on disk: 1


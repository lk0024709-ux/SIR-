# SIR Dataset Policy

<!-- Policy is enforced by code (training/data/manifest.py), not just stated here. Last verified 2026-10-05. -->

## Principles

1. **No invented sources.** Every dataset must be declared in `data/manifests/sources.yaml` before it can be used. The pipeline refuses unknown sources.
2. **No fabricated sizes.** `size_tokens` must be `null` in `sources.yaml`; actual sizes are measured by the fetcher/tokenizer and written to `data/manifests/acquisition.json`. A hand-typed number is treated as fabricated and fails validation.
3. **License is a gate, not a note.** `license: UNVERIFIED` makes a source unusable. `training/data/manifest.py` aborts. Each source records `license`, `license_notes`, `redistribution_allowed`, and `redistribution_notes`.
4. **Local-first, git-second.** Raw corpora, cleaned/processed JSONL, token streams, and checkpoints are git-ignored (`.gitignore`). Only configs, scripts, and data *cards* (`CARD.md`, `.gitkeep`) belong in git.
5. **Derived artifacts follow source license.** Model weights trained on CC BY-SA text inherit share-alike questions; that must be answered before publishing weights.
6. **Private user data is never training data.** `data/manifests/sources.yaml` contains a rejected entry `user_chat_logs` to make the prohibition auditable.

## Manifest Schema (`data/manifests/sources.yaml`)

```yaml
sources:
  - id: example_source
    name: Example Dataset
    url: "https://..."
    license: "CC-BY-SA-4.0"
    languages: [hi]
    size_tokens: null  # must be null; filled into acquisition.json
    redistribution_allowed: false
    status: planned  # available | planned | blocked | rejected
    notes: ""
    blocked_reason: ""  # required if status == blocked
    rejected_reason: "" # required if status == rejected
```

**Status vocabulary**

- `available` — fetched, hashed, locally present, licensed, acquisition-recorded.
- `planned` — desired, not yet fetched.
- `blocked` — desired, cannot be fetched now (reason required). In the current sandbox, all web corpora are blocked because egress to `dumps.wikimedia.org`, `huggingface.co`, etc. is closed by proxy (TLS EOF 2026-10-05). This is an access problem, not a licensing one.
- `rejected` — evaluated and unusable (license/quality).

**Gates (enforced)**

- G1: every `source_id` used by pipeline must be declared.
- G2: `license` must be a real string, not `UNVERIFIED`/empty.
- G3: `size_tokens` must be null; `status: available` requires an acquisition record.
- G4: `blocked` needs `blocked_reason`; `rejected` needs `rejected_reason`.
- G5: outputs from `redistribution_allowed: false` sources must land in git-ignored paths.

## Current Sources (2026-10-05)

| ID | Status | Languages | Natural? | License | Notes |
|----|--------|-----------|----------|---------|-------|
| `sir_fixture_v0` | available | hi, en, hinc-latn, hinc-deva | No (composed) | CC0-1.0 | Self-authored from seed sentences, 551 docs, 188k chars. Only corpus usable offline. |
| `hiwiki_dump` | blocked | hi | Yes | CC-BY-SA-4.0 | Wikimedia dump, attribution + share-alike. Blocked by egress. |
| `enwiki_dump` | blocked | en | Yes | CC-BY-SA-4.0 | English reference. Blocked. |
| `gutenberg_public_domain` | blocked | en | Yes | PublicDomain-US | Blocked. |
| `ai4bharat_ilci` | blocked | 12 Indic | Yes | UNVERIFIED | Requires manual access request. |
| `codemix_325k` | blocked | hinc-latn | Yes | UNVERIFIED | Social text; consent + license unverified. |
| `hinglish_proxy_romanized` | blocked | hinc-latn | No (derived) | CC-BY-SA-4.0 | Transliterated from hiwiki; not Hinglish. Blocked on source. |
| `commoncrawl_raw` | rejected | multi | Yes | Various-per-page | Cannot certify per-document licensing. |
| `user_chat_logs` | rejected | multi | Yes | none | Forbidden without per-user opt-in. |

## Pipeline

```
Raw Text
  ↓ Unicode Normalization (NFC, zero-width removal, mojibake recovery)
  ↓ Language Detection (script heuristic + lexicon for hinc-latn, mixed-script hinc-deva)
  ↓ Quality Filtering (min_chars, alnum ratio, symbol ratio, url ratio, repetition, boilerplate)
  ↓ Document Deduplication (exact hash)
  ↓ Near-Duplicate Filtering (MinHash/LSH, 64 perm, band 8, threshold 0.8)
  ↓ Train/Validation Split (leakage-safe, per-language, hashed, whole-component)
  ↓ Processed Dataset
```

Each stage writes a report (`dedup_report.json`, `split_report.json`, `provenance.json`). A result whose provenance cannot be regenerated is not a result.

### Cleaning

- Recovers mojibake (UTF-8 decoded as latin-1) when it increases Devanagari count.
- Strips wiki markup, templates, `<ref>`, markdown links, URLs, file headers.
- Rejects `too_short`, `too_long`, `low_alnum_ratio`, `line_repetition`, `link_farm`, `symbol_soup`, `no_punctuation_structure`, `boilerplate`.

### Deduplication

- Pure-Python MinHash/LSH, deterministic, earliest-id-wins. Adequate for fixture scale, **not** for a web crawl (tracked limitation: Phase 2 needs Spark/datasketches).
- Removes exact duplicates (sha256 of normalized text) then near-duplicates (Jaccard ≥0.8 on 24-char shingles).

### Splitting

- **Leakage-safe**: documents sharing a sentence or 8-gram are unioned into one component; whole components go to train or val by seeded hash, per language.
- Stratified: each language gets its own val budget (at least 1 doc if ≥2 docs). Achieved fractions are reported, not rounded.
- 8-gram overlap is guaranteed zero by construction and verified by `validate.py`. A nonzero result means grouping missed an edge.

## Tokenization and Corpus Size

- Token counts are written by `training/tokenize_corpus.py` into `acquisition.json` (`tokens_measured`, `tokens_measured_by`). This is the **only** sanctioned write of a size.
- Validation checks: tokenizer sha matches tokenized data; probe set must not appear in train (8-gram guard); EOS separators and doc ranges are recorded; windows never cross doc boundaries unless `pack_across_docs: true` (default false).

## Fixture Corpus (`sir_fixture_v0`)

- **Purpose**: exercise every pipeline stage without network or licensing risk. Not a release corpus.
- **Construction**: 551 docs composed from human-written seed sentences (HI, EN, Hinglish roman, Hinglish deva mix) via seeded RNG (seed 20261005). Topics: vigyan, shiksha, naagarik, kisan, khel, tech, etc. Includes deliberately dirty markup, mojibake, zero-width, exact+near+extended duplicates.
- **Limitations**: repeats a small sentence pool → token-efficiency numbers are biased, perplexity not comparable, generation meaningless as capability. Every report flags `provisional: true` and `natural: false`.
- **Reproducibility**: `python -m scripts.build_fixture_corpus --seed 20261005` → byte-identical `docs.json` (sha256 recorded).

## Provisional Threshold

`PROVISIONAL_BELOW_CHARS = 1_000_000`. Below this, every metric is labelled `pipeline-validation only`. Smoke corpus is 171k chars → all numbers are provisional.

## Acquisition Evidence (2026-10-05)

- Reachable: `pypi.org`, `files.pythonhosted.org`, `github.com`
- Blocked: `dumps.wikimedia.org`, `*.wikipedia.org`, `api.wikimedia.org`, `huggingface.co` (egress allowlist, TLS closed)
- Consequence: no natural web corpus could be obtained; all web sources are `blocked` with the exact command that would work once egress is available. Nothing was silently substituted.

## Record-Level Provenance (2026-10-07)

The corpus gates above govern *documents*. Curriculum and synthetic material is governed at
*record* level by `training/data/records.py`: every record carries `domain`, `subject`, `grade_level`,
`difficulty`, `language`, `source`, `provenance`, `teacher_models`, `verification_status`,
`error_tags`, `curriculum_node` and `license_status`, and nothing reaches a training split by default.

| Partition | Qualification |
|-----------|---------------|
| `trainable` | provenance in {self_authored, licensed, synthetic, public_domain} **and** licence cleared **and** verification not `unverified`/`rejected` |
| `quarantine` | uncertain provenance, unknown licence, or content nobody has verified yet |
| `rejected` | provenance/licence/verification explicitly rejected |

Synthetic records must name their `teacher_models`; identical content never straddles the train/val
split; and `check_against_graph` refuses records whose node or subject contradicts the curriculum
graph. Generated teacher-pipeline output lives in `data/generated/` and is not committed.

## What Is Still Missing

- A licensed natural Hindi/English/Hinglish corpus (≥200M tokens for M1).
- Pinned compute budget (M1 needs ~10 GPU-hours, sandbox is 2-core CPU, 3GB RAM).
- Frozen evaluation sets (probes exist but are small and fixture-derived).


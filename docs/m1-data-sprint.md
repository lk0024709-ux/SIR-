# SIR — M1 data acquisition sprint: order of operations, legal boundaries, offline fallback

This document is the operating manual for the M1 corpus work. It answers three questions the
sprint was required to answer explicitly:

1. **In what order does the work run?** (§1)
2. **Where does legal review stop and engineering start?** (§2)
3. **How is the pipeline exercised when the network is closed?** (§3)

Everything below is written against the *actual* repository state, not against an aspirational
one. Where a number is needed, it is produced by the pipeline
(`data/processed/m1_corpus/stats.json`, `evaluation/results/corpus_report.json`,
`docs/corpus-report.md`) and never typed in by hand.

---

## 1. Execution order

The order is fixed; each step consumes the artifact of the step before it and nothing may be
skipped. Steps 0–2 are *acquisition*, 3–6 are *corpus construction and measurement*, 7–8 are
*readiness and (only then) training*. A failure at any step stops the chain: the pipeline never
"works around" a missing licence, a blocked host or a failed gate.

| # | Step | Command | Produces | Stop condition |
|---|------|---------|----------|----------------|
| 0 | Environment + regression baseline | `python -m pytest -q` | test result | any regression → fix before touching data |
| 1 | Source discovery + licence review | edit `data/manifests/sources.yaml` | one record per candidate source with licence evidence and a status | a source whose training permission is unclear is `needs_review`, never `available` |
| 2 | Acquisition | `python -m training.data.acquire --all-available` (or `--source <ID>`) | `data/raw/<source_id>/…`, `data/manifests/acquisition.json`, `data/manifests/artifacts/<source_id>.jsonl`, `data/manifests/acquisition_log.jsonl` | a blocked host is recorded as BLOCKED with hostname/URL/error/timestamp and is not retried in a loop |
| 3 | Normalisation (raw → SIR documents) | `python -m training.data.normalize --source <ID>` (or via the build) | `data/normalized/<source_id>.jsonl` + `<source_id>.normalize_report.json` (sha256 of the normalized file) | unreadable artifacts are reported with a reason, never silently skipped |
| 4 | Corpus build | `python -m training.data.build_corpus --config configs/m1_corpus.yaml` | `data/processed/m1_corpus/{deduped,train,val}.jsonl`, `split_report.json`, `leak_repair_report.json`, `_work/*` (cleaning/dedup reports, per-document reject log), `stats.json`, `evaluation/results/corpus_report.json`, `docs/corpus-report.md` | gates G1–G7 decide; a failing gate is reported, not tuned away |
| 5 | Tokenizer bake-off | `python -m tokenizer.train_tokenizer --config <cfg>` then `python -m tokenizer.evaluate_tokenizer …` | candidate tokenizers + `tokenizer/artifacts/<corpus>/chosen.json` | selection is recorded with the measured slice scores |
| 6 | Token measurement (re-run of the tail of step 4) | `python -m training.data.build_corpus --config configs/m1_corpus.yaml --tokenize-only` | measured `tokens`, `tokens_by_language`, `tokens_by_source`, `tokens_train`, `tokens_validation` in `stats.json` + regenerated report | `--tokenize-only` never re-runs text stages; it refuses to run if their reports are missing |
| 7 | Gate report | `python -m training.data.gates --stats data/processed/m1_corpus/stats.json` | G1–G7 pass/fail + M1 status (exit code 1 on failure) | M1 is PASS only when measured tokens ≥ target **and** every gate passes |
| 8 | Training readiness | `training/gates.m1_status` inside `stats.json` | `READY FOR M1 TRAINING` / `READY FOR A DATA-SCALED EXPERIMENT (not M1)` / `NOT READY` | below 10M measured tokens nothing is trained and nothing is called a model |

Only after step 8 reports ready does training start, and even then in the documented cheap-first
order: one-step training → 100-step pilot → checkpoint/resume test → evaluation → *then* a longer
run. No 25M run is started from this sprint.

Resumability is part of the order, not an afterthought:

* `acquire` is idempotent — a second run re-hashes what is already on disk and skips it (measured:
  46 s for the whole 10-source set versus minutes for a fresh fetch).
* `build_corpus --resume` reuses `_work/{normalize,clean,dedup}_reports.json` and `split_report.json`
  instead of recomputing them.
* `--tokenize-only` exists so that selecting a tokenizer *after* a build does not force the text
  stages to run again.

## 2. Legal review boundaries

This section is **not legal advice** and does not replace counsel. It defines what SIR may
conclude on its own, and where it must stop.

### 2.1 What "reviewed" means here

A source may be marked `available` only when the manifest records all of the following:

* a licence name and a URL that was actually read (timestamped in `license_evidence`, with the
  decisive sentence quoted);
* `license_verified: true`, `model_training_allowed: true`;
* recorded answers for `redistribution_allowed`, `commercial_use`, `derivatives`,
  `attribution_required`, `share_alike`;
* a `provenance` string that says where the bytes come from (mirror → original);
* a pinned revision for the acquisition (commit SHA / dump date / archive hash).

If any of these cannot be established from the licence text, the status is `needs_review` — the
source is not downloaded. "Public", "free", "open access", "downloadable", "on GitHub" and
"indexed by a search engine" are **not** training permission.

### 2.2 What SIR does not do

* No re-interpretation of a licence in SIR's favour; ambiguous wording → `needs_review`.
* No downloads from endpoints that the environment blocks, no mirrors, no proxies, no scraping
  against a site's stated restrictions. A blocked host becomes a BLOCKED record with hostname,
  attempted URL, error and timestamp.
* No data with unclear rights; no user chat logs, personal data, or social-media dumps.
* No synthetic text is ever presented as natural: synthetic documents carry `synthetic: true`,
  are counted separately in every report, and are never mixed in silently.
* No licence text or third-party corpus text is committed to git (see §2.4).

### 2.3 Share-alike, non-commercial and no-derivatives

These are recorded per source rather than assumed. CC BY-SA sources are usable for training, but
the share-alike obligation propagates to derived *text*; model weights trained on them are a
question this project does not resolve by itself, so the report states the obligation and the
sprint treats SA data as **not silently redistributable**. Non-commercial or no-derivatives
sources are not used for a training corpus at all. The same rule applies to GPL-style licences on
text: they are recorded, not quietly used.

### 2.4 Licence separation (code ≠ docs ≠ data ≠ weights)

| Layer | Licence | Committed? |
|-------|---------|-----------|
| SIR source code (this repo) | as declared in `LICENSE` | yes |
| SIR docs and reports | same as code | yes |
| SIR-authored fixtures (`data/fixtures/sir_fixture_v0`) | CC0-1.0, with `CARD.md` | yes (small) |
| Third-party corpora (raw / normalized / processed) | each source's own licence, unchanged | **no** — git-ignored; only hashes, licences and metadata are committed |
| Model weights / tokenizer artifacts | separate, recorded in the model card | no (git-ignored) |

The manifest is the machine-readable version of this table; `training/data/manifest.py` rejects a
manifest where a source's licence fields contradict its status.

## 3. Offline fixture fallback

The whole pipeline must run with the network closed, and it does:

```bash
python -m training.data.build_corpus --config configs/m1_corpus.yaml \
    --offline-fixture-only --allow-id-fallback --outdir /tmp/m1_fixture_test
```

* `--offline-fixture-only` restricts the build to `sir_fixture_v0`, the CC0 fixture committed in
  `data/fixtures/sir_fixture_v0/` (built by `scripts/build_fixture_corpus.py`).
* `--allow-id-fallback` is required for this fixture only, because the fixture is deliberately
  repetitive; without it the splitter refuses the corpus (one content component) instead of
  producing a leaky split. Using the flag is recorded in the split report.
* The fixture is **composed, not natural**: every document is written with `synthetic: true` and
  `natural: false`, the report counts it separately, and the resulting M1 status is BLOCKED /
  NOT READY — which is the honest outcome for a few hundred kilobytes of constructed text.

What the fallback proves: acquisition gating, normalisation schema, cleaning and reject logging,
dedup, leakage-safe split, leakage measurement and repair, token measurement (when a tokenizer is
present), gates and report generation all work end to end with no network access. What it must
never be used for: any claim about natural-language coverage, Hinglish, or M1 size.

---

## 4. Current measured state

Generated by the pipeline; see `docs/corpus-report.md` for the full tables and
`evaluation/results/corpus_report.json` for the machine-readable copy. The status line here is
deliberately the same wording the gates produce — no aspiration, no rounding up.

* **M1 status:** see `docs/corpus-report.md` → *M1 status* (PASS / PARTIAL / BLOCKED with reason).
* **Training readiness:** see the same report → *Training readiness*.
* **Corpus, tokenizer and gates:** `data/processed/m1_corpus/stats.json`.
* **Environment limits observed in this sandbox:** 2 vCPU, ~3.9 GB RAM, Python 3.11; network
  reachable for `github.com` / `codeload.github.com` / `api.github.com` / PyPI only — Wikimedia,
  Hugging Face, Google Storage, archive.org and Indian government portals are blocked by the
  egress proxy (recorded per source in `data/manifests/sources.yaml`).

---

## 5. How the corpus is constructed (decisions and their reasons)

These are the choices that determine what the corpus *is*. Each one is implemented in a named
place, and each one is allowed to fail loudly.

**Exact deduplication** (`training/data/dedup_stream.py`) hashes whitespace-collapsed text
(SHA-256) and keeps the first occurrence in file order. Removed documents are written to a log
that names the document that replaced them, so "earliest source wins" is checkable rather than
asserted.

**Near deduplication** uses MinHash-LSH with a SQLite band index: 64 permutations, bands of 8,
Jaccard ≥ 0.8 on 24-character shingles. Candidates are verified by real Jaccard before anything is
removed, and a band shared by more than `max_band_size` documents is treated as boilerplate — the
truncation is counted in the report. Cross-source removals are counted separately, which is how the
report can answer "which source won a cross-source duplicate".

**The split** (`training/data/split_stream.py`) builds *components*: documents linked by any shared
normalized sentence (union-find in SQLite). Whole components move to train or validation, ordered
per language by `blake2b(seed:language:group)`. A corpus that collapses into one component is
refused, because a split that cannot separate content cannot be honest.

**Leakage** is measured, not assumed: hashed word 8-grams of train are sorted in memory (uint64,
not Python strings — that is what keeps this inside a 3 GB box) and validation is streamed against
them. If any overlap remains, the *whole* leaking validation document moves to train
(`moved_from_validation_by: leak-repair-N`); validation shrinks rather than being trimmed, and
documents are never deleted to improve a number.

**Token counts** exist only after a tokenizer actually ran over the corpus
(`training/data/build_corpus.py::tokenize_split`), producing a flat uint16/uint32 stream with one
EOS per document, plus per-language and per-source counts. Selecting a tokenizer later never
requires re-running the text stages: `--tokenize-only` reuses them and refuses to run if their
reports are missing.

**Synthetic documents** keep `synthetic: true` / `natural: false` through normalisation, cleaning,
dedup and the split; the report counts them separately and they never stand in for natural text.

# SIR Evaluation

<!-- All numbers here are measured, never typed. Last verified 2026-10-05. -->

## Principle

Every number is written to JSON by a script. README and model cards quote the JSON. No human retypes a benchmark figure anywhere in this project.

## What Is Measured (and What Is Not)

| Measured (mechanical, defensible) | Explicitly NOT measured |
|-----------------------------------|-------------------------|
| Training/val loss, perplexity, tokens/sec, memory, param count | Factual correctness, grammar quality, translation, instruction following |
| Token counts, chars/token, unk rate, round-trip, encode speed, context capacity | Model quality, Hindi fluency |
| 8-gram overlap, cross-split near-dups, probe contamination | Any comparison with Llama/Qwen etc. |
| Non-empty, non-degenerate generation (no >32-token single-token loop, no n-gram 3×, no glyph hammering) | Safety, refusal, production-readiness |
| CPU latency (128 tokens) with machine specs | Latency on phones (separate benchmark) |
| Reproducibility (±2% perplexity across two processes, identical traces) | Cross-machine reproduction |

All smoke results are `provisional: true` (corpus <1M chars, composed fixture).

## Harness

| Script | Output | What it does |
|--------|--------|--------------|
| `tokenizer/evaluate_tokenizer.py` | `evaluation/tokenizer_results.json` + `docs/tokenizer-evaluation.md` | Scores bake-off on held-out slices, tests acceptance criteria (Hindi ≥15%, Hinglish ≥10% fewer tokens vs byte BPE). |
| `evaluation/scripts/evaluate_lm.py` | `evaluation/results/lm_eval.json` | Per-document perplexity on held-out docs, per-language, plus trainer-agreement. |
| `evaluation/scripts/leakage.py` | `evaluation/results/leakage.json` | 8-gram overlap, cross-split near-dups, probe contamination. Criterion: zero 8-gram overlap. |
| `evaluation/scripts/probe_generation.py` | `evaluation/results/generation_probes.json` | 50 prompts (hi/en/hinc-latn/hinc-deva) graded for non-empty, non-degenerate, no control chars, deterministic. |
| `evaluation/scripts/reproducibility.py` | `evaluation/results/reproducibility.json` | Two independent training processes, compare ppl/trace/eval/generation. |
| `inference/bench_cpu.py` | `evaluation/results/cpu_bench.json` | 128 tokens latency, median/worst, first-token, machine specs. Target ≤2.0s. |

## Tokenizer Bake-Off (smoke)

- **Corpus**: 102 held-out docs (33051 chars) vs train 390 docs (138425 chars). All slices provisional (<50k chars per slice).
- **Candidates**: byte_bpe, sp_bpe, sp_unigram at 1024 and 2048 vocab. Trained on identical train text, scored on identical val.
- **Result**: No candidate met **both** acceptance criteria at comparable vocab. At 1024, sp_bpe passed Hindi (+27.7% fewer than byte BPE) but failed Hinglish (-0.1% vs +10% target); sp_unigram similar (+21.6% / -6.5%). At 2048, vocab deviation made comparison not like-for-like, reported as `not_comparable`.
- **Selection**: `sp_bpe @ 2048` (fewest tokens on India slices) but `met_acceptance: false`, `eligible_for_release: false` (fixture corpus). The failure is the result; the criterion stays.
- **Why Hinglish failed**: byte-level BPE already merges Latin-script words that dominate romanized Hinglish, so the Devanagari byte penalty that BPE suffers does not apply to this slice. True Hinglish evaluation needs organic code-switched data, not just romanized-Hindi proxy.

Full table: `docs/tokenizer-evaluation.md` (generated, do not hand-edit).

## Language Model (smoke, 0.9M params, 60 steps)

- **Perplexity** (token-weighted, per-document scoring): overall 679.0 (smoke repro 370.29 on windowed val; harness vs trainer delta +10% due to different windowing, flagged as `suspicious: false` because delta <15%).
  - Per language: hi 372.6, en 1188.5, hinc-deva 417.0, hinc-latn 1231.3 (fixture, not capability; high variance expected).
- **Leakage**: 0 8-gram overlaps (0 occurrences, 0.0 rate), 0 cross-split near-dups, 0/50 probe contamination. **PASS**.
- **Probe generation** (50 prompts, max 40 new tokens, greedy): 35/50 accepted (70%, target ≥50% → **PASS**). 15 degenerate, 0 empty, greedy determinism true for all 50. Per language: hi 10/17, en 10/17, hinc-latn 14/15, hinc-deva 1/1. Degeneracy includes phrase loops like `नियमान नियमान` and glyph hammering `----`, which a lenient grader would miss.
- **Reproducibility**: two runs ppl 370.2968 vs 370.2968 (spread 0.0000%, tolerance 2% → **REPRODUCIBLE**). Traces identical, checkpoint eval identical, greedy generation identical.
- **CPU latency**: median 0.323s for 128 tokens (target ≤2.0s → **MEETS**), 396 tok/s, first-token median 1.74ms, on 2-core Xeon 2.6GHz, 3GB RAM (no KV cache, O(t²)). Caveats recorded.

## Reading the Numbers Without Over-Claiming

- These are subword mechanics + pipeline plumbing on a tiny composed corpus. They say nothing about Hindi fluency, model quality, or SIR capabilities.
- If a criterion failed, the failure is the result. The criterion is not renegotiated.
- `chars/token` is reported for comparability, but the cost driver is total tokens for the same text (what the acceptance test uses).

## Test Sets

- **Probes**: `evaluation/probes/m1_probe_v0.jsonl` (50 prompts, CC0, held-out only, must never be trained on — enforced in `training/tokenize_corpus.py` probe guard). Categories: hindi_completion 12, english_completion 10, hinglish_completion 12, factual_short 8, reasoning_simple 8.
- **Future**: frozen test sets for Phase 1 need to be committed, versioned, and never used for training. Building them is on the critical path.

## Curated Suites (2026-10-07)

`evaluation/suite.py` evaluates *task behaviour* on curated cases, across the eleven axes the
promotion gate requires (knowledge, mathematics, science, language, reasoning, coding, application,
transfer, verification, brainstorming, reliability). Cases live in `evaluation/suites/*.jsonl`
(currently `dev_smoke_suite.jsonl`, 33 self-authored cases, all eleven axes covered).

Scoring honesty rules:

- every case declares a verification mode; deterministic modes (`exact`, `numeric`, `set`,
  `arithmetic_claims`, `integrity_only`) are scored offline through `teachers/verify.py`;
- `manual`/`none` cases are **UNSCORED** — counted, reported, excluded from the axis score, and never
  treated as passes;
- a case whose answerer crashes is recorded as `error`, not as a wrong answer;
- `verified_evidence: true` only when every *scored* case used a deterministic mode, so a promotion
  gate can distinguish execution-checked evidence from reviewer-graded evidence;
- `suite_hash` identifies the exact case set, and the regression gate refuses to compare runs with
  different hashes.

```bash
python -m evaluation.suite --suite evaluation/suites/dev_smoke_suite.jsonl --answers <answer_sheet.json>
```

Model answers come from a local checkpoint through `inference/` (`--checkpoint`) or from a recorded
answer sheet; there is no network path in this module.

## Reproducing

```bash
python -m tokenizer.evaluate_tokenizer --config configs/sir_nano_smoke.yaml
python -m evaluation.scripts.evaluate_lm --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml --out evaluation/results/lm_eval.json
python -m evaluation.scripts.leakage --config configs/sir_nano_smoke.yaml --out evaluation/results/leakage.json
python -m evaluation.scripts.probe_generation --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml
python -m evaluation.scripts.reproducibility --config configs/sir_nano_smoke.yaml --max-steps 60
python -m inference.bench_cpu --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml --out evaluation/results/cpu_bench.json
```

All produce JSON; the model card and README quote those JSON files.


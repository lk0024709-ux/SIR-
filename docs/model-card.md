# Model Card: SIR-Nano Smoke (Pipeline-Validation Prototype)

<!-- This is NOT SIR-Nano v0.1. It is a 0.9M-parameter research artifact that proves the pipeline runs. Last verified 2026-10-05. -->

## Model Details

- **Name**: SIR-Nano Smoke (`sir_nano_smoke`)
- **Version**: smoke (not `v0.1.0-sir-nano`)
- **Architecture**: Decoder-only transformer, 4 layers, 4 heads, d_model 128, d_ff 352, vocab 2048, max_seq_len 64, pre_layernorm, GELU, learned absolute positional, SDPA causal, tied embeddings (pad 0, eos 3).
- **Parameters**: 897,152 total (embedding 262,144, non-embedding 635,008, embedding share 29.22%).
- **Tokenizer**: `sp_bpe` @ 2048 (provisional; `tokenizer/artifacts/smoke/sp_bpe-2048/spec.json`, byte fallback, 15KB artifact). **Not release-grade**: chosen on fixture corpus, `eligible_for_release: false`.
- **Checkpoint**: `runs/sir_nano_smoke-dryrun2/final.pt` (10.8 MB with optimizer, ~3.6 MB weights-only estimate), step 60, seed 20261005, val_loss 5.914 (ppl 370.3), git `17b488f`-derived, provisional.

## Intended Use

- **Intended**: pipeline validation only — verify that data → tokenizer → pretrain → evaluate → inference runs end-to-end and reproduces from the same seed.
- **Not intended**: any use as a language model, chatbot, translator, or source of facts. Do not deploy, do not compare with other models, do not use for Hindi grammar evaluation.

## Training Data

- **Source**: `sir_fixture_v0` (self-authored, CC0, 551 docs, 188,908 chars, composed from seed sentences by `scripts/build_fixture_corpus.py` seed 20261005). **Natural: no**.
- **Pipeline**: clean 543 → dedup 492 → split 390/102 (leakage-safe, zero 8-gram overlap). Tokenized 45,021 tokens (train 30k, val 15k).
- **License**: CC0 for fixture; no natural corpus used. All web corpora (hiwiki, enwiki, Gutenberg, etc.) are blocked (egress) and unverified — not used.
- **Limitations**: tiny, repetitive, not natural; every metric is `provisional: true` (<1M chars). Model is under-parameterized for any real task and overfits after ~40 steps.

## Training Procedure

- **Config**: `configs/sir_nano_smoke.yaml` (window_stride 32, batch 16, lr 3e-3 cosine, warmup 30, 60 steps, 2-core CPU, deterministic, threads 1, amp off).
- **Hardware**: 2-core Xeon 2.6GHz, 3GB RAM (no GPU), wallclock ~5.5s for 60 steps.
- **Logs**: `runs/sir_nano_smoke-dryrun2/train_log.json` (full trace, eval every 40, best at step 40).
- **Reproducibility**: two independent processes gave identical ppl (370.2968, 0% spread, within ±2%), identical loss trace, identical eval/generation.

## Evaluation

| Axis | Metric | Smoke Result | Target | Verdict |
|------|--------|--------------|--------|---------|
| Tokenizer Hindi | tokens vs byte BPE @1024 | -27.7% | ≥15% fewer | PASS |
| Tokenizer Hinglish | tokens vs byte BPE @1024 | -0.1% | ≥10% fewer | **FAIL** (byte BPE already merges Latin words) |
| Overall tokenizer selection | — | no candidate met both | both | **FAIL** (reported, not hidden) |
| Data leakage | 8-gram overlap | 0 | 0 | PASS |
| Probe generation | accepted 35/50 (70%) | 70% | ≥50% | PASS (new strict grader) |
| Reproducibility | ppl spread | 0.00% | ≤2% | **REPRODUCIBLE** |
| CPU latency | 128 tokens median | 0.323s | ≤2.0s | **MEETS** |

- **Perplexity**: overall 679 (per-doc), hi 373, en 1188, hinc-deva 417, hinc-latn 1231 — provisional, not capability.
- **Probe details**: 15 degenerate (phrase loops, glyph hammering), 0 empty, greedy determinism true. Per language hi 10/17, en 10/17, hinc-latn 14/15, hinc-deva 1/1.
- **Artifacts**: `evaluation/results/` (tokenizer_results.json, leakage.json, generation_probes.json, reproducibility.json, cpu_bench.json, lm_eval.json if run).

## Limitations and Risks

- **Cannot be relied on for facts, instruction-following, safety, or Hindi grammar quality.** It is a statistical continuation of prompts, ungrammatical and occasionally looping (`नियमान नियमान`, `----`).
- **No instruction tuning, no safety guardrails, no RAG/search/memory/vision/voice/agent.**
- **Degeneracy**: 30% of generations are loops under strict rules; a lenient grader would overstate success.
- **Bias**: trained on a fixture that repeats a small sentence pool; any demographic or dialectal bias is unmeasured.
- **Privacy**: no user data used; no risk of memorization beyond the fixture.

## How to Run

```bash
python -m inference.generate --checkpoint runs/sir_nano_smoke-dryrun2/final.pt --prompt "भारत"
# greedy by default; add --temperature 0.8 --seed 42 for sampling
python -m inference.bench_cpu --checkpoint runs/sir_nano_smoke-dryrun2/final.pt
```

## Model Card Completeness

- This card describes a pipeline-validation prototype, not a milestone release. M1 (`SIR-Nano v0.1`, ~25M params, 200M tokens) is still blocked (no licensed corpus, no GPU budget) — see `configs/sir_nano_v0_1.yaml` (`status.trained: false`) and `docs/dataset-policy.md`.
- All claims are traceable to JSON in `evaluation/results/` and `data/processed/smoke/`. No hand-typed benchmark numbers.

## License

- **Code**: no `LICENSE` file yet (all rights reserved by default) — choosing one is the first Phase 1 box. Until then, no reuse rights granted.
- **Fixture data**: CC0-1.0.
- **Weights**: not licensed for release (provisional artifact, not a product).


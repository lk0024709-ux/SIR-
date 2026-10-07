# SIR Architecture

<!-- Generated from implemented code, not from roadmap aspirations. Last verified 2026-10-05. -->

## Overview

SIR (Super Intelligent Robo) is a modular AI intelligence platform. The architecture is deliberately boring at v0.1: every module has a narrow interface so a weak module can be replaced without rebuilding the system.

```
                    ┌──────────────────────────┐
                    │           SIR            │
                    │  SUPER INTELLENT ROBO  │
                    └────────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
        ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
        │  SIR    │         │  SIR    │         │  SIR    │
        │  Brain  │         │ Vision  │         │  Voice  │
        └────┬────┘         └─────────┘         └─────────┘
             │
       ┌─────┼───────────────┐
       │     │               │
  Reasoning Knowledge     Memory
       │     │               │
       └─────┼───────────────┘
             │
        ┌────▼─────┐
        │ SIR Agent│
        └────┬─────┘
             │
     ┌───────┼────────┐
     │       │        │
  Search   Tools   Android
```

SIR Brain is the only component every other module may depend on.

## Implemented Modules (2026-10-05)

| Module | Path | Status | What it does |
|--------|------|--------|--------------|
| Tokenizer | `tokenizer/` | ✅ Implemented | Common API over HuggingFace BPE and SentencePiece (BPE/Unigram), bake-off trainer and evaluator. See `docs/tokenizer-evaluation.md`. |
| Model | `model/` | ✅ Implemented | Decoder-only transformer, ~0.9M params in smoke run (vocab 2048, d_model 128, 4 layers, 4 heads, ff 352, ctx 64, tied embeddings, pre_layernorm, GELU, learned absolute pos). |
| Training Data Pipeline | `training/data/` | ✅ Implemented | Manifest-gated pipeline: clean → dedup → split (leakage-safe) → validate, with provenance. |
| Tokenized Dataset | `training/dataset.py` | ✅ Implemented | Flat token streams with doc-range boundaries, WindowDataset that respects doc boundaries. |
| Training Loop | `training/train.py` | ✅ Implemented | Deterministic, resumable, checkpointing, fixed seed, CPU/GPU. |
| Checkpoint | `training/checkpoint.py` | ✅ Implemented | Self-contained checkpoints with config, tokenizer sha, git commit, env. |
| Inference | `inference/` | ✅ Implemented | Greedy/sampling generation, no KV cache (O(t²), honest about it), CPU bench. |
| Evaluation | `evaluation/scripts/` | ✅ Implemented | LM perplexity, leakage (8-gram + near-dup + probe contamination), probe generation (50 prompts), reproducibility harness, CPU latency. |
| Tests | `tests/` | ✅ Implemented | 305 tests (305 passed, 3 skipped at this commit), CPU-only, hermetic. |

## Development Layer (2026-10-07)

| Module | Path | Status | What it does |
|--------|------|--------|--------------|
| Curriculum graph | `curriculum/graph.py`, `data/curriculum/curriculum_graph.yaml` | ✅ Implemented, ✅ Verified | 178 nodes / 221 edges / 254 topics; ladder, prerequisite and epistemic-note validation. |
| Requirement contract | `curriculum/requirements.py` | ✅ Implemented, ✅ Verified | Checks the graph against `configs/curriculum_requirements.yaml` (0 violations). |
| Coverage states | `curriculum/coverage.py` | ✅ Implemented, ✅ Verified | Assessment-driven `NOT_STARTED…MASTERED` / `REVIEW_REQUIRED`, with per-node blockers. |
| Spaced review | `curriculum/review.py` | ✅ Implemented, ✅ Verified | Interval schedule + stated forgetting-risk heuristic. |
| Curriculum sampler | `curriculum/sampler.py` | ✅ Implemented, ✅ Verified | Seeded, config-driven batch plan with honest shortfall reporting. |
| Teacher / critic / verifier pipeline | `teachers/` | ✅ Implemented, ✅ Verified | Recorded-transcript providers, deterministic critics, multi-teacher consensus, executable verification. No network client. |
| Training-record schema | `training/data/records.py` | ✅ Implemented, ✅ Verified | Metadata + provenance policy: nothing is trainable by default; unverified content is quarantined. |
| Generation contracts | `configs/generation_contracts.yaml`, `development/generations.py` | ✅ Implemented, ✅ Verified | Machine-readable promises for 9 generations; 9/9 UNVERIFIED, none trained. |
| Gates | `development/{regression,promotion,readiness,experiments}.py` | ✅ Implemented, ✅ Verified | Suite-hash-checked regression, evidence-only promotion, readiness levels, experiment records. |
| Error learning | `development/errors.py`, `data/errors/` | ✅ Implemented, ✅ Verified | 12-category taxonomy, 7-stage chain, skeleton flow; 8 committed cases, all unverified. |
| Brainstorming rubric | `development/brainstorming.py`, `evaluation/rubrics/` | ✅ Implemented, ✅ Verified | Weighted mechanical rubric with explicit UNSCORED semantics. |
| Curated evaluation suites | `evaluation/suite.py`, `evaluation/suites/` | ✅ Implemented, ✅ Verified | 33-case self-authored suite across 11 axes; unscored/error counts never become passes. |

See `docs/curriculum.md`, `docs/curriculum-data-pipeline.md`, `docs/generation-contract.md`,
`docs/error-learning.md`, `docs/brainstorming.md` and `docs/experiments.md`.

## Planned / Not Implemented

- RAG / retrieval (`rag/`) — Phase 4
- Search provider abstraction (`search/`) — Phase 4
- Vision (`vision/`) — Phase 5
- Voice (`voice/`) — Phase 5
- Agent / tools (`agent/`) — Phase 6
- Android / edge (`android/`) — Phase 7
- Instruction tuning (`instruction_tuning/`) — Phase 2
- Reasoning (`reasoning/`) — Phase 3
- Safety (`safety/`) — ongoing

Each planned module is scaffolded as an empty directory with `.gitkeep` so boundaries are visible to reviewers.

## Design Decisions (v0.1)

### Tokenizer

- Three candidates compared on identical text, same vocab, same rules. No winner declared before running.
- Byte-level BPE must round-trip exactly; unk rate must be 0 for byte-level (byte fallback).
- Selection is provisional until a natural corpus exists.

### Model

- **Plain decoder-only transformer**: no MoE, no RoPE, no ALiBi. Fewer ways to be subtly wrong.
- Pre-LayerNorm, causal SDPA, tied embeddings, learned absolute positional embeddings.
- Residual scaling (1/√(2L)) for stability.
- Causal mask is tested three ways: shape, no-future-influence, padded row.

### Data

- Manifest gates (G1-G5): every source must be declared, licensed, acquisition-recorded, measured (not typed), and git-ignored if non-redistributable.
- Leakage-safe split: documents sharing a sentence or 8-gram are forced to the same side. Zero overlap is guaranteed by construction and verified.
- Corpus sizes are measured by `tokenize_corpus.py` into `acquisition.json`, never typed by hand.

### Training

- One seed drives init, data order, eval order. `torch.use_deterministic_algorithms(True)`.
- Tokenizer artifact sha verified against tokenized data before first step.
- Validation on fixed window list, fixed order.
- Mixed precision only on CUDA (FP16 on CPU would change numerics for no gain).

### Inference

- No KV cache in v0.1: each token re-runs the visible sequence. Honest, auditable, quadratic. Cache is Phase 7.
- Greedy by default for reproducible probes; sampling optional with seed.

## Module Boundaries

- `tokenizer/` may not import `model/` or `training/`.
- `training/` may not import `inference/`.
- `sir_paths.py` is the only shared helper (repo-relative paths, config loading).
- No module imports a heavy dependency that would create a hard boundary violation.

## Scaling Notes

- Current dedup is pure-Python MinHash + LSH: fine for tens of thousands of docs, not for a web crawl. Phase 2 needs a JVM/Go-grade dedup (tracked as limitation).
- Tokenization is streaming; training uses memmapped token streams.
- Context length in smoke run is 64 (short docs would be starved at 128). Config `window_stride: 32` lets short docs contribute multiple windows.

## Reproducibility

- Every artifact writes `provenance.json` / `train_log.json` / `bakeoff_manifest.json` with git commit, script sha, config, and counts.
- Reproducibility harness runs training twice in separate processes and compares perplexity (tolerance ±2%), loss traces, eval, and generation.

## What This Document Is Not

- Not a capability claim. No Hindi fluency, no reasoning scores, no comparison with external models.
- Numbers here refer to the smoke run (fixture corpus, 0.9M params) and are labelled provisional.



## Human Development & Growth Architecture

SIR is developed through a staged cognitive-development program rather than a simple parameter ladder. The generation sequence is:

`SIR-Nano → SIR-Lite → SIR-Flash Lite → SIR-Flash → SIR-Pro → SIR-Pro+ → SIR-Pro Max → SIR-Ultra → SIR-Expert`

The developmental contract is defined in `docs/human-development.md` and `configs/sir_human_development.yaml`.

Each generation must grow across knowledge, understanding, reasoning, application, verification, experience, transfer, continual learning, and brainstorming. Generation promotion requires measured evaluation; model size alone is insufficient.

The long-term curriculum progresses from complete Class 1–10 broad education through Class 11–12, undergraduate foundations, advanced/professional study, research methodology, and domain specialization.

The target signature capability is **master brainstorming**: decompose problems, generate diverse hypotheses, connect domains, compare alternatives, challenge assumptions, verify, and synthesize. This remains an engineering objective and must not be reported as achieved until evaluated.

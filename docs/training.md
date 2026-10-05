# SIR Training

<!-- Documents the training pipeline that was actually executed (smoke run). Last verified 2026-10-05. -->

## Configurations

Two configs exist:

- **`configs/sir_nano_v0_1.yaml`** — TARGET for M1 (~25M params, 200M tokens, 10 GPU-hours). **Not executed** (`status.trained: false`, blocked on corpus + compute). Do not quote numbers against it.
- **`configs/sir_nano_smoke.yaml`** — ACTUALLY EXECUTED smoke run (pipeline validation). All results in `evaluation/results/` refer to this.

### Smoke Config (`sir_nano_smoke.yaml`)

```yaml
model:
  target_params: null  # pipeline test, not a claim
  vocab_size: 2048
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 352
  max_seq_len: 64
  dropout: 0.0
  tied_embeddings: true
  norm: pre_layernorm
  activation: gelu
training:
  batch_size: 16
  seq_len: 64
  window_stride: 32
  lr: 3.0e-3
  betas: [0.9, 0.95]
  warmup_steps: 30
  schedule: cosine
  max_steps: 400
  eval_every: 40
  eval_batches: 32
```

Actual parameter count: 897,152 (0.90M). Analytic estimate: 904,064 (gap 0.77%). Embedding share 29.22%.

## Data Flow

```
data/fixtures/sir_fixture_v0/docs.json  (CC0, 551 docs, 188k chars)
  ↓ clean (543 kept: 8 too_short, 4 language disagreements, 1 URL removed)
  ↓ dedup (492 kept: 18 exact, 25 near clusters removed)
  ↓ split (leakage-safe, 390 train / 102 val, zero 8-gram overlap)
  ↓ tokenize (smoke tokenizer sp_bpe-2048 chosen? Actually training uses sp_bpe 2048 provisional)
  → data/processed/smoke/tokenized/{train,val}/tokens.bin + ranges.npy + meta.json
```

Tokenized totals: 45,021 tokens (train 30,049, val 14,972) across hi/en/hinc-latn/hinc-deva.

## Training Loop (`training/train.py`)

```
python -m training.train --config configs/sir_nano_smoke.yaml
```

**Non-negotiables**

- Seed 20261005 drives init, data order, eval order. `torch.use_deterministic_algorithms(True)`, threads pinned to 1 in smoke.
- Tokenizer sha verified before first step.
- Validation on fixed window list, fixed order.
- Loss logged with token count; perplexity = exp(mean CE) over sum, not batch means.
- Mixed precision only on CUDA (disabled on CPU).
- Checkpoints to `runs/` (git-ignored), never uploaded.

**Smoke Run Results (runs/sir_nano_smoke-dryrun2, 60 steps, window_stride 32)**

- Steps 1..60, effective batch 16, seq_len 64
- `train_loss` trace: 7.65 → 6.19 → 3.93 → 2.81 (60 steps)
- `val_loss`: 5.84 at step 40 (ppl 346.6), 5.91 at step 60 (ppl 370.2), best at step 40
- Wallclock 5.5s, mean throughput ~11k tok/s, peak RSS 773 MB (2-core CPU)
- Reproducibility: two independent processes gave identical ppl 370.2968 (0.00% spread, within ±2%), identical loss traces, identical generation.

**What the loss means**

- These are pipeline measurements on a tiny fixture, not capability. The corpus is composed from a small sentence pool; perplexity is not comparable to any external number.
- Loss decreased but validation loss rose after step 40 (overfitting on tiny data) — expected and reported, not hidden.

## Tokenization Before Training

```
python -m training.tokenize_corpus --config configs/sir_nano_smoke.yaml
```

- Writes `tokenized/{train,val}/tokens.bin` (uint16), `ranges.npy` (doc boundaries), `meta.json` (tokenizer sha, counts).
- Probe contamination guard: if any probe prompt shares an 8-gram with train, the run is refused (3). Smoke: clean (0/50).

## Checkpoints (`training/checkpoint.py`)

Every checkpoint contains: `model_state`, `model_config`, `step`, `seed`, `validation_loss`, `tokenizer` (spec path, artifact sha, kind, vocab), `git` (commit, branch, dirty), `environment` (python, torch, platform, cpu), `run_config` (full resolved config), `optimizer_state`, `scheduler_state`.

- Format version 1, `*.pt` (torch.save).
- `assert_not_tracked` guard: `runs/` is git-ignored; a checkpoint in `git status` makes tests fail.
- `best.pt` is never rotated; `step-*.pt` keeps last N.

## How to Reproduce

```bash
# 1. Build fixture (deterministic)
python -m scripts.build_fixture_corpus --seed 20261005
python -m training.data.manifest --record sir_fixture_v0 "data/fixtures/sir_fixture_v0/docs.json"

# 2. Pipeline
python -m training.data.pipeline --source sir_fixture_v0 --config configs/sir_nano_smoke.yaml --outdir data/processed/smoke

# 3. Tokenizer bake-off
python -m tokenizer.train_tokenizer --config configs/sir_nano_smoke.yaml
python -m tokenizer.evaluate_tokenizer --config configs/sir_nano_smoke.yaml

# 4. Tokenize
python -m training.tokenize_corpus --config configs/sir_nano_smoke.yaml

# 5. Train
python -m training.train --config configs/sir_nano_smoke.yaml

# 6. Evaluate
python -m evaluation.scripts.evaluate_lm --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml
python -m evaluation.scripts.leakage --config configs/sir_nano_smoke.yaml
python -m evaluation.scripts.probe_generation --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml
python -m inference.bench_cpu --checkpoint runs/sir_nano_smoke/final.pt --config configs/sir_nano_smoke.yaml
```

All important parameters are configurable, not hard-coded. The same config reproduces the experiment.

## Limitations and Next Steps

- Corpus is fixture-only → no release model. M1 needs ≥200M tokens of licensed natural text (blocked).
- Model is 0.9M, not 25M. Scaling to 25M on a tiny corpus would be a hollow model (over-parameterized, no signal).
- No instruction tuning, no validation perplexity that means anything beyond "machinery works".
- Next: obtain a licensed corpus, re-run bake-off on natural text, then train the 25M target config.


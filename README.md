<div align="center">

# 🇮🇳 SIR — Super Intelligent Robo

**An Indian AI intelligence platform built for language, reasoning, knowledge, and intelligent assistance.**

`Status: Pipeline Validation` · `Phase: 1 — Smoke Run` · `Smoke checkpoint (0.9M, provisional)`

</div>

---

SIR stands for **Super Intelligent Robo**.

SIR is an ambitious Indian AI project focused on building an independent, modular AI system with strong
support for **Hindi, Hinglish, English, and Indian languages**, while providing reasoning, knowledge retrieval,
coding, vision, voice, memory, and agent capabilities.

> **Read this first:** the sections marked **Planned** below describe intent, not shipped software.
> As of the date in the status snapshot, SIR is a design and a repository — it has no trained model,
> no published benchmark, and no demo. Nothing in this file should be read as a capability claim.

---

## 📌 Current Status Snapshot

_As of 2026-10-05 · checked against the repository contents, not against the roadmap._

**Overall: Phase 1 — the research pipeline exists and has been exercised end-to-end on a tiny fixture corpus. The model is a 0.9M-parameter smoke prototype, not the 25M M1 target.**

| # | Component | State | What actually exists today | What that means for you |
|---|-----------|:-----:|----------------------------|--------------------------|
| 1 | Repository & layout | ✅ Exists | Git repo, this README, directory scaffold, `.gitignore` | Usable as a working base to build on |
| 2 | Project design & roadmap | ✅ Exists | Documented architecture, phases, policies in this file | Reviewable and criticizable; not implemented |
| 3 | Tokenizer | ✅ Implemented | Bake-off (byte_bpe / sp_bpe / sp_unigram × 1024/2048), scored on held-out slices, selection provisional (fixture corpus) — see `docs/tokenizer-evaluation.md` | Usable for research; not release-grade |
| 4 | Language model | 🟨 Smoke prototype (0.9M, not 25M) | Decoder-only transformer (vocab 2048, d128, 4 layers, ctx 64, tied embeddings) — checkpoint `runs/sir_nano_smoke-dryrun2/final.pt` (provisional) | Generates text, but ungrammatical and not a capability claim |
| 5 | Training pipeline | ✅ Implemented | `training/data/` (manifest, clean→dedup→split→validate), `training/train.py`, `training/checkpoint.py`, `training/dataset.py`, `training/tokenize_corpus.py`; deterministic, resumable, provenance | Reproducible: 0.00% ppl spread across 2 runs |
| 6 | Datasets | 🟨 Fixture only (CC0, provisional) | `sir_fixture_v0` (551 docs, 188k chars, composed, not natural); web corpora blocked (egress) — see `data/manifests/sources.yaml` | No natural corpus yet; M1 needs ≥200M tokens |
| 7 | Instruction tuning | ⬜ Not started | — | SIR is not a chatbot today |
| 8 | Reasoning system | ⬜ Not started | — | No math/logic capability |
| 9 | RAG / retrieval | ⬜ Not started | — | No document or KB answering |
| 10 | Web search integration | ⬜ Not started | Interface sketch only | SIR cannot search the web |
| 11 | Vision | ⬜ Not started | — | SIR cannot see images |
| 12 | Voice (ASR/TTS) | ⬜ Not started | — | SIR has no microphone or speech output |
| 13 | Memory | ⬜ Not started | — | No short- or long-term memory |
| 14 | Agent / tools | ⬜ Not started | — | SIR executes no actions |
| 15 | Android / edge runtime | ⬜ Not started | — | No APK, no on-device model |
| 16 | Evaluation harness & benchmarks | ✅ Implemented | Leakage (0 8-grams), 50-probe generation (70% non-degenerate), reproducibility (±2%), CPU latency (0.32s/128 tok), tokenizer bake-off — all JSON, never hand-typed | Provisional numbers exist (smoke), not capability; see `docs/evaluation.md` |
| 17 | Security controls | ⬜ Not started | Threat model listed below | Nothing to harden yet; define controls before tool use exists |
| 18 | License | ⚠️ **Undecided / blocker** | No `LICENSE` file in the repo | Until a license is chosen, **no reuse rights are granted** for code or docs |
| 19 | Public demo / hosted API | ⬜ Out of scope today | — | Any SIR endpoint you see elsewhere is unofficial |

**Legend:** ✅ exists in this repo · 🟨 designed, not implemented · ⬜ not started · 🔬 open research decision · ⚠️ blocker / risk

### Known blockers

1. **No license.** Still the first blocker (see [License](#license)). Until a license is chosen, **no reuse rights are granted** for code or docs — and it is decided per-component, not once.
2. **No natural corpus.** Fixture `sir_fixture_v0` is CC0 but not natural (551 docs, provisional). M1 needs ≥200M tokens of licensed Hindi/English/Hinglish — all web corpora are `blocked` (egress closed, not yet fetched). No 200M-token run has been done.
3. **No M1-scale compute.** Smoke was 60 steps, 0.9M params, 2-core CPU, ~5s, 11k tok/s, 773 MB RSS. M1 target is ~25M params, 400M tokens, ~10 GPU-hours — unfunded on this sandbox.
4. **Evaluation sets are small.** 50 probes exist (held-out, never trained on) but are fixture-derived; frozen, larger, natural test sets are still needed for a non-provisional benchmark.

---

## 🧭 Available vs. Planned

Kept deliberately separate so that nothing below is mistaken for a feature.

### ✅ Available today

- A public Git repository with implemented pipeline (71 tests, all passing) and defined module layout (see [Repository Structure](#repository-structure)).
- A written system design: architecture, training pipeline, retrieval flow, agent flow, memory model — now accompanied by executed code and measured artifacts.
- Written **policies** that bind the project: no invented sources (G1-G5 manifest gates), web pages are data not instructions, no private user data in training without consent, no secrets in weights or APKs, honest reproducible evaluation.
- A reproducible smoke run: `configs/sir_nano_smoke.yaml` → fixture → tokenizer bake-off → tokenize → train (60 steps, 0.9M) → evaluate → infer, with provenance and JSON results in `evaluation/results/` and `data/processed/smoke/`.
- Docs: `docs/architecture.md`, `docs/dataset-policy.md`, `docs/training.md`, `docs/evaluation.md`, `docs/model-card.md`, `docs/tokenizer-evaluation.md` (all generated/measured, not aspirational).
- The first working milestone, still defined with acceptance criteria: [M1](#first-working-milestone-m1) — smoke validates the machinery, not the milestone.

### 🔬 Open research decisions (nothing chosen yet)

| Decision | Options on the table | Decided by |
|----------|----------------------|------------|
| Tokenization scheme | BPE · Unigram · SentencePiece · byte-level | Smoke bake-off (sp_bpe@2048 provisional, no candidate met both criteria — see `docs/tokenizer-evaluation.md`) |
| Base vocab size | 8k · 16k · 32k | M1, from Hindi/Hinglish token efficiency |
| Training framework | PyTorch native · HF Trainer · lightweight custom loop | M1 |
| Attention/norm variants | RoPE, GQA, RMSNorm, tied embeddings | Phase 2, after M1 baseline |
| Retrieval stack | SearXNG · DuckDuckGo · others | Phase 4 |
| Vector store | any · none-yet | Phase 4 |
| Edge runtime | GGUF/llama.cpp · ONNX Runtime · ExecuTorch · LiteRT | Phase 7 benchmarks |
| Model/data license | TBD | **Before Phase 1 publication** |

### ⬜ Planned, not available (the long list)

Every capability in the sections below — Hindi/English/Hinglish conversation, reasoning, knowledge,
coding, RAG, search, memory, vision, voice, agents, Android deployment, safety systems — is **planned**.
A 0.9M-parameter smoke prototype exists (`runs/sir_nano_smoke-dryrun2/final.pt`) and generates text, but it is **not** a capability claim — it is a research artifact that proves the pipeline runs. Sizes like `SIR-1B` or `SIR-7B` are **targets, not releases**, and the 25M `SIR-Nano v0.1` milestone is still unrealized.

---

## First Working Milestone M1

**Working title: `SIR-Nano v0.1` — a tiny trilingual model that completes the whole loop once, end to end.**

### Why this milestone

M1's job is **not** to be impressive. It is to be *real*: to prove that data → tokenizer → pretraining →
evaluation → inference actually runs, is reproducible, and produces measurable numbers on Indian-language text.
Every later phase is only trustworthy if this loop exists. M1 also settles the two questions that most
determine SIR's viability — **token efficiency for Devanagari and Hinglish**, and **cost per training run**.

### Scope

| Deliverable | Concrete output |
|-------------|-----------------|
| D1 · Corpus | Cleaned Hindi + English + Hinglish text, **≥ 200M tokens** total, deduplicated, with a per-source license manifest (`data/CARD.md`) |
| D2 · Tokenizer bake-off | 3 candidates (SentencePiece-BPE, Unigram, byte-level baseline) at 8k and 16k vocab, scored on the same eval slice |
| D3 · Model | Decoder-only transformer, **~25M params** (`SIR-Nano`), config committed at `configs/sir-nano.yaml` |
| D4 · Training | Single-GPU script, ≤ 12 GPU-hours, resumable, fixed seed, logged losses, `train.py` + `configs/` only |
| D5 · Evaluation | Held-out perplexity per language, token-efficiency table, degeneration-rate check, contamination/dedup-overlap report |
| D6 · Inference | CPU-runnable CLI: `python -m inference.cli --model SIR-Nano --prompt "..."` |
| D7 · Release notes | Model card, run log (exact config + seed + wall-clock + GPU), and a **Limitations** section that says what it is bad at |

### Acceptance criteria — M1 is closed only if all pass

1. **Reproducibility:** a fresh clone + `configs/sir-nano.yaml` + recorded seed reproduces held-out
   perplexity within ±2% and the same token-count report.
2. **Tokenizer win, measured:** the chosen tokenizer encodes a 10 MB Hindi eval slice with **≥ 15% fewer tokens**
   than the byte-level BPE baseline, and Hinglish with **≥ 10% fewer**; both numbers published in a table.
3. **Non-degenerate generation:** on a 50-prompt Hindi/English/Hinglish probe set, **≥ 50%** of samples contain
   no repeated n-gram loop beyond 32 tokens (measured by a committed script, not by hand).
4. **Contamination hygiene:** zero 8-gram overlap between train and held-out sets, output of the dedup report committed.
5. **Real inference:** 128 tokens generated in **≤ 2.0 s** on a CPU-only laptop (machine recorded in the model card).
6. **Licensing hygiene:** every dataset used has a recorded license; non-redistributable sources are excluded, and
   `SIR-Nano` weights are released under whatever the code/data licenses permit — or a written reason they are not.
7. **Honesty check:** the model card states plainly that `SIR-Nano` is a research artifact that **cannot** be relied
   on for facts, instruction-following, safety, or Hindi grammar quality.

### Explicitly out of scope for M1

Instruction tuning · chat UI · safety guardrails · RAG, search, memory, vision, voice, agents · Android export ·
any run above 100M params · any claim of comparison with external models. If it is not in the table above, it is
not M1 — it moves to Phase 2+.

### Cost, time, exit

- **Effort:** ~4–8 weeks part-time, one person; ~30–60 GPU-hours total on a single T4/L4-class GPU (Colab/Kaggle).
- **Exit:** tag `v0.1.0-sir-nano` with weights (or a checksum + rebuild recipe), eval JSON, and run log.
- **If M1 fails** (e.g. Hindi token inflation or loss instability), the failure is written up in `docs/` as a
  finding and the next milestone is rescoped — not quietly replaced with a demo.

---

## 🚀 Vision

The goal of SIR is not to create another chatbot wrapper.

The long-term objective is to develop a complete AI intelligence platform with its own:

- 🧠 Language models
- 🔤 Tokenization technology
- 📚 Training pipeline
- 🧩 Reasoning system
- 🇮🇳 Indian-language capabilities
- 🔎 Knowledge retrieval
- 🌐 Web search integration
- 👁️ Vision system
- 🎙️ Voice system
- 🧠 Memory architecture
- 🤖 Agent framework
- 📱 Edge/Android deployment

SIR is designed to grow from a small research model into a complete AI ecosystem.

---

## 🧠 SIR Architecture

```text
                    ┌──────────────────────────┐
                    │           SIR            │
                    │  SUPER INTELLIGENT ROBO  │
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

Every box above is a separate module with a narrow interface, so that a weak module can be replaced without
rebuilding SIR. `SIR Brain` is the only component every other module may depend on.

---

## 🎯 Core Goals

### 💬 Natural Language

Hindi · English · Hinglish · other Indian languages · multilingual conversation · translation · summarization.

### 🧠 Reasoning

Mathematical reasoning · logical reasoning · multi-step problem solving · planning · code reasoning ·
Indian-context reasoning.

### 📚 Knowledge

General knowledge · Indian history · Indian geography · Indian education · Indian science and technology ·
Indian culture · civics and constitutional knowledge.

### 💻 Coding

Code generation · debugging · code explanation · refactoring · programming questions · software architecture.

### 🔎 Retrieval

SIR can be connected to external knowledge sources through a modular RAG/search architecture.

```text
User Question
      ↓
SIR
      ↓
Query Understanding
      ↓
Retriever / Search
      ↓
Evidence
      ↓
SIR Brain
      ↓
Verified Response
```

SIR must distinguish between:

- model knowledge
- retrieved information
- user-provided information

**SIR must never invent search results or sources.**

---

## 🇮🇳 Indian Language Focus

Indian-language support is a core part of SIR rather than a secondary feature.

| Stage | Languages |
|-------|-----------|
| Initial priority | Hindi · English · Hinglish |
| Planned expansion | Sanskrit · Marathi · Gujarati · Bengali · Tamil · Telugu · Kannada · Malayalam · Punjabi · Odia · Assamese · Urdu |

The tokenizer and training pipeline will be evaluated specifically for Indian scripts and multilingual token
efficiency. **Devanagari-mixing and romanized-Hindi code-switching are treated as first-class input, not as noise
to be normalized away.**

---

## 🔤 Tokenizer

SIR will investigate tokenizer architectures including:

- BPE
- Unigram
- SentencePiece
- Byte-level tokenization

Evaluation will include token efficiency on Hindi, Hinglish, English, Sanskrit, and other Indian scripts.
The objective is to avoid the unnecessarily high token counts that Indian languages suffer under
English-centric vocabularies — inflated token counts silently cost SIR training compute, latency, and context
budget, so this is measured at every milestone, starting with [M1](#first-working-milestone-m1).

---

## 🏗️ Model Roadmap

SIR will be developed incrementally. **Status column is the truth; the sizes are engineering targets, not claims
of existing or benchmarked releases.**

| Model | Target scale | Purpose | Status |
|-------|--------------|---------|:------:|
| `SIR-Nano` | ~25M | Pipeline proof, tokenizer bake-off, fast iteration | 🟨 planned as [M1](#first-working-milestone-m1) |
| `SIR-100M` | ~100M | Research prototype, scaling sanity check | ⬜ not started |
| `SIR-300M` | ~300M | Small language model, first useful Hindi/Hinglish behavior | ⬜ not started |
| `SIR-1B` | ~1B | Practical research model | ⬜ not started |
| `SIR-3B` | ~3B | Advanced model | ⬜ not started |
| `SIR-7B` | ~7B | Large open model | ⬜ not started |
| `SIR-Large` | TBD | Future large-scale research | ⬜ not started |

Possible variants once a base model is real: `SIR-1B-Base`, `SIR-1B-Instruct`, `SIR-1B-Reasoning`.

---

## 🧪 Training Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Language Detection
   ↓
Quality Filtering
   ↓
Deduplication
   ↓
Safety Filtering
   ↓
Tokenization
   ↓
Pretraining
   ↓
Instruction Tuning
   ↓
Reasoning Training
   ↓
Evaluation
   ↓
SIR Release
```

Rules that bind this pipeline:

- Training data must be obtained and used according to applicable licenses and permissions.
- **Private user data must not be secretly incorporated into model training.**
- Each stage is a separate, logged, reversible step with a committed config; no manual, undocumented data edits.

---

## 📖 Instruction Training

SIR will be trained on high-quality instruction datasets covering general conversation, Hindi conversation,
Hinglish, mathematics, science, history, geography, programming, reasoning, translation, summarization,
structured output, and tool calling.

**Quality is more important than simply increasing dataset size.** Every instruction set carries a data card:
source, license, size, dedup method, known biases, and how it may *not* be used.

---

## 🧠 Reasoning

SIR will include a dedicated reasoning-development pipeline.

```text
Understand
    ↓
Decompose
    ↓
Reason
    ↓
Verify
    ↓
Answer
```

Reasoning evaluation will cover mathematics, logic, planning, coding, multi-step tasks, and Indian-context problems.

**Private chain-of-thought should not be unnecessarily exposed to users.** SIR stores and verifies intermediate
steps internally; what it shows is a summary, the answer, and the citations — not a raw internal monologue.

---

## 🔎 SIR Search

SIR will use a provider-independent search interface.

```text
SearchProvider
├── SearXNG
├── DuckDuckGo
└── Future Providers
```

Search results must be: **deduplicated · ranked · timestamped · injection-filtered · attributed.**

Retrieved content is rendered into a fenced evidence block that the model is never allowed to treat as system
instructions. **Web pages are data, not instructions.** A citation is only ever produced from a real fetch; if a
fetch fails or a source is unavailable, SIR says so instead of guessing.

---

## 🧠 SIR Memory

SIR will support two primary memory layers.

| Layer | Contains | Lifetime |
|-------|----------|----------|
| Short-term | Current conversation context | Cleared with the session |
| Long-term | Only user-approved persistent information | Until the user inspects or deletes it |

Memory must provide: user control · privacy protection · deletion capability · clear separation from model
weights · secure storage.

**User data must not automatically become global training data.** Deletion is a first-class operation, and
whatever is in memory is visible and exportable to the user who owns it.

---

## 👁️ SIR Vision

Planned capabilities: image understanding · OCR · screenshot analysis · document understanding · visual question
answering · object understanding.

Vision remains modular so different vision models can be integrated without rebuilding the entire SIR system.
OCR for Devanagari and mixed-script documents is a stated requirement, not an afterthought.

---

## 🎙️ SIR Voice

```text
Microphone
     ↓
Voice Activity Detection
     ↓
ASR
     ↓
SIR Brain
     ↓
Response
     ↓
TTS
     ↓
Speaker
```

Initial priority: Hindi · English · Hinglish. Later versions may support wake-word detection and continuous
voice interaction. No audio is retained by default; recording is opt-in, local-first, and never a training
source without explicit consent.

---

## 🤖 SIR Agent

```text
User
 ↓
SIR Brain
 ↓
Planner
 ↓
Permission Check
 ↓
Tool
 ↓
Tool Result
 ↓
Verifier
 ↓
Final Response
```

Potential tools: calculator · search · browser · file system · code execution · Android actions · database ·
external APIs · smart-device APIs.

**Dangerous or irreversible actions must require appropriate authorization.** Each tool declares a side-effect
class, and anything that writes, deletes, spends money, or sends data off-device needs an explicit user-approved
grant with a recorded audit entry.

---

## 📱 Android / Edge AI

A major goal is to make smaller SIR models usable on Android and other edge devices.

- **Candidate runtimes:** GGUF · llama.cpp · ONNX Runtime · ExecuTorch · LiteRT · Android acceleration APIs
- **Deployment targets** depend on model size and device hardware; no device is assumed today.
- **Metrics that must be reported for every quantized build:** RAM usage · model size · tokens/sec ·
  first-token latency · battery consumption · temperature · context length.

---

## 🔐 Security

Security is a first-class part of SIR, not a launch-time layer.

| Threat | Control direction |
|--------|-------------------|
| Prompt injection via retrieved web pages | retrieved text is data, fenced and instruction-stripped |
| Prompt injection via documents/uploads | same untrusted-input channel as search |
| Tool abuse | capability grants per tool, dry-run mode, no implicit `sudo` |
| Unsafe autonomous actions | irreversible steps require explicit authorization + verifier |
| Data leakage | memory isolated from weights, exportable and deletable |
| Credential exposure | no API keys or secrets in weights, configs, or logs |
| Jailbreaks | evaluated, not assumed solved; results published whatever they say |
| Privacy violations | consent required for anything that persists or trains |

**API keys and secrets must never be embedded in model weights or publicly distributed Android APKs.**

---

## 📊 Evaluation

SIR will maintain a dedicated, reproducible evaluation framework. The test sets themselves are **not yet
committed** — building them is part of Phase 1, because a benchmark that cannot be rerun is not a benchmark.

| Axis | Coverage |
|------|----------|
| Language | Hindi · English · Hinglish · other Indian languages |
| Knowledge | general knowledge · Indian knowledge |
| Reasoning | mathematics · logic · planning |
| Coding | generation · debugging · understanding |
| Safety | prompt injection · jailbreak resistance · privacy |
| Agents | tool selection · tool execution · error recovery |

Every published SIR result must ship with: the frozen test set, the exact model and config, the seed, the
harness version, and failure cases. **No manipulated benchmarks, no cherry-picked samples, no misleading model
claims** — including when the honest number is embarrassing.

---

## Repository Structure

Status marks are the real state of this repository, not of the plan.

```text
SIR/
│
├── README.md              # ✅ this file
├── LICENSE                # ⚠️ still missing — first blocker, per-component decision (see License)
│
├── configs/               # ✅ sir_nano_smoke.yaml (executed) + sir_nano_v0_1.yaml (target, not executed)
│
├── data/                  # ✅ fixture + processed (git-ignored), manifests committed
│   ├── fixtures/sir_fixture_v0/  # CC0, 551 docs, 188k chars, provisional
│   ├── manifests/sources.yaml + acquisition.json
│   ├── cleaned/           # local only (git-ignored)
│   └── processed/smoke/   # train.jsonl 390 / val 102, tokenized, provenance.json
│
├── tokenizer/             # ✅ api.py, train_tokenizer.py, evaluate_tokenizer.py, artifacts/smoke/ (6 candidates)
├── model/                 # ✅ config.py, transformer.py (0.9M smoke, 25M target)
├── training/              # ✅ data/ (clean, dedup, split, validate, pipeline), dataset.py, tokenize_corpus.py, train.py, checkpoint.py
├── instruction_tuning/    # ⬜ empty — Phase 2+
├── reasoning/             # ⬜ empty — Phase 3
├── evaluation/            # ✅ probes (50), scripts (leakage, lm, probe, reproducibility), results/ (json)
├── inference/             # ✅ engine.py, generate.py, bench_cpu.py (no KV cache)
├── rag/                   # ⬜ empty — Phase 4
├── search/                # ⬜ empty — Phase 4
├── vision/                # ⬜ empty — Phase 5
├── voice/                 # ⬜ empty — Phase 5
├── agent/                 # ⬜ empty — Phase 6
├── safety/                # ⬜ empty — policies and probes
├── android/               # ⬜ empty — Phase 7
├── scripts/               # ✅ build_fixture_corpus.py (deterministic, CC0)
├── tests/                 # ✅ 71 tests (manifest, cleaning, dedup, split, tokenizer, model, training, inference)
│
└── docs/                  # ✅ architecture.md, dataset-policy.md, training.md, evaluation.md, model-card.md, tokenizer-evaluation.md
```

Scaffold directories for planned modules remain empty (with `.gitkeep`) so boundaries are visible to reviewers; implemented modules contain audited, tested code and measured artifacts.

---

## 🛣️ Development Roadmap

### Phase 1 — Research Prototype

M1 (`SIR-Nano v0.1`, 25M, 200M tokens) is still the milestone. The smoke run below validates the machinery that M1 will use.

- [ ] License decided and `LICENSE` committed (**blocker**, do this first)
- [x] Repository setup
- [x] Tokenizer bake-off (6 candidates, smoke, provisional — no release; see `docs/tokenizer-evaluation.md`)
- [x] Dataset pipeline (manifest-gated clean→dedup→leakage-safe split→validate, with fixture)
- [x] Small transformer (≈1M smoke, not 25M target — architecture proven, scaling still to do)
- [x] Training pipeline (deterministic, resumable, checkpointed, 60 steps smoke)
- [x] Inference engine (CPU greedy/sampling, bench 0.32s/128 tok, no KV cache)
- [x] Initial benchmark (leakage 0, probes 35/50, reproducibility 0% spread, all JSON)
- [ ] **M1 proper** (licensed natural corpus, 200M tokens, 25M model, 10 GPU-hours)

### Phase 2 — SIR-300M / 1B

- [ ] Larger dataset
- [ ] Improved tokenizer
- [ ] Pretraining
- [ ] Instruction tuning
- [ ] Hindi/Hinglish optimization
- [ ] Evaluation

### Phase 3 — Reasoning

- [ ] Reasoning datasets
- [ ] Math training
- [ ] Logic training
- [ ] Code reasoning
- [ ] Verification system

### Phase 4 — Knowledge

- [ ] RAG
- [ ] Search
- [ ] Document retrieval
- [ ] Source attribution

### Phase 5 — Multimodal

- [ ] Vision
- [ ] OCR
- [ ] Voice
- [ ] TTS

### Phase 6 — Agent

- [ ] Tool system
- [ ] Planner
- [ ] Permission system
- [ ] Tool verification

### Phase 7 — Edge AI

- [ ] Quantization
- [ ] GGUF export
- [ ] Android runtime
- [ ] Mobile benchmarking

### Phase 8 — Larger SIR models

- [ ] SIR-3B
- [ ] SIR-7B
- [ ] Larger research models

Smoke artifacts live in `evaluation/results/`, `tokenizer/artifacts/smoke/`, `data/processed/smoke/`, `runs/sir_nano_smoke-*/` (git-ignored where they contain derivable text/weights).

---

## 🧰 Technology Direction

SIR stays technology-independent where possible: Python · PyTorch · Hugging Face ecosystem · SentencePiece ·
CUDA · llama.cpp · ONNX Runtime · ExecuTorch · LiteRT · FastAPI · vector databases · Android/Kotlin.

Choices are made by benchmarking, and each one is reversible: no module is allowed to hard-depend on a runtime
that only Phase 7 will compare.

---

## 💻 Hardware Strategy

| Environment | Use |
|-------------|-----|
| Local | small experiments, inference, tests |
| Google Colab / Kaggle | training and experimentation on available GPUs (M1 target) |
| Cloud GPU | larger-scale training, Phase 2+ |
| Edge devices | quantized inference on Android and other supported hardware |

The project optimizes compute usage and avoids unnecessary large-scale training. **A 300M model that is well
tokenized and well evaluated is worth more to SIR than a 7B model trained on unverified data.**

---

## 🌐 Philosophy

| Principle | Meaning |
|-----------|---------|
| 🇮🇳 **Indian First** | Indian languages and context are fundamental, not translation-layer afterthoughts |
| 🔬 **Research First** | Claims must be supported by measurable results |
| 🧩 **Modular** | Every major capability must be replaceable |
| 🔐 **Privacy** | Private information belongs to the user |
| ⚡ **Efficient** | Small, efficient models are taken seriously |
| 📊 **Honest Evaluation** | No manipulated benchmarks or misleading model claims |
| 🌍 **Open Development** | Where licensing permits, code, datasets, and evaluation are documented openly |

**Status is part of honesty.** A planned capability written as a shipped one is the most common way an open AI
project loses credibility, so this README states what does not exist as often as what will.

---

## ⚠️ Project Status

- **Status:** Pipeline Validation — Phase 1 smoke run completed (2026-10-05).
- SIR is an evolving project. The roadmap describes intended capabilities and does **not** imply that every listed
  capability currently exists. Where numbers exist, they are from the smoke prototype (0.9M, fixture, provisional) and are labelled as such; they are not M1 results.
- Model sizes, benchmark results, training requirements, and performance are published **only after actual
  testing** — and the smoke results are now published as JSON in `evaluation/results/` and `docs/`. Any number quoted about SIR elsewhere that is not traceable to those JSON files is not from this project.
- No SIR model is released, hosted, or recommended for production, personal, legal, medical, or educational use. The smoke checkpoint is a research artifact, not a product (`docs/model-card.md`).

---

## 🤝 Contributing

Contributions are welcome in: dataset engineering · Indian-language NLP · tokenization · model architecture ·
training · evaluation · reasoning · RAG · search · vision · voice · Android/edge inference · security ·
documentation.

**The fastest useful contributions right now target M1, not the later phases:**

1. Build or vet a Hindi/Hinglish corpus and document its license (D1).
2. Train and score tokenizers on Devanagari + code-switched text (D2) — small compute, high value.
3. Write the frozen evaluation test sets and harness (D5) — this is on the critical path for every phase.
4. Review this README for over-claims; if any line reads like a finished feature, that is a bug.

Before contributing datasets or models, verify their licenses and usage restrictions. Open an issue describing the
approach before large code submissions, and include reproduction commands with every PR.

---

## License

The project license will be determined according to the licensing requirements of:

- source code
- training data
- model weights
- third-party dependencies
- external models

**Do not assume that all components can automatically be released under the same license.** A permissive code
license is compatible with restrictive or non-commercial data and may still block weight release, and vice versa;
each artifact gets its own license record.

> ⚠️ There is currently **no `LICENSE` file in this repository**. Until one is added, no permission is granted to
> copy, modify, or redistribute SIR's code or documentation. Choosing one is the first unchecked box in Phase 1.

---

<div align="center">

## 🇮🇳 SIR

**Super Intelligent Robo**

*Think Indian. Understand Everyone. Build Intelligence.*

SIR is not just a chatbot. It is an attempt to build an Indian AI intelligence platform from the ground up —
one measured milestone at a time. The pipeline now runs; [M1](#first-working-milestone-m1) (25M, licensed corpus) is next.

</div>

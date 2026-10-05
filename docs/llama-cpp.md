# llama.cpp integration

SIR uses [llama.cpp](https://github.com/ggml-org/llama.cpp) as an optional local-inference runtime for GGUF-compatible model artifacts, including future Android/edge inference work.

## Dependency policy

- Upstream: https://github.com/ggml-org/llama.cpp
- Pinned commit: `9871df5` (release b11418)
- Dependency is fetched at CMake configure/build time; the upstream source is not vendored into this repository.
- Do not silently move to `master`; update the pin deliberately and record the new version in the changelog/experiment manifest.

## Scope

This integration does **not** claim that the current SIR-Nano PyTorch checkpoint is directly loadable by llama.cpp. A model-to-GGUF conversion path must be implemented and validated separately before using llama.cpp for SIR-Nano inference.

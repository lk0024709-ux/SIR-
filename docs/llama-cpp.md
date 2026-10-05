# llama.cpp integration

SIR uses [llama.cpp](https://github.com/ggml-org/llama.cpp) as an optional local-inference runtime for GGUF-compatible model artifacts, including future Android/edge inference work.

## Dependency policy

- Upstream: https://github.com/ggml-org/llama.cpp
- Pinned commit: `9871df5911a03a813518bcd623ee2eba91bf32aa` (release b11418; short prefix `9871df5`)
- Dependency is fetched at CMake configure/build time; the upstream source is not vendored into this repository.
- The CMake integration uses the full commit SHA and does not request a shallow clone: CMake's `GIT_SHALLOW` mode supports branch/tag names, not a raw commit ID. This preserves the existing upstream commit while making the fetch mode valid and reproducible.
- Do not silently move to `master`; update the pin deliberately and record the new version in the changelog/experiment manifest.

## Scope

This integration does **not** claim that the current SIR-Nano PyTorch checkpoint is directly loadable by llama.cpp. A model-to-GGUF conversion path must be implemented and validated separately before using llama.cpp for SIR-Nano inference.

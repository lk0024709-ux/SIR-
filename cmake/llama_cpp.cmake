# SIR: reproducible llama.cpp dependency
# Pinned to a known upstream release commit; do not track moving master in builds.
include(FetchContent)

set(SIR_LLAMA_CPP_GIT_REPOSITORY "https://github.com/ggml-org/llama.cpp.git" CACHE STRING "llama.cpp repository")
# Keep a full commit SHA. CMake's GIT_SHALLOW mode only accepts branch/tag names, not raw commit IDs.
set(SIR_LLAMA_CPP_GIT_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp commit")

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY ${SIR_LLAMA_CPP_GIT_REPOSITORY}
    GIT_TAG        ${SIR_LLAMA_CPP_GIT_TAG}
)

# Disable optional heavyweight backends for the initial CPU/Android integration.
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

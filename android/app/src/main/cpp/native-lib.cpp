#include <jni.h>
#include <llama.h>

#include <mutex>

namespace {
std::once_flag backend_init_once;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_sir_ai_MainActivity_nativeBuildStatus(JNIEnv* env, jobject /* instance */) {
    // Touch the pinned llama.cpp API so the native smoke build verifies actual linkage.
    // No model is loaded, no prompt is accepted, and no inference is performed.
    std::call_once(backend_init_once, [] { llama_backend_init(); });
    return env->NewStringUTF(
        "Pinned llama.cpp CPU runtime linked. No SIR model is loaded; inference is not implemented.");
}

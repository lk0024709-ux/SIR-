# Android build foundation

This document describes the reproducible Android APK/JNI build scaffold. It does **not** describe a working SIR inference app.

## Migration status and repository inspection

At the inspected `main` baseline (`b06405c`, after PR #3), `android/` contained only `.gitkeep`: there was no existing Android application, Gradle project, manifest, Java/Kotlin source, or Android CMake target to migrate. This change is therefore a **greenfield Android scaffold**, not a migration or replacement of an existing app. The placeholder `MainActivity` can be extended or replaced in a later Android product task.

The existing llama.cpp integration was added by merged [PR #3](https://github.com/lk0024709-ux/SIR-/pull/3). It is consumed through the repository-owned `cmake/llama_cpp.cmake`, which fetches the pinned upstream commit at CMake configure time; upstream source is not vendored. This Android module includes that same CMake entry point. It does not create a second dependency or change the upstream commit. The CMake declaration now uses the full SHA and omits `GIT_SHALLOW`: CMake's shallow-clone mode is not valid for a raw commit ID, so the fetch mode was corrected while preserving the PR #3 pin.

## Pinned toolchain

| Component | Version / value | How it is pinned |
|---|---|---|
| JDK | 17 (Temurin in GitHub Actions) | `actions/setup-java` explicitly requests Java 17 |
| Gradle | 9.1.0 | `android/gradle/wrapper/gradle-wrapper.properties` |
| Gradle distribution SHA-256 | `a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806` | Gradle's published checksum for `gradle-9.1.0-bin.zip` |
| Gradle wrapper JAR SHA-256 | `76805e32c009c0cf0dd5d206bddc9fb22ea42e84db904b764f3047de095493f3` | Gradle's published wrapper-JAR checksum |
| Android Gradle Plugin | 9.0.1 | `android/build.gradle.kts` |
| Kotlin | 2.2.10 KGP, provided by AGP 9 built-in Kotlin | No separate `org.jetbrains.kotlin.android` plugin is applied |
| Android SDK Platform | 35 (`compileSdk` and `targetSdk`) | `android/app/build.gradle.kts` |
| Android SDK Build Tools | 36.0.0 | Explicitly pinned in the app module and CI |
| Android NDK | 28.2.13676358 | Explicitly pinned in the app module and CI |
| CMake | 3.22.1 | Explicitly pinned in the app module and CI |
| Android command-line tools | 22.0 / build `15859902` | Explicitly pinned in GitHub Actions |
| `minSdk` | 26 | `android/app/build.gradle.kts` |
| llama.cpp | `9871df5911a03a813518bcd623ee2eba91bf32aa` (release b11418) | `cmake/llama_cpp.cmake` full-SHA pin; upstream commit preserved |

The AGP pair is selected from Google's published compatibility table: AGP 9.0.1 requires Gradle 9.1.0 and JDK 17, and its documented default KGP, Build Tools, and NDK versions are 2.2.10, 36.0.0, and 28.2.13676358. AGP 9 has built-in Kotlin enabled by default, so the app compiles Kotlin without applying the incompatible legacy Kotlin Android plugin. `compileSdk`/`targetSdk` remain 35 as requested. `minSdk` 26 is retained as the native-runtime baseline rather than lowering the API floor for convenience; it is a build/deployment floor, not a claim about model support.

References: [AGP 9.0.1 compatibility and built-in Kotlin](https://developer.android.com/build/releases/agp-9-0-0-release-notes), [migrate to built-in Kotlin](https://developer.android.com/build/migrate-to-built-in-kotlin), [Gradle 9.1 Java compatibility](https://docs.gradle.org/9.1.0/userguide/compatibility.html), [Gradle published checksums](https://gradle.org/release-checksums/), and [Android NDK/CMake configuration](https://developer.android.com/studio/projects/install-ndk).

## What the app and native build currently do

- `com.sir.ai` is a minimal Kotlin Android application with a launchable `MainActivity`.
- Gradle invokes Android CMake for `arm64-v8a` and `x86_64`.
- The native target reuses the one pinned root llama.cpp CMake integration, builds the CPU backend, links a JNI shared library, and calls `llama_backend_init()` only to make the native linkage observable.
- The screen reports build/runtime status. It does not accept prompts, load a model, generate text, or return chatbot answers.
- No PyTorch checkpoint, GGUF file, converted weights, quantized weights, or production signing key is included.

The current SIR-Nano checkpoint is a PyTorch artifact and is **not automatically compatible with llama.cpp/GGUF**. There is no conversion path in this change. Consequently, SIR-Nano local inference on Android remains **not implemented and not claimed**.

## Native build acceptance criteria

The Android build foundation is accepted only when all of these are true:

1. Gradle runs with the pinned wrapper and explicitly provisioned JDK 17.
2. `assembleDebug` configures the pinned NDK/CMake toolchain and compiles the Kotlin application plus native C++ for both configured ABIs.
3. The native target includes the repository's pinned llama.cpp source and links the `llama_backend_init` API; the APK contains `lib/arm64-v8a/libsir_android.so` and `lib/x86_64/libsir_android.so`.
4. The resulting non-empty APK exists at `android/app/build/outputs/apk/debug/app-debug.apk` and is uploaded by CI only after those checks pass.
5. The existing Python test suite remains passing.

These criteria validate Gradle, Kotlin, JNI, NDK/CMake, and linkage only. They do **not** validate SIR model loading, GGUF conversion, text generation, model quality, device performance, emulator launch, or release signing. Those require separate future acceptance tests with a genuinely compatible model artifact.

## GitHub Actions

`.github/workflows/android-apk.yml` runs on relevant pull requests and pushes, configures JDK 17, installs the pinned Android command-line tools and SDK packages, enables Gradle caching, runs `./gradlew assembleDebug`, verifies the APK and both JNI libraries, and uploads the real build output as `sir-android-debug-apk`. The workflow pins the GitHub Actions used by commit SHA and fails if the APK or either JNI library is absent.

The first Android CI run is the authoritative build check for this change. A local run must not be reported as successful unless the commands below actually complete.

## Local requirements and setup

- JDK 17 (set `JAVA_HOME` to that JDK; Gradle 9 requires Java 17 or newer, and this project standardizes on 17).
- Android SDK command-line tools build `15859902` (22.0) and an SDK root exported as `ANDROID_SDK_ROOT`.
- SDK packages: Platform 35, Build Tools 36.0.0, NDK 28.2.13676358, and CMake 3.22.1.
- Network access to the official Gradle distribution service and GitHub, because the wrapper verifies and downloads Gradle and the pinned llama.cpp source is fetched by CMake.

Check Java and the wrapper:

```bash
java -version
cd android
./gradlew --version
```

The Java output must identify version 17. `./gradlew --version` must report Gradle 9.1.0.

Install the pinned SDK packages (after installing command-line tools and setting `ANDROID_SDK_ROOT`):

```bash
sdkmanager --install \
  "platforms;android-35" \
  "build-tools;36.0.0" \
  "ndk;28.2.13676358" \
  "cmake;3.22.1"
```

Build and verify the debug APK:

```bash
cd android
./gradlew --no-daemon assembleDebug
test -s app/build/outputs/apk/debug/app-debug.apk
```

The generated APK is `android/app/build/outputs/apk/debug/app-debug.apk` (relative to the repository root). To install it on a connected Android 8.0/API 26-or-newer device or emulator, install Android Platform Tools (`sdkmanager --install "platform-tools"` if needed), enable USB debugging, then run from `android/`:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.sir.ai/.MainActivity
```

## Change boundaries

This Android foundation is deliberately limited to:

- `android/**`: Gradle wrapper/project, one Kotlin launch screen, manifest, and native CMake/JNI smoke target.
- `.github/workflows/android-apk.yml`: reproducible debug APK build and artifact upload.
- `cmake/llama_cpp.cmake`: switch the existing pin from an abbreviated to a full SHA and remove the incompatible shallow-clone option; the upstream commit is unchanged.
- `docs/android-build.md`, `docs/llama-cpp.md`, and the README link to these guides.

It reuses the existing llama.cpp dependency rather than adding another. It does not modify Python training, corpus acquisition, tokenizer, model, checkpoint, evaluation, or inference code; add dataset text or credentials; produce/rename model weights; add GGUF conversion; or implement chatbot/agent behavior. Production signing and release distribution are also out of scope.

## Validation record

- GitHub Actions PR run [37343152291](https://github.com/lk0024709-ux/SIR-/actions/runs/37343152291) **passed** on 2026-10-05 in 4m7s. JDK setup, pinned SDK installation, Gradle version check, `assembleDebug`, APK/JNI payload verification, and artifact upload all completed successfully.
- The workflow verified a non-empty `app/build/outputs/apk/debug/app-debug.apk` containing `lib/arm64-v8a/libsir_android.so` and `lib/x86_64/libsir_android.so`; artifact `sir-android-debug-apk` was uploaded (7,823,876-byte artifact archive).
- The existing Python suite in the Agent sandbox: **117 passed, 3 skipped**.
- A local Android APK build remains **NOT VERIFIED** in the Agent sandbox: it had no `java`, Gradle, CMake, or Android SDK executable, and direct downloads from the Gradle/Google SDK hosts were unavailable. The CI build above is the verified build result; local absence is not counted as a pass.

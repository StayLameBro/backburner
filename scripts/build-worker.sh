#!/usr/bin/env bash
# build-worker.sh - the engine's command-line worker (llama-split-prefill --serve-tail), plus llama-server and llama-bench,
# for a machine that isn't the Mac host: docs/LINUX.md, docs/ANDROID.md. UNTESTED on Linux and Android so far.
#   scripts/build-worker.sh linux-cpu | linux-cuda | linux-vulkan        (on that Linux machine)
#   ANDROID_NDK=<ndk dir> scripts/build-worker.sh android-vulkan | android-opencl | android-cpu   (cross-compile)
#   scripts/build-worker.sh mac                                          (a spare Mac as a worker; Metal)
# JOBS=N limits parallel compile jobs (default: all cores). Output: build-worker[-android]/bin/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${ROOT}/llama.cpp"
TARGET="${1:-}"
NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
JOBS="${JOBS:-$NCPU}"
[ -f "${LLAMA}/CMakeLists.txt" ] || { echo "missing ${LLAMA}: git submodule update --init" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is missing" >&2; exit 1; }

ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF)
OUT="${ROOT}/build-worker"
case "$TARGET" in
  linux-cpu)      ARGS+=(-DGGML_NATIVE=ON) ;;
  linux-cuda)     ARGS+=(-DGGML_NATIVE=ON -DGGML_CUDA=ON) ;;
  linux-vulkan)   ARGS+=(-DGGML_NATIVE=ON -DGGML_VULKAN=ON) ;;
  mac)            ARGS+=(-DGGML_METAL=ON) ;;
  android-*)
    [ -n "${ANDROID_NDK:-}" ] && [ -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ] || {
      echo "set ANDROID_NDK to an Android NDK (r27 or later)" >&2; exit 1; }
    OUT="${ROOT}/build-worker-android"
    # arm64 only: every phone fast enough to matter is arm64-v8a; armv8.7-a covers the i8mm/dotprod kernels of 2022+ cores
    ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a
           -DANDROID_PLATFORM=android-31 -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.7-a -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF)
    case "$TARGET" in
      android-vulkan) ARGS+=(-DGGML_VULKAN=ON) ;;
      android-opencl) ARGS+=(-DGGML_OPENCL=ON) ;;
      android-cpu)    ;;
      *) echo "unknown target $TARGET" >&2; exit 1 ;;
    esac ;;
  *) sed -n '2,8p' "$0" >&2; exit 1 ;;
esac

echo "build-worker: $TARGET -> $OUT (-j$JOBS)"
cmake -S "$LLAMA" -B "$OUT" "${ARGS[@]}"
cmake --build "$OUT" --config Release -j "$JOBS" --target llama-split-prefill llama-server llama-bench
echo "build-worker: done. Worker: $OUT/bin/llama-split-prefill --serve-tail 50060 --tail-model <tail.gguf>"
echo "build-worker: it accepts loopback only. Read docs/LINUX.md before setting SPT_ALLOW_REMOTE=1: the link has no authentication."

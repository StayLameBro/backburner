#!/usr/bin/env bash
# build-android.sh - Android port of scripts/build-iphone.sh.
#   scripts/build-android.sh [--release] [-PllamaDir=/path/to/llama.cpp]
#
# Builds the Backburner Android app (arm64-v8a) and installs it with adb.
# Needs: ANDROID_SDK_ROOT (or ANDROID_HOME) + NDK r26+, JDK 17, adb on PATH,
# and the same llama.cpp fork as iOS (StayLameBro/backburner-llama.cpp).
# The llama.cpp Android static libs (llama/ggml/ggml-rpc + Vulkan or OpenCL)
# are linked via LLAMA_BUILD_DIR; without them the app still builds and the
# phone-attn path (:50062) works, but the tail (:50060) and RPC (:50052)
# report "not linked" until wired (see android/app/src/main/cpp/CMakeLists.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANDROOT="$ROOT/android"
MODE="assembleDebug"
for a in "$@"; do [ "$a" = "--release" ] && MODE="assembleRelease"; done

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
[ -n "$SDK" ] || { echo "set ANDROID_SDK_ROOT to your Android SDK"; exit 1; }
command -v adb >/dev/null || { echo "adb not on PATH (platform-tools)"; exit 1; }
command -v java >/dev/null || { echo "JDK 17 required"; exit 1; }

LLAMA_DIR="${LLAMA_DIR:-$ROOT/llama.cpp}"
[ -f "$LLAMA_DIR/include/llama.h" ] || { echo "missing $LLAMA_DIR (git submodule update --init --recursive)"; exit 1; }

echo "building Backburner Android app ($MODE), llama=$LLAMA_DIR"
cd "$ANDROOT"
GRADLE_ARGS=("$MODE")
if [ -n "${LLAMA_BUILD_DIR:-}" ]; then
  GRADLE_ARGS+=("-PllamaBuildDir=$LLAMA_BUILD_DIR")
fi
# shellcheck disable=SC2128
EXTRA_ARGS=("${GRADLE_EXTRA_ARGS[@]:-}")
if [ -x ./gradlew ]; then
  ./gradlew "${GRADLE_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"
else
  gradle "${GRADLE_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"
fi

APK=$(find "$ANDROOT/app/build/outputs/apk" -name "*.apk" | head -1)
[ -n "$APK" ] || { echo "no APK produced"; exit 1; }
echo "apk: $APK"
if adb get-state 1>/dev/null 2>&1; then
  adb install -r "$APK"
  echo "installed. Open Backburner, keep it foreground, then:"
  echo "  adb reverse tcp:50060 tcp:50060; adb reverse tcp:50061 tcp:50061; adb reverse tcp:50062 tcp:50062"
else
  echo "no adb device: install $APK by hand."
fi

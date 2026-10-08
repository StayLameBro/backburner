#!/usr/bin/env bash
# build-iphone.sh - the iPhone app: llama.xcframework for iOS (Metal + ggml-rpc), the SME2 attention kernel, then the app.
#   DEVELOPMENT_TEAM=<your Apple team id> [UDID=<iPhone UDID>] scripts/build-iphone.sh
#   IPA=1 scripts/build-iphone.sh      # no team id: ios/build/Backburner.ipa for AltStore (docs/INSTALL-IPHONE.md)
# With UDID set (and the phone wired and unlocked) it installs the app; otherwise it prints where the .app is.
# Xcode 27 public cannot debug iOS 27.2 (DDI). It can still archive + devicectl install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# LLAMA_DIR: build the framework from another llama.cpp tree (e.g. a snapshot without in-progress kernel edits)
LLAMA="${LLAMA_DIR:-${ROOT}/llama.cpp}"
IOS="${ROOT}/ios/Backburner"
FW="${IOS}/Frameworks"
MIN_IOS="${MIN_IOS:-16.4}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
IPA="${IPA:-0}"
TEAM="${DEVELOPMENT_TEAM:-}"
if [[ "${IPA}" != 1 && -z "${TEAM}" ]]; then
  echo "set DEVELOPMENT_TEAM to your Apple team id (Xcode > Settings > Accounts), or IPA=1 for an unsigned AltStore build"
  exit 1
fi
UDID="${UDID:-}"

if [[ ! -d "${LLAMA}" ]]; then
  echo "missing ${LLAMA}"
  exit 1
fi
# what a stock Mac lacks (GitHub #23): say how to get it before a long build fails half way
command -v cmake >/dev/null || { echo "cmake is missing: brew install cmake (or https://cmake.org/download)"; exit 1; }
if ! xcrun -sdk iphoneos metal -v >/dev/null 2>&1; then
  echo "the Metal Toolchain is missing (xcrun metal fails): xcodebuild -downloadComponent MetalToolchain (~840 MB, once)"
  exit 1
fi

echo "building iOS-device llama.xcframework (Metal + RPC, RDMA off) -j${JOBS}"
cd "${LLAMA}"

COMMON_CMAKE_ARGS=(
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_COMMON=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_MTMD=OFF
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_METAL_TARGET_OS=ios
  -DGGML_BLAS_DEFAULT=ON
  -DGGML_OPENMP=OFF
  -DGGML_NATIVE=OFF
  -DGGML_RPC=ON
  -DGGML_RPC_RDMA=OFF
  -DLLAMA_OPENSSL=OFF
)

rm -rf build-ios-sidecar
cmake -B build-ios-sidecar -G Xcode \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MIN_IOS}" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
  -DCMAKE_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -g -ffile-prefix-map=${LLAMA}=llama.cpp" \
  -DCMAKE_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -g -ffile-prefix-map=${LLAMA}=llama.cpp" \
  -DCMAKE_OBJC_FLAGS="-ffile-prefix-map=${LLAMA}=llama.cpp" \
  -S .
cmake --build build-ios-sidecar --config Release -j "${JOBS}" -- -quiet

REL="Release-iphoneos"
B="build-ios-sidecar"
mkdir -p "${B}/framework/llama.framework/Headers" "${B}/framework/llama.framework/Modules" "${B}/dSYMs"

cp include/llama.h \
   ggml/include/ggml.h ggml/include/ggml-opt.h ggml/include/ggml-alloc.h \
   ggml/include/ggml-backend.h ggml/include/ggml-metal.h ggml/include/ggml-cpu.h \
   ggml/include/ggml-blas.h ggml/include/gguf.h ggml/include/ggml-rpc.h \
   "${B}/framework/llama.framework/Headers/"

cat > "${B}/framework/llama.framework/Modules/module.modulemap" <<'EOF'
framework module llama {
    umbrella "Headers"
    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"
    export *
}
EOF

cat > "${B}/framework/llama.framework/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>llama</string>
    <key>CFBundleIdentifier</key><string>org.ggml.llama</string>
    <key>CFBundleName</key><string>llama</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>MinimumOSVersion</key><string>${MIN_IOS}</string>
    <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
</dict>
</plist>
EOF

LIBS=(
  "${B}/src/${REL}/libllama.a"
  "${B}/ggml/src/${REL}/libggml.a"
  "${B}/ggml/src/${REL}/libggml-base.a"
  "${B}/ggml/src/${REL}/libggml-cpu.a"
  "${B}/ggml/src/ggml-metal/${REL}/libggml-metal.a"
  "${B}/ggml/src/ggml-blas/${REL}/libggml-blas.a"
  "${B}/ggml/src/ggml-rpc/${REL}/libggml-rpc.a"
)
OPTIONAL=(
  "${B}/vendor/hash/${REL}/libvendor-hash.a"
)
EXISTING=()
for l in "${LIBS[@]}" "${OPTIONAL[@]}"; do
  if [[ -f "${l}" ]]; then
    EXISTING+=("${l}")
  else
    echo "missing ${l}"
    if [[ "${l}" == *libggml-rpc.a ]]; then
      echo "RPC static lib missing — GGML_RPC did not build. Cannot build the app."
      exit 1
    fi
  fi
done

TMP="${B}/temp"
mkdir -p "${TMP}"
xcrun libtool -static -o "${TMP}/combined.a" "${EXISTING[@]}" 2>/dev/null
SDKROOT="$(xcrun --sdk iphoneos --show-sdk-path)"
OUT="${B}/framework/llama.framework/llama"
xcrun -sdk iphoneos clang++ -dynamiclib \
  -isysroot "${SDKROOT}" \
  -arch arm64 \
  -mios-version-min="${MIN_IOS}" \
  -Wl,-force_load,"${TMP}/combined.a" \
  -framework Foundation -framework Metal -framework Accelerate \
  -install_name "@rpath/llama.framework/llama" \
  -o "${OUT}"
xcrun vtool -set-build-version ios "${MIN_IOS}" "${MIN_IOS}" -replace -output "${OUT}" "${OUT}"
xcrun dsymutil "${OUT}" -o "${B}/dSYMs/llama.dSYM"
xcrun strip -S "${OUT}" -o "${TMP}/stripped"
mv "${TMP}/stripped" "${OUT}"
rm -rf "${TMP}"

rm -rf "${LLAMA}/build-apple/llama-sidecar.xcframework" "${FW}/llama.xcframework"
mkdir -p "${LLAMA}/build-apple" "${FW}"
xcrun xcodebuild -create-xcframework \
  -framework "$(pwd)/${B}/framework/llama.framework" \
  -debug-symbols "$(pwd)/${B}/dSYMs/llama.dSYM" \
  -output "$(pwd)/build-apple/llama-sidecar.xcframework"
rm -rf "${FW}/llama.xcframework"
cp -R "$(pwd)/build-apple/llama-sidecar.xcframework" "${FW}/llama.xcframework"
echo "xcframework → ${FW}/llama.xcframework"

echo "building the SME2 attention kernel for iOS"
xcrun -sdk iphoneos clang -c -O3 -isysroot "$(xcrun --sdk iphoneos --show-sdk-path)" -target arm64-apple-ios16.4 \
  -mcpu=apple-a18 -ffile-prefix-map="${ROOT}"=. -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN "${ROOT}/scripts/sme/sme_attn.c" -o "${IOS}/Sidecar/sme_attn_ios.o"
echo "archiving the app (no debugger, no DDI)"
cd "${IOS}"
mkdir -p "${ROOT}/ios/build"
if [[ "${IPA}" == 1 ]]; then
  # unsigned: AltStore signs it on the phone with the user's own Apple ID (and appends that team id to the bundle id)
  SIGN_ARGS=(CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY= DEVELOPMENT_TEAM= PRODUCT_BUNDLE_IDENTIFIER=app.backburner.sidecar)
else
  SIGN_ARGS=(-allowProvisioningUpdates DEVELOPMENT_TEAM="${TEAM}" CODE_SIGN_STYLE=Automatic)
fi
archive() {
  xcodebuild \
    -project Sidecar.xcodeproj \
    -scheme Sidecar \
    -configuration Release \
    -destination "generic/platform=iOS" \
    "${SIGN_ARGS[@]}" \
    -archivePath "${ROOT}/ios/build/Sidecar.xcarchive" \
    archive 2>&1 | tee "${ROOT}/ios/build/archive.log"
  return "${PIPESTATUS[0]}"
}
if ! archive; then
  # a free (personal) team gets a profile only for devices Xcode has registered, and archiving for a generic device
  # registers none: build once for the wired phone (which registers it), then archive again (GitHub #23)
  if [[ "${IPA}" != 1 && -n "${UDID}" ]] && grep -q "has no devices" "${ROOT}/ios/build/archive.log"; then
    echo "registering ${UDID} with your team (free teams need this once per device), then archiving again"
    xcodebuild -project Sidecar.xcodeproj -scheme Sidecar -configuration Release -destination "id=${UDID}" \
      "${SIGN_ARGS[@]}" build -quiet
    archive
  else
    [[ "${IPA}" != 1 && -z "${UDID}" ]] && grep -q "has no devices" "${ROOT}/ios/build/archive.log" && \
      echo "free team: wire the phone, unlock it and run again with UDID=<its UDID> so Xcode can register it"
    exit 1
  fi
fi

APP="${ROOT}/ios/build/Sidecar.xcarchive/Products/Applications/Sidecar.app"
if [[ ! -d "${APP}" ]]; then
  echo "archive produced no Sidecar.app"
  ls -R "${ROOT}/ios/build" | head
  exit 1
fi
echo "app: ${APP}"

if [[ "${IPA}" == 1 ]]; then
  # Ad-hoc sign with the entitlements so AltStore can read them: without increased-memory-limit the phone gets ~3 GB, not ~6.
  P="${ROOT}/ios/build/ipa"
  rm -rf "${P}" && mkdir -p "${P}/Payload"
  cp -R "${APP}" "${P}/Payload/"
  for f in "${P}"/Payload/Sidecar.app/Frameworks/*.framework; do codesign -f -s - "${f}"; done
  codesign -f -s - --entitlements "${IOS}/Sidecar/Sidecar.entitlements" "${P}/Payload/Sidecar.app"
  codesign -d --entitlements - "${P}/Payload/Sidecar.app" 2>/dev/null | grep -q increased-memory-limit \
    || { echo "IPA is missing the increased-memory-limit entitlement"; exit 1; }
  rm -f "${ROOT}/ios/build/Backburner.ipa"
  (cd "${P}" && zip -qry "${ROOT}/ios/build/Backburner.ipa" Payload)
  rm -rf "${P}"
  echo "ipa: ${ROOT}/ios/build/Backburner.ipa ($(du -h "${ROOT}/ios/build/Backburner.ipa" | cut -f1)). Install: docs/INSTALL-IPHONE.md"
  exit 0
fi

if [[ -n "${UDID}" ]] && xcrun devicectl device info details --device "${UDID}" >/dev/null 2>&1; then
  echo "installing onto ${UDID} (DDI may fail; install can still work)"
  xcrun devicectl device install app --device "${UDID}" "${APP}" || {
    echo "devicectl install failed. Open Finder, drag ${APP}, or use Apple Configurator."
    exit 0
  }
  echo "installed. Open Backburner, keep it foreground, then scripts/serve.sh"
else
  echo "device ${UDID} not visible. Sidecar.app is ready to sideload."
fi

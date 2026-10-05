#!/bin/bash
# build-release.sh VERSION - the release assets in build/release/, built so they carry no personal data:
#   backburner-mac-arm64.tar.gz   llama-server + llama-quantize, static, portable (any Apple-silicon Mac, macOS 14+)
#   Backburner.ipa                the phone app, unsigned, for AltStore (scripts/build-iphone.sh IPA=1)
#   ane-template.tar.gz           the phone's Neural Engine page model (ANE_TEMPLATE=path to reuse a previous release's)
#   SHA256SUMS
# Source paths are rewritten at compile time (-ffile-prefix-map), and the build stops if any binary still contains a
# home-folder path or one of your private strings (scripts/hooks/check-sensitive.py fingerprints).
# Then: version entry in altstore/source.json (size printed here), `gh release create vVERSION --prerelease ...`.
set -euo pipefail
VERSION=${1:?usage: scripts/build-release.sh VERSION}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/llama.cpp"
OUT="$ROOT/build/release"
mkdir -p "$OUT"
cd "$ROOT"

echo "== Mac engine (static, GGML_NATIVE=OFF, macOS 14+)"
MAP="-ffile-prefix-map=$LLAMA=llama.cpp"
cmake -S "$LLAMA" -B "$LLAMA/build-release" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_C_FLAGS="$MAP" -DCMAKE_CXX_FLAGS="$MAP" -DCMAKE_OBJC_FLAGS="$MAP" >/dev/null
cmake --build "$LLAMA/build-release" --target llama-server llama-quantize -j "$(sysctl -n hw.ncpu)" >/dev/null
"$LLAMA/build-release/bin/llama-server" --version 2>&1 | grep -q 'version' || { echo "llama-server does not run"; exit 1; }
tar -C "$LLAMA/build-release/bin" -czf "$OUT/backburner-mac-arm64.tar.gz" llama-server llama-quantize

echo "== phone app (unsigned IPA)"
IPA=1 scripts/build-iphone.sh >/dev/null
cp ios/build/Backburner.ipa "$OUT/Backburner.ipa"

echo "== Neural Engine template"
if [ -n "${ANE_TEMPLATE:-}" ]; then
  cp "$ANE_TEMPLATE" "$OUT/ane-template.tar.gz"
else
  tar -C build/ane-kv/tmpl -czf "$OUT/ane-template.tar.gz" kv_N16384_R48_fp16_pfix_cinput.mlmodelc
fi

echo "== privacy check of every binary"
CHK=$(mktemp -d)
trap 'rm -rf "$CHK"' EXIT
tar -C "$CHK" -xzf "$OUT/backburner-mac-arm64.tar.gz"
(cd "$CHK" && unzip -q "$OUT/Backburner.ipa")
mkdir -p "$CHK/ane" && tar -C "$CHK/ane" -xzf "$OUT/ane-template.tar.gz"
bad=0
while IFS= read -r -d '' f; do
  n=$(strings -a "$f" | grep -cE '/(Users|home)/[^/ ]+/' || true)
  [ "$n" = 0 ] || { echo "  $f: $n home-folder paths"; bad=1; }
  if strings -a "$f" | python3 -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cs", sys.argv[1]); cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
sys.exit(1 if any(cs.private_hit(l) for l in sys.stdin) else 0)' "$ROOT/scripts/hooks/check-sensitive.py"; then :; else
    echo "  $f: contains one of your private strings"; bad=1
  fi
done < <(find "$CHK" -type f -print0)
[ $bad = 0 ] || { echo "release assets contain personal data: not finished"; exit 1; }
echo "  no home-folder paths or private strings in any asset"

(cd "$OUT" && shasum -a 256 Backburner.ipa backburner-mac-arm64.tar.gz ane-template.tar.gz > SHA256SUMS)
echo "== done: $OUT (v$VERSION)"
echo "  IPA size for altstore/source.json: $(/usr/bin/stat -f %z "$OUT/Backburner.ipa")"
cat "$OUT/SHA256SUMS"

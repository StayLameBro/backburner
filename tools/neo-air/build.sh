#!/bin/bash
# build.sh: build sdref into tools/neo-air/bin against an existing llama.cpp build (default llama.cpp/build-metal)
#   BUILD=<llama.cpp build dir>  tools/neo-air/build.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); LC=$(cd "$D/../../llama.cpp" && pwd); B=${BUILD:-$LC/build-metal}
mkdir -p "$D/bin"
xcrun clang++ -std=c++17 -O2 -I"$LC/include" -I"$LC/common" -I"$LC/ggml/include" -o "$D/bin/sdref" "$D/sdref.cpp" \
  -L"$B/bin" -lllama -lllama-common -lggml -lggml-base -Wl,-rpath,"$B/bin"
echo "built: $D/bin/sdref"

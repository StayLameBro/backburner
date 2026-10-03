#!/bin/bash
# install.sh - set up Backburner on an Apple Silicon Mac in one go:
#   curl -fsSL https://raw.githubusercontent.com/StayLameBro/backburner/main/install.sh | bash
# Safe to re-run: every step is skipped when it is already done. Then run `backburner`.
#   1. checks the Mac (Apple Silicon, macOS 14+, 24 GB+ memory, free disk)
#   2. the repo in ~/backburner (BACKBURNER_DIR=...)
#   3. the Mac engine from the latest release (BUILD=1: build from source instead; needs cmake and the Xcode command line tools)
#   4. a Python venv (.venv) with numpy for the model tools
#   5. the models in ~/Models (MODELS=...): Qwen3.8-27B IQ4_XS (15.5 GB) and the DFlash2 draft model (converted here, ~1.3 GB)
#   6. the phone's half of the model (layers 41-64, ~5.1 GB) and the phone's Neural Engine page template
#   7. the `backburner` command in ~/.local/bin
set -euo pipefail
REPO=StayLameBro/backburner
DIR=${BACKBURNER_DIR:-$HOME/backburner}
MODELS=${MODELS:-$HOME/Models}
MODEL_URL=${MODEL_URL:-https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-IQ4_XS.gguf}
DRAFT_REPO=${DRAFT_REPO:-z-lab/Qwen3.8-27B-DFlash2}
TAIL_L=${TAIL_L:-40}   # the phone runs layers TAIL_L+1..64: 40 for the A19 Pro (iPhone 17 Pro), 52 for the A18 Pro (16 Pro)

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# a release asset's URL: the newest release (pre-releases included) that has it. RELEASE_DIR=folder: local files (testing)
asset_url() {
  if [ -n "${RELEASE_DIR:-}" ]; then [ -s "$RELEASE_DIR/$1" ] && echo "file://$RELEASE_DIR/$1"; return; fi
  curl -fsSL "https://api.github.com/repos/$REPO/releases" | python3 -c '
import json, sys
for r in json.load(sys.stdin):
    for a in r["assets"]:
        if a["name"] == sys.argv[1]:
            print(a["browser_download_url"]); sys.exit()' "$1"
}
fetch() {   # URL OUT: resumable download through OUT.part
  [ -s "$2" ] && return 0
  curl -fL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

# 1. the Mac
[ "$(uname -m)" = arm64 ] || die "Backburner needs an Apple Silicon Mac (M1 or newer)"
MACOS=$(sw_vers -productVersion); [ "${MACOS%%.*}" -ge 14 ] || die "macOS 14 or newer needed (this Mac has $MACOS)"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if [ "$MEM_GB" -lt 24 ] && [ "${FORCE:-0}" != 1 ]; then
  die "the 27B model needs a Mac with 24 GB or more (this one has $MEM_GB GB). FORCE=1 to try anyway"
fi
command -v git >/dev/null && command -v python3 >/dev/null || die "install the command line tools first: xcode-select --install"
FREE_GB=$(df -g "$HOME" | awk 'NR==2{print $4}')
[ "$FREE_GB" -ge 30 ] || warn "only $FREE_GB GB free; a fresh install downloads/creates ~28 GB"
say "Mac: $(sysctl -n machdep.cpu.brand_string), $MEM_GB GB, macOS $MACOS"

# 2. the repo
if [ -d "$DIR/.git" ]; then
  say "updating $DIR"
  git -C "$DIR" pull --ff-only -q || warn "could not update $DIR (local changes?): keeping it as is"
  git -C "$DIR" submodule update --init --depth 1 -q
else
  say "cloning into $DIR"
  git clone -q --depth 1 --recurse-submodules --shallow-submodules "https://github.com/$REPO" "$DIR"
fi
cd "$DIR"

# 3. the Mac engine
BIN=llama.cpp/build-metal/bin
if [ ! -x "$BIN/llama-server" ] || [ ! -x "$BIN/llama-quantize" ]; then
  mkdir -p "$BIN"
  if [ "${BUILD:-0}" = 1 ]; then
    say "building the Mac engine from source (~10 min)"
    cmake -S llama.cpp -B llama.cpp/build-metal -DCMAKE_BUILD_TYPE=Release
    cmake --build llama.cpp/build-metal --target llama-server llama-quantize -j
  else
    say "downloading the Mac engine"
    URL=$(asset_url backburner-mac-arm64.tar.gz) || true
    [ -n "${URL:-}" ] || die "no prebuilt Mac engine in the releases yet: re-run with BUILD=1"
    curl -fsSL "$URL" | tar -xz -C "$BIN"
    xattr -dr com.apple.quarantine "$BIN" 2>/dev/null || true
  fi
fi
"$BIN/llama-server" --version 2>&1 | grep -q version || die "$BIN/llama-server does not run on this Mac: re-run with BUILD=1"

# 4. Python for the model tools (llama.cpp's gguf-py needs 3.10+; macOS's own python3 is 3.9: then uv fetches 3.12, no sudo)
PYDEPS=(numpy pyyaml tqdm requests)
if ! .venv/bin/python3 -c 'import numpy, yaml' 2>/dev/null; then
  say "Python environment (.venv)"
  rm -rf .venv
  PYBIN=
  for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PYBIN=$(command -v "$p"); break
    fi
  done
  if [ -n "$PYBIN" ]; then
    "$PYBIN" -m venv .venv
    .venv/bin/pip install -q "${PYDEPS[@]}"
  else
    UV=$(command -v uv || echo "$HOME/.local/bin/uv")
    [ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
    "$UV" venv -q --python 3.12 .venv
    "$UV" pip install -q --python .venv/bin/python3 "${PYDEPS[@]}"
  fi
fi
export PATH="$DIR/.venv/bin:$PATH"

# 5. the models
mkdir -p "$MODELS"
MODEL=$MODELS/Qwen3.8-27B-IQ4_XS.gguf
if [ ! -s "$MODEL" ]; then
  say "downloading Qwen3.8-27B IQ4_XS (15.5 GB; resumes if interrupted)"
  fetch "$MODEL_URL" "$MODEL"
fi
DRAFT=$MODELS/dflash2-v2-q4km-self16.gguf
if [ ! -s "$DRAFT" ]; then
  say "downloading and converting the DFlash2 draft model"
  SRC=$MODELS/qwen38-27b-dflash2; mkdir -p "$SRC"
  for f in config.json model.safetensors; do fetch "https://huggingface.co/$DRAFT_REPO/resolve/main/$f" "$SRC/$f"; done
  TARGET=$MODEL scripts/make-drafter.sh "$SRC" "$DRAFT"
fi

# 6. the phone's share
TAIL=$MODELS/tail-iq4xs-L$TAIL_L-nohead.gguf
if [ ! -s "$TAIL" ]; then
  say "making the phone's half of the model (layers $((TAIL_L + 1))-64)"
  python3 scripts/split-gguf.py "$MODEL" "$TAIL" -L "$TAIL_L" --no-head
fi
TMPL=build/ane-kv/tmpl/kv_N16384_R48_fp16_pfix_cinput.mlmodelc
if [ ! -d "$TMPL" ]; then
  URL=$(asset_url ane-template.tar.gz) || true
  if [ -n "${URL:-}" ]; then
    mkdir -p build/ane-kv/tmpl && curl -fsSL "$URL" | tar -xz -C build/ane-kv/tmpl
  else
    warn "no Neural Engine page template in the releases: writing past 64k uses the phone's GPU only"
  fi
fi

# 7. the backburner command
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/backburner" <<EOF
#!/bin/bash
# backburner - run Backburner (made by install.sh). Settings go in front: CTX=131072 backburner
#   backburner            start the server (OpenAI-compatible, http://127.0.0.1:8080/v1), with the phone if it's plugged in
#   backburner phone      put the phone's half of the model on the plugged-in phone (once per phone)
#   backburner update     re-run the installer (updates the repo and the engine)
set -eu
DIR="$DIR"
export PATH="\$DIR/.venv/bin:\$PATH" PY="\$DIR/.venv/bin/python3"
MEM_MB=\$(( \$(sysctl -n hw.memsize) / 1048576 ))
WANT=\$(( MEM_MB - 4096 ))
if [ "\$MEM_MB" -le 32768 ] && [ "\$(sysctl -n iogpu.wired_limit_mb)" -lt "\$WANT" ]; then
  echo "backburner: letting the GPU keep the model in memory (macOS resets this at every reboot; asks for your password)"
  sudo sysctl iogpu.wired_limit_mb=\$WANT >/dev/null
fi
case "\${1:-}" in
  phone)  exec "\$DIR/scripts/phone-tail.sh" "$TAIL" ;;
  update) curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash ;;
  *)      exec "\$DIR/scripts/serve.sh" ;;
esac
EOF
chmod +x "$HOME/.local/bin/backburner"

say "done"
echo
echo "  iPhone app:  $DIR/docs/INSTALL-IPHONE.md (free Apple ID + AltStore)"
echo "  phone half:  plug the phone in, open Backburner, then: backburner phone"
echo "  start:       backburner      (OpenAI-compatible at http://127.0.0.1:8080/v1, any model name and API key)"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo; warn "add ~/.local/bin to your PATH: echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.zshrc" ;; esac

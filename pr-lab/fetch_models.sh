#!/usr/bin/env bash
# Fetch test models (run in WSL). Tiny for smoke + 7B Q4_K_M for perf.
# Usage: bash fetch_models.sh
set -euo pipefail
DIR=~/llama-pr-lab/models
mkdir -p "$DIR"
cd "$DIR"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "installing huggingface_hub..."
  pip3 install --break-system-packages -q --upgrade huggingface_hub
fi

echo "== tiny smoke model (~638MB, TheBloke, curl-direct, no HF auth needed) =="
if [ ! -f tinyllama.gguf ]; then
  curl -L --fail -o tinyllama.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"
fi

echo "== 7B perf model (~4.7GB, Qwen, curl-direct) =="
if [ ! -f qwen2.5-7b.gguf ]; then
  curl -L -C - --fail -o qwen2.5-7b.gguf "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf?download=true"
fi

ls -lh "$DIR"

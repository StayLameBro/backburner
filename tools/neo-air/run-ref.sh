#!/bin/bash
# run-ref.sh NAME [extra args]: Mac-only reference for prompts/p_NAME.txt, greedy SDREF_N steps (64), no split engine.
# On an 8 GB Mac the whole model doesn't fit on the GPU, so pass the split-mode layout and the CPU computes layers >= L:
#   tools/neo-air/run-ref.sh short -ot '^blk\.(20|[1-9][0-9][0-9]+|[3-9][0-9]|2[1-9])\.=CPU' -ot '^output=CPU' --no-repack -t 6 -tb 6
# Writes $OUT/ref_NAME.{tok,argmax,logits}. ~10 s per token on the Neo with the CPU half: budget ~20 min per prompt.
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$D/out}; mkdir -p "$OUT"
env -u LLAMA_SPLIT_TAIL -u LLAMA_SPLIT_DECODE LLAMA_LAZY_EMBD=1 GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 GGML_METAL_FA_PREFILL_GQA=1 \
  SDREF_PROMPT="$D/prompts/p_$1.txt" SDREF_OUT="$OUT/ref_$1" SDREF_N=${N:-64} \
  "$D/bin/sdref" -m "${MODEL:-$HOME/Models/Qwen3.8-27B-IQ2_XS.gguf}" -ngl 999 -fa on -c 8192 -ctk q8_0 -ctv q8_0 -ub 256 -t 2 -tb 2 "${@:2}"

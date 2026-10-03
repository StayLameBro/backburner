#!/bin/bash
# run-split.sh NAME greedy|force [extra args]: split decode at L (LLAMA_SPLIT_L, 20) against the phone tail at TAIL=<ip>:50060,
# with serve.sh SPLIT_DECODE_L's model flags. "force" feeds the reference's tokens ($OUT/ref_NAME.tok: teacher forcing), so
# cmp.py can compare the logits position by position. Writes $OUT/${TAG:-sd}_MODE_NAME.{tok,argmax,logits}.
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$D/out}; mkdir -p "$OUT"
[ -n "${TAIL:-}" ] || { echo "set TAIL=<phone ip>:50060 (scripts/phone-up.sh prints it)" >&2; exit 1; }
L=${LLAMA_SPLIT_L:-20}
GE=$(python3 -c "import sys; L=int(sys.argv[1]); print('|'.join([str(L)] + [f'{d}[0-9]' for d in range(L//10+1, 10)] + ([f'{L//10}[{L%10+1}-9]'] if L%10<9 else []) + ['[1-9][0-9][0-9]+']))" "$L")
F=; [ "$2" = force ] && F=$OUT/ref_$1.tok
env LLAMA_LAZY_EMBD=1 GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 GGML_METAL_FA_PREFILL_GQA=1 LLAMA_SPLIT_GPU_WARM_US=${LLAMA_SPLIT_GPU_WARM_US:-1000} \
  LLAMA_SPLIT_DECODE=1 LLAMA_SPLIT_L=$L LLAMA_SPLIT_TAIL=$TAIL \
  SDREF_PROMPT="$D/prompts/p_$1.txt" SDREF_OUT="$OUT/${TAG:-sd}_$2_$1" SDREF_N=${N:-64} ${F:+SDREF_FORCE=$F} \
  "$D/bin/sdref" -m "${MODEL:-$HOME/Models/Qwen3.8-27B-IQ2_XS.gguf}" -ngl 999 -fa on -c 8192 -ctk q8_0 -ctv q8_0 -ub 256 -t 2 -tb 2 \
  -ot "^blk\.($GE)\.=CPU" -ot '^output=CPU' --no-repack "${@:3}"

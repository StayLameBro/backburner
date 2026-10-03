#!/bin/bash
# serve.sh - Backburner's OpenAI-compatible server for Qwen3.8-27B (omp or any client); uses the wired iPhone when it's there.
#   scripts/serve.sh                 # IQ4_XS, 64k context, port 8080
#   CTX=98304 PORT=8081 MODEL=/path.gguf scripts/serve.sh
#   CTX=131072 scripts/serve.sh     # 128k: q4_0 KV, no-mmap load, embeddings on CPU (see the BIG block)
# Client: base URL http://127.0.0.1:${PORT}/v1, any model name, API key anything.
# Notes:
# - Speculative decoding = n-gram (copy) + DFlash2 drafter on the Mac GPU. Greedy (temperature 0) is fastest: the batched
#   argmax path only engages for greedy requests; other temperatures still work, via the normal sampler chain.
# - Memory (24 GB Mac): replay rollback (-1 GB) + drafter ubatch 64 (-1.3 GB) are what make IQ4_XS + drafter fit at 64k.
#   128k fits Mac-only with q4_0 KV (BIG block below); 128k q8_0 and 262k need the phone (README: "Phone-held context").
# - Prompt cache on the SSD (scripts/proxy.py, PROXY=0 to turn off): the proxy owns PORT, the server runs on PORT+100.
#   A new conversation restores its saved preamble (system prompt + tools, ~25k tokens for omp) in ~0.2 s instead of re-reading it
#   for ~4 minutes; the first time a preamble is seen it is read once and saved. Conversations you switch away from are saved too.
# - SME co-attention (the Mac CPU matrix units take ~20-35% of the keys at depth) is ON by default from 40k keys: see the SME= line.
set -u
cd "$(dirname "$0")/.."
# CLEAN=1 (screen recordings): every log line goes through a filter that hides the home folder, cable addresses, device ids and
# phone names. The filter ignores Ctrl-C so the last lines still print.
if [ "${CLEAN:-0}" = 1 ]; then
  exec > >(trap '' INT; exec sed -l -E -e "s#$HOME#~#g" -e 's/169\.254\.[0-9]+\.[0-9]+/169.254.x.y/g' \
    -e 's/0000[0-9]{4}-[0-9A-Fa-f]{16}/<device id>/g' \
    -e "s/iphone \([0-9]+\)/iPhone/gI") 2>&1
fi
B=${BIN:-$PWD/llama.cpp/build-metal/bin}
MODEL=${MODEL:-$HOME/Models/Qwen3.8-27B-IQ4_XS.gguf}
DRAFT=${DRAFT-$HOME/Models/dflash2-v2-q4km-self16.gguf}   # selector kept in f16: +7.5% acceptance vs the all-Q4_K file; DRAFT= (empty): no drafter
CTX=${CTX:-65536}
if [ "$CTX" -gt 65536 ]; then KV=${KV:-q4_0}; else KV=${KV:-q8_0}; fi
# SPLIT_DECODE_L=N (split decode, small Mac + phone): the Mac holds and runs only layers [0,N); the phone's tail (made with
# scripts/split-gguf.py -L N, head kept: no --no-head) runs layers N+ and the output head for every token and returns the logits.
# The Mac keeps the full GGUF mapped but sends layers N+ and output* to the CPU (-ot), where they stay as never-read file pages,
# so the GPU maps only [0,N). Needs llama.cpp split decode (LLAMA_SPLIT_DECODE); without it decode still runs every layer on
# the Mac and pages layers N+ in from the SSD. No speculation (the phone can't rewind its recurrent state), no drafter, no
# checkpoints, no prompt-cache proxy (a saved slot would lack the phone's half), no phone-held KV, SME off. docs/SPLIT-DECODE.md
SD_ARGS=()
if [ -n "${SPLIT_DECODE_L:-}" ]; then
  SDL=$SPLIT_DECODE_L
  case "$SDL" in ''|*[!0-9]*|0*) echo "serve: SPLIT_DECODE_L must be a positive layer number, got '$SDL'" >&2; exit 1 ;; esac
  [ "$CTX" -gt 65536 ] && { echo "serve: SPLIT_DECODE_L needs CTX <= 65536 (the CTX > 65536 block loads with -lm none)" >&2; exit 1; }
  [ -n "${PHONE_KV:-}" ] && { echo "serve: SPLIT_DECODE_L and PHONE_KV can't be combined" >&2; exit 1; }
  if [ -z "${LLAMA_SPLIT_TAIL:-}" ]; then   # the phone's tail is required: find it on the cable
    PUP=$(TAIL_WAIT=${TAIL_WAIT:-90} "$(dirname "$0")/phone-up.sh") || exit 1
    read -r PIP PTAIL _ <<< "$(head -1 <<< "$PUP")"
    [ "$PTAIL" = 1 ] || { echo "serve: SPLIT_DECODE_L needs the phone's tail (:50060) up: scripts/phone-tail.sh <tail-...-L$SDL-head.gguf>" >&2; exit 1; }
    LLAMA_SPLIT_TAIL=$PIP:50060
  fi
  export LLAMA_SPLIT_TAIL LLAMA_SPLIT_L=$SDL LLAMA_SPLIT_DECODE=1
  # keep the Mac GPU clocked up while the phone computes (a trivial dispatch every 1 ms, only during the wait): the A18 Pro
  # otherwise drops its clock in the ~160 ms gap and the next head pass runs slow. Head-only bench, L=20 IQ2_XS, 160 ms gap per
  # token: 98.7-101.0 -> 88.0-88.1 ms per token, p95 139-141 -> 95-96 ms, +0.5 W average GPU power. 2026-10-03.
  export LLAMA_SPLIT_GPU_WARM_US=${LLAMA_SPLIT_GPU_WARM_US:-1000}
  PHONE=0 PHONE_ANE=0 PHONE_IP=${LLAMA_SPLIT_TAIL%:*}
  LOAD_MODE=mmap SPEC_TYPE=none DRAFT= SPEC_REPLAY=0 CTX_CHECKPOINTS=0 PROXY=0 SME=0 MM_SME=0
  # layer numbers >= N as a regex (no upper bound, no commas: -ot splits its value on ','); anchored so blk.2. != blk.20.
  digits() { local k; for (( k = 0; k < $1; k++ )); do printf '[0-9]'; done; }
  ge() {   # N itself, every longer number, and per digit: same prefix, a bigger digit, any digits after
    local n=$1 d=${#1} i c alt
    alt="$n|[1-9]$(digits "$d")+"
    for (( i = 0; i < d; i++ )); do
      c=${n:i:1}
      [ "$c" = 9 ] && continue
      alt="$alt|${n:0:i}$([ "$c" = 8 ] && echo 9 || echo "[$((c + 1))-9]")$(digits $((d - i - 1)))"
    done
    echo "$alt"
  }
  # --no-repack: the CPU backend would otherwise read layers N+ at load and keep an anonymous repacked copy (847 MiB at
  # IQ2_XS L=20, measured). The warmup decode goes through the phone like any other batch, so it stays on: it also checks
  # the tail has its head (a head-less tail fails it with "worker returned 0 logits").
  SD_ARGS=(-ot "^blk\.($(ge "$SDL"))\.=CPU" -ot "^output=CPU" --no-repack)
  echo "serve: SPLIT DECODE mode ON, L=$SDL: Mac GPU maps layers [0,$SDL), layers $SDL+ and output stay CPU-mapped (never read), phone tail at $LLAMA_SPLIT_TAIL runs the rest + head; spec off, no drafter, no checkpoints, no proxy, SME off" >&2
  if ! grep -qa LLAMA_SPLIT_DECODE "$B"/llama-server "$B"/*.dylib 2>/dev/null; then
    echo "serve: ************************************************************************************************" >&2
    echo "serve: WARNING: this llama.cpp has NO split decode (LLAMA_SPLIT_DECODE). Decode will NOT be split:" >&2
    echo "serve: every generated token runs layers $SDL+ and the head on the Mac CPU, paging them from the SSD (seconds per" >&2
    echo "serve: token). Rebuild llama.cpp (git submodule update, then the cmake build in the README) before sending requests." >&2
    echo "serve: ************************************************************************************************" >&2
  fi
fi
# PHONE=auto (default): find the iPhone on the USB cable (scripts/phone-up.sh: its current address, relaunch the Backburner app if it
# stopped, wait for the prefill tail) and use it for split prefill (LLAMA_SPLIT_TAIL) and for the old KV past the Mac's cells
# (PHONE_KV: nothing changes below 64k). PHONE=0: Mac only. An explicit LLAMA_SPLIT_TAIL / PHONE_KV wins.
if [ "${PHONE:-auto}" != 0 ] && [ -z "${LLAMA_SPLIT_TAIL:-}${PHONE_KV:-}" ]; then
  if PUP=$(PHONES_ALL=1 "$(dirname "$0")/phone-up.sh"); then
    read -r PIP PTAIL PVER PAVAIL PWIRED PNAME <<< "$(head -1 <<< "$PUP")"
    # The phone's Neural Engine takes part of the old-key attention while writing past 64k (docs/ANE.md: 279 -> 176 ms per token
    # at 140k). It needs the page template in the app's Documents, which installing the app doesn't bring: push it once
    # (scripts/phone-ane.sh builds it first if needed, with coremltools). PHONE_ANE=0: don't check.
    ane_on() { printf 'mem\n' | nc -G 2 "$1" 50061 2>/dev/null | grep -q '"pa_ane"'; }
    PANE=unchecked
    if [ "${PHONE_ANE:-1}" != 0 ]; then
      if ane_on "$PIP"; then PANE=on
      else
        echo "serve: the phone's ANE pages are off (no template on the phone yet): pushing it with scripts/phone-ane.sh" >&2
        PANE=off
        "$(dirname "$0")/phone-ane.sh" "$PIP" >&2   # relaunches the app; phone-up below waits for it either way
        if PUP=$(PHONES_ALL=1 "$(dirname "$0")/phone-up.sh"); then
          read -r PIP PTAIL PVER PAVAIL PWIRED PNAME <<< "$(head -1 <<< "$PUP")"
          ane_on "$PIP" && PANE=on
        fi
        [ $PANE = off ] && echo "serve: ANE pages still off: old keys run on the phone GPU only (writing past 64k is slower). Check scripts/phone-ane.sh $PIP" >&2
      fi
    fi
    [ "${CLEAN:-0}" = 1 ] && PNAME=iPhone
    PHONE_IP=$PIP
    [ "$PTAIL" = 1 ] && export LLAMA_SPLIT_TAIL=$PIP:50060
    # Phone-attn's shared Metal KV counts toward both the app footprint limit and system wired memory.
    # q8_0 costs 34,816 bytes/remote token, q4_0 18,432, f16 65,536.
    PHONE_WIRED_MAX_MB=${PHONE_WIRED_MAX_MB:-9400}
    PHONE_APP_RESERVE_MB=${PHONE_APP_RESERVE_MB:-512}
    if [ "${PWIRED:-0}" -le 0 ] || [ "${PWIRED:-0}" -ge "$PHONE_WIRED_MAX_MB" ] || [ "${PAVAIL:-0}" -le "$PHONE_APP_RESERVE_MB" ]; then
      echo "serve: phone wired/app memory is unknown or too low; remote KV disabled" >&2
    else
      PHONE_KV=$PIP:50062
      REMOTE_BPT=34816
      [ "$KV" = q4_0 ] && REMOTE_BPT=18432
      [ "$KV" = f16 ] && REMOTE_BPT=65536
      PHONE_CAP_WIRED=$(( CTX + (PHONE_WIRED_MAX_MB - PWIRED) * 1048576 / REMOTE_BPT / 4096 * 4096 ))
      PHONE_CAP_APP=$(( CTX + (PAVAIL - PHONE_APP_RESERVE_MB) * 1048576 / REMOTE_BPT / 4096 * 4096 ))
      PHONE_CAP=$PHONE_CAP_WIRED
      [ "$PHONE_CAP_APP" -lt "$PHONE_CAP" ] && PHONE_CAP=$PHONE_CAP_APP
      # more phones (another cable): each holds an equal share of the old keys, so the smallest phone sets the share
      NPH=1; SHARE=$(( PHONE_CAP - CTX ))
      while read -r ip2 _ _ a2 w2 _; do
        [ -n "$ip2" ] || continue
        if [ "${w2:-0}" -le 0 ] || [ "${w2:-0}" -ge "$PHONE_WIRED_MAX_MB" ] || [ "${a2:-0}" -le "$PHONE_APP_RESERVE_MB" ]; then
          echo "serve: the iPhone at $ip2 has too little memory free: not used" >&2; continue
        fi
        s_w=$(( (PHONE_WIRED_MAX_MB - w2) * 1048576 / REMOTE_BPT / 4096 * 4096 ))
        s_a=$(( (a2 - PHONE_APP_RESERVE_MB) * 1048576 / REMOTE_BPT / 4096 * 4096 ))
        [ "$s_a" -lt "$s_w" ] && s_w=$s_a
        [ "$s_w" -lt "$SHARE" ] && SHARE=$s_w
        PHONE_KV=$PHONE_KV,$ip2:50062; NPH=$((NPH + 1))
      done <<< "$(tail -n +2 <<< "$PUP")"
      PHONE_CAP=$(( CTX + NPH * SHARE ))
      [ "$PHONE_CAP" -gt 262144 ] && PHONE_CAP=262144
      if [ -n "${CTX_TOTAL:-}" ] && [ "$CTX_TOTAL" -gt "$PHONE_CAP" ]; then
        echo "serve: requested CTX_TOTAL=$CTX_TOTAL exceeds the phone's safe $KV cap $PHONE_CAP (${PWIRED} MiB wired, ${PAVAIL} MiB app budget)" >&2
        exit 1
      fi
      CTX_TOTAL=${CTX_TOTAL:-$PHONE_CAP}
    fi
    [ "${NPH:-1}" -gt 1 ] && echo "serve: $NPH iPhones share the old keys" >&2
    echo "serve: $PNAME at $PIP: split prefill $([ "$PTAIL" = 1 ] && echo on || echo off); context up to ${CTX_TOTAL:-${CTX:-65536}} tokens (remote KV $([ -n "${PHONE_KV:-}" ] && echo on || echo off), ANE pages $PANE, v$PVER)" >&2
  else
    echo "serve: no phone: Mac only, 64k context" >&2
  fi
fi
# PHONE_KV=ip:port (the Backburner app phone-attn, :50062): the phone holds the OLDEST KV pages (README: "Phone-held context"). CTX is then the
# number of cells the Mac keeps (default 65536: with --cache-ram 0 / 6 checkpoints the 64k config is 18.4 GB, no swap; the phone
# only takes what goes past 64k, so nothing slows down below that) and a conversation may grow to
# CTX_TOTAL tokens. The phone's app and system memory determine the safe limit. KV may be q8_0, q4_0, or f16.
if [ -n "${PHONE_KV:-}" ]; then
  CTX=${CTX:-65536}
  export LLAMA_KV_REMOTE=$PHONE_KV LLAMA_KV_REMOTE_CTX=${CTX_TOTAL:-262144}
  # pipelined prefill once the phone holds keys: ubatches of 512 run as two staggered halves of 256, so the Mac computes one
  # half's layers while the phone computes the other half's old keys (same math as 256-token ubatches). 140k q4_0: 58.0 ->
  # 67.4-68.3 tok/s, tokens identical, 2026-09-29. LLAMA_REMOTE_PIPE=0 to turn off.
  export LLAMA_REMOTE_PIPE=${LLAMA_REMOTE_PIPE:-1} LLAMA_UBATCH_REMOTE=${LLAMA_UBATCH_REMOTE:-512}
fi
PORT=${PORT:-8080}
# SME co-attention (the Mac CPU's matrix units take the oldest keys of each attention layer; README: "SME2 on the Mac CPU"): ON by default
# since 2026-09-25 (user: fine as long as output quality holds; they run only light apps beside the model). With llama.cpp 6bb2bffef+
# (per-cluster SME worker roles, spin while jobs are queued): 51k -3.5 ms/round, 140k -9.8 ms/round; greedy output token-identical,
# top-5 probability changes of the same size as a GPU split-count change. CAUTION: keeps ~8 P-cores busy while generating (server
# ~790% CPU) and the GPU waits on the CPU's share: heavy CPU load beside it (big builds) can stall rounds; on 2026-09-23 (older code)
# that once hit a Metal GPU timeout. SME=0 turns it off. SME_MIN_KV: engaged only from this many keys (below 40k untested since the fixes).
SME=${SME:-0.35}
SME_MIN_KV=${SME_MIN_KV:-40960}
# 128k (CTX > 65536), measured 2026-09-23: q8_0 KV is ~0.3 GB over the 20.5 GB Metal cap with the drafter on the Mac, q4_0 fits
# (~18.8 GB) once the model is loaded without mmap (a mapped file puts all 15.6 GB in the GPU working set) and the 644 MiB token
# embedding table stays on the CPU (8 lookups per round, free). Batches of 256 shrink the compute buffer at no prefill cost.
BIG=0; [ "$CTX" -gt 65536 ] && BIG=1
if [ -n "${PHONE_KV:-}" ] && [ "$KV" != q8_0 ] && [ "$KV" != q4_0 ] && [ "$KV" != f16 ]; then
  echo "serve: remote KV supports q8_0, q4_0, or f16" >&2
  exit 1
fi
EXTRA=(); [ $BIG = 1 ] && EXTRA=(-lm none -ot token_embd.weight=CPU -ub 256)
# LOAD_MODE=none loads the weights into wired memory: macOS then can't evict them under memory pressure (with the default mmap,
# pressure from other apps made the server re-read 3-16 GB from the SSD per 160 tokens: 5-8 instead of ~23 tok/s, measured 2026-09-23)
LOAD_MODE=${LOAD_MODE:-none}   # default: wired (set LOAD_MODE=mmap to go back)
# the wired load needs the GPU wired limit raised; it resets to the default (~16 GB) on every reboot, and then the model +
# drafter + KV page and crawl (or crash the Mac). Refuse instead of running slow.
WL=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [ "$LOAD_MODE" != mmap ] && [ "${WL:-0}" -lt 20000 ]; then
  echo "serve: iogpu.wired_limit_mb is ${WL} (reset by a reboot). Run: sudo sysctl iogpu.wired_limit_mb=20480" >&2
  exit 1
fi
[ $BIG = 0 ] && [ "$LOAD_MODE" != mmap ] && EXTRA+=(-lm "$LOAD_MODE")
# PHONE_DRAFT=ip:port (the Backburner app's ggml-rpc, e.g. 169.254.x.y:50052): the drafter runs on the phone. Frees ~2.8 GB of Mac memory
# (the 64k Mac-only config swaps ~1.5-3 GB), but the draft is serial, so rounds are slower.
# -dev MTL0 is mandatory: without it --rpc puts part of the 27B on the phone (3 tok/s).
[ -n "${PHONE_DRAFT:-}" ] && EXTRA+=(--rpc "$PHONE_DRAFT" -dev MTL0 --spec-draft-device RPC0)
# replay rollback rows per recurrent cell (sent to the phone too); SPEC_REPLAY=0 leaves the flag off (no speculation needs no rollback)
SPEC_REPLAY=${SPEC_REPLAY:-8}
REPLAY=(); [ "$SPEC_REPLAY" != 0 ] && REPLAY=(--spec-gdn-replay "$SPEC_REPLAY")

export GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 GGML_METAL_FA_PREFILL_GQA=1 LLAMA_BATCHED_ARGMAX=1 SPEC_DRAFT_UBATCH=64
# lossless speculative sampling for sampled (temperature > 0) requests, e.g. omp: +12% on the omp replay
export LLAMA_SPEC_SAMPLE=${LLAMA_SPEC_SAMPLE:-1}
# split prefill with the phone (LLAMA_SPLIT_TAIL=IP:50060): split messages from 512 tokens (a 2k tool result would never reach
# the old 2048 minimum), and keep the prompt in one batch so the phone gets all but one ubatch (the server's 4 + n_ubatch
# checkpoint stop otherwise leaves the last ubatch Mac-only). L is learned from the phone's tail. 2026-09-26, A18 L=52 at 51k.
# Batches of 256 with the phone: twice the pipeline chunks, so the fill/drain and the Mac-only last batch halve (A19 L=44 at 51k:
# 93.5 -> 97.3 tok/s; Mac-only pays 3% at 256, so only when a phone is set). SPLIT_UB=512 to go back. 2026-09-26.
if [ -n "${LLAMA_SPLIT_TAIL:-}" ]; then
  export LLAMA_SPLIT_MIN=${LLAMA_SPLIT_MIN:-512} LLAMA_SPLIT_ONE_BATCH=${LLAMA_SPLIT_ONE_BATCH:-1}
  [ $BIG = 0 ] && EXTRA+=(-ub "${SPLIT_UB:-256}")
fi
# depth-aware verify width: past 40k tokens draft at most 4 (verify 5 rows). Measured 2026-09-24 at 140k (synthetic depth, q4_0):
# 221.4 -> 181.9 ms/round; with the real per-position acceptance (3.16 -> 2.90 tokens/round) that is +12% tok/s. From 389 real
# rounds x the FA cost table, cap 4 breaks even at ~39k and pays +1% at 44k, +3% at 60k, +4% at 64k; below 40k it loses.
# 2026-09-26 (integration build, real 51k edit turn, one server, 8 seeds ABAB, LLAMA_SPEC_NMAX 4 vs 7): 7 drafts won on all 8
# paired seeds, 27.0 -> 31.8 tok/s (+18%), 3.24 -> 4.07 tokens/round, 120 -> 128 ms/round (bench-out/lever1-ab.txt). So the cap
# now starts above the 64k config (the 128k q4_0 config keeps cap 4 until it is re-measured there).
export LLAMA_SPEC_DEPTH_CAP=${LLAMA_SPEC_DEPTH_CAP-65537:4}
# block verification (Sun et al., ICLR 2025): lossless, never accepts fewer tokens than token-by-token verification.
# 2026-09-24, real 26k omp turn, one server, 6 seeds ABAB: 3.15 -> 3.31 tokens/round (+5%), 22.8 -> 23.6 tok/s.
export LLAMA_SPEC_BLOCK=${LLAMA_SPEC_BLOCK:-1}
# confidence-dependent q sharpening (lossless): +0.6-0.7% tokens/round, fitted on one capture and confirmed on another
export LLAMA_SPEC_Q_CONF=${LLAMA_SPEC_Q_CONF:-1}
[ "$SME" != 0 ] && export GGML_METAL_FA_SME=$SME GGML_METAL_FA_SME_MIN_KV=$SME_MIN_KV
# SME2 prefill matmuls (llama.cpp ggml-metal-mmsme.m, 2026-09-27): the CPU's SME units take ~30% of the tokens of every prefill
# ubatch's big IQ4_XS matmul while the GPU does the rest (adaptive per shape; decode/verify never use it). Same weights, no extra
# memory. Measured (scripts/prefill-bench.py, 4 appends of 2030 tokens each, medians): 51k Mac-only 79.5 -> 92.1 tok/s,
# 51k with the A19 105.3 -> 117.1, ~6k with the A19 111.9 (Mac-only) -> 179. Output: summation order differs from the GPU
# (full-model KL 2-6e-4 vs off, the size of q8_0-vs-f16 KV). MM_SME=0 turns it off.
# MM_SME_ROWS=1 (default): split each matmul by rows (the CPU dequantizes only its own rows) instead of by tokens (the CPU
# dequantizes the whole matrix per matmul): Mac-only pp2048 ub256 121.6 (off) -> 148.2 (tokens) -> 157.1 (rows), 2026-09-27.
MM_SME=${MM_SME:-0.30}
MM_SME_ROWS=${MM_SME_ROWS:-1}
[ "$MM_SME" != 0 ] && export GGML_METAL_MM_SME=$MM_SME GGML_METAL_MM_SME_ROWS=$MM_SME_ROWS

# Saved prompt prefixes: one directory per KV type (a q8_0 state can't load into a q4_0 cache).
# CACHE_DIR override: a saved state includes the drafter's own KV (.dft), so A/B a different drafter in its own directory
# (fastbench --cache must name the same directory; before 2026-09-24 the server always saved here and overwrote the baseline).
CACHE_DIR=${CACHE_DIR:-$PWD/cache/$KV}
mkdir -p "$CACHE_DIR"
# Host memory (2026-09-26): llama-server defaults keep an 8 GB RAM prompt cache (--cache-ram 8192) and up to 32 recurrent-state
# checkpoints per slot (~150-190 MB each for this hybrid model): a live omp session reached a 21.1 GB server footprint with
# ~2.8 GB of idle host allocations swapped out. The proxy already caches prompts on the SSD, so the RAM cache is off
# (CACHE_RAM to override) and checkpoints are capped (CTX_CHECKPOINTS, 3 since 2026-10-01).
# 2026-10-01: the 64k phone config still swapped 1.7-2.4 GB. The token-embedding table now stays in the mapped model file
# (LLAMA_LAZY_EMBD: clean pages, never swapped, one row read per token) and 3 checkpoints are kept (agent turns append to the
# prompt; a divergence is near the end, where the newest checkpoints are). With the llama.cpp fixes (scheduler buffer mapped
# on demand, freed heap returned to macOS): server footprint after a read 19.7 -> 18.7 GB.
export LLAMA_LAZY_EMBD=${LLAMA_LAZY_EMBD:-1}
SPORT=$PORT
[ "${PROXY:-1}" != 0 ] && SPORT=$((PORT + 100))

# the context this server can hold, Mac + phone (a client can read it to size its context window)
TOTAL=$CTX; [ -n "${PHONE_KV:-}" ] && TOTAL=${CTX_TOTAL:-262144}
mkdir -p cache && echo "$TOTAL" > cache/server-context

# the the Backburner app screen follows the server: starting (the Mac loads the model), ready (it answers; the phone wires its layers into
# GPU memory now), stopped. The proxy reports each request (reading / thinking / writing / done).
phone_note() { [ -n "${PHONE_IP:-}" ] && printf 'mac %s\n' "$*" | nc -G 1 -w 2 "$PHONE_IP" 50061 >/dev/null 2>&1; true; }
phone_note starting

"$B/llama-server" -m "$MODEL" -ngl 999 -fa on -c "$CTX" -np 1 -ctk "$KV" -ctv "$KV" -t 2 -tb 2 \
  --spec-type "${SPEC_TYPE:-ngram-simple,draft-dflash}" ${DRAFT:+-md "$DRAFT" -ngld 999 --spec-draft-n-max 7} ${REPLAY[@]+"${REPLAY[@]}"} \
  --slot-save-path "$CACHE_DIR/" --jinja --host 127.0.0.1 --port "$SPORT" \
  --cache-ram "${CACHE_RAM:-0}" --ctx-checkpoints "${CTX_CHECKPOINTS:-3}" \
  ${SD_ARGS[@]+"${SD_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} ${SERVER_ARGS:-} "$@" &
SRV=$!
PRX=
if [ "${PROXY:-1}" != 0 ]; then
  python3 scripts/proxy.py --listen "$PORT" --upstream "$SPORT" --cache "$CACHE_DIR" ${PHONE_IP:+--phone "$PHONE_IP"} &
  PRX=$!
fi
if [ -n "${PHONE_IP:-}" ]; then
  ( for _ in $(seq 1 600); do
      [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SPORT/health")" = 200 ] && { phone_note ready; exit 0; }
      kill -0 $SRV 2>/dev/null || exit 0
      sleep 1
    done ) &
fi
trap 'kill $SRV $PRX 2>/dev/null; wait $SRV 2>/dev/null; phone_note stopped; exit 0' INT TERM
wait $SRV
kill $PRX 2>/dev/null
phone_note stopped

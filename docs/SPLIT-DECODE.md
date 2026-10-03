# Split decode: an 8 GB Mac + the iPhone (2026-10-03)

Status: **built and measured** on a MacBook Neo (A18 Pro, 8 GB) with an iPhone Air (A19 Pro, 12 GB), on the stock v0.0.3
app. Not the default: `SPLIT_DECODE_L` turns it on.

## When to use it

The normal setup needs a Mac that holds the whole 27B (24 GB tested). An 8 GB Mac can't: even the 9.1 GB IQ2_XS file
doesn't fit. Split decode gives the Mac only the first L layers and the phone the rest, **for every token**, prefill and
decode:

- The Mac maps and runs layers [0, L) on its GPU, with their KV and recurrent state only.
- The phone's tail runs layers [L, 64) and the output head, holds their KV / state, and returns the last row's logits.
- Per token the link carries the residual entering layer L out (20 KB) and 248,320 logits back (1 MB).

Use it only when the Mac can't hold the model. On a Mac that can, the normal setup is faster (speculation, the drafter,
the Mac computing all of decode).

## Run

```bash
SPLIT_DECODE_L=20 CTX=16384 MODEL=~/Models/Qwen3.8-27B-IQ2_XS.gguf scripts/serve.sh
```

OpenAI-compatible on :8080 as usual. The mode finds the phone on the cable and refuses to start without its tail; it sets
mmap, the `-ot` overrides, `--no-repack`, no drafter or speculation, no checkpoints, no proxy, phone KV and ANE off, SME off,
and `LLAMA_SPLIT_GPU_WARM_US=1000` (see below). `LLAMA_SPLIT_VERBOSE=1` logs the Mac / phone / link time per token.

## The phone's tail, with the head

Split decode needs the tail **with** the output head (the phone computes the logits). The normal setup's tail is cut with
`--no-head`, and `phone-tail.sh L40` resolves to that file, so cut this one without the flag and pass its path:

```bash
# the IQ2_XS file from the same Hugging Face repo install.sh uses (bartowski/Qwen3.8-27B-GGUF)
python3 scripts/split-gguf.py ~/Models/Qwen3.8-27B-IQ2_XS.gguf ~/Models/tail-iq2xs-L20-head.gguf -L 20   # 6.5 GB
TAIL_WAIT=120 scripts/phone-tail.sh ~/Models/tail-iq2xs-L20-head.gguf                                     # ~3 min over USB 2
```

A head-less tail fails serve.sh's startup warmup: `worker returned 0 logits ... (a head-less tail?)`.

## Memory (IQ2_XS, L=20, 16k context, q8_0 KV)

| | |
|---|---|
| Mac GPU-mapped weights | layers 0-19: 2.24 GiB |
| Mac KV + recurrent state | 170 + 47 MiB (5 attention + 15 GDN layers, vs 544 + 150 for all 64) |
| Mac wired after load | +2.54 to +2.57 GiB; server footprint ~410-460 MB; no swap growth |
| Phone | 8,510-8,820 MiB system wired with the Mac connected; the app's gate is 9,100 |

L=20 is the lowest L that fits the phone at IQ2_XS: L=16 projects over 9,100. **Headroom on the phone is ~300 MiB**, and
its system-wide wired reading is noisy (single readings up to 9,700 with nothing jetsammed), so don't go lower.

## Measured (MacBook Neo A18 Pro 8 GB + iPhone Air A19 Pro 12 GB, USB 2 cable, iOS 27.2 beta, app v0.0.3)

Every run starts with the phone at thermal ≤ 1 **and** the Mac at thermal 0, at least 120 s after the last request. Both
devices are fanless, and either one warm slows the run by a third or more.

| | tok/s |
|---|---|
| decode, short prompt (128 tokens) | 3.97 / 4.02 / 4.00 |
| decode after a 4k prompt (prefill, cool down, append + 128) | 3.73 / 3.85 / 3.88 |
| prefill 2k, cold | 62.2 / 63.2 / 62.2 |

- **Per token:** Mac ~87 ms (layers 0-19) + phone ~135 ms (layers 20-63 + head) + link ~28 ms.
- **Prefill is Mac-bound:** ~3.7 s per 256-token ubatch on the Mac, ~2.4 s on the phone (pipelined).
- **Same answers:** against the same model run on the Mac alone, teacher-forced over 64 tokens on three prompts (19, 1,059
  and 5,275 tokens): top-1 agreement 93.8% / 100% / 100%, mean KL ≤ 0.0074. The four misses are near-ties in the
  reference. Reproduce with `tools/neo-air/` (`sdref` + `cmp.py`).

### GPU warm

While the Mac waits ~160 ms per token for the phone, the A18 Pro GPU clocks down, and the next head pass runs at ~1.1 GHz
instead of 1.47. `LLAMA_SPLIT_GPU_WARM_US=1000` commits a trivial Metal dispatch every 1 ms, only during that wait: the
Mac's share drops from 98.7-101.0 to 88.0-88.1 ms per token (p95 139-141 → 95-96), for +0.5 W. Every 5 ms is worse than
none (127 ms: the governor reads sparse wakes as low load).

## Limits

- **One sequence**, no speculation, no drafter: the phone holds the only copy of layers ≥ L and can't rewind its recurrent
  state.
- **Any rewind is a full re-prefill.** A prompt that diverges from the cached one resets both sides and prefills from 0.
  Appending to the conversation (the usual chat case) doesn't.
- **Heat.** A 4k prefill takes the phone to thermal 1-2 and the Mac to 1; long sessions run slower than the table.
- **Phone memory headroom ~300 MiB** at L=20 (above).
- A phone error fails the request (no fallback to the Mac, which doesn't hold layers ≥ L); the next request from position
  0 reconnects.

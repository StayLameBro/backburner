# Linux

**Status: not supported yet.** Nothing on this page has been run by the maintainer. It records what the code already
allows, what is missing, and how to help.

## What should already work

- **The engine builds on Linux.** `llama.cpp/` is a llama.cpp fork; the CPU, CUDA and Vulkan backends are upstream's.
  The Apple-only parts (Metal kernels, SME2, the phone app) are behind `APPLE` / `__APPLE__` checks.
- **A Linux machine as a tail worker.** The split-prefill tail server (`llama.cpp/tools/split-prefill/tail-server.h`) is
  plain POSIX sockets, and `llama-split-prefill --serve-tail PORT --tail-model tail.gguf` runs it as a command-line
  program. A Mac running `scripts/serve.sh` with `LLAMA_SPLIT_TAIL=<ip>:<port>` streams layers L..64 of every prefill
  batch to it, exactly as it does to a phone.

```bash
git clone --recursive https://github.com/StayLameBro/backburner && cd backburner
scripts/build-worker.sh linux-cuda      # or linux-vulkan, linux-cpu
python3 scripts/split-gguf.py Qwen3.8-27B-IQ4_XS.gguf tail-L40.gguf -L 40 --no-head
build-worker/bin/llama-split-prefill --serve-tail 50060 --tail-model tail-L40.gguf
```

## Why it isn't on by default: the link has no authentication yet

The tail protocol has no authentication. By default the server accepts only loopback. `SPT_ALLOW_REMOTE=1` lets anyone who
can reach the port use it: they can read the state it holds (your prompt's context) and make it load files. Use it only
on a direct cable between two machines or an isolated network, never on a shared LAN or Wi-Fi.

The phone app solves this with a cable-only rule and, for Wi-Fi, a Noise NNpsk0 tunnel paired over the cable
(docs/WIFI.md). The worker needs the same before Linux support ships (docs/ROADMAP.md, "Authenticated links").

## What's missing

| Piece | Status |
|---|---|
| Worker build script for Linux (`scripts/build-worker.sh`) | Written, untested on Linux |
| Authenticated worker link (PSK/Noise) | Not started; blocks a release |
| `serve.sh` on a Linux host (the Mac's role) | Not started: it uses macOS tools (sysctl, ioreg, devicectl) |
| Discovery (`phone-up.sh` finds only USB phones) | Not started |
| A device profile on Linux (`bench/device-probe.py`) | Host part works; no phones |
| Measured speed and same-answers gate on a Linux worker | Not done |

## How to help

Build the worker on your machine, run the same-answers check against the Mac alone (CONTRIBUTING, "Same answers") over a
direct cable, and post the result with your `bench/device-probe.py` output. Open an issue before starting on the
authenticated link so the design can be agreed first.

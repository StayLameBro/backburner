# Android port: design

This is the Android twin of the iPhone Sidecar (`ios/Backburner`). The Mac
server, the bench scripts, and the wire protocols do not change; only the
phone end is re-implemented. Status: **v1 scaffold, CPU path** — wire-compatible,
accuracy-exact, GPU/NNAPI stubs isolated to two files.

## Why this shape

- `phone-attn/phone-attn.h` is already platform-agnostic (POSIX sockets,
  `aligned_alloc` pages when `TARGET_OS_IPHONE` is unset). It is included
  **verbatim** — no fork. The `pa::engine` / `pa::page_engine` interfaces are
  the seam: iOS plugs in Metal + CoreML, Android plugs in Vulkan + NNAPI.
- `scripts/sme/sme_attn.c` is ARM code with a runtime `sme2_available()` gate.
  It builds under the NDK unchanged; non-SME SoCs take the same plain path as
  an iPhone with SME2 off.
- The split-prefill tail (`llama.cpp/tools/split-prefill/tail-server.h`) and
  `ggml-rpc` come from the same llama.cpp fork. Only the backend flag changes
  (`GGML_METAL` -> `GGML_VULKAN` or `GGML_OPENCL`).
- Discovery differs: iOS hunts the USB link-local IP by broadcast ping
  (`phone-up.sh`); Android uses `adb reverse` (`phone-up-android.sh`), which
  forwards Mac-localhost to phone-localhost over the one cable. No Wi-Fi, no
  broadcast, no per-OEM tethering subnet handling.
- Push is the same protocol: `scripts/phone-push.py PHONE_IP FILE REMOTE_NAME`
  serves the file over HTTP and issues `fetch <url> <dst>` on :50061. The
  Android `:50061` implements `mem` + `fetch` + `mac PHASE` with identical
  JSON/lines, so the script is reused untouched (with `127.0.0.1` after
  `adb reverse`).

## Files

- `android/` — Gradle app (`app.backburner`), Compose UI, `NsdManager`
  advertiser (same `_infernet-rpc._tcp` + TXT), JNI bridge, stub engines.
- `android/app/src/main/cpp/rpcbridge.cc` — the `RPCBridge.h` equivalent.
- `android/app/src/main/cpp/vulkan_engine.{h,cc}` — `pa-metal.mm` equivalent
  (v1 stub: `begin()` false -> CPU/SME for all keys).
- `android/app/src/main/cpp/nnapi_page_engine.{h,cc}` — `pa-ane.mm` equivalent
  (v1 stub: `ready()` 0 -> no ANE pages).
- `scripts/build-android.sh` — `build-iphone.sh` equivalent (Gradle + adb).
- `scripts/phone-up-android.sh` — `phone-up.sh` equivalent (adb + HELLO probe,
  same output line).
- `scripts/phone-tail-android.sh` — `phone-tail.sh` equivalent (adb push).
- `scripts/phone-relaunch-android.sh` — `phone-relaunch.sh` equivalent
  (force-stop + monkey launch).

## What deliberately differs from iOS

1. No dim-screen/OLED animation (`ContentView` LayerStack/Canvas). The Compose
   UI keeps headline/subline/stats/counters; the 64-layer bar is a progress row.
2. No brightness ramp / `isIdleTimerDisabled`: `keepScreenOn` + foreground service.
3. No Bonjour `NetService`: `NsdManager` with the same type/TXT.
4. No `os_proc_available_memory`/jetsam wired ceiling: `sysinfo` + `/proc`,
   conservative caps in `serve.sh` (see below).
5. No ANE page template push (`phone-ane.sh` + `coremltools` + `.mlmodelc`):
   the stub reports `pa_ane:false` in `mem`, so `serve.sh` prints "ANE pages
   off" and runs GPU/CPU only. Pushing an NNAPI blob is future work in the same
   `fetch` channel.

## serve.sh with Android (no fork needed)

```bash
scripts/phone-up-android.sh                      # sets adb reverse, prints IP=127.0.0.1 ...
LLAMA_SPLIT_TAIL=127.0.0.1:50060 PHONE_KV=127.0.0.1:50062 scripts/serve.sh
```

## Windows host <> Android (this machine)

`serve.sh` does not run on Windows (bash, `nc -G`, `sysctl iogpu`,
`GGML_METAL_*`, `-dev MTL0`). Use the PowerShell host instead; the phone app
and the wire protocols are unchanged.

```powershell
# 0. one-time: host engine (CUDA or Vulkan) + adb + JDK 17 + Python 3
git submodule update --init --recursive
cmake -S llama.cpp -B llama.cpp/build-windows -DGGML_CUDA=ON   # or -DGGML_VULKAN=ON
cmake --build llama.cpp/build-windows --config Release --target llama-server -j
#    needs CUDA toolkit (CUDA path) or Vulkan SDK (Vulkan path)
adb devices   # one `device` line, phone unlocked, USB debugging on

# 1. app (same as Mac flow, Windows wrapper)
powershell -ExecutionPolicy Bypass -File scripts\build-android.ps1

# 2. cable + probes (pure Python, no nc, no /dev/tcp)
python scripts\phone_up_android.py
#    prints: 127.0.0.1 TAIL_UP VER AVAIL WIRED NAME  (or .\scripts\phone-up-android.ps1)

# 3. tail (same split file as iOS/macOS)
python scripts\split-gguf.py %USERPROFILE%\Models\Qwen3.8-27B-IQ4_XS.gguf %USERPROFILE%\Models\tail-iq4xs-L40-nohead.gguf -L 40
python scripts\phone_tail_android.py L40
#    adb push -> /sdcard/Download/tail.gguf, then Import in the app

# 4. serve (Windows launcher; auto-detects the phone like serve.sh)
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
#    Windows alone:  scripts\serve.ps1 -NoPhone
#    explicit:       $env:LLAMA_SPLIT_TAIL='127.0.0.1:50060'; $env:PHONE_KV='127.0.0.1:50062'; scripts\serve.ps1
#    client: base URL http://127.0.0.1:8080/v1
```

Notes:

- `serve.ps1` keeps the portable tunables (`LLAMA_SPEC_*`, `SPEC_DRAFT_UBATCH`,
  `LLAMA_KV_REMOTE`, `LLAMA_REMOTE_PIPE`, prompt cache via `proxy.py`) and
  drops the Apple-only ones (`iogpu` gate, `-lm none` default, `GGML_METAL_*`,
  M4-SME matmul/co-attention exports). No `-dev MTL0` default: with
  `PHONE_DRAFT` unset the drafter stays on the host GPU.
- `proxy.py --phone` works on Windows as-is (raw socket, no `nc`).
- `phone-push.py` behind `adb reverse` serves on a random port the phone's
  `fetch http://127.0.0.1:PORT` cannot reach; use
  `python scripts\phone_push_android.py 127.0.0.1 <file>` (adb push, or
  `--via fetch` which reverse-forwards the HTTP port first).

`serve.sh` auto-detects iOS USB only when `LLAMA_SPLIT_TAIL`/`PHONE_KV` are
unset; explicit env wins, so no script change is required. The iOS-only bits it
touches (`iogpu.wired_limit_mb`, `GGML_METAL_*`, `PHONE_WIRED_MAX_MB=9400`,
`PHONE_APP_RESERVE_MB=512`) stay Mac-side and remain valid: keep the 512 MiB
app reserve, and treat the Android `sys_wired_mb` (= MemAvailable) as the
wired number — capacity math then under-claims rather than over-claims.

## Bring-up order

1. `scripts/build-android.sh`, open the app, keep it foreground.
2. `scripts/phone-up-android.sh` -> `127.0.0.1 0 <ver> <avail> <wired> ...`
   (TAIL 0 until the tail links; attention HELLO is the gate).
3. `pa-tool test` equivalent against `127.0.0.1:50062` from a Mac loopback
   build to gate accuracy (`max|O-ref|/max|ref| < 5e-3`, same as `pa-tool.cpp`),
   then `PA_BIG_TOK=256` for the ATTN_BIG path.
4. Push the tail (`phone-tail-android.sh L40`), `LLAMA_SPLIT_TAIL=... serve.sh`,
   reproduce `bench/turn-bench.py --config phone`.
5. Port `vulkan_engine.cc` (MSL -> GLSL), retune `L`/`PA_BIG_CFG`/`PA_GPU_MIN_KEYS`
   per SoC, then implement `nnapi_page_engine.cc` and re-run `bench/long-bench.py`.

## Limits (carry over from README, plus Android deltas)

- Small reads stay on the Mac (< ~512 tokens); one request at a time; 60 s
  phone failure -> Mac-only retry: all Mac-side, unchanged.
- Past 64k the phone does old-key attention only (no split-prefill + KV
  simultaneously): unchanged; second-phone sharing (`docs/TWO-PHONES.md` A)
  works with one Android + one iPhone since `LLAMA_KV_REMOTE` takes a list.
- v1 Android adds: no GPU matrix units yet (expect iOS-with-them-off speed for
  the phone's layers), no ANE pages (expect the 279 ms/token end of the 140k
  range, not 176), tail needs the llama.cpp Android link before it loads.

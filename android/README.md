# Backburner for Android (Sidecar port)

Android port of `ios/Backburner/Sidecar`. It speaks the same wire protocols as
the iPhone app, so the Mac server (`scripts/serve.sh`) needs no changes besides
pointing at the Android device's address (or `adb reverse` + `127.0.0.1`).

## What maps to what

| iPhone | Android (this directory) | Notes |
|---|---|---|
| `ContentView.swift` (SwiftUI) | `app/.../MainActivity.kt` (Compose) | Same headline/subline/stats logic, simplified. No dim-screen burn-in animation. |
| `RPCBridge.h/.mm` | `app/src/main/cpp/rpcbridge.cc` + `SidecarRpc.kt` | Same method surface (`startHost`, `startTail`, `startPhoneAttn`, `cableAddress`, `metalStats` -> `gpuStats`, `memoryStats`, `linkStats`, `tailStatus`, `phoneAttnStatus`, `macStatus`, `envNote`, `deviceModel`). |
| `ggml RPC server` on :50052 | same `ggml-rpc` from `llama.cpp`, built for Android NDK | `LLAMA_DIR` tree must have `GGML_RPC=ON`. `-dev MTL0` on the Mac stays; the device selector names the remote backend `RPC0`. |
| split-prefill tail on :50060 (`tail-server.h`) | same `tail-server.h` compiled into `librpcbridge.so` | Needs a tail GGUF in the app's files dir (`tail.gguf`), same `phone-tail.sh` flow but over `adb push` / `phone-push.py`. Vulkan or OpenCL backend, not Metal. |
| `phone-attn/phone-attn.h` server on :50062 | **unchanged** — included verbatim | POSIX sockets + `sme_attn.c`. Compiles on Bionic as-is. |
| `pa-metal.mm` (Metal matrix-unit attention) | `cpp/vulkan_engine.cc` (`vulkan_engine : pa::engine`) | v1 is a compile-safe stub: reports no GPU pages (`begin()` returns false) so every key runs on the CPU/SME path. Port the MSL kernels (`pa_attn_q8`, `pa_attn_na*`, `pa_attn_big`) to Vulkan compute or OpenCL to enable it. The `config_req.gpu_permille` gate already handles "no GPU". |
| `pa-ane.mm` (CoreML 16k-key pages) | `cpp/nnapi_page_engine.cc` (`nnapi_page_engine : pa::page_engine`) | v1 returns `ready()=0` (no ANE pages), so accuracy is exact and speed is CPU/GPU only. Implement with NNAPI/QNN delegate later; the `page_engine` interface does not change. |
| `scripts/sme/sme_attn.c` (Apple M4/A18 SME2) | same file, built by CMake here | Guarded by `sme2_available()` at runtime. Snapdragon 8 Elite / Dimensity with SME2 use it; older SoCs fall back to the plain path (same as iOS when `sme2_available()==0`). |
| `NetService _infernet-rpc._tcp` (Bonjour) | `NsdAdvertiser.kt` (`NsdManager`, same service type) | Same TXT keys (`model`, `chip`, `mem`, `addr`, `port`). |
| `isIdleTimerDisabled`, brightness dim | `FLAG_KEEP_SCREEN_ON` | No brightness ramp; OLED burn-in drift omitted. |
| `os_proc_available_memory`, `task_vm_info`, `if_data`, `utsname` | `ActivityManager`, `TrafficStats`, `Build.*`, `getifaddrs` | `getifaddrs` works on Bionic; the 169.254 filter is kept for USB tethering, plus `adb reverse` which needs no IP at all. |

## Requirements

- Android 10+ (API 29), arm64-v8a device (Snapdragon 8 Gen 2 / 8 Elite, Tensor G3+, Dimensity 9200+ recommended).
- A 10 Gb/s USB-C data cable (same as iOS: the charge-only cable in the box is too slow).
- ll turnover: this app is a *compute node*, not a standalone LLM app. The host runs `scripts/serve.ps1` (Windows) or `scripts/serve.sh` (Mac); the phone only serves layers / attention.
- Host: Android Studio or `ANDROID_SDK_ROOT` + NDK r26+, CMake 3.22+, Python 3, adb. Windows host additionally needs a `llama-server.exe` build (CUDA or Vulkan, see `docs/ANDROID.md`).

## Quick start

### Windows host (PowerShell)

```powershell
# 1. build + install (phone wired, unlocked, USB debugging on)
powershell -ExecutionPolicy Bypass -File scripts\build-android.ps1

# 2. cable + probes (no IP hunting, no Wi-Fi)
python scripts\phone_up_android.py
#    127.0.0.1 TAIL_UP VER AVAIL WIRED NAME

# 3. push the tail (same split file as iOS: scripts/split-gguf.py output)
python scripts\split-gguf.py %USERPROFILE%\Models\Qwen3.8-27B-IQ4_XS.gguf %USERPROFILE%\Models\tail-iq4xs-L40-nohead.gguf -L 40
python scripts\phone_tail_android.py L40
#    adb push -> /sdcard/Download/tail.gguf, then Import in the app

# 4. serve (llama-server.exe + proxy.py; auto-detects the phone)
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
#    Windows alone: scripts\serve.ps1 -NoPhone
```

Host engine: `cmake -S llama.cpp -B llama.cpp/build-windows -DGGML_CUDA=ON`
(or `-DGGML_VULKAN=ON`) then build `llama-server`. Details in `docs/ANDROID.md`.

### Mac/Linux host (bash)

```bash
# 1. build + install (phone wired, unlocked, USB debugging on)
scripts/build-android.sh            # assembleDebug + adb install, needs ANDROID_SDK_ROOT/ANDROID_HOME

# 2. forward the three ports over the one cable (no IP hunting, no Wi-Fi)
adb reverse tcp:50052 tcp:50052     # ggml RPC (optional drafter path)
adb reverse tcp:50060 tcp:50060     # split-prefill tail
adb reverse tcp:50061 tcp:50061     # ANE/command port (mem, fetch, mac PHASE reports)
adb reverse tcp:50062 tcp:50062     # phone-attn

# 3. push the tail (same split file as iOS: scripts/split-gguf.py output)
python3 scripts/split-gguf.py ~/Models/Qwen3.8-27B-IQ4_XS.gguf ~/Models/tail-iq4xs-L40-nohead.gguf -L 40
adb push ~/Models/tail-iq4xs-L40-nohead.gguf /sdcard/Download/tail.gguf
# ... then in the app: ⋮ -> "Import tail.gguf", or:
python3 scripts/phone-push.py 127.0.0.1 ~/Models/tail-iq4xs-L40-nohead.gguf tail.gguf

# 4. run the Mac server against localhost (serve.sh auto-detects iOS USB; override for Android)
LLAMA_SPLIT_TAIL=127.0.0.1:50060 PHONE_KV=127.0.0.1:50062 scripts/serve.sh
# Mac alone for comparison:
PHONE=0 scripts/serve.sh
```

`scripts/phone-up-android.sh` does step 2 + the `HELLO`/`STATS` checks and prints
the same `IP TAIL_UP VER AVAIL WIRED NAME` line as `phone-up.sh`, so scripts that
wrap it keep working.

## Performance expectations (not yet measured)

The iOS numbers (README: +29-44% prefill at 16-48k, 128k at 12.6 tok/s) were
measured with A18 Pro/A19 Pro matrix units + SME2 + ANE pages. On Android v1:

- Split prefill works (tail layers run on Vulkan/OpenCL + CPU), but without the
  matrix-unit kernels it will be closer to "same phone with them off" (README:
  2.4x slower than with them on for the phone's layers). Tune `L` (first phone
  layer) per SoC: start with the A18 values (L=52) and move down while the tail
  fits in memory.
- Phone-held attention past 64k works on the CPU/SME path (`pa-tool test`
  passes), at CPU speed. Port `vulkan_engine.cc` for GPU speed, then add an
  NNAPI page engine for the oldest pages (mirrors `pa-ane.mm`).
- The `PA_GPU_*` / `PA_ANE_SHARE*` env knobs in `phone-attn/phone-attn.h` are
  honored verbatim; `Documents/env.txt` (`phone-env.sh` flow) works the same.

## Porting checklist (what is stubbed)

- [x] Wire protocol compatible (`PATN` framing, tail HELLO, `fetch`/`mem` on :50061, NSD TXT).
- [x] CPU attention path (`sme_attn.c` + `phone-attn.h` server) — full accuracy.
- [ ] `vulkan_engine.cc`: port `pa-metal.mm` MSL -> Vulkan compute (or OpenCL). Interface is fixed; `serve.sh` needs no change when it lands.
- [ ] `nnapi_page_engine.cc`: 16k-key page models via NNAPI/QNN. Interface is fixed (`page_engine`); `phone-ane.sh` flow ports by pushing an NNAPI blob instead of `.mlmodelc`.
- [ ] SoC tuning: `L` split, `PA_BIG_CFG` tile sizes, `PA_GPU_MIN_KEYS`, wired-limit equivalents (`ActivityManager.getMemoryInfo`, `VK_EXT_memory_budget`).

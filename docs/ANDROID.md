# Android

**Status: not supported yet.** Nothing on this page has been run on an Android device. It records the plan, what carries
over from the iPhone work, and what doesn't.

## The plan

An Android phone joins as a **worker**, like a Linux machine (LINUX.md), not as an app port of the iPhone's Backburner:

1. **A native worker built with the NDK.** The split-prefill tail server is plain POSIX and builds as a command-line
   program. llama.cpp has Vulkan and OpenCL (Adreno) backends for Android GPUs, and the CPU backend with KleidiAI for
   recent Arm cores. `scripts/build-worker.sh android-vulkan` cross-compiles it (untested).
2. **The link: USB tethering.** An Android phone with USB tethering on puts a network interface on the computer side of
   the cable, so the Mac can reach the worker's port without Wi-Fi. Round trip and bandwidth on real phones are unmeasured.
3. **Then a small app** that runs the worker in a foreground service (so Android doesn't stop it), shows its state and
   holds the pairing key, once the authenticated link exists (ROADMAP.md, "Authenticated links").

```bash
# the worker (untested): needs ANDROID_NDK pointing at an NDK r27 or later
ANDROID_NDK=~/Library/Android/sdk/ndk/<version> scripts/build-worker.sh android-vulkan
adb push build-worker-android/bin/llama-split-prefill /data/local/tmp/
adb push tail-L52.gguf /data/local/tmp/
adb shell /data/local/tmp/llama-split-prefill --serve-tail 50060 --tail-model /data/local/tmp/tail-L52.gguf
```

A worker started from `adb shell` listens on loopback only. Reaching it from the Mac without opening it to a network needs
`adb forward tcp:50060 tcp:50060`, which goes through adb over the cable, or the authenticated link once it exists.

## What carries over and what doesn't

| iPhone piece | Android |
|---|---|
| Split prefill tail (layers L..64 on the phone GPU) | Same protocol; needs a fast Vulkan or OpenCL path for the 27B's layers on Adreno / Mali / Xclipse |
| Phone-held context (`phone-attn/`) | Its kernels are Metal and Apple SME: needs a Vulkan port |
| Neural Engine pages | Apple only. An NPU path (QNN, NNAPI) would be new work |
| Cable-only rule, paired Wi-Fi tunnel | To be built for the worker (ROADMAP.md) |
| AltStore install, memory entitlement | Not needed: Android apps get much more memory, but a foreground service is required |

## How to help

If you have a recent Snapdragon, Dimensity, Tensor or Exynos phone, the most useful first numbers are its GPU's prompt
speed with each backend: build with `android-vulkan` and `android-opencl`, then run
`llama-bench -m <Qwen3.5-0.8B or another small Qwen3.5/3.8 GGUF> -p 512 -n 128` on the phone (adb shell) for each, and
post both tables with the phone model and its thermal behaviour over 5 minutes. Open an issue first if you want to take on
the Vulkan port of `phone-attn`.

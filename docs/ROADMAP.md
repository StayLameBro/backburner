# Roadmap

**Backburner turns the devices you already own into one inference machine.** The goal is to run models too big for any
one of them, faster than any one of them could alone, with the same answers you'd get from a single machine big enough to
hold the whole model.

Today that means a Mac plus iPhones and iPads over USB-C, with Windows PCs in development. The aim is every device in a
home: spare Macs, iPads, Linux and Windows PCs, Android phones, an Apple TV.

## Why not just llama.cpp `-rpc`?

`-rpc` is a good tool with a different shape. One host runs the whole model and the other machines are remote compute
backends, so every hand-off is a blocking network round trip, and phones can't take part. Backburner is built differently:

- **Each device owns its layers and its own memory.** Only a few KB of hidden state cross the link per layer block, not
  weights or graphs.
- **Phones and iPads are real workers.** They run layers on their GPUs and hold old context with their own GPU and Neural
  Engine (README, "Who does what").
- **Lossless by default.** Every speed change is checked against the Mac alone for identical greedy output
  (CONTRIBUTING, "Same answers"). A change that alters the output distribution is opt-in and labelled.

A same-house, same-model comparison against `-rpc`, set up as well as it can be, will go in this file when it is measured
for each release. Until then this section describes the design, not a result.

## Now: 0.0.5

- Security hardening of the phone app. Details come with the release; everyone should update the app then.
- #18: one `LLAMA_SPLIT_TIMEOUT_S` for split prefill (20 s, then the batch reruns on the Mac) and split decode (120 s).
- #23 and #1 setup fixes: iPads named as iPads, the app budget capped by the device's RAM, the installer never lowering the
  GPU memory limit, SME defaults that follow the chip, a phone with no tail model reported as down, free-team Xcode builds.
- `bench/device-probe.py`: a device profile (no personal data) for results issues.

## In development: Qwen3.8-Flash-Next across a house

Qwen3.8-Flash-Next is a ~180B-parameter mixture-of-experts model: no Mac, PC or phone here can hold it alone. It runs end to
end in development builds across a Mac, two Windows PCs and two iPhones. Each device owns whole layers with their experts,
keeps the hot experts in memory and reads the rest from its own SSD. The code and measured numbers come to this repository
with its release.

## In development: the 27B on two phones

Mac → iPhone A → iPhone B prefill chain (docs/TWO-PHONES.md), measured on two A19 Pros: speed at 16k-128k, same-answers
gate, thermal behaviour over long runs, and what each device was doing. Then the decode levers.

## Then: any device

1. **One worker for every device.** The phone's tail protocol (`llama.cpp/tools/split-prefill/tail-server.h`) is plain
   POSIX. The same server already runs on a Mac (`llama-split-prefill --serve-tail`). Linux and Android builds are next
   ([LINUX.md](LINUX.md), [ANDROID.md](ANDROID.md)). A Mac or PC worker has no 6 GB app limit, and Macs and iPads can link
   over Thunderbolt Bridge.
2. **Authenticated links.** Over USB the phone app answers only the cable. A worker on a LAN needs the same guarantee before
   it ships: the Wi-Fi tunnel's Noise NNpsk0 handshake (docs/WIFI.md), ported to the worker. **No worker will listen
   unauthenticated on a network by default.**
3. **Discovery and automatic layout.** Find every worker on the cable, Thunderbolt, LAN or tunnel, probe it for 30 s
   (bandwidth, compute, memory, link round trip), then choose who runs which layers and which part of the context.
4. **Devices, in the order people asked for them:** M4/M5 iPad Pro (works today, needs profiles), extra Macs (MacBook Air,
   mini), Linux PCs with CUDA or Vulkan, older and USB 2 phones (jobs that need little bandwidth, such as holding old
   context or drafting), Android, Apple TV (tvOS runs the same Metal code; with no USB data port it joins over Ethernet,
   once the authenticated link exists).
5. **Community profiles.** `bench/device-probe.py` output attached to results issues, used by the planner and by the README
   table "Will it work on my …?".

## Upstream

Small, general llama.cpp pull requests with measurements, for anything that helps people who never use a phone: merged
expert copies, layered prefill for offloaded MoE, GPU keep-warm. The phone, multi-device and planner work stays here.

## What people asked for

| Ask | Where | Status |
|---|---|---|
| iPad Pro (M4/M5, Thunderbolt) | X, #1 | Works: M5 iPad Pro with an 8 GB MacBook Neo, 8 tok/s decode, 155 tok/s prefill at 2k (#1, user-reported). Profiles wanted |
| More phones at once | X | Two phones work (docs/TWO-PHONES.md). More phones need the planner |
| Smaller Macs (8-18 GB) | #1, #23 | Split decode for 8 GB (docs/SPLIT-DECODE.md); 18 GB with IQ2_XS (#23). README "Smaller Macs" |
| A20 Pro / iPhone 18 Pro | #14 | Works (#14, user-reported); long runs at 128k still being debugged there |
| No timeout when the app is suspended | #18 | Fixed in 0.0.5 |
| Android (Pixel, Galaxy) | X | Not yet: [ANDROID.md](ANDROID.md) |
| Linux / Windows PCs | X | Not yet for the 27B: [LINUX.md](LINUX.md) |
| Wireless / mesh | X | Paired Wi-Fi tunnel exists (docs/WIFI.md); the cable is faster |
| Keep working when the phone locks | X | iOS stops GPU work in the background; Guided Access keeps it in front (INSTALL-IPHONE.md) |
| Probabilities with speculative decoding | #12 | `launch-quality.py` reports coverage; server-side fix later |

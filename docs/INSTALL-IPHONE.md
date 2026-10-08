# Installing the iPhone (or iPad) app

Two ways. Both give the same app; the Mac scripts find it on the phone either way (`scripts/bundle-id.sh`).

| | AltStore + the release IPA | Xcode |
|---|---|---|
| Apple account | your free Apple ID | an Apple developer team id |
| Build anything for the phone | no | yes (`scripts/build-iphone.sh`, ~10 min) |
| Re-sign | every 7 days (AltStore does it) | every 7 days on a free team, yearly on a paid one |

You need an iPhone 15 Pro or newer, or an iPad with an M-series chip (M1 or newer). The A19 Pro (iPhone 17 Pro / Pro Max)
has the GPU matrix units that make the phone's half of prefill 2.4x faster; the A18 Pro works but helps less.

**iPads** install the same app (v0.0.2 and newer) and every step below is the same. Users have run an M5 iPad Pro 16 GB
(app budget ~16,260 MiB, #1) and an M1 iPad Pro 8 GB (#23). An 8 GB iPad reports an app budget above its RAM, so
`serve.sh` caps it at 74% of RAM (0.0.5). The app runs full screen on iPad (no Split View), because it has to stay in
front while the Mac uses it.

**Local Network permission.** The first time the Mac sends the phone a file (`scripts/phone-tail.sh`), iOS asks whether
Backburner may use the local network. Until you allow it, the push fails with "The Internet connection appears to be
offline". Allow it, then run the command again (#23).

## With AltStore (no developer account)

> **Confirmed by users:** AltStore keeps the memory entitlement when it signs the IPA (6,073 MiB app budget on an iPhone
> Air, #1; ~16,260 MiB on an M5 iPad Pro). The Mac scripts still call `xcrun devicectl`, so the Mac needs Xcode installed
> (not a developer account).

1. **AltServer on the Mac.** Download it from [altstore.io](https://altstore.io) and open it. It has no window, only a
   diamond-shaped icon in the menu bar. No icon? On a MacBook with a notch, a full menu bar hides icons behind it: quit a
   menu-bar app or two (or hold ⌘ and drag icons off the bar) until it appears. Also check System Settings → Menu Bar →
   Allow in the Menu Bar → AltServer.
2. **AltStore on the iPhone.** Plug the phone in, unlock it, then in the AltServer menu: *Install AltStore* → your iPhone.
   AltServer asks for your Apple ID: that is how free sideloading signs apps. Use **AltStore 2.2 or newer**.
3. **Developer Mode.** On the iPhone: Settings → Privacy & Security → Developer Mode → on, then restart when asked.
   If iOS says the developer isn't trusted: Settings → General → VPN & Device Management → your Apple ID → Trust.
4. **Backburner.** In AltStore: **Sources → +**, paste this source URL, then install Backburner from it. AltStore then tells you
   when there's a new version:

   ```
   https://raw.githubusercontent.com/StayLameBro/backburner/main/altstore/source.json
   ```

   Or download `Backburner.ipa` from the [releases page](https://github.com/StayLameBro/backburner/releases) onto the iPhone
   (or AirDrop it over), then in AltStore: My Apps → **+** → `Backburner.ipa`.
5. **Check the memory budget.** Open Backburner, keep it in front, plug the phone into a 10 Gb/s USB-C port, and on the Mac:

   ```bash
   scripts/phone-up.sh
   # phone-up: wired iPhone at 169.254.x.x: phone-attn v…, prefill tail down, … MiB system wired, 6xxx MiB app budget
   ```

   The app budget should be around 6,000 MiB on a 17 Pro Max. Around 3,000 means the app lost its increased-memory-limit
   entitlement on the way: the phone then has no room for its half of the model. That happens with **SideStore** (free
   accounts, [SideStore#1616](https://github.com/SideStore/SideStore/issues/1616)) and AltStore older than 2.2. Reinstall with
   AltStore 2.2+.

Then continue with step 4 of the README's Setup (the phone's half of the model).

**Every 7 days** a free Apple ID signature expires and the app stops opening. AltStore refreshes it in the background when the
phone and a Mac running AltServer are on the same Wi-Fi, or tap *Refresh All* in AltStore. Free accounts can have 3 sideloaded
apps at a time (AltStore counts as one).

## Keeping it in front

Backburner works only while it is open and in front: iOS doesn't let an iPhone app use its GPU in the background. The app
keeps the screen from locking on its own and dims itself, but the side button still locks it. Below 64k that costs nothing
but speed (the Mac takes the work back); past 64k the phone holds the oldest part of the conversation, and a phone that stops
answering for 15 s stops the server. Guided Access locks the phone into Backburner and can turn the side button off:

1. Settings → Accessibility → Guided Access: turn it on (and set a passcode).
2. Open Backburner and triple-click the side button.
3. Options → turn off **Side Button**, then Start.

Triple-click again (and enter the passcode) to leave.

## With Xcode (developer team id)

What a stock Mac needs first (the script checks and says so):
- **cmake**: `brew install cmake`
- **the Metal Toolchain** (~840 MB, once): `xcodebuild -downloadComponent MetalToolchain`

```bash
export DEVELOPMENT_TEAM=<your team id>       # Xcode > Settings > Accounts (a free personal team works)
UDID=<your iPhone's UDID> scripts/build-iphone.sh
```

It builds llama.cpp for iOS, the SME2 attention kernel and the app, signs it as `app.backburner.<team id>` and installs it.
A free team gets a provisioning profile only for devices Xcode has registered: with `UDID` set and the phone wired and
unlocked, the script registers it on the first run (#23). `JOBS=4` keeps the build from taking every core. iOS 26 and
iPadOS 26 work too (#23).

## Making the IPA (maintainers)

```bash
IPA=1 scripts/build-iphone.sh    # -> ios/build/Backburner.ipa
```

The IPA is unsigned except for an ad-hoc signature that carries the entitlements (`ios/Backburner/Sidecar/Sidecar.entitlements`),
so AltStore can request increased-memory-limit when it signs the app with the user's Apple ID. The script stops if the
entitlement is missing.

For a new release: bump `MARKETING_VERSION` (and `CURRENT_PROJECT_VERSION`) in the Xcode project, build, attach the IPA to the
GitHub release, then add a version to `altstore/source.json` with the same version and build numbers, the IPA's size in bytes
and its download URL. AltStore refuses an install whose version or permissions don't match the source.

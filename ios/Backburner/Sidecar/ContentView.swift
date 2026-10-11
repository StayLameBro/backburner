import SwiftUI
import UIKit

// "iPad" or "iPhone", for the text on screen (the hardware model name, e.g. "iPad16,3", not UIDevice: no main actor needed)
private let device: String = {
    var u = utsname()
    uname(&u)
    let model = withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self) }
    return model.hasPrefix("iPad") ? "iPad" : "iPhone"
}()

// One line per service the Mac can use, with what it is doing and when the Mac last used it.
private struct Service: Identifiable {
    let id: String        // short name
    let port: Int32
    var state: String     // one word: Starting, Ready, Working, Idle, Error, No model
    var detail: String    // one short line under it
    var lastUsed: Date?
}

struct ContentView: View {
    @State private var cable = ""
    @State private var machine = ""
    @State private var thermal = ProcessInfo.ThermalState.nominal
    @State private var memAvail: UInt64 = 0
    @State private var memFootprint: UInt64 = 0
    @State private var gpuAlloc: UInt64 = 0
    @State private var rxRate: Double = 0
    @State private var txRate: Double = 0
    @State private var envNote = ""

    @State private var rpcRunning = false
    @State private var rpcError = ""
    @State private var tailRunning = false
    @State private var tailError = ""
    @State private var tail = Service(id: "Prefill tail", port: 50060, state: "Starting", detail: "", lastUsed: nil)
    @State private var attn = Service(id: "Phone attention", port: 50062, state: "Starting", detail: "", lastUsed: nil)
    @State private var ane = Service(id: "Neural Engine", port: 50061, state: "Ready", detail: "", lastUsed: nil)
    @State private var rpc = Service(id: "GPU (ggml RPC)", port: 50052, state: "Starting", detail: "", lastUsed: nil)
    @State private var lastTailChunks: UInt64 = 0
    @State private var lastAttnCalls: UInt64 = 0
    @State private var tailTokS: Double = 0
    @State private var heldKeys: UInt64 = 0
    @State private var attnLastMs: Double = 0
    @State private var attnBusy = false
    @State private var aneOn = false
    @State private var showDetails = false
    @State private var tailFirst = 40
    @State private var appearAt = Date()
    @State private var brightnessRamp = 0
    @State private var loadProgress = 0.0
    @State private var modelBytes: UInt64 = 0
    @State private var macPhase = ""            // what the Mac reports: starting, ready, reading, thinking, writing, done, stopped
    @State private var macN1 = 0.0
    @State private var macN2 = 0.0
    @State private var macCtx = 0.0
    @State private var macPhaseAt = Date()      // when the phase last changed, on this phone's clock
    @State private var activations: UInt64 = 0
    @State private var activateAt: Date? = nil  // the Mac said it's up: this iPhone started wiring its layers into GPU memory
    @State private var activating = false
    @State private var tailTokens: UInt64 = 0
    @State private var readStartTokens: UInt64 = 0
    @State private var readStartChunks: UInt64 = 0
    @State private var writeRate = 0.0

    // dim mode: black screen, low brightness, a few dim lines that move a little (OLED burn-in, heat)
    @State private var dim = false
    // nothing to show for a while (no cable, waiting, or ready with the Mac idle): the animations stop, as with Reduce Motion.
    // A screen redrawn every frame while the app just waits warms thin phones (GitHub #1).
    @State private var still = false
    @State private var restSince: Date? = nil
    private let stillAfter: TimeInterval = 8
    @State private var lastTouch = Date()
    @State private var savedBrightness: CGFloat = -1
    @State private var drift = CGSize.zero


    private static var lastRx: UInt64 = 0
    private static var lastTx: UInt64 = 0
    private static var lastAt: Date? = nil

    private let rpcPort: Int32 = 50052
    private let tailPort: Int32 = 50060
    private let dimAfter: TimeInterval = 120

    // The split-prefill TAIL model: Documents/tail.gguf (scripts/phone-push.py IP FILE tail.gguf from the Mac).
    private var tailPath: String {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("tail.gguf").path
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            // the full screen stays alive under the dim view (no rebuild, no replayed draw-in); only opacity changes
            fullView
                .opacity(dim ? 0 : 1)
                .allowsHitTesting(!dim)
            dimView
                .opacity(dim ? 1 : 0)
                .allowsHitTesting(false)
        }
        .contentShape(Rectangle())
        .onTapGesture { toggleDim() }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            machine = SidecarRPC.deviceModel()
            envNote = SidecarRPC.envNote()
            startRPC()
            startTail()
            SidecarRPC.startANEBench(port: 50061)
            SidecarRPC.startPhoneAttn(port: 50062)
            WifiTunnel.startIfPaired()
            refresh()
            appearAt = Date()
        }
        .onReceive(Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()) { _ in
            refresh()
            let resting = [Mode.noCable, .waiting, .ready].contains(mode) && !attnBusy && macFlow.rate == 0
            if resting {
                let since = restSince ?? Date()
                restSince = since
                if !still && Date().timeIntervalSince(since) > stillAfter { still = true }
            } else {
                restSince = nil
                if still { still = false }
            }
            if !dim && Date().timeIntervalSince(lastTouch) > dimAfter { setDim(true) }
            if dim && Int(Date().timeIntervalSince1970) % 60 == 0 {
                withAnimation(.easeInOut(duration: 4)) {
                    drift = CGSize(width: .random(in: -12...12), height: .random(in: -40...40))
                }
            }
        }
    }

    // MARK: what the phone is doing, in plain words

    // One story at a time, from what the Mac reports (its proxy: reading, thinking, writing, done; its server script: starting,
    // ready, stopped) and what this iPhone sees itself. A state holds for its whole phase: the pauses inside a long read (every
    // 2,048 tokens the Mac saves its place, ~2 s) and the stretch while the Mac writes never flip the screen to "nothing".
    private enum Mode: Equatable { case loading, noCable, waiting, starting, activating, ready, readingTogether, readingAlone, writing }

    // the Mac's server is up: it said so, or it is talking to this iPhone (the app was reopened while it ran)
    private var linked: Bool {
        macPhase == "stopped" ? false : !macPhase.isEmpty || tail.state == "Connected" || tail.state == "Working"
    }
    private var holding: Bool { heldKeys > 0 }
    private var writingPhase: Bool { macPhase == "thinking" || macPhase == "writing" }
    private static let activateMin = 2.6

    private var mode: Mode {
        if cable.isEmpty { return .noCable }
        if !linked { return .waiting }
        if tail.state == "Loading" { return .loading }
        if macPhase == "starting" { return .starting }
        if activating || activateAt.map({ Date().timeIntervalSince($0) < Self.activateMin }) == true { return .activating }
        if macPhase == "reading" { return lastTailChunks > readStartChunks ? .readingTogether : .readingAlone }
        if writingPhase { return .writing }
        return .ready
    }

    private var phoneLayers: Int { 64 - tailFirst }
    private var macRange: String { "1–\(tailFirst)" }
    private var phoneRange: String { "\(tailFirst + 1)–64" }

    private func count(_ n: Double) -> String { n >= 1000 ? String(format: "%.1fk", n / 1000) : String(format: "%.0f", n) }

    private var headline: String {
        switch mode {
        case .loading: return "Opening the model"
        case .noCable: return "Plug into your Mac"
        case .waiting: return "Waiting for your Mac"
        case .starting: return "Your Mac is starting up"
        case .activating: return "Getting ready"
        case .ready: return holding ? "Holding the start of your chat" : "Ready"
        case .readingTogether: return "Reading your prompt together"
        case .readingAlone: return "Your Mac is reading"
        case .writing: return macPhase == "thinking" ? "Your Mac is thinking" : "Your Mac is writing"
        }
    }

    private var subline: String {
        switch mode {
        case .loading:
            return "This \(device) is opening its part of the model, the last \(phoneLayers) layers."
        case .noCable:
            return "Use a USB-C cable that carries data. A 10 Gb/s cable runs at full speed. The charge-only cable from the box won't work."
        case .waiting:
            return "Start the server on your Mac. This \(device) gets its \(phoneLayers) layers ready as soon as the server is up."
        case .starting:
            return "Your Mac is loading the model. That takes about a minute, then this \(device) gets ready."
        case .activating:
            return "Moving this \(device)'s \(phoneLayers) layers into GPU memory, so it can start on your first long prompt right away."
        case .ready:
            if holding {
                return "Your Mac keeps the newest 64k tokens of the chat. This \(device) keeps the \(count(Double(heldKeys))) before them and answers your Mac's questions about them."
            }
            if macPhase == "done" && macN1 + macN2 > 0 {
                return "Last turn: read \(Int(macN1)) tokens, wrote \(Int(macN2)). Send a long prompt and this \(device) reads it with your Mac."
            }
            return "Send a long prompt and your Mac runs layers \(macRange) while this \(device) runs \(phoneRange), at the same time."
        case .readingTogether:
            return "Your Mac runs layers \(macRange) while this \(device) runs \(phoneRange), at the same time. Every 2,048 tokens they pause for a moment to save their place."
        case .readingAlone:
            return "Short prompts are quicker on your Mac alone. This \(device) joins in from 512 tokens."
        case .writing:
            if holding { return "For every token, your Mac asks this \(device) about the oldest \(count(Double(heldKeys))) tokens of the chat." }
            return "Writing goes one token at a time, and that's fastest on your Mac alone. This \(device) waits for the next long prompt."
        }
    }

    private struct Stat { let value: String; var unit = ""; let label: String; let lit: Bool }

    // how far the activation looks: paced over at least activateMin, and never done while the phone is still wiring
    private static func activationShown(_ at: Date?, busy: Bool) -> Double {
        guard let at else { return 1 }
        let p = min(1, Date().timeIntervalSince(at) / activateMin)
        return busy ? min(p, 0.94) : p
    }

    // the three numbers that matter in each state, shown large under the headline
    private var stats: [Stat] {
        let quiet = rxRate + txRate < 2e5
        let mbs = String(format: "%.0f", (rxRate + txRate) / 1_048_576)
        let cable = quiet ? Stat(value: "Quiet", label: "cable", lit: false) : Stat(value: mbs, unit: "MB/s", label: "over the cable", lit: false)
        let layers = Stat(value: "\(phoneLayers)", unit: "of 64", label: "layers here", lit: false)
        let chat = Stat(value: macCtx > 0 ? count(macCtx) : "0", label: "tokens in this chat", lit: false)
        let held = Stat(value: count(Double(heldKeys)), label: "older tokens held", lit: true)
        let since = Int(Date().timeIntervalSince(macPhaseAt))
        switch mode {
        case .noCable: return []
        case .loading, .activating:
            let p = mode == .loading ? loadProgress : Self.activationShown(activateAt, busy: activating)
            return [Stat(value: "\(Int((p * 100).rounded()))", unit: "%", label: mode == .loading ? "opened" : "in GPU memory", lit: true),
                    Stat(value: "\(Int(p * Double(phoneLayers)))", unit: "of \(phoneLayers)", label: "layers ready", lit: false),
                    cable]
        case .waiting:
            let free = memAvail > 0 ? String(format: "%.1f", Double(memAvail) / 1_073_741_824) : "-"
            return [Stat(value: free, unit: "GB", label: "memory free", lit: false), layers, cable]
        case .starting:
            return [Stat(value: "\(since)", unit: "s", label: "starting", lit: false), layers, cable]
        case .ready:
            return holding ? [held, chat, cable] : [chat, layers, cable]
        case .readingTogether:
            // the headline number is the whole prompt read, end to end (the server's own count, sent by the Mac's proxy: new
            // tokens over the server's prefill time). tailTokS is only this phone's 24 layers per chunk and overstates the
            // combined speed, so it is labeled as such and kept second.
            return [Stat(value: macN2 > 0 ? "\(Int(macN2.rounded()))" : "–", label: "tokens a second, Mac + \(device)", lit: true),
                    Stat(value: count(macN1), label: "tokens read", lit: false),
                    Stat(value: tailTokS > 0 ? "\(Int(tailTokS.rounded()))" : "–", label: "tok/s on this \(device)'s layers only", lit: false)]
        case .readingAlone:
            return [macN2 > 0 ? Stat(value: "\(Int(macN2.rounded()))", label: "tokens a second", lit: false)
                              : Stat(value: "\(since)", unit: "s", label: "reading", lit: false),
                    holding ? held : layers, cable]
        case .writing:
            return [Stat(value: count(macN1), label: macPhase == "thinking" ? "tokens of thinking" : "tokens written", lit: true),
                    Stat(value: writeRate > 0 ? "\(Int(writeRate.rounded()))" : "–", label: "tokens a second", lit: false),
                    holding ? held : macCtx > 0 ? chat : layers]
        }
    }

    private var inUseList: [String] {
        var e: [String] = []
        if mode == .readingTogether || attnBusy { e.append("GPU") }
        if attnBusy && aneOn { e.append("Neural Engine") }
        return e
    }

    private var inUse: String {
        if !inUseList.isEmpty { return inUseList.joined(separator: " and ") }
        return linked && mode != .starting ? "Standing by" : "Nothing yet"
    }

    private var hot: Bool { thermal == .serious || thermal == .critical }

    // heat colors everything the phone lights up: white when cool, warm white, amber, then ember
    private var glow: Color {
        switch thermal {
        case .nominal: return light
        case .fair: return warm
        case .serious: return amber
        default: return ember
        }
    }

    private var heatLevel: Int {
        switch thermal {
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        default: return 3
        }
    }

    // MARK: full screen

    private var fullView: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .firstTextBaseline) {
                    Text(Self.appName)
                        .font(.system(size: 22, weight: .semibold, design: .serif))
                        .foregroundStyle(light)
                    Spacer()
                    HStack(spacing: 6) {
                        Circle().fill(cable.isEmpty ? amber : glow).frame(width: 6, height: 6)
                        Text(cable.isEmpty ? "\(chip), not connected" : "\(chip), connected")
                    }
                    .font(.system(size: 14))
                    .foregroundStyle(cable.isEmpty ? amber : mid)
                }

                ZStack(alignment: .topLeading) {
                    if holding && linked && mode != .starting {
                        ConversationBar(frozen: dim || still, phoneTokens: heldKeys, macTokens: 65_536, busy: attnBusy,
                                        glow: glow, light: light, mid: mid, dark: dark)
                            .transition(.opacity)
                    } else {
                        HStack(alignment: .top, spacing: 14) {
                            VStack(alignment: .leading, spacing: 0) {
                                Text("Mac").frame(height: CGFloat(tailFirst) * 5, alignment: .center)
                                Text(device)
                                    .foregroundStyle([.readingTogether, .loading, .activating].contains(mode) ? glow : mid)
                                    .frame(height: CGFloat(phoneLayers) * 5, alignment: .center)
                            }
                            .font(.system(size: 14))
                            .foregroundStyle(mid)
                            .frame(width: 52, alignment: .leading)
                            LayerStack(frozen: dim || still, prefill: mode == .readingTogether, firstPhoneLayer: tailFirst, tokS: tailTokS,
                                       appearAt: appearAt, loading: mode == .loading || mode == .activating,
                                       phoneEmpty: [.noCable, .waiting, .starting].contains(mode), progress: loadTarget,
                                       macFlow: macFlow.rate, macGain: macFlow.gain, glow: glow, mid: mid, dark: dark)
                                .frame(height: 64 * 5)
                        }
                        .transition(.opacity)
                    }
                }
                .frame(height: 64 * 5, alignment: .top)
                .padding(.top, 36)

                Text(headline)
                    .font(.system(size: 30, weight: .regular, design: .serif))
                    .foregroundStyle(light)
                    .fixedSize(horizontal: false, vertical: true)
                    .id(headline)
                    .transition(.opacity)
                    .padding(.top, 28)
                Text(subline)
                    .font(.system(size: 16).monospacedDigit())
                    .foregroundStyle(mid)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .numericTransition()
                    .padding(.top, 10)

                if !stats.isEmpty {
                    HStack(alignment: .top, spacing: 12) {
                        ForEach(stats, id: \.label) { st in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(alignment: .firstTextBaseline, spacing: 4) {
                                    Text(st.value)
                                        .font(.system(size: 34, weight: .regular, design: .serif).monospacedDigit())
                                        .foregroundStyle(st.lit ? glow : light)
                                        .numericTransition()
                                    if !st.unit.isEmpty {
                                        Text(st.unit).font(.system(size: 14)).foregroundStyle(mid)
                                    }
                                }
                                .lineLimit(1)
                                .fixedSize()
                                Text(st.label)
                                    .font(.system(size: 13))
                                    .foregroundStyle(mid)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.top, 18)
                    .overlay(alignment: .top) { Rectangle().fill(dark).frame(height: 1) }
                    .padding(.top, 24)
                    .transition(.opacity)
                }

                if hot {
                    VStack(alignment: .leading, spacing: 10) {
                        note("It's running hot, so it has slowed down. A fan or a cool surface brings the speed back.", glow)
                    }
                    .padding(.top, 18)
                    .transition(.opacity)
                }

                VStack(spacing: 0) {
                    HStack {
                        Text("Heat").font(.system(size: 16)).foregroundStyle(mid)
                        Spacer()
                        HeatMeter(level: heatLevel, on: glow, off: dark)
                        Text(thermalWord)
                            .font(.system(size: 16))
                            .foregroundStyle(heatLevel >= 2 ? glow : light)
                            .frame(minWidth: 58, alignment: .trailing)
                    }
                    .padding(.vertical, 12)
                    .overlay(alignment: .top) { Rectangle().fill(dark).frame(height: 1) }
                    if mode != .waiting { row("Memory free", gib(memAvail), memAvail > 0 && memAvail < 512 << 20 ? amber : light) }
                    row("In use", inUse, inUseList.isEmpty ? mid : light)
                }
                .padding(.top, 24)

                Button {
                    lastTouch = Date()
                    withAnimation(.easeOut(duration: 0.25)) { showDetails.toggle() }
                } label: {
                    Text(showDetails ? "Hide details" : "Show details")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(mid)
                        .padding(.vertical, 12)
                }
                .padding(.top, 8)

                if showDetails {
                    VStack(alignment: .leading, spacing: 14) {
                        if !cable.isEmpty {
                            Text("Cable address \(cable)")
                                .font(.system(size: 12).monospacedDigit())
                                .foregroundStyle(mid)
                                .textSelection(.enabled)
                        }
                        ForEach([tail, attn, ane, rpc]) { s in serviceRow(s) }
                        if !envNote.isEmpty {
                            Text("Settings: \(envNote)")
                                .font(.system(size: 12))
                                .foregroundStyle(mid)
                        }
                        Text("App memory \(gib(memFootprint)), GPU memory \(gib(gpuAlloc))")
                            .font(.system(size: 12).monospacedDigit())
                            .foregroundStyle(mid)
                    }
                    .padding(.bottom, 16)
                    .transition(.opacity)
                }

                Text("Leave this open and unlocked. If you switch apps or lock the phone, your Mac carries on by itself. Tap anywhere to dim the screen.")
                    .font(.system(size: 13))
                    .foregroundStyle(mid)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 8)
            }
            .animation(.easeInOut(duration: 0.5), value: mode)
            .animation(.easeInOut(duration: 0.6), value: heatLevel)
            .padding(.horizontal, 24)
            .padding(.top, 12)
            .padding(.bottom, 32)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func row(_ k: String, _ v: String, _ c: Color) -> some View {
        HStack {
            Text(k).font(.system(size: 16)).foregroundStyle(mid)
            Spacer()
            Text(v).font(.system(size: 16).monospacedDigit()).foregroundStyle(c)
                .lineLimit(1).minimumScaleFactor(0.7)
                .numericTransition()
        }
        .padding(.vertical, 12)
        .overlay(alignment: .top) { Rectangle().fill(dark).frame(height: 1) }
    }

    private var linkText: String {
        if rxRate + txRate < 2e5 { return "Quiet" }
        return "\(rate(rxRate)) in, \(rate(txRate)) out"
    }

    private func serviceRow(_ s: Service) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Circle().fill(stateColor(s.state)).frame(width: 7, height: 7)
                Text(s.id)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(light)
                Text("port \(String(s.port))")
                    .font(.system(size: 12).monospacedDigit())
                    .foregroundStyle(mid)
                Spacer(minLength: 4)
                Text(s.state)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(stateColor(s.state))
            }
            let line = [s.detail, ago(s.lastUsed)].filter { !$0.isEmpty }.joined(separator: ", ")
            if !line.isEmpty {
                Text(line)
                    .font(.system(size: 12))
                    .foregroundStyle(mid)
                    .lineLimit(2)
                    .padding(.leading, 15)
            }
        }
    }

    // MARK: dim screen

    private var dimView: some View {
        Text(headline)
            .font(.system(size: 17, design: .serif))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(white: 0.2))
            .padding(.horizontal, 48)
            .offset(drift)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func toggleDim() {
        lastTouch = Date()
        setDim(!dim)
    }

    private func setDim(_ on: Bool) {
        guard on != dim else { return }
        withAnimation(.easeInOut(duration: on ? 0.9 : 0.45)) { dim = on }
        if on {
            savedBrightness = UIScreen.main.brightness
            rampBrightness(to: 0.05, over: 0.9)
        } else if savedBrightness >= 0 {
            rampBrightness(to: savedBrightness, over: 0.45)
            lastTouch = Date()
        }
    }

    // UIScreen.brightness can't be animated; step it along an ease curve so the screen doesn't snap
    private func rampBrightness(to target: CGFloat, over seconds: Double) {
        brightnessRamp += 1
        let ramp = brightnessRamp
        let from = UIScreen.main.brightness
        let steps = max(1, Int(seconds * 60))
        for k in 1...steps {
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds * Double(k) / Double(steps)) {
                guard ramp == brightnessRamp else { return }   // a newer ramp took over
                let x = Double(k) / Double(steps)
                let e = x * x * (3 - 2 * x)
                UIScreen.main.brightness = from + (target - from) * CGFloat(e)
            }
        }
    }

    // MARK: text

    static let appName = "Backburner"

    private var chip: String {
        switch machine {
        case "iPhone17,1", "iPhone17,2": return "A18 Pro"
        case "iPhone18,1", "iPhone18,2": return "A19 Pro"
        default: return machine
        }
    }

    private var thermalWord: String {
        switch thermal {
        case .nominal: return "Cool"
        case .fair: return "Warm"
        case .serious: return "Hot"
        case .critical: return "Too hot"
        @unknown default: return "Unknown"
        }
    }

    // palette: true black (OLED); the phone's work drawn as light whose color follows its heat; amber for warnings
    private let light = Color(red: 0.933, green: 0.945, blue: 0.949)   // #EEF1F2
    private let warm  = Color(red: 0.969, green: 0.859, blue: 0.678)   // #F7DBAD
    private let amber = Color(red: 0.949, green: 0.702, blue: 0.239)   // #F2B33D
    private let ember = Color(red: 1.0,   green: 0.416, blue: 0.239)   // #FF6A3D
    private let mid   = Color(red: 0.42,  green: 0.447, blue: 0.459)   // #6B7275
    private let dark  = Color(red: 0.149, green: 0.165, blue: 0.173)   // #262A2C

    private func stateColor(_ s: String) -> Color {
        switch s {
        case "Working", "Ready", "Idle", "Connected": return light
        case "Error", "No model": return amber
        default: return mid
        }
    }

    private func note(_ t: String, _ c: Color) -> some View {
        Text(t)
            .font(.system(size: 15, weight: .medium))
            .foregroundStyle(c)
            .lineSpacing(2)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func gib(_ b: UInt64) -> String {
        if b == 0 { return "-" }
        let g = Double(b) / 1_073_741_824
        return g >= 1 ? String(format: "%.1f GB", g) : String(format: "%.0f MB", Double(b) / 1_048_576)
    }

    private func rate(_ bps: Double) -> String {
        if bps < 1024 { return "0" }
        if bps >= 1_073_741_824 { return String(format: "%.2f GB/s", bps / 1_073_741_824) }
        if bps >= 1_048_576 { return String(format: "%.0f MB/s", bps / 1_048_576) }
        return String(format: "%.0f KB/s", bps / 1024)
    }

    private func ago(_ d: Date?) -> String {
        guard let d else { return "" }
        let s = Int(Date().timeIntervalSince(d))
        if s < 2 { return "used now" }
        if s < 60 { return "used \(s)s ago" }
        if s < 3600 { return "used \(s / 60)m ago" }
        return "used \(s / 3600)h ago"
    }

    // MARK: refresh (1 Hz)

    private func refresh() {
        cable = SidecarRPC.cableAddress()
        WifiTunnel.startIfPaired()   // after `pair` over the cable (scripts/phone-wifi.sh); it stops itself on `unpair`
        if machine.isEmpty { machine = SidecarRPC.deviceModel() }
        thermal = ProcessInfo.processInfo.thermalState

        let mem = SidecarRPC.memoryStats()
        memAvail     = (mem["availableBytes"] as? NSNumber)?.uint64Value ?? 0
        memFootprint = (mem["footprintBytes"] as? NSNumber)?.uint64Value ?? 0
        gpuAlloc = (SidecarRPC.metalStats()["allocatedBytes"] as? NSNumber)?.uint64Value ?? 0

        let l = SidecarRPC.linkStats()
        let rx = (l["rxBytes"] as? NSNumber)?.uint64Value ?? 0
        let tx = (l["txBytes"] as? NSNumber)?.uint64Value ?? 0
        let now = Date()
        if let prev = Self.lastAt {
            let dt = now.timeIntervalSince(prev)
            if dt > 0.05 {
                rxRate = rx >= Self.lastRx ? Double(rx - Self.lastRx) / dt : 0
                txRate = tx >= Self.lastTx ? Double(tx - Self.lastTx) / dt : 0
            }
        }
        Self.lastRx = rx
        Self.lastTx = tx
        Self.lastAt = now

        // prefill tail
        let t = SidecarRPC.tailStatus()
        let ts = (t["state"] as? String) ?? ""
        let chunks = (t["chunks"] as? NSNumber)?.uint64Value ?? 0
        let tokS = (t["lastTokS"] as? NSNumber)?.doubleValue ?? 0
        tailTokS = tokS
        if chunks != lastTailChunks { tail.lastUsed = now; lastTailChunks = chunks }
        loadProgress = (t["loadProgress"] as? NSNumber)?.doubleValue ?? 0
        modelBytes = (t["modelBytes"] as? NSNumber)?.uint64Value ?? 0
        tailTokens = (t["tokens"] as? NSNumber)?.uint64Value ?? 0
        let rawModel = (t["model"] as? String) ?? ""
        if let r = rawModel.range(of: #"tail L=(\d+)"#, options: .regularExpression),
           let n = Int(rawModel[r].dropFirst(7)) { tailFirst = n }
        let model = shortModel(rawModel)
        if !tailRunning && !tailError.isEmpty {
            tail.state = "Error"; tail.detail = tailError
        } else {
            switch ts {
            case "no model": tail.state = "No model"; tail.detail = "Copy tail.gguf from the Mac (scripts/phone-push.py)"
            case "busy": tail.state = "Working"; tail.detail = String(format: "%@  %.0f tok/s", model, tokS)
            case "connected": tail.state = "Connected"; tail.detail = chunks > 0 ? String(format: "%@  %llu chunks", model, chunks) : model
            case "ready": tail.state = "Ready"; tail.detail = model
            case "error": tail.state = "Error"; tail.detail = (t["detail"] as? String) ?? ""
            case "": tail.state = "Starting"
            default: tail.state = ts.capitalized; tail.detail = model
            }
        }

        // phone attention
        let a = SidecarRPC.phoneAttnStatus()
        let calls = (a["calls"] as? NSNumber)?.uint64Value ?? 0
        let held = (a["heldKeys"] as? NSNumber)?.uint64Value ?? 0
        let lastMs = (a["lastMs"] as? NSNumber)?.doubleValue ?? 0
        let astate = (a["state"] as? String) ?? ""
        if calls != lastAttnCalls { attn.lastUsed = now; lastAttnCalls = calls }
        attnBusy = attn.lastUsed.map { now.timeIntervalSince($0) < 3 } ?? false   // held through the gaps between calls
        heldKeys = held
        attnLastMs = lastMs
        aneOn = astate.contains("ane") || astate.contains("ANE")
        if astate.hasPrefix("error") {
            attn.state = "Error"; attn.detail = astate
        } else {
            attn.state = attnBusy ? "Working" : (calls > 0 || held > 0 ? "Idle" : "Ready")
            attn.detail = held > 0 ? String(format: "%llu keys held  last call %.1f ms", held, lastMs)
                                   : (astate.hasPrefix("gpu off") ? "CPU only (\(astate))" : "")
        }

        // ggml RPC: no counters; say working while traffic flows and the tail and attention are quiet
        if !rpcRunning && !rpcError.isEmpty {
            rpc.state = "Error"; rpc.detail = rpcError
        } else if rpcRunning {
            let rpcBusy = (rxRate > 1e6 || txRate > 1e6) && tail.state != "Working" && !attnBusy
            rpc.state = rpcBusy ? "Working" : "Ready"
            if rpcBusy { rpc.lastUsed = now }
        }

        // what the Mac is doing
        let m = SidecarRPC.macStatus()
        if Self.demo.isEmpty { applyMac((m["phase"] as? String) ?? "", n1: (m["n1"] as? NSNumber)?.doubleValue ?? 0, n2: (m["n2"] as? NSNumber)?.doubleValue ?? 0,
                 ctx: (m["ctx"] as? NSNumber)?.doubleValue ?? 0, activations: (m["activations"] as? NSNumber)?.uint64Value ?? 0,
                 activating: (m["activating"] as? NSNumber)?.boolValue ?? false) }
        applyDemo()
    }

    private func applyMac(_ phase: String, n1: Double, n2: Double, ctx: Double, activations a: UInt64, activating busy: Bool) {
        let now = Date()
        if phase != macPhase {
            if phase == "reading" { readStartTokens = tailTokens; readStartChunks = lastTailChunks }
            if (phase == "thinking" || phase == "writing") && !writingPhase { writeRate = 0 }
            macPhase = phase
            macPhaseAt = now
        }
        if writingPhase && n2 > 0 { writeRate = n2 }   // the proxy's own count: tokens since the first, over that time
        macN1 = n1
        macN2 = n2
        if ctx > 0 { macCtx = ctx }
        if a != activations { activations = a; if a > 0 { activateAt = now } }
        activating = busy
    }

    // how the Mac's rows move: a slow sweep while it loads the model, a quicker stream while it reads or writes alone
    private var macFlow: (rate: Double, gain: Double) {
        switch mode {
        case .starting: return (0.45, 0.55)
        case .readingAlone: return (1.1, 0.8)
        case .writing: return (min(2.4, 0.6 + writeRate / 20), 0.7)
        default: return (0, 0)
        }
    }

    // the phone rows' fill while loading: the real open progress, or the activation (paced, held short of full while wiring)
    private var loadTarget: () -> Double {
        if mode == .loading { return Self.liveLoadProgress }
        let at = activateAt
        return { Self.activationShown(at, busy: (SidecarRPC.macStatus()["activating"] as? NSNumber)?.boolValue ?? false) }
    }

    private static var demo: String { getenv("SIDECAR_DEMO").map { String(cString: $0) } ?? "" }
    private static let demoCycle = 24.0

    // read every frame while loading (a mutex and a float), so the load front moves with the real progress
    private static func liveLoadProgress() -> Double {
        return (SidecarRPC.tailStatus()["loadProgress"] as? NSNumber)?.doubleValue ?? 0
    }

    // SIDECAR_DEMO=flow|prefill|context|nocable|hot in Documents/env.txt: fake a state for screenshots and videos. flow loops the
    // whole story every 24 s: waiting, starting, getting ready, reading together, writing, done
    private func applyDemo() {
        guard let d = getenv("SIDECAR_DEMO").map({ String(cString: $0) }), !d.isEmpty else { return }
        switch d {
        case "flow":
            let c = Date().timeIntervalSince1970.truncatingRemainder(dividingBy: Self.demoCycle)
            let ph = c < 3 ? "" : c < 7 ? "starting" : c < 11 ? "ready" : c < 17 ? "reading" : c < 22 ? "writing" : "done"
            let acts = activations + (ph == "ready" && macPhase == "starting" ? 1 : 0)
            let wrote = ph == "writing" ? (c - 17) * 24 : ph == "done" ? 120 : 0
            applyMac(ph, n1: ph == "done" ? 6_864 : ph == "reading" ? (c - 11) * 157 : wrote, n2: ph == "writing" ? 24 : ph == "reading" ? 157 : wrote, ctx: ph == "done" ? 7_054 : macCtx, activations: acts, activating: false)
            if ph == "reading" {
                tail.state = "Working"; lastTailChunks = readStartChunks + 1; tailTokS = 168 + Double.random(in: -6...6)
                tailTokens = readStartTokens + UInt64((c - 11) * 170); rxRate = 2.4e8; txRate = 2.3e8
            } else {
                tail.state = ph.isEmpty ? "Ready" : "Connected"; rxRate = 0; txRate = 0
            }
        case "prefill":
            applyMac("reading", n1: 2_048, n2: 157, ctx: 0, activations: activations, activating: false)   // 157 tok/s: measured end to end at 16k, 2026-10-01
            tail.state = "Working"; lastTailChunks = readStartChunks + 1; tailTokS = 168 + Double.random(in: -6...6); rxRate = 2.4e8; txRate = 2.3e8
        case "context":
            applyMac("writing", n1: 212, n2: 0, ctx: 145_000, activations: activations, activating: false)
            heldKeys = 79_616; attnBusy = true; attnLastMs = 3; aneOn = true; writeRate = 16.5; rxRate = 1.9e8; txRate = 1.8e8
        case "nocable":
            cable = ""
        case "hot":
            tail.state = "Working"; tailTokS = 121; thermal = .serious; rxRate = 1.6e8; txRate = 1.5e8
        default:
            break
        }
    }

    private func shortModel(_ m: String) -> String {
        // "Qwen3.8 27B [tail L=44] | qwen35 ?B IQ4_XS - 4.25 bpw" -> "tail L=44 IQ4_XS"
        guard !m.isEmpty else { return "" }
        var parts: [String] = []
        if let r = m.range(of: #"tail L=\d+"#, options: .regularExpression) { parts.append(String(m[r])) }
        if let r = m.range(of: #"(IQ|Q)\d\w*"#, options: .regularExpression) { parts.append(String(m[r])) }
        if m.contains("ANE") { parts.append("+ANE") }
        return parts.isEmpty ? m : parts.joined(separator: " ")
    }

    // MARK: servers

    // each server runs on its own thread and is restarted after a failure (a USB replug, an abort in a request)
    private func startRPC() {
        guard !rpcRunning else { return }
        rpcRunning = true
        rpcError = ""
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("rpc", isDirectory: true)
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let port = rpcPort
        DispatchQueue.global(qos: .userInitiated).async {
            let err = SidecarRPC.start(host: "127.0.0.1", port: port, cacheDir: cache.path)
            DispatchQueue.main.async {
                rpcRunning = false
                rpcError = (err ?? "stopped") + ". Restarting."
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) { startRPC() }
            }
        }
    }

    private func startTail() {
        guard !tailRunning else { return }
        tailRunning = true
        tailError = ""
        let path = tailPath
        let port = tailPort
        DispatchQueue.global(qos: .userInitiated).async {
            let err = SidecarRPC.startTail(port: port, modelPath: path)
            DispatchQueue.main.async {
                tailRunning = false
                tailError = (err ?? "stopped") + ". Restarting."
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) { startTail() }
            }
        }
    }
}

private extension View {
    // numbers roll instead of jumping (iOS 17+)
    @ViewBuilder func numericTransition() -> some View {
        if #available(iOS 17.0, *) { self.contentTransition(.numericText()) } else { self }
    }
}

private func smooth(_ x: Double) -> Double { let t = min(1, max(0, x)); return t * t * (3 - 2 * t) }

// Keeps motion continuous across frames: a phase that advances at a rate that may change (no jump when it does), an
// activity level that eases toward 0 or 1 so motion ramps up and tails off instead of switching, and a followed value
// (load progress) that glides after its target. A new clock starts the followed value where it is, so a rebuilt view
// never replays a load.
private final class MotionClock {
    private var last: Date?
    private(set) var phase = 0.0
    private(set) var activity = 0.0
    private(set) var followed = -1.0
    private(set) var phase2 = 0.0       // a second flow (the Mac's rows on their own)
    private(set) var activity2 = 0.0
    private(set) var wakeAt: Date?      // set when a fill completes: the rows light up and a pulse rises to the Mac
    private var filling = false

    func step(_ now: Date, rate: Double, target: Double, ease: Double = 0.5, follow: Double = 1, rate2: Double = 0) {
        let dt = last.map { min(0.1, max(0, now.timeIntervalSince($0))) } ?? 0
        last = now
        phase += dt * rate
        activity += (target - activity) * (1 - exp(-dt * 3 / ease))
        phase2 += dt * rate2
        activity2 += ((rate2 > 0 ? 1 : 0) - activity2) * (1 - exp(-dt * 3 / 0.6))
        if followed < 0 { followed = follow }
        else { followed += (follow - followed) * (1 - exp(-dt * (follow > followed ? 6 : 2))) }   // empties slower than it fills
        if followed < 0.97 { filling = true }
        else if filling && followed >= 0.999 { filling = false; wakeAt = now }
    }
}

private func hash01(_ a: Int, _ b: Int) -> Double {
    var x = UInt32(truncatingIfNeeded: a &* 73_856_093 ^ b &* 19_349_663)
    x ^= x >> 13; x = x &* 0x5bd1_e995; x ^= x >> 15
    return Double(x & 0xffff) / 65_536
}

// rises smoothly to 1 at t = peak, then decays: a single swell with no flicker
private func swell(_ t: Double, peak: Double) -> Double { t <= 0 ? 0 : (t / peak) * exp(1 - t / peak) }

// The model as 64 lines, one per layer. The Mac's layers are hairlines, this iPhone's are bars lit in its heat color.
// On appear the lines draw in from the left, top to bottom.
// Loading: this iPhone's layers start as rows of faint dots, empty slots in memory. The weights arrive in file order: a
// slanted front sweeps down the rows, and each dot it reaches swells into the bar with a brief glint, at a slightly
// different moment from its neighbors, so the front reads as grain settling rather than a progress bar. The rows just
// ahead of the front brighten a little in anticipation. The front follows the real load progress, read every frame.
// Wake-up: when the load finishes, the block swells from the bottom row up and sends one pulse up through the Mac's
// layers (ready, telling the Mac). When the Mac links up, a pulse falls from the Mac's first layer and lands in the
// block, which swells as it arrives. Both leave the same afterglow as a prefill chunk.
// Prefill: chunks of the prompt flow down through all 64 layers one after another, like the real pipeline: a band of
// light that is faint on the Mac's layers and flares on the iPhone's. The flow speed follows the real chunk rate and eases
// in and out. At rest the iPhone's lines breathe very slowly.
private struct LayerStack: View {
    var frozen = false
    let prefill: Bool
    let firstPhoneLayer: Int
    let tokS: Double
    let appearAt: Date
    var loading = false
    var phoneEmpty = false   // this iPhone's layers aren't in GPU memory yet: rows of dots
    var progress: () -> Double = { 1 }
    var macFlow = 0.0        // passes a second through the Mac's rows on their own (it reads or writes alone), 0 = still
    var macGain = 0.0
    let glow: Color, mid: Color, dark: Color

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var clock = MotionClock()

    private static let wakeDur = 1.25

    // when the wake pulse reaches line i (seconds after it starts); it rises from the last row and slows as it reaches the Mac
    private func arrival(_ i: Int) -> Double {
        Self.wakeDur * (1 - pow(max(0, 1 - Double(63 - i) / 67), 1 / 1.6))
    }

    var body: some View {
        TimelineView(.animation(paused: reduceMotion || frozen)) { tl in
            Canvas { ctx, size in
                let now = tl.date
                let t = now.timeIntervalSinceReferenceDate
                // one chunk (256 tokens) crosses all 64 layers per `travel` seconds; the next starts as it leaves
                let travel = min(2.4, max(1.1, 256 / max(1, tokS)))
                let target = phoneEmpty ? 0 : loading ? progress() : 1
                clock.step(now, rate: 1 / travel, target: prefill && !reduceMotion ? 1 : 0, follow: target,
                           rate2: reduceMotion ? 0 : macFlow)
                let act = clock.activity
                let u = clock.phase
                let loaded = reduceMotion ? target : clock.followed
                let since = now.timeIntervalSince(appearAt)
                let pitch = size.height / 64
                let h = max(2, pitch * 0.46)
                let fm = Double(firstPhoneLayer), fp = Double(64 - firstPhoneLayer)
                let wake = reduceMotion ? 99 : (clock.wakeAt.map { now.timeIntervalSince($0) } ?? 99)
                let waking = wake < Self.wakeDur + 1.6
                let wakeFade = 1 - smooth((wake - Self.wakeDur) / 1.6)
                let wavePos = 63 - 67 * (1 - pow(1 - min(1, wake / Self.wakeDur), 1.6))
                // the load front: row j fills while F passes from j to j + K
                let K = 4.0
                let F = loaded * (fp + K)
                let cell = 5.0, soft = 64.0

                for i in 0..<64 {
                    let drawIn = reduceMotion ? 1 : smooth((since - Double(i) * 0.012) / 0.7)
                    if drawIn <= 0 { continue }
                    let w = size.width * drawIn
                    let y = CGFloat(i) * pitch + (pitch - h) / 2
                    let cy = y + h / 2

                    var e = 0.0
                    if act > 0.001 {
                        let base = Int(floor(u))
                        for k in (base - 3)...base {
                            let tau = u - Double(k)
                            let pos = tau < 1 ? fm * tau : fm + fp * (tau - 1)
                            let d = pos - Double(i)                                   // > 0: the band has passed line i
                            let head = exp(-d * d / 2.2)
                            e = max(e, d < 0 ? head : max(head, 0.7 * exp(-d / 6)))
                        }
                    }
                    e *= act
                    var bloom = 0.0
                    if waking {
                        let d = Double(i) - wavePos                                   // > 0: the pulse has passed
                        let pulse = exp(-d * d / 2.5) + (d > 0 ? 0.55 * exp(-d / 5) : 0)
                        e = max(e, pulse * wakeFade)
                        bloom = swell(wake - arrival(i), peak: 0.22) * wakeFade
                    }

                    if i < firstPhoneLayer {
                        if clock.activity2 > 0.001 {   // the Mac alone: one soft band per pass down its rows, with a short trail
                            let d = (fm + 8) * clock.phase2.truncatingRemainder(dividingBy: 1) - 4 - Double(i)   // enters above, leaves below
                            let band = exp(-d * d / 3) + (d > 0 ? 0.45 * exp(-d / 3) : 0)
                            e = max(e, band * macGain * clock.activity2)
                        }
                        let c = dark.mix(mid, 0.25 + 1.1 * e + 0.4 * bloom)
                        let lh = 1 + CGFloat(bloom)
                        ctx.fill(Path(CGRect(x: 0, y: cy - lh / 2, width: w, height: lh)), with: .color(c))
                        continue
                    }

                    let j = Double(i - firstPhoneLayer)
                    let breathe = reduceMotion ? 0.34 : 0.31 + 0.05 * sin(t * 2 * .pi / 6.5 - j * 0.2)
                    let a = min(1, max(breathe * (1 - act) + 0.22 * act + 0.78 * e, breathe + 0.55 * e) + 0.45 * bloom)
                    let bh = h * CGFloat(1 + 0.3 * bloom)                    // the row swells a little as it lights
                    let lp = min(1, max(0, (F - j) / K))                       // how far the front is along this row

                    if lp >= 1 {
                        ctx.fill(Path(CGRect(x: 0, y: cy - bh / 2, width: w, height: bh)), with: .color(glow.opacity(a)))
                        continue
                    }
                    // a row still loading: cells from empty dots to the bar
                    let lead = exp(-max(0, j - F) / 2) * 0.08                  // rows just ahead of the front wake a little
                    let xf = lp * (Double(w) + soft) - soft / 2
                    let n = Int(ceil(Double(w) / cell))
                    var dots = Path()
                    for c in 0..<n {
                        let x = Double(c) * cell
                        let jitter = (hash01(i, c) - 0.5) * soft * 0.9
                        let k = smooth((xf - x + jitter) / (soft * 0.35))
                        if k <= 0.001 {
                            dots.addRect(CGRect(x: x + cell / 2 - 0.75, y: cy - 0.75, width: 1.5, height: 1.5))
                            continue
                        }
                        let glint = 4 * k * (1 - k) * (0.55 + 0.45 * hash01(c, i + 101))
                        let cw = 1.5 + (cell + 0.4 - 1.5) * k
                        let ch = 1.5 + (Double(h) - 1.5) * k
                        let op = min(1, 0.14 + (breathe - 0.14) * k + 0.75 * glint)
                        ctx.fill(Path(CGRect(x: x + cell / 2 - cw / 2, y: cy - ch / 2, width: min(cw, Double(w) - x), height: ch)),
                                 with: .color(glow.opacity(op)))
                    }
                    ctx.fill(dots, with: .color(glow.opacity(0.14 + lead)))
                }
            }
        }
        .accessibilityLabel(phoneEmpty ? "The model's 64 layers. The last \(64 - firstPhoneLayer) will run on this \(device)."
                            : loading ? "Loading the last \(64 - firstPhoneLayer) layers of the model"
                            : prefill ? "This \(device) runs the last \(64 - firstPhoneLayer) of the model's 64 layers"
                                      : "The model's 64 layers. The last \(64 - firstPhoneLayer) run on this \(device).")
    }
}

private extension Color {
    // linear mix toward another color, k in 0...1
    func mix(_ o: Color, _ k: Double) -> Color {
        let k = min(1, max(0, k))
        let a = UIColor(self), b = UIColor(o)
        var r1: CGFloat = 0, g1: CGFloat = 0, b1: CGFloat = 0, a1: CGFloat = 0
        var r2: CGFloat = 0, g2: CGFloat = 0, b2: CGFloat = 0, a2: CGFloat = 0
        a.getRed(&r1, green: &g1, blue: &b1, alpha: &a1)
        b.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)
        let f = CGFloat(k)
        return Color(red: Double(r1 + (r2 - r1) * f), green: Double(g1 + (g2 - g1) * f), blue: Double(b1 + (b2 - b1) * f))
    }
}

// The conversation as the 16 attention layers, oldest on the left. The part this iPhone holds is lit in its heat color
// and grows in; the Mac's newest part is dim. While the Mac asks about the older part, a soft shimmer drifts along the
// iPhone's lines from newest to oldest, staggered per line like eyes reading down a page. It is a continuous flow that
// enters and leaves at the edges of the iPhone's part, and eases in and out with the Mac's questions.
private struct ConversationBar: View {
    var frozen = false
    let phoneTokens: UInt64
    let macTokens: UInt64
    let busy: Bool
    let glow: Color, light: Color, mid: Color, dark: Color

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var appeared = Date()
    @State private var clock = MotionClock()

    private var f: Double { Double(phoneTokens) / Double(max(1, phoneTokens + macTokens)) }

    private func count(_ n: UInt64) -> String {
        n >= 10_000 ? String(format: "%.1fk tokens", Double(n) / 1000) : "\(n) tokens"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Your conversation, oldest to newest")
                .font(.system(size: 14))
                .foregroundStyle(mid)
            TimelineView(.animation(paused: reduceMotion || frozen)) { tl in
                Canvas { ctx, size in
                    let now = tl.date
                    clock.step(now, rate: 1, target: busy && !reduceMotion ? 1 : 0, ease: 0.8)
                    let act = clock.activity
                    let grow = reduceMotion ? 1 : smooth(now.timeIntervalSince(appeared) / 1.1)
                    let pw = size.width * f * grow
                    let pitch = size.height / 16
                    let h = max(2, pitch * 0.5)
                    let spacing: CGFloat = 150        // distance between shimmer crests
                    let speed = 70.0                  // points per second, right to left
                    for i in 0..<16 {
                        let y = CGFloat(i) * pitch + (pitch - h) / 2
                        ctx.fill(Path(CGRect(x: min(size.width, pw + 3), y: y, width: max(0, size.width - pw - 3), height: h)),
                                 with: .color(dark))
                        guard pw > 1 else { continue }
                        // one smooth gradient per line: sample the crest pattern and let the gradient interpolate
                        var stops: [Gradient.Stop] = []
                        let n = 28
                        for k in 0...n {
                            let fx = Double(k) / Double(n)
                            let x = fx * Double(pw)
                            let q = (x + clock.phase * speed + Double(i) * 6).truncatingRemainder(dividingBy: Double(spacing))
                            let c = q - Double(spacing) / 2
                            let crest = exp(-c * c / (2 * 30 * 30))
                            stops.append(.init(color: glow.opacity(0.42 + 0.36 * crest * act), location: fx))
                        }
                        ctx.fill(Path(CGRect(x: 0, y: y, width: pw, height: h)),
                                 with: .linearGradient(Gradient(stops: stops), startPoint: CGPoint(x: 0, y: y), endPoint: CGPoint(x: pw, y: y)))
                    }
                }
            }
            .frame(height: 204)
            GeometryReader { g in
                HStack(alignment: .top, spacing: 0) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("On this \(device)").font(.system(size: 16, weight: .semibold)).foregroundStyle(light)
                        Text(count(phoneTokens)).font(.system(size: 14).monospacedDigit()).foregroundStyle(mid).numericTransition()
                    }
                    .frame(width: max(120, g.size.width * f), alignment: .leading)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("On your Mac").font(.system(size: 16, weight: .semibold)).foregroundStyle(light)
                        Text(count(macTokens)).font(.system(size: 14).monospacedDigit()).foregroundStyle(mid)
                    }
                    .padding(.leading, 8)
                }
            }
            .frame(height: 44)
        }
        .onAppear { appeared = Date() }
        .accessibilityElement(children: .combine)
    }
}

// four short bars, lit up to the phone's thermal state
private struct HeatMeter: View {
    let level: Int
    let on: Color, off: Color

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<4, id: \.self) { i in
                Capsule().fill(i <= level ? on : off).frame(width: 12, height: 4)
            }
        }
        .padding(.trailing, 8)
        .accessibilityHidden(true)
    }
}

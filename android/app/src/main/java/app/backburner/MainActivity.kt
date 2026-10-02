package app.backburner

import android.app.Activity
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

/**
 * Android port of ios/Backburner/Sidecar/ContentView.swift.
 *
 * Same story, condensed: one headline + subline (from the Mac's PHASE report
 * and this phone's own counters), three headline numbers, heat/memory rows,
 * and a details section (per-service state, cable address, env note).
 * The full LayerStack/Canvas animation is intentionally not ported; the
 * 64-layer bar is a static progress row (loading/activating/prefill).
 *
 * Keep the screen on via AndroidManifest keepScreenOn (== isIdleTimerDisabled).
 * Tap-to-dim + brightness ramp are omitted; FLAG_KEEP_SCREEN_ON holds the wake.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        startForegroundService(Intent(this, BackburnerService::class.java))
        setContent { BackburnerScreen() }
    }
}

private data class Snapshot(
    val cable: String = "",
    val model: String = "",
    val thermal: String = "Cool",
    val memAvail: Long = 0,
    val memFootprint: Long = 0,
    val sme2: Int = 0,
    val tailState: String = "Starting",
    val tailDetail: String = "",
    val tailChunks: Long = 0,
    val tailTokS: Double = 0.0,
    val attnState: String = "Starting",
    val attnDetail: String = "",
    val attnCalls: Long = 0,
    val heldKeys: Long = 0,
    val attnLastMs: Double = 0.0,
    val macPhase: String = "",
    val macN1: Double = 0.0,
    val macN2: Double = 0.0,
    val macCtx: Double = 0.0,
    val rxRate: Double = 0.0,
    val txRate: Double = 0.0,
    val envNote: String = "",
    val nsdFailed: Boolean = false,
)

@Composable
private fun BackburnerScreen() {
    var snap by remember { mutableStateOf(Snapshot()) }
    var showDetails by remember { mutableStateOf(false) }
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val advertiser = remember { NsdAdvertiser(ctx) }
    var lastRx by remember { mutableStateOf(0L) }
    var lastTx by remember { mutableStateOf(0L) }
    var lastAt by remember { mutableStateOf(0L) }

    LaunchedEffect(Unit) {
        while (isActive) {
            snap = refresh(lastRx, lastTx, lastAt)?.let { (s, rx, tx, at) ->
                lastRx = rx; lastTx = tx; lastAt = at
                if (s.cable.isNotEmpty()) {
                    advertiser.publish(50052, s.model, advertiser.chipLabel(), s.memAvail, s.cable)
                }
                s.copy(nsdFailed = advertiser.failed)
            } ?: snap
            delay(1000)
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(Modifier.fillMaxSize()) {
            Column(
                Modifier.padding(24.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Backburner", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
                Text(if (snap.cable.isEmpty()) "${snap.model}, not connected" else "${snap.model}, connected")
                LinearProgressIndicator(
                    progress = { layerProgress(snap) },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(headline(snap), style = MaterialTheme.typography.headlineMedium)
                Text(subline(snap), style = MaterialTheme.typography.bodyMedium)
                stats(snap).forEach { (v, label) ->
                    Column {
                        Text(v, style = MaterialTheme.typography.displaySmall)
                        Text(label, style = MaterialTheme.typography.labelMedium)
                    }
                }
                if (snap.thermal != "Cool" || snap.nsdFailed) {
                    if (snap.thermal != "Cool") Text("It's running hot (${snap.thermal}), so it has slowed down. A fan or a cool surface brings the speed back.")
                    if (snap.nsdFailed) Text("Your Mac can't find this phone by name. Use adb reverse (see android/README.md).")
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Heat: ${snap.thermal}")
                    Text("Memory free: ${gib(snap.memAvail)}")
                }
                TextButton(onClick = { showDetails = !showDetails }) {
                    Text(if (showDetails) "Hide details" else "Show details")
                }
                if (showDetails) {
                    if (snap.cable.isNotEmpty()) Text("Cable address ${snap.cable}")
                    Text("Prefill tail :50060 — ${snap.tailState} ${snap.tailDetail}")
                    Text("Phone attention :50062 — ${snap.attnState} ${snap.attnDetail}")
                    Text("GPU (ggml RPC) :50052 — see Mac logs")
                    if (snap.envNote.isNotEmpty()) Text("Settings: ${snap.envNote}")
                    Text("SME2: ${if (snap.sme2 == 1) "yes" else "no"}  App memory ${gib(snap.memFootprint)}")
                }
                Text("Leave this open and unlocked. If you switch apps or lock the phone, your Mac carries on by itself.")
            }
        }
    }
}

private fun gib(b: Long): String {
    if (b <= 0) return "-"
    val g = b.toDouble() / 1_073_741_824
    return if (g >= 1) "%.1f GB".format(g) else "%.0f MB".format(b.toDouble() / 1_048_576)
}

private fun count(n: Double): String = if (n >= 1000) "%.1fk".format(n / 1000) else "%.0f".format(n)

private fun linked(s: Snapshot) =
    s.macPhase != "stopped" && (s.macPhase.isNotEmpty() || s.tailState == "Connected" || s.tailState == "Working")

private fun headline(s: Snapshot): String {
    if (s.cable.isEmpty()) return "Plug into your Mac"
    if (!linked(s)) return "Waiting for your Mac"
    if (s.tailState == "Loading") return "Opening the model"
    if (s.macPhase == "starting") return "Your Mac is starting up"
    if (s.macPhase == "reading") return if (s.tailChunks > 0) "Reading your prompt together" else "Your Mac is reading"
    if (s.macPhase == "thinking") return "Your Mac is thinking"
    if (s.macPhase == "writing") return "Your Mac is writing"
    if (s.heldKeys > 0) return "Holding the start of your chat"
    return "Ready"
}

private fun subline(s: Snapshot): String {
    if (s.cable.isEmpty()) return "Use a USB-C cable that carries data. A 10 Gb/s cable runs at full speed. Prefer adb reverse (android/README.md) over Wi-Fi."
    if (!linked(s)) return "Start the server on your Mac (LLAMA_SPLIT_TAIL=127.0.0.1:50060 PHONE_KV=127.0.0.1:50062 scripts/serve.sh)."
    if (s.macPhase == "reading" && s.tailChunks > 0) return "Your Mac runs the first layers while this phone runs the last ones, at the same time."
    if (s.macPhase == "reading") return "Short prompts are quicker on your Mac alone. This phone joins in from ~512 tokens."
    if (s.macPhase == "thinking" || s.macPhase == "writing") {
        return if (s.heldKeys > 0) "For every token, your Mac asks this phone about the oldest ${count(s.heldKeys.toDouble())} tokens."
        else "Writing goes one token at a time, fastest on your Mac alone."
    }
    if (s.heldKeys > 0) return "Your Mac keeps the newest tokens; this phone keeps the ${count(s.heldKeys.toDouble())} before them."
    return "Send a long prompt and your Mac runs the first layers while this phone runs the last ones."
}

private fun stats(s: Snapshot): List<Pair<String, String>> {
    val quiet = s.rxRate + s.txRate < 2e5
    val cable = if (quiet) "Quiet" else "%.0f MB/s".format((s.rxRate + s.txRate) / 1_048_576)
    return when {
        s.cable.isEmpty() -> emptyList()
        s.macPhase == "reading" && s.tailChunks > 0 -> listOf(
            (if (s.macN2 > 0) "${s.macN2.toInt()}" else "–") to "tokens a second, Mac + phone",
            count(s.macN1) to "tokens read",
            (if (s.tailTokS > 0) "${s.tailTokS.toInt()}" else "–") to "tok/s on this phone's layers only",
        )
        s.heldKeys > 0 -> listOf(
            count(s.heldKeys.toDouble()) to "older tokens held",
            (if (s.macCtx > 0) count(s.macCtx) else "0") to "tokens in this chat",
            cable to "over the cable",
        )
        else -> listOf(
            (if (s.macCtx > 0) count(s.macCtx) else "0") to "tokens in this chat",
            cable to "over the cable",
            gib(s.memAvail) to "memory free",
        )
    }
}

private fun layerProgress(s: Snapshot): Float = when {
    s.tailState == "Loading" -> 0.3f
    s.macPhase == "starting" -> 0.6f
    linked(s) -> 1.0f
    else -> 0.1f
}

/** One 1 Hz poll of every native status endpoint (mirrors ContentView.refresh). */
private fun refresh(lastRx: Long, lastTx: Long, lastAt: Long): Quad? {
    return try {
        val mem = runCatching { SidecarRpc.memoryStats() }.getOrDefault(emptyMap())
        val link = runCatching { SidecarRpc.linkStats() }.getOrDefault(emptyMap())
        val tail = runCatching { SidecarRpc.tailStatus() }.getOrDefault(emptyMap())
        val attn = runCatching { SidecarRpc.phoneAttnStatus() }.getOrDefault(emptyMap())
        val mac = runCatching { SidecarRpc.macStatus() }.getOrDefault(emptyMap())
        val now = System.currentTimeMillis()
        val rx = (link["rxBytes"] as? Number)?.toLong() ?: 0L
        val tx = (link["txBytes"] as? Number)?.toLong() ?: 0L
        var rxRate = 0.0
        var txRate = 0.0
        if (lastAt > 0) {
            val dt = (now - lastAt) / 1000.0
            if (dt > 0.05) {
                if (rx >= lastRx) rxRate = (rx - lastRx) / dt
                if (tx >= lastTx) txRate = (tx - lastTx) / dt
            }
        }
        val s = Snapshot(
            cable = runCatching { SidecarRpc.cableAddress() }.getOrDefault(""),
            model = "${Build.MANUFACTURER} ${Build.MODEL}",
            memAvail = (mem["availableBytes"] as? Number)?.toLong() ?: 0L,
            memFootprint = (mem["footprintBytes"] as? Number)?.toLong() ?: 0L,
            sme2 = runCatching { SidecarRpc.sme2Available() }.getOrDefault(0),
            tailState = (tail["state"] as? String) ?: "Starting",
            tailDetail = (tail["detail"] as? String) ?: "",
            tailChunks = (tail["chunks"] as? Number)?.toLong() ?: 0L,
            tailTokS = (tail["lastTokS"] as? Number)?.toDouble() ?: 0.0,
            attnState = (attn["state"] as? String) ?: "",
            attnDetail = (attn["detail"] as? String) ?: "",
            attnCalls = (attn["calls"] as? Number)?.toLong() ?: 0L,
            heldKeys = (attn["heldKeys"] as? Number)?.toLong() ?: 0L,
            attnLastMs = (attn["lastMs"] as? Number)?.toDouble() ?: 0.0,
            macPhase = (mac["phase"] as? String) ?: "",
            macN1 = (mac["n1"] as? Number)?.toDouble() ?: 0.0,
            macN2 = (mac["n2"] as? Number)?.toDouble() ?: 0.0,
            macCtx = (mac["ctx"] as? Number)?.toDouble() ?: 0.0,
            rxRate = rxRate, txRate = txRate,
            envNote = runCatching { SidecarRpc.envNote() }.getOrDefault(""),
        )
        Quad(s, rx, tx, now)
    } catch (_: Throwable) {
        null
    }
}

private data class Quad(val s: Snapshot, val rx: Long, val tx: Long, val at: Long)

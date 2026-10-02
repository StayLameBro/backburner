package app.backburner

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.os.Build

/**
 * Android port of the iOS RPCAdvertiser (NetService _infernet-rpc._tcp).
 * Same service type and TXT keys so serve.sh / phone-up.sh need no changes:
 * model, chip, mem (16 MiB bucket), addr (cable IP), port.
 */
class NsdAdvertiser(private val ctx: Context) {
    private val nsd: NsdManager? = ctx.getSystemService(NsdManager::class.java)
    var failed = false
        private set
    private var registered: NsdServiceInfo? = null
    private var lastTxt = ""
    private val listener = object : NsdManager.RegistrationListener {
        override fun onServiceRegistered(info: NsdServiceInfo) {
            failed = false
        }
        override fun onRegistrationFailed(info: NsdServiceInfo, err: Int) {
            failed = true
        }
        override fun onServiceUnregistered(info: NsdServiceInfo) {}
        override fun onUnregistrationFailed(info: NsdServiceInfo, err: Int) {}
    }

    fun publish(port: Int, model: String, chip: String, memBytes: Long, addr: String) {
        val bucket = memBytes / (16L * 1024 * 1024) * (16L * 1024 * 1024)
        val stamp = "addr=$addr;chip=$chip;mem=$bucket;model=$model;port=$port"
        if (stamp == lastTxt && registered != null) return
        lastTxt = stamp
        try {
            registered?.let { runCatching { nsd?.unregisterService(listener) } }
        } catch (_: Exception) {
        }
        val info = NsdServiceInfo().apply {
            serviceName = "infernet"
            serviceType = "_infernet-rpc._tcp."
            setPort(port)
            // NsdServiceInfo TXT: setAttribute is API 21+; keys must match iOS TXT exactly.
            setAttribute("model", model)
            setAttribute("chip", chip)
            setAttribute("mem", bucket.toString())
            setAttribute("addr", addr)
            setAttribute("port", port.toString())
        }
        registered = info
        try {
            nsd?.registerService(info, NsdManager.PROTOCOL_DNS_SD, listener)
        } catch (_: Exception) {
            failed = true
        }
    }

    fun chipLabel(): String {
        // Best-effort SoC label for the TXT "chip" + UI, mirroring the iOS
        // utsname.machine -> "A18 Pro"/"A19 Pro" mapping in ContentView.chip.
        // NOTE: Build.SOC_MODEL is API 31+; use HARDWARE/BOARD so this runs on minSdk 29.
        return Build.HARDWARE ?: Build.BOARD ?: Build.MODEL
    }
}

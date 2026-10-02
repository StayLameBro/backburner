package app.backburner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service owning the four listeners, mirroring the iOS
 * startRPC/startTail + startANEBench/startPhoneAttn blocks in ContentView.
 *
 * Each server restarts after a failure (USB replug, abort in a request):
 * the native accept loop returns an error string and we sleep 3 s, same as
 * the iOS DispatchQueue.main.asyncAfter(deadline: .now() + 3) restart.
 */
class BackburnerService : Service() {
    private val pool = Executors.newFixedThreadPool(4)
    private val running = AtomicBoolean(false)

    override fun onCreate() {
        super.onCreate()
        startForeground(1, notification())
        if (running.compareAndSet(false, true)) {
            val cache = File(cacheDir, "rpc").apply { mkdirs() }
            val tail = File(filesDir, "tail.gguf")
            // ggml RPC (drafter path, :50052). Optional: serve.sh only uses it with PHONE_DRAFT=.
            pool.execute { loop("rpc") { SidecarRpc.startHost("0.0.0.0", 50052, cache.path) } }
            // Split-prefill tail (:50060). Loads tail.gguf if present, else reports "no model".
            pool.execute { loop("tail") { SidecarRpc.startTail(50060, tail.path) } }
            // Command/bench port (:50061: mem, fetch, ANE probe, mac PHASE notes).
            pool.execute { SidecarRpc.startCmdPort(50061) }
            // Phone-held KV attention (:50062, phone-attn.h server + sme_attn.c + stubs).
            pool.execute { SidecarRpc.startPhoneAttn(50062) }
        }
    }

    private fun loop(tag: String, fn: () -> String?) {
        while (running.get()) {
            val err = try {
                fn()
            } catch (t: Throwable) {
                t.message ?: t.toString()
            } ?: "$tag stopped"
            android.util.Log.w("Backburner", "$tag exited: $err; restarting in 3 s")
            try {
                Thread.sleep(3000)
            } catch (_: InterruptedException) {
                return
            }
        }
    }

    private fun notification(): Notification {
        val mgr = getSystemService(NotificationManager::class.java)
        mgr.createNotificationChannel(
            NotificationChannel("bb", "Backburner", NotificationManager.IMPORTANCE_LOW),
        )
        return Notification.Builder(this, "bb")
            .setContentTitle("Backburner")
            .setContentText("Serving your Mac: tail :50060, attention :50062")
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .build()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        running.set(false)
        pool.shutdownNow()
        super.onDestroy()
    }
}

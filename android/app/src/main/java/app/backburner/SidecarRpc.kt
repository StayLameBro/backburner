package app.backburner

/**
 * JNI surface mirroring ios/Backburner/Sidecar/RPCBridge.h.
 *
 * Each method is implemented in app/src/main/cpp/rpcbridge.cc. The native
 * servers (ggml RPC :50052, tail :50060, phone-attn :50062, command :50061)
 * are blocking accept loops, so call them off the main thread (see
 * BackburnerService) exactly like the iOS DispatchQueue.global blocks.
 */
object SidecarRpc {
    init {
        System.loadLibrary("rpcbridge")
    }

    /** Blocking. Returns null on clean exit; an error string otherwise. */
    external fun startHost(host: String, port: Int, cacheDir: String): String?

    /** USB-cable address only: one 169.254.x.x IPv4, or "". Wi-Fi/IPv6 excluded. */
    external fun cableAddress(): String

    /** GPU stats: deviceName, allocatedBytes (best effort), hasUnifiedMemory. */
    external fun gpuStats(): Map<String, Any>

    /** Memory: availableBytes, footprintBytes, physicalBytes. */
    external fun memoryStats(): Map<String, Long>

    /** Cumulative rx/tx bytes+packets across non-loopback interfaces. */
    external fun linkStats(): Map<String, Long>

    /** Blocking tail worker (llama.cpp tools/split-prefill/tail-server.h). */
    external fun startTail(port: Int, modelPath: String): String?

    /** Tail status: state, detail, model, chunks, tokens, lastChunkMs, lastTokS. */
    external fun tailStatus(): Map<String, Any>

    /** Hardware string (Build.MODEL/FINGERPRINT), for the NSD TXT record. */
    external fun deviceModel(): String

    /** Command/bench server (:50061: mem, fetch, mac PHASE reports). Starts once. */
    external fun startCmdPort(port: Int)

    /** Phone-held KV attention (:50062, phone-attn/phone-attn.h + sme_attn.c). Starts once. */
    external fun startPhoneAttn(port: Int)

    /** Phone attention status: state, calls, heldKeys, lastMs. */
    external fun phoneAttnStatus(): Map<String, Any>

    /** What the Mac reported (serve.sh phone_note / proxy): phase, n1, n2, ctx, age, activations, activating, activateMs. */
    external fun macStatus(): Map<String, Any>

    /** What files/env.txt changed at launch ("" if nothing). */
    external fun envNote(): String

    /** 1 if the CPU has SME2 with 512-bit streaming vectors, else 0. */
    external fun sme2Available(): Int
}

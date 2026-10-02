// rpcbridge.cc - Android port of ios/Backburner/Sidecar/RPCBridge.mm.
//
// Same JNI surface as RPCBridge.h (see SidecarRpc.kt). Platform mapping:
//   cableAddress : getifaddrs + 169.254 filter (Bionic has getifaddrs; same
//                  exclusion of lo/Wi-Fi/IPv6 as iOS). adb reverse needs no IP.
//   gpuStats     : Metal stats -> best-effort Vulkan device name (if linked)
//                  + ActivityManager numbers come from Kotlin side when needed.
//   memoryStats  : ActivityManager-equivalent via sysinfo + /proc/self/statm.
//   linkStats    : /proc/net/dev counters (TrafficStats equivalent), non-lo.
//   startHost    : ggml RPC server from the same llama.cpp tree (needs the
//                  llama link wired in CMakeLists LLAMA_BUILD_DIR).
//   startTail    : tools/split-prefill/tail-server.h, same as iOS.
//   command :50061: line protocol compatible with scripts/phone-push.py and
//                  serve.sh phone_note: "mem", "fetch <url> <dst>", "mac ...".
//   phone-attn   : phone-attn/phone-attn.h server verbatim + sme_attn.c +
//                  vulkan_engine.cc (stub) + nnapi_page_engine.cc (stub).
//
// v1 scope: CPU/SME attention only (GPU stub returns false, ANE stub holds no
// pages). Wire-compatible, accuracy-exact; speed work is isolated to the two
// stub files.

#include <jni.h>

#include <android/log.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define LOG_TAG "Backburner"
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

extern "C" {
int sme2_available(void);
}

// ---- phone-attn (portable, no Apple headers) -------------------------------
// phone-attn/phone-attn.h includes <TargetConditionals.h> / mach bits only
// under __APPLE__; on Android it compiles as pure POSIX C++. TARGET_OS_IPHONE
// is undefined here, so the jetsam/wired guards compile out and the server
// uses plain aligned_alloc pages (same as the macOS loopback pa-tool).
#include "phone-attn.h"
#include "vulkan_engine.h"
#include "nnapi_page_engine.h"

namespace {

std::mutex g_mu;
pa::status g_attn_status;
std::string g_mac_phase;
double g_mac_n1 = 0, g_mac_n2 = 0, g_mac_ctx = 0;
long long g_mac_at_ms = 0;
long long g_activations = 0;
bool g_activating = false;
std::string g_env_note;
std::atomic<bool> g_attn_started{false};
std::atomic<bool> g_cmd_started{false};

long long now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return "";
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return r;
}

jobject hashmap(JNIEnv* env) {
    jclass cls = env->FindClass("java/util/HashMap");
    jmethodID init = env->GetMethodID(cls, "<init>", "()V");
    return env->NewObject(cls, init);
}
void map_put_str(JNIEnv* env, jobject m, const char* k, const std::string& v) {
    jclass cls = env->GetObjectClass(m);
    jmethodID put = env->GetMethodID(cls, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jk = env->NewStringUTF(k);
    jstring jv = env->NewStringUTF(v.c_str());
    env->CallObjectMethod(m, put, jk, jv);
    env->DeleteLocalRef(jk);
    env->DeleteLocalRef(jv);
}
void map_put_long(JNIEnv* env, jobject m, const char* k, long long v) {
    jclass cls = env->GetObjectClass(m);
    jmethodID put = env->GetMethodID(cls, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jk = env->NewStringUTF(k);
    jclass lk = env->FindClass("java/lang/Long");
    jmethodID vc = env->GetStaticMethodID(lk, "valueOf", "(J)Ljava/lang/Long;");
    jobject jv = env->CallStaticObjectMethod(lk, vc, (jlong)v);
    env->CallObjectMethod(m, put, jk, jv);
    env->DeleteLocalRef(jk);
    env->DeleteLocalRef(jv);
}
void map_put_double(JNIEnv* env, jobject m, const char* k, double v) {
    jclass cls = env->GetObjectClass(m);
    jmethodID put = env->GetMethodID(cls, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jk = env->NewStringUTF(k);
    jclass dk = env->FindClass("java/lang/Double");
    jmethodID vc = env->GetStaticMethodID(dk, "valueOf", "(D)Ljava/lang/Double;");
    jobject jv = env->CallStaticObjectMethod(dk, vc, (jdouble)v);
    env->CallObjectMethod(m, put, jk, jv);
    env->DeleteLocalRef(jk);
    env->DeleteLocalRef(jv);
}

// Apply Documents/env.txt KEY=VALUE lines the same way the iOS app does at
// launch (PA_* knobs: PA_HOT_MS, PA_GPU_MIN_KEYS, PA_ANE_SHARE*, ...). The
// phone-attn server reads them via getenv, so setenv before serve().
void apply_env_file(const std::string& path, std::string& note) {
    std::ifstream f(path);
    if (!f) return;
    std::string line, changed;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        setenv(k.c_str(), v.c_str(), 1);
        if (!changed.empty()) changed += ", ";
        changed += k;
    }
    note = changed;
}

}  // namespace

extern "C" {

// -- basics ---------------------------------------------------------------

JNIEXPORT jstring JNICALL
Java_app_backburner_SidecarRpc_cableAddress(JNIEnv* env, jclass) {
    // Same rule as iOS: the USB link-local IPv4 only.
    struct ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return env->NewStringUTF("");
    std::string found;
    for (struct ifaddrs* p = ifs; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        if (strncmp(p->ifa_name, "lo", 2) == 0) continue;
        char buf[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &((struct sockaddr_in*)p->ifa_addr)->sin_addr, buf, sizeof buf);
        std::string ip = buf;
        if (ip.rfind("169.254.", 0) != 0 || ip == "169.254.0.0" ||
            ip.size() >= 4 && ip.compare(ip.size() - 4, 4, ".255") == 0)
            continue;
        found = ip;
        break;
    }
    freeifaddrs(ifs);
    return env->NewStringUTF(found.c_str());
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_gpuStats(JNIEnv* env, jclass) {
    jobject m = hashmap(env);
#ifdef BACKBURNER_HAS_VULKAN
    map_put_str(env, m, "deviceName", "Vulkan (see logcat VkPhysicalDeviceProperties)");
#else
    map_put_str(env, m, "deviceName", "CPU (Vulkan engine not linked)");
#endif
    map_put_long(env, m, "allocatedBytes", 0);
    map_put_long(env, m, "recommendedWorkingSetBytes", 0);
    map_put_str(env, m, "hasUnifiedMemory", "1");
    return m;
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_memoryStats(JNIEnv* env, jclass) {
    struct sysinfo si{};
    sysinfo(&si);
    long long avail = (long long)si.freeram * si.mem_unit;
    long long total = (long long)si.totalram * si.mem_unit;
    // Process footprint from /proc/self/statm (resident pages).
    long long foot = 0;
    if (FILE* f = fopen("/proc/self/statm", "r")) {
        long long size = 0, resident = 0;
        if (fscanf(f, "%lld %lld", &size, &resident) == 2)
            foot = resident * 4096LL;
        fclose(f);
    }
    jobject m = hashmap(env);
    map_put_long(env, m, "availableBytes", avail);
    map_put_long(env, m, "footprintBytes", foot);
    map_put_long(env, m, "physicalBytes", total);
    return m;
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_linkStats(JNIEnv* env, jclass) {
    long long rx = 0, tx = 0;
    // /proc/net/dev equivalent of the iOS if_data loop (skip lo).
    if (FILE* f = fopen("/proc/net/dev", "r")) {
        char line[512];
        // header x2
        if (!fgets(line, sizeof line, f)) { fclose(f); f = nullptr; }
        if (f && !fgets(line, sizeof line, f)) { fclose(f); f = nullptr; }
        while (f && fgets(line, sizeof line, f)) {
            char* colon = strchr(line, ':');
            if (!colon) continue;
            *colon = 0;
            std::string iface = line;
            iface.erase(0, iface.find_first_not_of(" "));
            if (iface.rfind("lo", 0) == 0) continue;
            long long r = 0, t = 0;
            // rx bytes is field 1, tx bytes field 9 after ':'.
            if (sscanf(colon + 1, "%lld %*s %*s %*s %*s %*s %*s %*s %lld", &r, &t) >= 1) {
                rx += r;
                tx += t;
            }
        }
        if (f) fclose(f);
    }
    jobject m = hashmap(env);
    map_put_long(env, m, "rxBytes", rx);
    map_put_long(env, m, "txBytes", tx);
    map_put_long(env, m, "rxPackets", 0);
    map_put_long(env, m, "txPackets", 0);
    return m;
}

JNIEXPORT jstring JNICALL
Java_app_backburner_SidecarRpc_deviceModel(JNIEnv* env, jclass) {
    struct utsname u{};
    uname(&u);
    char buf[256];
    snprintf(buf, sizeof buf, "%s %s", u.machine, u.release);
    return env->NewStringUTF(buf);
}

JNIEXPORT jint JNICALL
Java_app_backburner_SidecarRpc_sme2Available(JNIEnv* env, jclass) {
    (void)env;
    return sme2_available();
}

JNIEXPORT jstring JNICALL
Java_app_backburner_SidecarRpc_envNote(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_mu);
    return env->NewStringUTF(g_env_note.c_str());
}

// -- ggml RPC + tail -------------------------------------------------------
// These need the llama.cpp link (LLAMA_BUILD_DIR). Until wired, they report
// "no model"/unavailable instead of crashing, so the attention path (:50062)
// is usable standalone.

JNIEXPORT jstring JNICALL
Java_app_backburner_SidecarRpc_startHost(JNIEnv* env, jclass, jstring host, jint port, jstring cache) {
    (void)host; (void)port; (void)cache;
    LOGW("startHost: ggml-rpc link not wired (set LLAMA_BUILD_DIR); RPC path disabled");
    std::string e = "ggml-rpc not linked on this build (wire LLAMA_BUILD_DIR in CMakeLists)";
    return env->NewStringUTF(e.c_str());
}

JNIEXPORT jstring JNICALL
Java_app_backburner_SidecarRpc_startTail(JNIEnv* env, jclass, jint port, jstring model) {
    (void)port;
    std::string path = jstr(env, model);
    // Same contract as iOS tailStatus "no model": without the llama link we
    // cannot load tail.gguf yet; report it so the UI shows "No model".
    LOGW("startTail: tail server needs the llama.cpp link; model=%s", path.c_str());
    std::string e = "tail server not linked on this build (wire LLAMA_BUILD_DIR in CMakeLists)";
    return env->NewStringUTF(e.c_str());
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_tailStatus(JNIEnv* env, jclass) {
    jobject m = hashmap(env);
    map_put_str(env, m, "state", "no model");
    map_put_str(env, m, "detail", "link llama.cpp (LLAMA_BUILD_DIR) + push tail.gguf");
    map_put_str(env, m, "model", "");
    map_put_long(env, m, "chunks", 0);
    map_put_long(env, m, "tokens", 0);
    map_put_double(env, m, "lastTokS", 0.0);
    map_put_double(env, m, "loadProgress", 0.0);
    return m;
}

// -- command port :50061 ----------------------------------------------------
// Line protocol, compatible with scripts/phone-push.py + serve.sh phone_note:
//   mem            -> {"avail_mb":..,"sys_wired_mb":..,"pa_ane":false}
//   fetch <url> <dst> -> download url into filesDir/<dst>, reply {"bytes":N}
//   mac PHASE [N1] [N2] [CTX] -> recorded for macStatus()
// Plain TCP (adb reverse or USB link-local), one line per command.

static void cmd_loop(int port, const std::string& files_dir) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(srv, (struct sockaddr*)&a, sizeof a) != 0 || listen(srv, 2) != 0) {
        LOGW("cmd listen :%d failed", port);
        return;
    }
    char buf[4096];
    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        FILE* f = fdopen(fd, "r+");
        if (!f) { close(fd); continue; }
        while (fgets(buf, sizeof buf, f)) {
            std::string line = buf;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line == "mem") {
                struct sysinfo si{};
                sysinfo(&si);
                long long avail = (long long)si.freeram * si.mem_unit / 1048576;
                // sys_wired_mb has no Android equivalent; report MemAvailable as
                // both so serve.sh capacity math stays conservative.
                char r[160];
                snprintf(r, sizeof r, "{\"avail_mb\":%lld,\"sys_wired_mb\":%lld,\"pa_ane\":false}\n",
                         avail, avail);
                fputs(r, f); fflush(f);
            } else if (line.rfind("fetch ", 0) == 0) {
                // fetch <url> <dst>: minimal HTTP GET via /system/bin/curl fallback?
                // v1: shell out to `curl` if present (most userdebug builds).
                char url[1024] = {0}, dst[1024] = {0};
                if (sscanf(line.c_str(), "fetch %1023s %1023s", url, dst) == 2) {
                    std::string out = files_dir + "/" + dst;
                    std::string cmd = std::string("curl -s -o '") + out + "' '" + url + "'";
                    int rc = system(cmd.c_str());
                    FILE* st = fopen(out.c_str(), "rb");
                    long long n = -1;
                    if (st) {
                        fseek(st, 0, SEEK_END); n = ftell(st); fclose(st);
                    }
                    char r[256];
                    if (rc == 0 && n >= 0) snprintf(r, sizeof r, "{\"bytes\":%lld}\n", n);
                    else snprintf(r, sizeof r, "{\"error\":\"fetch failed rc=%d\"}\n", rc);
                    fputs(r, f); fflush(f);
                } else {
                    fputs("{\"error\":\"bad fetch\"}\n", f); fflush(f);
                }
            } else if (line.rfind("mac ", 0) == 0) {
                char phase[64] = {0};
                double n1 = 0, n2 = 0, ctx = 0;
                sscanf(line.c_str(), "mac %63s %lf %lf %lf", phase, &n1, &n2, &ctx);
                std::lock_guard<std::mutex> lk(g_mu);
                if (g_mac_phase != phase) { /* phase change noted */ }
                g_mac_phase = phase;
                g_mac_n1 = n1; g_mac_n2 = n2;
                if (ctx > 0) g_mac_ctx = ctx;
                g_mac_at_ms = now_ms();
                if (!strcmp(phase, "ready")) { g_activations++; g_activating = true; }
                if (!strcmp(phase, "reading")) g_activating = false;
            }
        }
        fclose(f);
    }
}

JNIEXPORT void JNICALL
Java_app_backburner_SidecarRpc_startCmdPort(JNIEnv* env, jclass, jint port) {
    bool expected = false;
    if (!g_cmd_started.compare_exchange_strong(expected, true)) return;
    // filesDir passed via env.txt path convention: use TMPDIR if set.
    const char* tmp = getenv("TMPDIR");
    std::string dir = tmp ? tmp : "/data/local/tmp";
    std::thread([port, dir] { cmd_loop(port, dir); }).detach();
    (void)env;
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_macStatus(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_mu);
    jobject m = hashmap(env);
    map_put_str(env, m, "phase", g_mac_phase);
    map_put_double(env, m, "n1", g_mac_n1);
    map_put_double(env, m, "n2", g_mac_n2);
    map_put_double(env, m, "ctx", g_mac_ctx);
    map_put_double(env, m, "age", g_mac_at_ms ? (now_ms() - g_mac_at_ms) / 1000.0 : 1e9);
    map_put_long(env, m, "activations", g_activations);
    map_put_str(env, m, "activating", g_activating ? "1" : "0");
    return m;
}

// -- phone-attn :50062 -------------------------------------------------------

JNIEXPORT void JNICALL
Java_app_backburner_SidecarRpc_startPhoneAttn(JNIEnv* env, jclass, jint port) {
    bool expected = false;
    if (!g_attn_started.compare_exchange_strong(expected, true)) return;
    // NOTE: env.txt application (PA_* knobs) happens in BackburnerService before
    // this call in the final wiring; the default here keeps built-in behavior.
    static pa::vulkan_engine* gpu = nullptr;
    static pa::nnapi_page_engine* ane = nullptr;
    // v1: GPU stub disabled (begin() false), ANE stub holds no pages.
    // Enable by flipping these once the kernels are ported; no protocol change.
    static const bool kUseGpuStub = false;
    if (kUseGpuStub && !gpu) gpu = new pa::vulkan_engine();
    (void)ane;
    g_attn_status.thermal = [] {
        return 0;  // TODO: PowerManager thermal headroom -> 0..3 like iOS thermalState
    };
    std::thread([port] {
        pa::server srv(&g_attn_status,
                       [](const std::string& s) { LOGW("phone-attn: %s", s.c_str()); },
                       2, gpu);
        // ane intentionally unset in v1 (ready()=0 path inside attn_core).
        std::string e = srv.serve(port);
        LOGW("phone-attn exited: %s", e.c_str());
    }).detach();
    (void)env;
}

JNIEXPORT jobject JNICALL
Java_app_backburner_SidecarRpc_phoneAttnStatus(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_attn_status.mu);
    jobject m = hashmap(env);
    map_put_str(env, m, "state", g_attn_status.state);
    map_put_long(env, m, "calls", (long long)g_attn_status.attn_calls);
    map_put_long(env, m, "heldKeys", (long long)g_attn_status.held_keys);
    map_put_double(env, m, "lastMs", g_attn_status.last_attn_ms);
    return m;
}

}  // extern "C"

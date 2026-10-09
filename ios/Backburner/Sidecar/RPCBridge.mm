#import "RPCBridge.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <vector>
#include <map>
#include <chrono>
#include <string>

#include "ggml-backend.h"
#include "llama.h"
#include "ggml-metal.h"
// shared with the Mac client (llama-split-prefill); both must come from the same llama.cpp tree
#include "../../../llama.cpp/tools/split-prefill/tail-server.h"
#include "../../../llama.cpp/tools/split-prefill/ane-ffn.mm"
#include "../../../phone-attn/phone-attn.h"
#include "../../../phone-attn/pa-metal.mm"
#include "../../../phone-attn/pa-ane.mm"
#include "CableOnly.h"
#import <Security/Security.h>

#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
#import <mach/mach.h>
#import <os/proc.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>
#include <fcntl.h>
#include <unistd.h>

// scripts/sme/sme_attn.c, prebuilt as Sidecar/sme_attn_ios.o (linked via OTHER_LDFLAGS):
//   xcrun -sdk iphoneos clang -c -O3 -target arm64-apple-ios16.4 -mcpu=apple-a18 -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN \
//     scripts/sme/sme_attn.c -o ios/Sidecar/Sidecar/sme_attn_ios.o
extern "C" int sme2_available(void);
extern "C" int sme_bench(int argc, char **argv);

// ---- Wi-Fi pairing (Tunnel.swift / WifiTunnel.swift) ----
// One 32-byte key in this device's Keychain (never in Documents, which Finder and Files can read; never in a backup:
// ThisDeviceOnly). `pair` on :50061 makes a new one and hands it to the Mac, over the cable only; `unpair` deletes it and the
// Wi-Fi listener closes. Without the key nothing on Wi-Fi gets past the tunnel's first message.
static NSDictionary *wifi_key_query(void) {
    return @{ (__bridge id)kSecClass : (__bridge id)kSecClassGenericPassword,
              (__bridge id)kSecAttrService : @"backburner.wifi-tunnel", (__bridge id)kSecAttrAccount : @"pairing-key" };
}
static NSData *wifi_key_load(void) {
    NSMutableDictionary *q = [wifi_key_query() mutableCopy];
    q[(__bridge id)kSecReturnData] = @YES;
    q[(__bridge id)kSecMatchLimit] = (__bridge id)kSecMatchLimitOne;
    CFTypeRef out = NULL;
    if (SecItemCopyMatching((__bridge CFDictionaryRef)q, &out) != errSecSuccess || !out) return nil;
    NSData *d = (__bridge_transfer NSData *)out;
    return d.length == 32 ? d : nil;
}
static NSData *wifi_key_new(void) {
    NSMutableData *k = [NSMutableData dataWithLength:32];
    if (SecRandomCopyBytes(kSecRandomDefault, 32, k.mutableBytes) != errSecSuccess) return nil;
    SecItemDelete((__bridge CFDictionaryRef)wifi_key_query());
    NSMutableDictionary *a = [wifi_key_query() mutableCopy];
    a[(__bridge id)kSecValueData] = k;
    a[(__bridge id)kSecAttrAccessible] = (__bridge id)kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly;
    return SecItemAdd((__bridge CFDictionaryRef)a, NULL) == errSecSuccess ? k : nil;
}
static std::mutex g_wifi_mu;
static std::string g_wifi_status = "not paired";
static thread_local bool g_conn_via_cable = false;   // the control-port connection being served came over the cable

@implementation SidecarRPC

+ (nullable NSData *)wifiKey {
    return wifi_key_load();
}

+ (void)setWifiTunnelStatus:(NSString *)status {
    std::lock_guard<std::mutex> lk(g_wifi_mu);
    g_wifi_status = status.UTF8String ?: "";
}


+ (NSString *)cableAddress {
    // USB NCM gives this phone one link-local IPv4. Wi-Fi, cellular, and every
    // IPv6 address are a different network and must not appear as the endpoint.
    struct ifaddrs *ifs = NULL;
    if (getifaddrs(&ifs) != 0) {
        return @"";
    }
    NSString *found = @"";
    for (struct ifaddrs *p = ifs; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        if (strncmp(p->ifa_name, "lo", 2) == 0) {
            continue;
        }
        char buf[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &((struct sockaddr_in *)p->ifa_addr)->sin_addr, buf, sizeof(buf));
        NSString *ip = [NSString stringWithUTF8String:buf];
        if (![ip hasPrefix:@"169.254."] || [ip hasSuffix:@".255"] || [ip isEqualToString:@"169.254.0.0"]) {
            continue;
        }
        found = ip;
        break;
    }
    freeifaddrs(ifs);
    return found;
}

// The phone's Wi-Fi IPv4 (the interface iOS types as Wi-Fi infrastructure), or empty. Reported over the cable in `mem` so
// scripts/check-phone-exposure.sh can confirm, from the Mac, that the raw ports refuse Wi-Fi.
+ (NSString *)wifiAddress {
    struct ifaddrs *ifs = NULL;
    if (getifaddrs(&ifs) != 0) {
        return @"";
    }
    NSString *found = @"";
    for (struct ifaddrs *p = ifs; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        if (bb::if_functional_type(p->ifa_name) != IFRTYPE_FUNCTIONAL_WIFI_INFRA) {
            continue;
        }
        char buf[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &((struct sockaddr_in *)p->ifa_addr)->sin_addr, buf, sizeof(buf));
        found = [NSString stringWithUTF8String:buf];
        break;
    }
    freeifaddrs(ifs);
    return found;
}

+ (NSDictionary<NSString *, id> *)metalStats {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();   // once: creating it every refresh cost main-thread time
    if (!dev) {
        return @{};
    }
    return @{
        @"deviceName"                 : dev.name ?: @"?",
        @"allocatedBytes"             : @(dev.currentAllocatedSize),
        @"recommendedWorkingSetBytes" : @(dev.recommendedMaxWorkingSetSize),
        @"hasUnifiedMemory"           : @(dev.hasUnifiedMemory),
    };
}

+ (NSDictionary<NSString *, id> *)memoryStats {
    uint64_t footprint = 0;
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        footprint = info.phys_footprint;
    }
    return @{
        // what iOS will still let THIS process allocate before it jetsams us
        @"availableBytes" : @(os_proc_available_memory()),
        @"footprintBytes" : @(footprint),
        @"physicalBytes"  : @([[NSProcessInfo processInfo] physicalMemory]),
    };
}

+ (NSDictionary<NSString *, id> *)linkStats {
    uint64_t rx = 0, tx = 0, rxp = 0, txp = 0;
    struct ifaddrs *ifs = NULL;
    if (getifaddrs(&ifs) == 0) {
        for (struct ifaddrs *p = ifs; p; p = p->ifa_next) {
            if (!p->ifa_addr || p->ifa_addr->sa_family != AF_LINK) {
                continue;
            }
            if (strncmp(p->ifa_name, "lo", 2) == 0) {
                continue;
            }
            const struct if_data *d = (const struct if_data *)p->ifa_data;
            if (!d) {
                continue;
            }
            rx  += d->ifi_ibytes;
            tx  += d->ifi_obytes;
            rxp += d->ifi_ipackets;
            txp += d->ifi_opackets;
        }
        freeifaddrs(ifs);
    }
    return @{ @"rxBytes" : @(rx), @"txBytes" : @(tx), @"rxPackets" : @(rxp), @"txPackets" : @(txp) };
}

// ggml's RPC server has no authentication and lets its client read and write the phone's GPU memory, and it binds a single
// address and blocks in accept, so no filter can be added inside it. It listens on 127.0.0.1:(port + 1000) only, and this
// gate owns the public port on IPv4 and IPv6: every connection must pass the USB-cable check (CableOnly.h) and is then
// spliced to the loopback server, one thread per connection. Wi-Fi goes through the paired, encrypted tunnel instead.
static int connect_local_v4(int port) {
    for (int attempt = 0; attempt < 50; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return -1;
        }
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        usleep(100 * 1000);
    }
    return -1;
}

static void splice_pair(int a, int b) {
    int fds[2] = { a, b };
    while (true) {
        struct pollfd pf[2] = {
            { fds[0], POLLIN, 0 },
            { fds[1], POLLIN, 0 },
        };
        if (poll(pf, 2, -1) < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        bool dead = false;
        for (int i = 0; i < 2; i++) {
            if ((pf[i].revents & (POLLERR | POLLHUP | POLLNVAL)) && !(pf[i].revents & POLLIN)) {
                dead = true;
            }
            if (!(pf[i].revents & POLLIN)) {
                continue;
            }
            char buf[16 * 1024];
            ssize_t n = recv(fds[i], buf, sizeof(buf), 0);
            if (n <= 0) {
                dead = true;
                break;
            }
            ssize_t off = 0;
            while (off < n) {
                ssize_t w = send(fds[i ^ 1], buf + off, (size_t)(n - off), 0);
                if (w <= 0) {
                    dead = true;
                    break;
                }
                off += w;
            }
            if (dead) {
                break;
            }
        }
        if (dead) {
            break;
        }
    }
}

struct rpc_gate_arg { int port, family; };
static constexpr int RPC_INTERNAL_OFFSET = 1000;

static void *rpc_gate_main(void *raw) {
    std::unique_ptr<rpc_gate_arg> arg((rpc_gate_arg *)raw);
    const int port = arg->port, fam = arg->family;
    int srv = socket(fam, SOCK_STREAM, 0);
    if (srv < 0) {
        return nullptr;
    }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    int rc;
    if (fam == AF_INET6) {
        setsockopt(srv, IPPROTO_IPV6, IPV6_V6ONLY, &one, sizeof(one));
        struct sockaddr_in6 addr = {};
        addr.sin6_family = AF_INET6;
        addr.sin6_port = htons((uint16_t)port);
        addr.sin6_addr = in6addr_any;   // cable-gated below
        rc = bind(srv, (struct sockaddr *)&addr, sizeof(addr));
    } else {
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);   // cable-gated below
        rc = bind(srv, (struct sockaddr *)&addr, sizeof(addr));
    }
    if (rc != 0 || listen(srv, 4) != 0) {
        fprintf(stderr, "rpc gate (%s) :%d failed: %s\n", fam == AF_INET6 ? "v6" : "v4", port, strerror(errno));
        close(srv);
        return nullptr;
    }
    fprintf(stderr, "rpc gate (%s) on :%d -> 127.0.0.1:%d, cable only\n", fam == AF_INET6 ? "v6" : "v4", port, port + RPC_INTERNAL_OFFSET);
    while (true) {
        int client = accept(srv, nullptr, nullptr);
        if (client < 0) {
            // a USB replug fails accept (e.g. ECONNABORTED): keep listening instead of leaving the port dead until relaunch
            if (errno != EINTR) { fprintf(stderr, "rpc gate accept: %s (retrying)\n", strerror(errno)); usleep(100000); }
            continue;
        }
        std::string why;
        if (!bb::cable_accept(client, why)) {
            fprintf(stderr, "rpc gate: refused %s\n", why.c_str());
            close(client);
            continue;
        }
        std::thread([client, port] {
            int upstream = connect_local_v4(port + RPC_INTERNAL_OFFSET);
            if (upstream >= 0) {
                splice_pair(client, upstream);
                close(upstream);
            }
            close(client);
        }).detach();
    }
    close(srv);
    return nullptr;
}

+ (NSString *)deviceModel {
    struct utsname u = {};
    if (uname(&u) != 0) {
        return @"?";
    }
    return [NSString stringWithUTF8String:u.machine] ?: @"?";
}

static spt::server_status g_tail_status;
static std::string g_tail_work;   // the last on_work that did something: ANE suspend / tail prewarm ms (under g_tail_status.mu)

// tail FFN on the ANE ("tailane" command): models loaded by the command thread, switched on at the tail's next HELLO
static std::mutex g_tail_ane_mu;
static spt::ane_ffn * g_tail_ane = nullptr;
static int g_tail_ane_n = 0;

// Metal kernel defaults for the phone, set at image load before any ggml/Metal work (a launch environment variable wins).
// GQA-packed prefill attention: the split-prefill tail at 51k depth, A18 L=56 chunk 4.8 -> 3.15 s (2026-09-26).
// Prefill attention on the matrix units (A19 and later; no effect on the A18): kernel_flash_attn_ext_pna (2026-09-26).
// Documents/env.txt (KEY=VALUE lines, # comments) overrides any of these at launch, so a setting can be changed from the
// Mac with phone-push.py + relaunch, no reinstall. Documents/metal/ggml-metal-embed-<kind>.metal replaces that kind's
// built-in Metal kernels (GGML_METAL_KERNELS_DIR; scripts/phone-kernels.sh pushes them).
static char g_env_note[160];   // plain buffer: filled by the constructor below, before C++ globals may exist
__attribute__((constructor)) static void sidecar_env_defaults() {
    setenv("GGML_METAL_FA_PREFILL_GQA", "1", 0);
    // the prefill tail's weights (~5 GB for L40) stay wired while idle unless the residency is ENDED. Released 5 s after the last
    // graph, but ONLY while the ANE page engine wants the memory (phone-held KV: ggml_backend_metal_set_residency_release from
    // on_want_memory): iOS takes ~3 s to wire the tail again (first chunk 4.7 vs ~1.2 s, 2026-09-28), a cost only long
    // contexts should pay
    setenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "5", 0);
    setenv("GGML_METAL_FA_PREFILL_NA", "1", 0);
    @autoreleasepool {
        NSString *docs = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents"];
        NSString *metal = [docs stringByAppendingPathComponent:@"metal"];
        BOOL isDir = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:metal isDirectory:&isDir] && isDir) {
            setenv("GGML_METAL_KERNELS_DIR", metal.UTF8String, 0);
            strlcat(g_env_note, "kernels from Documents/metal", sizeof(g_env_note));
        }
        NSString *txt = [NSString stringWithContentsOfFile:[docs stringByAppendingPathComponent:@"env.txt"] encoding:NSUTF8StringEncoding error:nil];
        int n = 0;
        for (NSString *raw in [txt componentsSeparatedByString:@"\n"]) {
            NSString *line = [raw stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            NSRange eq = [line rangeOfString:@"="];
            if (line.length == 0 || [line hasPrefix:@"#"] || eq.location == NSNotFound || eq.location == 0) continue;
            setenv([line substringToIndex:eq.location].UTF8String, [line substringFromIndex:eq.location + 1].UTF8String, 1);
            n++;
        }
        if (n > 0) {
            char b[64];
            snprintf(b, sizeof(b), "%s%d setting%s from env.txt", g_env_note[0] ? ", " : "", n, n == 1 ? "" : "s");
            strlcat(g_env_note, b, sizeof(g_env_note));
        }
    }
}
static bool g_tail_ane_on = false;
// What the Mac is doing (its proxy and serve-infernet.sh send "mac PHASE [N1] [N2] [CTX]" to :50061): starting, ready,
// reading, thinking, writing, done, stopped. The screen tells its story from this. "ready" (the server is up) also wires
// the tail's weights into GPU memory now, so the first long prompt doesn't wait for it.
static std::mutex g_mac_mu;
static std::string g_mac_phase;
static double g_mac_n1 = 0, g_mac_n2 = 0, g_mac_ctx = 0, g_mac_activate_ms = 0;
static std::chrono::steady_clock::time_point g_mac_at;
static uint64_t g_mac_activations = 0;
static std::atomic<bool> g_mac_activating { false };
static pa::status g_pa_status;   // phone-attn's status (STATS; the mem op reads last_big)
static std::atomic<pa::ane_engine *> g_pa_ane { nullptr };   // phone-attn's ANE page engine (the tail suspends it)

+ (NSString *)startTailPort:(int)port modelPath:(NSString *)modelPath {
    static dispatch_once_t once;
    dispatch_once(&once, ^{ llama_backend_init(); });
    spt::tail_server srv(modelPath.UTF8String, &g_tail_status, [](const std::string & s) {
        fprintf(stderr, "[tail] %s\n", s.c_str());
    }, std::string(NSTemporaryDirectory().UTF8String));
    // the tail runs: unload the ANE page models first (both need memory; the ANE reloads on the next decode). The unload waits
    // only while the tail (its GGUF's size) would not fit under the wired ceiling. The tail's residency is requested on a
    // background queue (PA_TAIL_PREWARM_ASYNC=0: inline), so the SYNC is answered while iOS wires it; its first graph takes the
    // same lock and waits for the rest.
    static double tail_mb = 0;
    if (NSDictionary *fa = [[NSFileManager defaultManager] attributesOfItemAtPath:modelPath error:nil]) tail_mb = [fa fileSize] / 1048576.0;
    srv.on_work = [] {
        static const bool async = !getenv("PA_TAIL_PREWARM_ASYNC") || atoi(getenv("PA_TAIL_PREWARM_ASYNC")) != 0;
        const auto t0 = std::chrono::steady_clock::now();
        pa::ane_engine * a = g_pa_ane.load();
        if (a) a->suspend(tail_mb);
        const double sus_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        auto prewarm = [sus_ms] {
            const auto t1 = std::chrono::steady_clock::now();
            ggml_backend_metal_residency_prewarm();   // start wiring the tail now (a SYNC comes ~1 s before the first chunk)
            const double pw_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            if (sus_ms > 1 || pw_ms > 1) {   // only the calls that did something (the rest just note the time)
                char b[128]; snprintf(b, sizeof b, "suspend %.0f ms, prewarm %.0f ms", sus_ms, pw_ms);
                std::lock_guard<std::mutex> lk(g_tail_status.mu); g_tail_work = b;
            }
        };
        if (async) dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ prewarm(); });
        else prewarm();
    };
    srv.on_hello = [](llama_context * ctx, int n_layer_tail) -> std::string {
        std::lock_guard<std::mutex> lk(g_tail_ane_mu);
        // the ANE may cover only the first N tail layers (split-gguf.py --no-ffn-first N): the rest keep their FFN on the GPU
        if (g_tail_ane_on && g_tail_ane && g_tail_ane_n > 0 && g_tail_ane_n <= n_layer_tail) {
            llama_set_ffn_offload(ctx, 0, g_tail_ane_n, spt::ane_ffn_run, g_tail_ane);
            return g_tail_ane_n == n_layer_tail ? std::string(" +ANE FFN") : " +ANE FFN x" + std::to_string(g_tail_ane_n);
        }
        llama_set_ffn_offload(ctx, 0, 0, nullptr, nullptr);
        char buf[16];
        if (llama_model_meta_val_str(llama_get_model(ctx), "split.no_ffn", buf, sizeof(buf)) > 0) {
            return "!this tail GGUF has no FFN weights (split-gguf.py --no-ffn): load them on the ANE first (tailane DIR 44 20)";
        }
        return "";
    };
    srv.set_accept_filter(bb::cable_accept);   // the USB cable (and the in-app Wi-Fi tunnel on loopback) only
    // A missing model is not fatal: HELLO retries the load, so the user can copy tail.gguf later.
    srv.load();
    std::string err = srv.serve(port);
    g_tail_status.set("error", err);
    return [NSString stringWithUTF8String:err.c_str()];
}

+ (NSDictionary<NSString *, id> *)tailStatus {
    std::lock_guard<std::mutex> lk(g_tail_status.mu);
    return @{
        @"state"       : [NSString stringWithUTF8String:g_tail_status.state.c_str()] ?: @"",
        @"detail"      : [NSString stringWithUTF8String:g_tail_status.detail.c_str()] ?: @"",
        @"model"       : [NSString stringWithUTF8String:g_tail_status.model_desc.c_str()] ?: @"",
        @"chunks"      : @(g_tail_status.chunks),
        @"tokens"      : @(g_tail_status.tokens),
        @"sessions"    : @(g_tail_status.sessions),
        @"lastChunkMs" : @(g_tail_status.last_chunk_ms),
        @"lastTokS"    : @(g_tail_status.last_tok_s),
        @"loadProgress": @(g_tail_status.load_progress),
        @"modelBytes"  : @(g_tail_status.model_bytes),
    };
}

+ (NSString *)startHost:(NSString *)host port:(int)port cacheDir:(NSString *)cacheDir {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        for (int fam : { AF_INET, AF_INET6 }) {
            auto *arg = new rpc_gate_arg{ port, fam };
            pthread_t thread;
            if (pthread_create(&thread, nullptr, rpc_gate_main, arg) == 0) {
                pthread_detach(thread);
            } else {
                delete arg;
            }
        }
    });

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        // Expose only the GPU by default. BLAS (Accelerate) used to be exposed too, but a default
        // `--rpc` client then splits layers onto it, sends RMS_NORM, and the app aborts
        // (ggml_backend_blas_graph_compute: unsupported op). Set SIDECAR_RPC_BLAS=1 to expose it.
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU || (t == GGML_BACKEND_DEVICE_TYPE_ACCEL && getenv("SIDECAR_RPC_BLAS"))) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu) {
            devices.push_back(cpu);
        }
    }
    if (devices.empty()) {
        return @"no ggml devices (Metal missing?)";
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        return @"RPC backend not linked — rebuild with GGML_RPC=ON";
    }
    using start_fn = void (*)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *);
    auto fn = (start_fn)ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!fn) {
        return @"ggml_backend_rpc_start_server missing";
    }

    // loopback only, whatever `host` says: the gate above is the only way in (see rpc_gate_main)
    (void)host;
    std::string endpoint = "127.0.0.1:" + std::to_string(port + RPC_INTERNAL_OFFSET);
    unsigned n_threads = std::max(1u, (unsigned)[[NSProcessInfo processInfo] processorCount] / 2);
    fn(endpoint.c_str(), cacheDir.UTF8String, n_threads, devices.size(), devices.data());
    return @"rpc server returned";
}

@end

// ---------------------------------------------------------------------------------------------
// ANE bench (:50061). Line protocol, one JSON line per reply. Loads compiled CoreML models
// (.mlmodelc) from Documents and times prediction. For measuring the phone's Neural Engine only;
// nothing in the prefill path uses it yet. Commands:
//   ls | mem | load ALIAS FILE [ne|gpu|cpu|all] | run ALIAS N | loop ALIAS SECS | plan FILE [units]
//   | unload ALIAS|all | pred ALIAS IN=B64.. (one prediction on given fp16 inputs, outputs back as fp32 base64)
// ---------------------------------------------------------------------------------------------
static uint64_t ane_footprint() {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) return info.phys_footprint;
    return 0;
}
static NSMutableDictionary *ane_mem() {
    // thermal: 0 nominal, 1 fair, 2 serious, 3 critical (NSProcessInfoThermalState); label every timing with it
    NSMutableDictionary *r = [@{ @"footprint_mb" : @(ane_footprint() / 1048576.0), @"avail_mb" : @(os_proc_available_memory() / 1048576.0),
               @"thermal" : @((int) [NSProcessInfo processInfo].thermalState) } mutableCopy];
    // what the Mac needs to size this device: its RAM (an 8 GB iPad can report an app budget above its RAM) and its model
    // identifier (iPhone18,2 / iPad16,3: scripts name the device and pick its profile from it)
    r[@"phys_mb"] = @([NSProcessInfo processInfo].physicalMemory / 1048576.0);
    r[@"machine"] = [SidecarRPC deviceModel];
    // system-wide pages: the ANE's weights and Metal's resident buffers are wired but not in the app footprint, and
    // jetsam kills on system page shortage (~10.5 GB wired on the 12 GB A19)
    vm_statistics64_data_t vs; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) == KERN_SUCCESS) {
        const double pg = (double) vm_kernel_page_size / 1048576.0;
        r[@"sys_wired_mb"] = @(vs.wire_count * pg);
        r[@"sys_free_mb"]  = @(vs.free_count * pg);
        r[@"sys_compressed_mb"] = @(vs.compressor_page_count * pg);
    }
    return r;
}
static MLComputeUnits ane_units(NSString *u) {
    if ([u isEqualToString:@"gpu"]) return MLComputeUnitsCPUAndGPU;
    if ([u isEqualToString:@"cpu"]) return MLComputeUnitsCPUOnly;
    if ([u isEqualToString:@"all"]) return MLComputeUnitsAll;
    return MLComputeUnitsCPUAndNeuralEngine;
}
static NSURL *ane_doc(NSString *f) {
    NSURL *d = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask][0];
    return [d URLByAppendingPathComponent:f];
}

struct ane_model { MLModel *m; id<MLFeatureProvider> in; double flops; int S; };
static std::map<std::string, ane_model> g_ane;

// small random fp16 values for EVERY input (a model with a second input, e.g. ane-kv's score offset c, used to fail to run);
// FLOPs / S come from the first input
static id<MLFeatureProvider> ane_input(MLModel *m, double *flops, int *S_out) {
    NSDictionary *ins = m.modelDescription.inputDescriptionsByName;
    NSMutableDictionary *feed = [NSMutableDictionary dictionary];
    uint32_t r = 12345;
    for (NSString *name in [ins.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
        NSArray<NSNumber *> *shape = ((MLFeatureDescription *)ins[name]).multiArrayConstraint.shape;
        MLMultiArray *x = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:nil];
        __fp16 *p = (__fp16 *)x.dataPointer;
        for (NSInteger i = 0; i < x.count; i++) { r = r * 1664525u + 1013904223u; p[i] = (__fp16)(((int)(r >> 9) % 2001 - 1000) * 1e-4f); }
        feed[name] = x;
    }
    NSArray<NSNumber *> *shape = ((MLFeatureDescription *)ins[ins.allKeys.firstObject]).multiArrayConstraint.shape;
    int C = shape.count > 1 ? shape[1].intValue : 0, S = shape.lastObject.intValue;
    *S_out = S;
    *flops = C == 5120 ? 2.0 * 3 * 5120 * 17408 * S : 0;  // 27B FFN shape; 0 = unknown model
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:nil];
}

static NSDictionary *ane_cmd(NSArray<NSString *> *a) {
    NSString *op = a.count ? a[0] : @"";
    NSMutableDictionary *r = ane_mem();
    // every argument that names a file must stay inside Documents, and fetch may only download from the Mac over the cable
    // (CableOnly.h; tests/security/cable-policy-test.cpp)
    {
        int path_arg = -1;
        if ([op isEqualToString:@"fetch"] && a.count >= 3) path_arg = 2;
        if ([op isEqualToString:@"load"] && a.count >= 3) path_arg = 2;
        if ([op isEqualToString:@"plan"] && a.count >= 2) path_arg = 1;
        if ([op isEqualToString:@"tailane"] && a.count >= 4) path_arg = 1;
        if (path_arg >= 0 && !bb::safe_doc_path(std::string(a[path_arg].UTF8String ?: ""))) {
            r[@"error"] = @"refused: the path must be relative to Documents ([A-Za-z0-9._-] names, no '..' or hidden names)";
            return r;
        }
        if ([op isEqualToString:@"fetch"] && a.count >= 3 && !bb::fetch_url_ok(std::string(a[1].UTF8String ?: ""))) {
            r[@"error"] = @"refused: fetch downloads only from http://169.254.x.y (the Mac on the cable)";
            return r;
        }
    }
    r[@"op"] = op;
    // pair: a new Wi-Fi pairing key, returned once, over the cable only (scripts/phone-wifi.sh pair). unpair: delete it.
    if ([op isEqualToString:@"pair"] || [op isEqualToString:@"unpair"]) {
        if (!g_conn_via_cable) { r[@"error"] = @"refused: pairing works over the USB cable only"; return r; }
        if ([op isEqualToString:@"unpair"]) {
            SecItemDelete((__bridge CFDictionaryRef)wifi_key_query());
            r[@"paired"] = @NO;
            return r;
        }
        NSData *k = wifi_key_new();
        if (!k) { r[@"error"] = @"could not store a key in the Keychain"; return r; }
        return @{ @"op" : op, @"paired" : @YES, @"wifi_key" : [k base64EncodedStringWithOptions:0], @"wifi_port" : @50070,
                  @"wifi_ip" : [SidecarRPC wifiAddress] };
    }
    if ([op isEqualToString:@"mac"] && a.count >= 2) {
        bool activate = false;
        {
            std::lock_guard<std::mutex> lk(g_mac_mu);
            g_mac_phase = a[1].UTF8String;
            g_mac_n1  = a.count > 2 ? a[2].doubleValue : 0;
            g_mac_n2  = a.count > 3 ? a[3].doubleValue : 0;
            if (a.count > 4) g_mac_ctx = a[4].doubleValue;
            g_mac_at = std::chrono::steady_clock::now();
            if (g_mac_phase == "ready") { g_mac_activations++; activate = true; }
        }
        if (activate && !g_mac_activating.exchange(true)) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_metal_residency_prewarm();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                { std::lock_guard<std::mutex> lk(g_mac_mu); g_mac_activate_ms = ms; }
                g_mac_activating = false;
            });
        }
        r[@"mac"] = a[1];
        return r;
    }
    if ([op isEqualToString:@"mem"]) {
        std::lock_guard<std::mutex> lk(g_tail_status.mu);
        r[@"tail_sync"] = [NSString stringWithUTF8String:g_tail_status.last_sync.c_str()] ?: @"";
        r[@"tail_last_chunk_ms"] = @(g_tail_status.last_chunk_ms);
        r[@"tail_work"] = [NSString stringWithUTF8String:g_tail_work.c_str()] ?: @"";
        r[@"tail_state"] = [NSString stringWithUTF8String:g_tail_status.state.c_str()] ?: @"";
        r[@"tail_detail"] = [NSString stringWithUTF8String:g_tail_status.detail.c_str()] ?: @"";
        r[@"tail_chunks"] = @(g_tail_status.chunks);
        r[@"tail_sessions"] = @(g_tail_status.sessions);
        // phone-attn's ANE pages and prefill-call totals here too: this port answers while a llama-server holds phone-attn's
        // connection threads (a STATS on :50062 then waits)
        if (pa::ane_engine * a = g_pa_ane.load()) r[@"pa_ane"] = [NSString stringWithUTF8String:a->describe().c_str()] ?: @"";
        { std::lock_guard<std::mutex> lk2(g_pa_status.mu); r[@"pa_big"] = [NSString stringWithUTF8String:g_pa_status.last_big.c_str()] ?: @"";
          r[@"pa_dec"] = [NSString stringWithUTF8String:g_pa_status.last_dec.c_str()] ?: @"";
          r[@"pa_calls"] = @(g_pa_status.attn_calls); r[@"pa_held"] = @(g_pa_status.held_keys); }
        // the connection gate (CableOnly.h): scripts/check-phone-exposure.sh reads these
        { auto & g = bb::stats(); std::lock_guard<std::mutex> lk3(g.mu);
          r[@"gate_allowed"] = @(g.allowed.load()); r[@"gate_denied"] = @(g.denied.load());
          r[@"gate_last_denied"] = [NSString stringWithUTF8String:g.last_denied.c_str()] ?: @"";
          r[@"cable_if"] = [NSString stringWithUTF8String:g.cable_if.c_str()] ?: @"";
          r[@"cable_if_type"] = @(g.cable_if_type == ~0u ? -1 : (int) g.cable_if_type); }
        r[@"wifi_ip"] = [SidecarRPC wifiAddress];
        r[@"wifi_paired"] = @(wifi_key_load() != nil);
        { std::lock_guard<std::mutex> lk4(g_wifi_mu); r[@"wifi_tunnel"] = [NSString stringWithUTF8String:g_wifi_status.c_str()] ?: @""; }
        return r;
    }
    // fetch URL DEST: download URL (the Mac, over the USB link) to Documents/DEST, creating directories
    if ([op isEqualToString:@"fetch"] && a.count >= 3) {
        NSURL *dst = ane_doc(a[2]);
        [[NSFileManager defaultManager] createDirectoryAtURL:[dst URLByDeletingLastPathComponent] withIntermediateDirectories:YES attributes:nil error:nil];
        auto t0 = std::chrono::steady_clock::now();
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block NSString *ferr = nil;
        __block long long bytes = 0;
        NSURLSessionDownloadTask *t = [[NSURLSession sharedSession] downloadTaskWithURL:[NSURL URLWithString:a[1]]
            completionHandler:^(NSURL *loc, NSURLResponse *resp, NSError *e) {
                NSInteger code = [resp isKindOfClass:[NSHTTPURLResponse class]] ? ((NSHTTPURLResponse *)resp).statusCode : 0;
                if (e || !loc || code != 200) { ferr = e.localizedDescription ?: [NSString stringWithFormat:@"HTTP %ld", (long)code]; }
                else {
                    [[NSFileManager defaultManager] removeItemAtURL:dst error:nil];
                    NSError *me = nil;
                    if (![[NSFileManager defaultManager] moveItemAtURL:loc toURL:dst error:&me]) ferr = me.localizedDescription ?: @"move failed";
                    else bytes = [[[NSFileManager defaultManager] attributesOfItemAtPath:dst.path error:nil] fileSize];
                }
                dispatch_semaphore_signal(sem);
            }];
        [t resume];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (ferr) r[@"error"] = ferr;
        r[@"bytes"] = @(bytes);
        r[@"ms"] = @(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        return r;
    }
    // tailane DIR IL_OFFSET N_LAYERS: load Documents/DIR/ffn_L<il+offset>.mlmodelc for the tail's layers [0, N) and run the
    // tail's FFN on the ANE from the next HELLO on. tailane off: back to the GPU (models stay loaded). tailane unload.
    if ([op isEqualToString:@"tailane"] && a.count >= 2) {
        if ([a[1] isEqualToString:@"off"] || [a[1] isEqualToString:@"unload"]) {
            std::lock_guard<std::mutex> lk(g_tail_ane_mu);
            g_tail_ane_on = false;
            if ([a[1] isEqualToString:@"unload"]) { spt::ane_ffn_close(g_tail_ane); g_tail_ane = nullptr; g_tail_ane_n = 0; }
            r[@"tailane"] = @"off";
            return r;
        }
        if ([a[1] isEqualToString:@"on"]) {
            std::lock_guard<std::mutex> lk(g_tail_ane_mu);
            g_tail_ane_on = g_tail_ane != nullptr;
            r[@"tailane"] = g_tail_ane_on ? @"on" : @"not loaded";
            return r;
        }
        if (a.count < 4) { r[@"error"] = @"usage: tailane DIR IL_OFFSET N | on | off | unload"; return r; }
        const int off = a[2].intValue, n = a[3].intValue;
        auto t0 = std::chrono::steady_clock::now();
        std::string err;
        spt::ane_ffn * m = spt::ane_ffn_open(std::string(ane_doc(a[1]).path.UTF8String), 0, n, off, err);
        r[@"load_ms"] = @(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (!m) { r[@"error"] = [NSString stringWithUTF8String:err.c_str()]; return r; }
        {
            std::lock_guard<std::mutex> lk(g_tail_ane_mu);
            spt::ane_ffn_close(g_tail_ane);
            g_tail_ane = m; g_tail_ane_n = n; g_tail_ane_on = true;
        }
        NSMutableDictionary *r2 = ane_mem();
        [r2 addEntriesFromDictionary:@{ @"op" : op, @"tailane" : @"on", @"layers" : @(n), @"load_ms" : r[@"load_ms"] }];
        return r2;
    }
    if ([op isEqualToString:@"tailane-stats"]) {
        std::lock_guard<std::mutex> lk(g_tail_ane_mu);
        const auto st = spt::ane_ffn_get_stats(g_tail_ane);
        r[@"calls"] = @(st.calls); r[@"blocks"] = @(st.blocks); r[@"ms_total"] = @(st.ms_total); r[@"ms_pred"] = @(st.ms_pred);
        r[@"last_error"] = [NSString stringWithUTF8String:spt::ane_ffn_last_error(g_tail_ane).c_str()];
        return r;
    }
    if ([op isEqualToString:@"ls"]) {
        NSURL *d = ane_doc(@"");
        r[@"files"] = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:d.path error:nil] ?: @[];
        return r;
    }
    if ([op isEqualToString:@"load"] && a.count >= 3) {
        MLModelConfiguration *cfg = [MLModelConfiguration new];
        cfg.computeUnits = ane_units(a.count > 3 ? a[3] : @"ne");
        double fp0 = ane_footprint() / 1048576.0, av0 = os_proc_available_memory() / 1048576.0;
        NSError *err = nil;
        auto t0 = std::chrono::steady_clock::now();
        MLModel *m = [MLModel modelWithContentsOfURL:ane_doc(a[2]) configuration:cfg error:&err];
        double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!m) { r[@"error"] = err.localizedDescription ?: @"load failed"; return r; }
        ane_model e = { m, nil, 0, 0 };
        e.in = ane_input(m, &e.flops, &e.S);
        double fp1 = ane_footprint() / 1048576.0, av1 = os_proc_available_memory() / 1048576.0;
        t0 = std::chrono::steady_clock::now();
        id<MLFeatureProvider> o = [m predictionFromFeatures:e.in error:&err];
        double first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!o) r[@"predict_error"] = err.localizedDescription ?: @"?";
        g_ane[a[1].UTF8String] = e;
        [r addEntriesFromDictionary:@{ @"load_ms" : @(load_ms), @"first_ms" : @(first_ms), @"S" : @(e.S),
            @"fp_before_mb" : @(fp0), @"fp_loaded_mb" : @(fp1), @"avail_before_mb" : @(av0), @"avail_loaded_mb" : @(av1) }];
        return r;
    }
    if (([op isEqualToString:@"run"] || [op isEqualToString:@"loop"]) && a.count >= 3) {
        auto it = g_ane.find(a[1].UTF8String);
        if (it == g_ane.end()) { r[@"error"] = @"not loaded"; return r; }
        ane_model &e = it->second;
        bool loop = [op isEqualToString:@"loop"];
        double lim = a[2].doubleValue;
        std::vector<double> ts;
        std::vector<int> per_sec;
        auto start = std::chrono::steady_clock::now();
        for (int w = 0; w < (loop ? 0 : 3); w++) @autoreleasepool { [e.m predictionFromFeatures:e.in error:nil]; }
        while (true) {
            @autoreleasepool {
                auto t0 = std::chrono::steady_clock::now();
                NSError *err = nil;
                if (![e.m predictionFromFeatures:e.in error:&err]) { r[@"error"] = err.localizedDescription ?: @"?"; return r; }
                auto t1 = std::chrono::steady_clock::now();
                ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                double el = std::chrono::duration<double>(t1 - start).count();
                if (loop) {
                    size_t sec = (size_t)el;
                    if (per_sec.size() <= sec) per_sec.resize(sec + 1, 0);
                    per_sec[sec]++;
                    if (el >= lim) break;
                } else if ((double)ts.size() >= lim) break;
            }
        }
        std::vector<double> s = ts;
        std::sort(s.begin(), s.end());
        double med = s[s.size() / 2];
        NSMutableArray *ps = [NSMutableArray array];
        for (int c : per_sec) [ps addObject:@(c)];
        [r addEntriesFromDictionary:@{ @"n" : @(ts.size()), @"med_ms" : @(med), @"min_ms" : @(s.front()), @"max_ms" : @(s.back()),
            @"p90_ms" : @(s[s.size() * 9 / 10]), @"S" : @(e.S), @"tflops_med" : @(e.flops / (med * 1e-3) / 1e12), @"per_sec" : ps }];
        if (loop) {  // TFLOPS per second-bucket, for before/during/after comparisons
            NSMutableArray *tf = [NSMutableArray array];
            for (int c : per_sec) [tf addObject:@(c * e.flops / 1e12)];
            r[@"tflops_per_sec"] = tf;
        }
        return r;
    }
    // pred NAME IN=BASE64 [IN2=BASE64 ..]: one prediction of a loaded model on the Mac's inputs (raw fp16, C order, every input),
    // returns each output as {shape, b64 = raw fp32 C order} and the call time. For accuracy checks (phone-attn/ane-kv).
    if ([op isEqualToString:@"pred"] && a.count >= 3) {
        auto it = g_ane.find(a[1].UTF8String);
        if (it == g_ane.end()) { r[@"error"] = @"not loaded"; return r; }
        MLModel *m = it->second.m;
        NSDictionary *ins = m.modelDescription.inputDescriptionsByName;
        NSMutableDictionary *feed = [NSMutableDictionary dictionary];
        for (NSUInteger i = 2; i < a.count; i++) {
            NSRange eq = [a[i] rangeOfString:@"="];
            if (eq.location == NSNotFound) continue;
            NSString *nm = [a[i] substringToIndex:eq.location];
            NSData *d = [[NSData alloc] initWithBase64EncodedString:[a[i] substringFromIndex:eq.location + 1] options:0];
            MLFeatureDescription *fd = ins[nm];
            if (!fd || !d) { r[@"error"] = [NSString stringWithFormat:@"bad input %@", nm]; return r; }
            MLMultiArray *x = [[MLMultiArray alloc] initWithShape:fd.multiArrayConstraint.shape dataType:MLMultiArrayDataTypeFloat16 error:nil];
            if ((NSInteger)d.length != x.count * 2) {
                r[@"error"] = [NSString stringWithFormat:@"input %@: %lu bytes, want %ld", nm, (unsigned long)d.length, (long)x.count * 2];
                return r;
            }
            memcpy(x.dataPointer, d.bytes, d.length);   // a fresh MLMultiArray is contiguous, C order
            feed[nm] = x;
        }
        if (feed.count != ins.count) { r[@"error"] = [NSString stringWithFormat:@"need all %lu inputs", (unsigned long)ins.count]; return r; }
        NSError *err = nil;
        MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&err];
        auto t0 = std::chrono::steady_clock::now();
        id<MLFeatureProvider> o = fp ? [m predictionFromFeatures:fp error:&err] : nil;
        r[@"ms"] = @(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (!o) { r[@"error"] = err.localizedDescription ?: @"predict failed"; return r; }
        NSMutableDictionary *outs = [NSMutableDictionary dictionary];
        for (NSString *on in o.featureNames) {
            MLMultiArray *y = [o featureValueForName:on].multiArrayValue;
            if (!y) continue;
            const NSInteger n = y.count, nd = (NSInteger)y.shape.count;
            std::vector<NSInteger> shp(nd), str(nd), idx(nd, 0);
            for (NSInteger k = 0; k < nd; k++) { shp[k] = y.shape[k].integerValue; str[k] = y.strides[k].integerValue; }
            std::vector<float> v(n);
            const char *base = (const char *)y.dataPointer;   // outputs can be padded (strides), so walk them
            for (NSInteger k = 0; k < n; k++) {
                NSInteger off = 0;
                for (NSInteger j = 0; j < nd; j++) off += idx[j] * str[j];
                switch (y.dataType) {
                    case MLMultiArrayDataTypeFloat16: v[k] = (float)((const __fp16 *)base)[off]; break;
                    case MLMultiArrayDataTypeFloat32: v[k] = ((const float *)base)[off]; break;
                    case MLMultiArrayDataTypeDouble:  v[k] = (float)((const double *)base)[off]; break;
                    default: v[k] = ((const int32_t *)base)[off]; break;
                }
                for (NSInteger j = nd - 1; j >= 0; j--) { if (++idx[j] < shp[j]) break; idx[j] = 0; }
            }
            outs[on] = @{ @"shape" : y.shape,
                          @"b64" : [[NSData dataWithBytes:v.data() length:(NSUInteger)n * 4] base64EncodedStringWithOptions:0] };
        }
        r[@"outputs"] = outs;
        return r;
    }
    if ([op isEqualToString:@"unload"] && a.count >= 2) {
        if ([a[1] isEqualToString:@"all"]) g_ane.clear(); else g_ane.erase(a[1].UTF8String);
        return ane_mem();
    }
    if ([op isEqualToString:@"plan"] && a.count >= 2) {
        if (@available(iOS 17.4, *)) {
            MLModelConfiguration *cfg = [MLModelConfiguration new];
            cfg.computeUnits = ane_units(a.count > 2 ? a[2] : @"ne");
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block NSMutableArray *ops = [NSMutableArray array];
            __block NSString *perr = nil;
            [MLComputePlan loadContentsOfURL:ane_doc(a[1]) configuration:cfg completionHandler:^(MLComputePlan *plan, NSError *err) {
                if (!plan) { perr = err.localizedDescription ?: @"plan failed"; dispatch_semaphore_signal(sem); return; }
                MLModelStructureProgram *prog = plan.modelStructure.program;
                MLModelStructureProgramFunction *fn = prog.functions[@"main"];
                for (MLModelStructureProgramOperation *o in fn.block.operations) {
                    MLComputePlanDeviceUsage *u = [plan computeDeviceUsageForMLProgramOperation:o];
                    MLComputePlanCost *c = [plan estimatedCostOfMLProgramOperation:o];
                    NSMutableArray *sup = [NSMutableArray array];
                    for (id<MLComputeDeviceProtocol> d in u.supportedComputeDevices) [sup addObject:NSStringFromClass([(NSObject *)d class])];
                    [ops addObject:@{ @"op" : o.operatorName ?: @"?",
                        @"device" : u ? NSStringFromClass([(NSObject *)u.preferredComputeDevice class]) : @"-",
                        @"supported" : sup, @"cost" : @(c ? c.weight : -1) }];
                }
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 120LL * NSEC_PER_SEC));
            if (perr) r[@"error"] = perr; else r[@"ops"] = ops;
        } else {
            r[@"error"] = @"MLComputePlan needs iOS 17.4";
        }
        NSMutableArray *devs = [NSMutableArray array];
        if (@available(iOS 17.0, *)) {
            for (id<MLComputeDeviceProtocol> d in MLAllComputeDevices()) {
                NSString *s = NSStringFromClass([(NSObject *)d class]);
                if ([(NSObject *)d isKindOfClass:[MLNeuralEngineComputeDevice class]])
                    s = [s stringByAppendingFormat:@" cores=%ld", (long)((MLNeuralEngineComputeDevice *)d).totalCoreCount];
                [devs addObject:s];
            }
        }
        r[@"devices"] = devs;
        return r;
    }
    if ([op isEqualToString:@"sme"]) {
        // sme                -> the CPU feature sysctls only (safe on any phone)
        // sme ARGS...        -> sme_bench(ARGS) from scripts/sme/sme_attn.c (e.g. "sme peak 0 1", "sme attn 16384 1 1 5 512 1");
        //                       sme_bench refuses to run without FEAT_SME2 + 512-bit SVL, so no SIGILL
        NSMutableDictionary *hw = [NSMutableDictionary dictionary];
        for (const char *k : { "hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2", "hw.optional.arm.FEAT_SME_F64F64",
                               "hw.optional.arm.FEAT_SME_I16I64", "hw.optional.arm.sme_max_svl_b", "hw.optional.arm.FEAT_BF16",
                               "hw.optional.arm.FEAT_I8MM", "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu" }) {
            int v = -1; size_t n = sizeof v;
            if (sysctlbyname(k, &v, &n, nullptr, 0) != 0) v = -1;
            hw[[NSString stringWithUTF8String:k]] = @(v);
        }
        r[@"sysctl"] = hw;
        r[@"sme2_available"] = @(sme2_available());
        if (a.count > 1) {
            std::vector<std::string> sargs = { "sme" };
            for (NSUInteger i = 1; i < a.count; i++) if (a[i].length) sargs.push_back(a[i].UTF8String);
            std::vector<char *> argv;
            for (auto & s : sargs) argv.push_back(s.data());
            // sme_bench prints to stdout: capture it through a temp file
            NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:@"sme_bench.txt"];
            fflush(stdout);
            int saved = dup(STDOUT_FILENO);
            int tfd = open(tmp.UTF8String, O_CREAT | O_TRUNC | O_WRONLY, 0644);
            dup2(tfd, STDOUT_FILENO); close(tfd);
            auto t0 = std::chrono::steady_clock::now();
            int rc = sme_bench((int)argv.size(), argv.data());
            fflush(stdout);
            dup2(saved, STDOUT_FILENO); close(saved);
            r[@"rc"] = @(rc);
            r[@"wall_ms"] = @(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            r[@"out"] = [NSString stringWithContentsOfFile:tmp encoding:NSUTF8StringEncoding error:nil] ?: @"";
        }
        return r;
    }
    r[@"error"] = @"usage: ls | mem | load ALIAS FILE [ne|gpu|cpu|all] | run ALIAS N | loop ALIAS SECS | plan FILE [units] | unload ALIAS|all | sme [ARGS]";
    return r;
}


static void *ane_server_main(void *raw) {
    int port = (int)(intptr_t)raw;
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);   // cable-gated after accept
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(srv, 2) != 0) {
        fprintf(stderr, "ane bench listen :%d failed: %s\n", port, strerror(errno));
        return nullptr;
    }
    fprintf(stderr, "ane bench on :%d\n", port);
    while (true) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) {   // survive a USB replug (see the rpc gate)
            if (errno != EINTR) { fprintf(stderr, "ane bench accept: %s (retrying)\n", strerror(errno)); usleep(100000); }
            continue;
        }
        {
            std::string why;
            if (!bb::cable_accept(fd, why)) {   // the USB cable (and the in-app Wi-Fi tunnel on loopback) only
                fprintf(stderr, "control port: refused %s\n", why.c_str());
                close(fd);
                continue;
            }
            // pair / unpair need the cable itself, not the Wi-Fi tunnel (which arrives on loopback)
            g_conn_via_cable = why.size() >= 7 && why.compare(why.size() - 7, 7, ": cable") == 0;
        }
        std::string buf;
        char c;
        while (recv(fd, &c, 1, 0) == 1) {
            if (c != '\n') { buf.push_back(c); continue; }
            @autoreleasepool {
                NSString *line = [[NSString stringWithUTF8String:buf.c_str()] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
                NSArray *parts = [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                NSDictionary *res = ane_cmd(parts);
                NSData *js = [NSJSONSerialization dataWithJSONObject:res options:0 error:nil];
                std::string out((const char *)js.bytes, js.length);
                out.push_back('\n');
                send(fd, out.data(), out.size(), 0);
            }
            buf.clear();
        }
        close(fd);
    }
    close(srv);
    return nullptr;
}

@implementation SidecarRPC (ANE)
+ (void)startPhoneAttnPort:(int)port {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        std::thread([port] {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            g_pa_status.thermal = [] { return (int) [NSProcessInfo processInfo].thermalState; };
            auto * eng = new pa::metal_engine();
            if (!eng->init()) {
                fprintf(stderr, "[phone-attn] GPU engine off: %s\n", eng->error.c_str());
                std::lock_guard<std::mutex> lk(g_pa_status.mu);
                g_pa_status.state = "gpu off: " + eng->error;
                delete eng; eng = nullptr;
            } else {
                std::lock_guard<std::mutex> lk(g_pa_status.mu);
                g_pa_status.state = "gpu on";
            }
            pa::server srv(&g_pa_status, [](const std::string & s) { fprintf(stderr, "[phone-attn] %s\n", s.c_str()); }, 2, eng);
            srv.set_accept_filter(bb::cable_accept);   // the USB cable (and the in-app Wi-Fi tunnel on loopback) only
            // ANE page engine (docs/ANE.md): on when Documents/anekv/tmpl16k.mlmodelc exists (scripts/phone-ane.sh
            // pushes it) and PA_ANE != 0. PA_ANE_MB (default 6000): the most page-model memory (67 MB per 16k keys x layer); in practice the wired-memory guard
            // (PA_WIRED_MAX_MB 9400 - 800) decides: 3-4 pages per layer. 140k (75.5k phone keys): 3 pages 176 ms/token, 2 pages 208, none 279.
            NSString *tmpl = [[NSHomeDirectory() stringByAppendingPathComponent:@"Documents/anekv"] stringByAppendingPathComponent:@"tmpl16k.mlmodelc"];
            const char *pa_ane = getenv("PA_ANE");
            if ((!pa_ane || atoi(pa_ane) != 0) && [[NSFileManager defaultManager] fileExistsAtPath:tmpl]) {
                NSString *cache = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject
                                   stringByAppendingPathComponent:@"anekv"];
                const size_t mb = getenv("PA_ANE_MB") ? (size_t) atoll(getenv("PA_ANE_MB")) : 6000;
                auto *ane = new pa::ane_engine();
                if (ane->init(tmpl.UTF8String, cache.UTF8String, mb)) {
                    ane->on_want_memory = [](bool want) { ggml_backend_metal_set_residency_release(want); };
                    // decode started (the prefill is over): unwire the tail now, not after its 5 s keep-alive, so the pages fit
                    ane->on_wake = [] { ggml_backend_metal_residency_release_now(); };
                    srv.set_page_engine(ane);
                    g_pa_ane.store(ane);
                    fprintf(stderr, "[phone-attn] ANE page engine: %u keys per page, budget %zu MB\n", ane->page_keys(), mb);
                } else {
                    fprintf(stderr, "[phone-attn] ANE page engine off: %s\n", ane->error.c_str());
                    delete ane;
                }
            }
            std::string err = srv.serve(port);
            std::lock_guard<std::mutex> lk(g_pa_status.mu);
            g_pa_status.state = "error: " + err;
        }).detach();
    });
}

+ (NSString *)envNote {
    return [NSString stringWithUTF8String:g_env_note] ?: @"";
}

+ (NSDictionary<NSString *, id> *)phoneAttnStatus {
    std::lock_guard<std::mutex> lk(g_pa_status.mu);
    return @{ @"state" : [NSString stringWithUTF8String:g_pa_status.state.c_str()] ?: @"",
              @"calls" : @(g_pa_status.attn_calls), @"heldKeys" : @(g_pa_status.held_keys),
              @"lastMs" : @(g_pa_status.last_attn_ms) };
}

+ (NSDictionary<NSString *, id> *)macStatus {
    std::lock_guard<std::mutex> lk(g_mac_mu);
    const double age = g_mac_phase.empty() ? 1e9 : std::chrono::duration<double>(std::chrono::steady_clock::now() - g_mac_at).count();
    return @{ @"phase" : [NSString stringWithUTF8String:g_mac_phase.c_str()] ?: @"", @"n1" : @(g_mac_n1), @"n2" : @(g_mac_n2),
              @"ctx" : @(g_mac_ctx), @"age" : @(age), @"activations" : @(g_mac_activations),
              @"activating" : @(g_mac_activating.load()), @"activateMs" : @(g_mac_activate_ms) };
}

+ (void)startANEBenchPort:(int)port {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        pthread_t t;
        if (pthread_create(&t, nullptr, ane_server_main, (void *)(intptr_t)port) == 0) pthread_detach(t);
    });
}
@end

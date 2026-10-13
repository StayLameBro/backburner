// Two identical copies: backburner/phone-attn/phone-attn.h (the app, pa-tool) and llama.cpp/ggml/src/ggml-metal/phone-attn.h
// (the Mac). Keep them identical; VERSION is checked at HELLO. tests/security/run.sh checks they match.

// phone-attn.h - the phone holds the OLDEST KV pages of every full-attention layer and computes their
// share of each attention op; the Mac merges the phone's partial with its own (log-sum-exp merge).
// Design: README.md, "Phone-held context". Header-only; built into two programs:
//   - ios/Sidecar (RPCBridge.mm)              the real phone, port 50062
//   - phone-attn/pa-tool.cpp (macOS)          "loopback phone" on the Mac CPU + the test client
// Both compute with scripts/sme/sme_attn.c (SME2, 512-bit SVL: M4, A18 Pro, A19 Pro).
//
// Framing: u32 magic 'PATN' | u32 type | u64 payload_len | payload. Little-endian (both ends arm64).
// One client at a time, strictly request -> reply.
//
//   HELLO                                                    -> HELLO_OK hello_rep
//   CONFIG   config_req                                      -> OK      (drops all held keys; is_q8: 0 f16, 1 q8_0, 2 q4_0)
//   APPEND   append_req + K[n*rs] + V[n*rs]                  -> OK u32 n_now   (keys pos0..pos0+n of one layer; pos0 == n_now)
//   TRUNCATE u32 n                                           -> OK      (every layer keeps its first n keys)
//   ATTN     attn_req + Q f16 [n_head_kv][48][256]           -> ATTN_OK attn_rep + O f16 [n_head_kv][48][256] + lse f32 [n_head_kv][48]
//              Q rows per KV head are r = g*8 + t (GQA group member g, token t), unscaled, rows t >= n_tok zero.
//              O is NORMALIZED (sum_j p_j v_j / sum_j p_j); lse = m + log(l) of the scaled scores, -inf if no keys.
//              For the co-attention slot: O_unnorm = O, S = 1, M = lse.
//   ATTN_BIG attn_req + Q f16 [ng][n_head_kv][48][256]       -> ATTN_OK attn_rep + O f16 [ng][n_head_kv][48][256] + lse f32 [ng][n_head_kv][48]
//              (v3) a whole prefill ubatch in one call: ng = ceil(n_tok / 8) groups of 8 tokens, each laid out exactly like
//              ATTN; the phone reads its keys for every group in one request (one round trip per layer per ubatch).
//   FETCH    fetch_req                                       -> OK rows[n*rs]   (v4) keys (which 0) or values (which 1) pos0..pos0+n
//              of one layer, exactly as APPEND received them: lets the Mac save a state while the phone holds keys
//   STATS                                                    -> OK text
//   PING     u32 reply_len + bytes                           -> OK reply_len bytes (link probe)
//   BYE                                                      -> closed
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

extern "C" {
void * sme_ws_new(void);
void   sme_attn_partial(void * w, const float * Q, int qrs, const uint8_t * K, const uint8_t * V, size_t rs,
                        int is_q8, int nk, float scale, int bk, float * O, float * m, float * l);
int    sme2_available(void);
void * sme_pipe_new(int nh);
void   sme_attn_pipe(void ** pp, int nw, int nh, const float * Q, const uint8_t * K, const uint8_t * V, size_t rs, size_t hb,
                     int is_q8, int nkv, int nk, float scale, float * O, float * m, float * l);
}

namespace pa {

constexpr uint32_t MAGIC        = 0x4E544150u; // "PATN"
constexpr uint32_t VERSION      = 4;           // v3: ATTN_BIG (v2 clients/servers still interoperate on ATTN); v4: FETCH
constexpr int      DEFAULT_PORT = 50062;
constexpr int      NR           = 48;          // query rows per KV head (8 tokens x GQA 6)
constexpr int      HD           = 256;         // head dim
constexpr int      KEY_ALIGN    = 64;          // the SME kernel works in multiples of 64 keys

enum msg : uint32_t { HELLO = 1, HELLO_OK, CONFIG, APPEND, TRUNCATE, ATTN, ATTN_OK, STATS, OK, ERR, BYE, PING, ATTN_BIG, FETCH };
constexpr int MAX_GROUPS = 64;                 // ATTN_BIG: up to 512 tokens per call

#pragma pack(push, 1)
struct hdr        { uint32_t magic, type; uint64_t len; };
struct hello_rep  { uint32_t version, sme2; char device[64]; };
struct config_req { uint32_t n_layer, n_head_kv, rs, hb, is_q8, sme_workers, sme_helpers, gpu_permille, gpu_chunk, store_f16, gpu_variant; };
// sme_helpers 0: plain SME kernel on sme_workers threads; gpu_permille: share of the pages for the GPU engine (if any); gpu_chunk: keys per GPU threadgroup
// store_f16: the phone stores q8_0 rows it receives as f16 (2x memory; lets the A19 neural accelerators read K/V directly)
struct append_req { uint32_t layer, pos0, n; };
struct fetch_req  { uint32_t layer, pos0, n, which; };                      // which: 0 keys, 1 values
struct attn_req   { uint32_t layer, n_tok, nk; float scale; };            // nk = 0: all held keys
struct attn_rep   { uint32_t nk; float phone_ms, gpu_ms, sme_ms; uint32_t gpu_pages, pages; };
#pragma pack(pop)

// ---------------------------------------------------------------- socket helpers
inline bool send_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n) { ssize_t k = ::send(fd, c, n, 0); if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; } c += k; n -= (size_t) k; }
    return true;
}
inline bool recv_all(int fd, void * p, size_t n) {
    char * c = (char *) p;
    while (n) { ssize_t k = ::recv(fd, c, n, 0); if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; } c += k; n -= (size_t) k; }
    return true;
}
inline bool send_msg(int fd, uint32_t type, const void * a, size_t na, const void * b = nullptr, size_t nb = 0,
                     const void * c = nullptr, size_t nc = 0) {
    hdr h = { MAGIC, type, (uint64_t) (na + nb + nc) };
    return send_all(fd, &h, sizeof h) && (!na || send_all(fd, a, na)) && (!nb || send_all(fd, b, nb)) && (!nc || send_all(fd, c, nc));
}
inline void tune_socket(int fd) {
    int one = 1, buf = 4 << 20;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
}

inline float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }
inline uint16_t f2h(float f) { __fp16 x = (__fp16) f; uint16_t h; memcpy(&h, &x, 2); return h; }

// Who may connect. The protocol has no authentication, so by default only this machine may (loopback: pa-tool tests);
// PA_ALLOW_REMOTE=1 lifts that for a deliberate setup. The Backburner app replaces it with its USB-cable check
// (set_accept_filter). Called after accept(), before anything is read; `why` says who it was and why.
using accept_filter_fn = std::function<bool(int fd, std::string & why)>;
inline bool accept_loopback_only(int fd, std::string & why) {
    sockaddr_storage p = {};
    socklen_t pl = sizeof p;
    if (getpeername(fd, (sockaddr *) &p, &pl) != 0) { why = "no peer address"; return false; }
    bool lo = false;
    char ip[INET6_ADDRSTRLEN] = "?";
    if (p.ss_family == AF_INET) {
        const in_addr a = ((sockaddr_in *) &p)->sin_addr;
        lo = (ntohl(a.s_addr) >> 24) == 127;
        inet_ntop(AF_INET, &a, ip, sizeof ip);
    } else if (p.ss_family == AF_INET6) {
        const in6_addr & a = ((sockaddr_in6 *) &p)->sin6_addr;
        lo = IN6_IS_ADDR_LOOPBACK(&a) || (IN6_IS_ADDR_V4MAPPED(&a) && a.s6_addr[12] == 127);
        inet_ntop(AF_INET6, &a, ip, sizeof ip);
    }
    const char * e = getenv("PA_ALLOW_REMOTE");
    if (lo || (e && atoi(e) != 0)) { why = std::string(ip) + (lo ? ": loopback" : ": PA_ALLOW_REMOTE"); return true; }
    why = std::string(ip) + ": not loopback (PA_ALLOW_REMOTE=1 allows it)";
    return false;
}

// ---------------------------------------------------------------- server
struct status {
    std::mutex mu;
    std::function<int()> thermal;   // optional: device thermal state (0 nominal .. 3 critical)
    std::string state = "idle";
    uint64_t attn_calls = 0, appended = 0, held_keys = 0;
    double last_attn_ms = 0, sum_attn_ms = 0, last_gpu_ms = 0, last_sme_ms = 0, last_ane_ms = 0;
    uint32_t last_ane_keys = 0;
    std::string last_big;           // the last ATTN_BIG's split and timing
    std::string last_dec;           // decode page choice: ms per call by (tokens, ANE pages)
};

constexpr uint32_t PAGE = 4096;   // keys per page (multiple of KEY_ALIGN)

// one page of PAGE keys of one layer: K and V rows (cfg.rs bytes each); h* = engine handles (e.g. id<MTLBuffer>)
struct page { uint8_t * k = nullptr, * v = nullptr; void * hk = nullptr, * hv = nullptr; };

// optional GPU engine (Metal on the phone): owns page memory so the GPU can read it without copies
struct engine {
    virtual ~engine() {}
    virtual bool alloc_page(size_t bytes, page & p) = 0;
    virtual void free_page(page & p) = 0;
    // start the GPU's share: pages[i] holds nkeys[i] keys (multiples of 64). Qs = f16(q*scale) [ng][nkv][48][256] (ng groups
    // of 8 tokens: ATTN_BIG sends a whole prefill ubatch, one command buffer for all of them).
    virtual bool begin(const page * pages, const uint32_t * nkeys, int n_pages, const uint16_t * Qs, const config_req & cfg, int ng = 1) = 0;
    // wait; unnormalised O [ng][nkv*48][256], m, l [ng][nkv*48]; returns the GPU ms
    virtual double end(float * O, float * m, float * l) = 0;
    virtual std::string describe() { return ""; }
    virtual void warm() {}   // between calls: keep the GPU clocked up (no wait)
    virtual double exec_ms() const { return 0; }   // the last command buffer's GPU execution time (0 = unknown)
    int ntok_hint = 8;   // one-group calls: the group's real tokens (rows t >= ntok_hint are zero; an engine may skip them)
    // after a one-group call: the next call will likely be this (the next layer, same split and tokens): an engine may prepare it
    virtual void arm(const page * pages, const uint32_t * nkeys, int n_pages, const config_req & cfg, int ntok) {
        (void) pages; (void) nkeys; (void) n_pages; (void) cfg; (void) ntok;
    }
};

// system-wide wired memory in MiB (iOS kills Sidecar at ~10.4-10.7 GB wired on the A19: the prefill tail, the GPU's KV and the
// ANE's page models all count, the ANE's NOT in the app footprint; 2026-09-28: 6.9 GB tail + 1.4 GB KV + 2.1 GB ANE was killed)
inline double wired_mb() {
#if defined(__APPLE__)
    vm_statistics64_data_t vs; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) == KERN_SUCCESS) {
        return (double) vs.wire_count * (double) vm_kernel_page_size / 1048576.0;
    }
#endif
    return 0;
}
// this device's RAM in MiB (0 if unknown)
inline double phys_mb() {
#if defined(__APPLE__)
    uint64_t m = 0; size_t l = sizeof m;
    if (sysctlbyname("hw.memsize", &m, &l, nullptr, 0) == 0) return (double) m / 1048576.0;
#endif
    return 0;
}
// memory the system could still hand out right now (free + purgeable pages), MiB. A Neural Engine compile runs in another
// process (ANECompilerService), so neither the app's own budget nor the wired count shows what it takes.
inline double sys_free_mb() {
#if defined(__APPLE__)
    vm_statistics64_data_t vs; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) == KERN_SUCCESS) {
        return (double) (vs.free_count + vs.purgeable_count) * (double) vm_kernel_page_size / 1048576.0;
    }
#endif
    return 0;
}
// what the Mac set with the control port's `wired MB` (scripts/serve.sh sends PHONE_WIRED_MAX_MB): 0 = not set
inline std::atomic<double> & wired_max_from_mac() { static std::atomic<double> v { 0 }; return v; }
// The wired ceiling everything on the phone stays under: PA_WIRED_MAX_MB if set, else the Mac's, else 76.5% of this device's
// RAM, at most 9400 (the 12 GB A19s: iOS killed the app at ~10.4-10.7 GB wired). It used to be 9400 on every device, and the
// Mac's PHONE_WIRED_MAX_MB never reached the phone (GitHub #14).
inline double wired_max_mb() {
    static const double env = getenv("PA_WIRED_MAX_MB") ? atof(getenv("PA_WIRED_MAX_MB")) : 0;
    if (env > 0) return env;
    const double mac = wired_max_from_mac().load(std::memory_order_relaxed);
    if (mac > 0) return mac;
    static const double def = [] { const double p = phys_mb(); return p > 0 ? std::min(9400.0, p * 0.765) : 9400.0; }();
    return def;
}

// optional page engine for the OLDEST keys (the Neural Engine, pa-ane.mm; docs/ANE.md): each layer's keys
// [p*page_keys(), (p+1)*page_keys()) become one page model once the layer holds all of them. Server-internal: the wire format
// does not change. Builds run on the engine's own thread, never inside ATTN.
struct page_engine {
    virtual ~page_engine() {}
    virtual uint32_t page_keys() const = 0;
    // CONFIG: drop every page (waits for a build in flight); sc = the storage row format (rs, hb, is_q8)
    virtual void configure(const config_req & sc) = 0;
    // after APPEND: ANE page p of `layer` is complete; src = the storage pages holding its keys (page_keys()/PAGE of them).
    // Their memory stays valid until truncate()/configure() returns.
    virtual void queue(uint32_t layer, uint32_t p, const std::vector<page> & src) = 0;
    // TRUNCATE(n): drop every page reaching past key n (waits for a build in flight)
    virtual void truncate(uint32_t n) = 0;
    // pages [0, k) of `layer` ready, k <= max_pages
    virtual uint32_t ready(uint32_t layer, uint32_t max_pages) = 0;
    // unnormalised partial over pages [0, n_pages): Qs = f16(q*scale) [nkv][48][256] in; O [nkv*48][256], m, l [nkv*48] out
    virtual bool run(uint32_t layer, uint32_t n_pages, const uint16_t * Qs, float * O, float * m, float * l) = 0;
    virtual std::string describe() = 0;
    virtual bool wait_builds() const { return false; }   // gates: APPEND waits for its builds (drain())
    virtual void drain() {}
    // memory is short (the KV comes first): unload one page model (the highest page of the layer holding the most); false if none
    virtual bool shed() { return false; }
    // held by the server from ready() to the end of run() (a host that unloads the models, e.g. for the prefill tail, takes it)
    virtual void begin_use() {}
    virtual void end_use() {}
    virtual void wake() {}   // a decode ATTN arrived: an engine suspended for a prefill can come back now
};

// merge partial (O2, m2, l2) into (O, m, l), rows of HD
inline void merge_partial(float * O, float * m, float * l, const float * O2, const float * m2, const float * l2, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        if (l2[r] <= 0) continue;
        if (l[r] <= 0) { memcpy(O + r * HD, O2 + r * HD, HD * sizeof(float)); m[r] = m2[r]; l[r] = l2[r]; continue; }
        const float M = std::max(m[r], m2[r]), a = expf(m[r] - M), b = expf(m2[r] - M);
        for (int d = 0; d < HD; d++) O[r * HD + d] = O[r * HD + d] * a + O2[r * HD + d] * b;
        m[r] = M; l[r] = l[r] * a + l2[r] * b;
    }
}

class server {
public:
    explicit server(status * st, std::function<void(const std::string &)> log = nullptr, int n_threads = 2, engine * eng = nullptr)
        : st_(st), log_(std::move(log)), n_thr_(n_threads), eng_(eng) {}

    // the ANE page engine (optional). PA_ANE_SHARE (permille, default 650, rounded to whole pages): this share of a call's keys goes to it,
    // the rest runs on the GPU / SME at the same time (A19: ANE 0.086 ms per 1k keys vs GPU q4_0 0.150, 2026-09-27)
    void set_page_engine(page_engine * pe) { ane_ = pe; }

    // who may connect (default: loopback only, see accept_loopback_only)
    void set_accept_filter(accept_filter_fn f) { accept_ok_ = std::move(f); }

    // Blocking accept loop. Returns an error string if the listener fails.
    std::string serve(int port) {
        int srv = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t) port); a.sin_addr.s_addr = htonl(INADDR_ANY);
        if (::bind(srv, (sockaddr *) &a, sizeof a) != 0 || ::listen(srv, 2) != 0) {
            std::string e = "phone-attn listen :" + std::to_string(port) + " failed: " + strerror(errno);
            ::close(srv); return e;
        }
        say("phone-attn on :" + std::to_string(port) + (sme2_available() ? " (SME2" : " (no SME2") + (eng_ ? ", GPU)" : ")"));
        for (;;) {
            int fd = ::accept(srv, nullptr, nullptr);
            if (fd < 0) {   // a USB replug fails accept: keep listening (breaking here left the port dead until relaunch)
                if (errno != EINTR) { say(std::string("phone-attn accept: ") + strerror(errno) + " (retrying)"); ::usleep(100000); }
                continue;
            }
            std::string why;
            if (!(accept_ok_ ? accept_ok_(fd, why) : accept_loopback_only(fd, why))) {
                say("refused a connection from " + why);
                ::close(fd);
                continue;
            }
            tune_socket(fd);
            session(fd);
            ::close(fd);
        }
        ::close(srv);
        return "accept failed";
    }

private:
    status * st_;
    std::function<void(const std::string &)> log_;
    int n_thr_;
    engine * eng_;
    page_engine * ane_ = nullptr;
    accept_filter_fn accept_ok_;
    config_req cfg_ = {};
    std::vector<std::vector<page>> pages_;      // per layer
    std::vector<uint32_t> n_;                   // keys held per layer
    std::vector<void *> ws_;                    // one SME workspace per thread (plain kernel)
    std::vector<void *> pipes_;                 // one pipe per SME worker (pipelined kernel)
    int pipe_nh_ = -1;

    // storage layout of the held rows (what the kernels see): f16 if cfg_.store_f16, else as received
    config_req store_cfg() const {
        config_req c = cfg_;
        if (cfg_.store_f16 && cfg_.is_q8 == 1) { c.is_q8 = 0; c.hb = HD * 2; c.rs = cfg_.n_head_kv * HD * 2; }
        return c;
    }
    // copy n received rows into storage (dequantizing q8_0 -> f16 if store_f16)
    void store_rows(uint8_t * dst, const uint8_t * src, uint32_t n) const {
        if (!(cfg_.store_f16 && cfg_.is_q8 == 1)) { memcpy(dst, src, (size_t) n * cfg_.rs); return; }
        const config_req sc = store_cfg();
        for (uint32_t j = 0; j < n; j++) {
            const uint8_t * row = src + (size_t) j * cfg_.rs;
            __fp16 * out = (__fp16 *) (dst + (size_t) j * sc.rs);
            for (uint32_t h = 0; h < cfg_.n_head_kv; h++) {
                for (int b = 0; b < HD / 32; b++) {
                    const uint8_t * blk = row + h * cfg_.hb + b * 34;
                    uint16_t dh; memcpy(&dh, blk, 2); const float d = h2f(dh);
                    for (int i = 0; i < 32; i++) out[h * HD + b * 32 + i] = (__fp16) (d * (float) (int8_t) blk[2 + i]);
                }
            }
        }
    }

    void say(const std::string & s) { if (log_) log_(s); }
    void set_state(const char * s) { if (st_) { std::lock_guard<std::mutex> lk(st_->mu); st_->state = s; } }
    bool err(int fd, const std::string & e) { say("ERR " + e); return send_msg(fd, ERR, e.data(), e.size()); }

    bool new_page(page & p) {
        const size_t bytes = (size_t) PAGE * store_cfg().rs;
#if TARGET_OS_IPHONE
        // iOS kills the app (jetsam) at its memory budget long before an allocation fails: refuse a page that would leave less
        // than PA_MEM_RESERVE_MB (default 400) instead, so APPEND answers "out of memory" and the Mac stops cleanly
        // (2026-09-27: 196,608 keys x 16 layers of q8_0 = 6.8 GB against a 6.0 GB app budget killed Sidecar at layer 14).
        static const size_t reserve = (size_t) (getenv("PA_MEM_RESERVE_MB") ? atoi(getenv("PA_MEM_RESERVE_MB")) : 400) << 20;
        if (os_proc_available_memory() < 2 * bytes + reserve) return false;
#endif
        if (eng_) return eng_->alloc_page(bytes, p);
        p.k = (uint8_t *) aligned_alloc(16384, (bytes + 16383) / 16384 * 16384);
        p.v = (uint8_t *) aligned_alloc(16384, (bytes + 16383) / 16384 * 16384);
        return p.k && p.v;
    }
    void drop_page(page & p) {
        if (eng_) eng_->free_page(p); else { free(p.k); free(p.v); }
        p = page();
    }
    void drop_all() {
        if (ane_) ane_->configure(store_cfg());   // before the storage it reads goes away
        for (auto & L : pages_) for (auto & p : L) drop_page(p);
        pages_.clear(); n_.clear();
    }

    void session(int fd) {
        std::vector<uint8_t> buf;
        // Hot window: after a message, spin on the socket for PA_HOT_MS (default 250) before blocking. Measured on the A18 Pro
        // (pa-tool PA_GAP_US, 4096 keys): a 5 ms idle gap between calls (a real decode) made the phone's compute 1.08 -> 2.62 ms
        // and the link 0.65 -> 1.97 ms; 20 ms idle, 7.3 ms compute. iOS parks and downclocks the cores between calls.
        static const int hot_ms = getenv("PA_HOT_MS") ? atoi(getenv("PA_HOT_MS")) : 250;
        auto hot_until = std::chrono::steady_clock::now();
        // PA_GPU_WARM_US=N (default 1000, 0 = off): while hot after an ATTN, a trivial GPU dispatch every N us. The GPU clocks down
        // within a few ms (A19, 5 ms gaps, q4_0: 4k keys 3.21 vs 0.75 ms back to back, 32k 5.43 vs 4.80); warm: 16k 3.09 -> 2.61,
        // 32k 5.44 -> 4.91 ms. 140k tier-3 gate (75.5k keys/layer on the phone, ANE 3 pages): 178.9 -> 167.0 ms/token, 8-token
        // verify rounds 277.8 -> 246.3 ms (Mac-only 243.6), same tokens; 2026-09-28
        static const int warm_us = getenv("PA_GPU_WARM_US") ? atoi(getenv("PA_GPU_WARM_US")) : 1000;
        auto last_warm = std::chrono::steady_clock::now();
        bool attn_hot = false;   // warm only in a decode (after ATTN calls), not after APPENDs
        for (;;) {
            while (hot_ms > 0 && std::chrono::steady_clock::now() < hot_until) {
                struct pollfd pf = { fd, POLLIN, 0 };
                if (::poll(&pf, 1, 0) != 0) break;
                if (warm_us > 0 && attn_hot && eng_) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last_warm >= std::chrono::microseconds(warm_us)) { eng_->warm(); last_warm = now; }
                }
                __builtin_arm_yield();
            }
            hdr h;
            if (!recv_all(fd, &h, sizeof h) || h.magic != MAGIC) return;
            if (h.len > (1ull << 31)) return;
            buf.resize(h.len);
            if (h.len && !recv_all(fd, buf.data(), h.len)) return;
            bool ok = true;
            switch (h.type) {
                case HELLO: {
                    hello_rep r = {}; r.version = VERSION; r.sme2 = (uint32_t) sme2_available();
                    snprintf(r.device, sizeof r.device, "phone-attn threads=%d gpu=%d ane=%u", n_thr_, eng_ ? 1 : 0,
                             ane_ ? ane_->page_keys() : 0);
                    ok = send_msg(fd, HELLO_OK, &r, sizeof r);
                } break;
                case CONFIG: {
                    if (h.len < sizeof(config_req)) { ok = err(fd, "short CONFIG"); break; }
                    config_req c; memcpy(&c, buf.data(), sizeof c);
                    if (c.n_head_kv == 0 || c.n_head_kv > 16 || c.n_layer == 0 || c.n_layer > 256 || c.rs < c.n_head_kv * c.hb) {
                        ok = err(fd, "bad CONFIG"); break;
                    }
                    drop_all();
                    cfg_ = c;
                    pages_.assign(cfg_.n_layer, {}); n_.assign(cfg_.n_layer, 0);
                    if (ane_) ane_->configure(store_cfg());
                    if (st_) { std::lock_guard<std::mutex> lk(st_->mu); st_->held_keys = 0; }
                    ok = send_msg(fd, OK, nullptr, 0);
                } break;
                case APPEND: {
                    append_req q; if (h.len < sizeof q) { ok = err(fd, "short APPEND"); break; }
                    memcpy(&q, buf.data(), sizeof q);
                    const size_t bytes = (size_t) q.n * cfg_.rs;
                    if (q.layer >= n_.size() || h.len != sizeof q + 2 * bytes) { ok = err(fd, "bad APPEND"); break; }
                    if (q.pos0 != n_[q.layer]) { ok = err(fd, "APPEND pos0 " + std::to_string(q.pos0) + " != held " + std::to_string(n_[q.layer])); break; }
                    auto & L = pages_[q.layer];
                    const uint8_t * K = buf.data() + sizeof q, * V = K + bytes;
                    const size_t srs = store_cfg().rs;
                    for (uint32_t done = 0; done < q.n; ) {
                        const uint32_t pos = n_[q.layer] + done, pi = pos / PAGE, off = pos % PAGE;
                        while (L.size() <= pi) {
                            // the KV comes first: shed ANE page models while a new page would cross the wired ceiling (TARGET_OS_IPHONE)
                            const double need = 2.0 * PAGE * store_cfg().rs / 1048576.0;
                            while (TARGET_OS_IPHONE && ane_ && wired_mb() + need > wired_max_mb() && ane_->shed()) {}
                            page p; if (!new_page(p)) { ok = false; break; } L.push_back(p);
                        }
                        if (!ok) break;
                        const uint32_t take = std::min(q.n - done, PAGE - off);
                        store_rows(L[pi].k + (size_t) off * srs, K + (size_t) done * cfg_.rs, take);
                        store_rows(L[pi].v + (size_t) off * srs, V + (size_t) done * cfg_.rs, take);
                        done += take;
                    }
                    if (!ok) { ok = err(fd, "out of memory at " + std::to_string(n_[q.layer]) + " keys"); break; }
                    n_[q.layer] += q.n;
                    if (ane_) {   // ANE pages this APPEND completed: queue their builds (the reply doesn't wait)
                        const uint32_t pk = ane_->page_keys(), per = pk / PAGE;
                        for (uint32_t p = q.pos0 / pk; p < n_[q.layer] / pk; p++) {
                            ane_->queue(q.layer, p, std::vector<page>(L.begin() + p * per, L.begin() + (p + 1) * per));
                        }
                        if (ane_->wait_builds()) ane_->drain();
                    }
                    if (st_) { std::lock_guard<std::mutex> lk(st_->mu); st_->appended += q.n; st_->held_keys = n_[0]; }
                    const uint32_t now = n_[q.layer];
                    ok = send_msg(fd, OK, &now, sizeof now);
                } break;
                case TRUNCATE: {
                    uint32_t n = 0; if (h.len >= 4) memcpy(&n, buf.data(), 4);
                    if (ane_) ane_->truncate(n);
                    for (size_t l = 0; l < n_.size(); l++) {
                        if (n_[l] <= n) continue;
                        n_[l] = n;
                        while (pages_[l].size() > (n + PAGE - 1) / PAGE) { drop_page(pages_[l].back()); pages_[l].pop_back(); }
                    }
                    ok = send_msg(fd, OK, nullptr, 0);
                } break;
                case FETCH: {
                    // the held rows as APPEND received them, for a state save on the Mac; rows stored as f16 (store_f16) are
                    // no longer the bytes the Mac sent, so that mode refuses rather than return something that isn't exact
                    fetch_req q; if (h.len < sizeof q) { ok = err(fd, "short FETCH"); break; }
                    memcpy(&q, buf.data(), sizeof q);
                    if (cfg_.store_f16 && cfg_.is_q8 == 1) { ok = err(fd, "FETCH: rows are stored as f16, not as received"); break; }
                    if (q.layer >= n_.size() || q.which > 1 || (uint64_t) q.pos0 + q.n > n_[q.layer]) {
                        ok = err(fd, "bad FETCH: layer " + std::to_string(q.layer) + " holds " +
                                 std::to_string(q.layer < n_.size() ? n_[q.layer] : 0) + " keys");
                        break;
                    }
                    const auto & L = pages_[q.layer];
                    std::vector<uint8_t> out((size_t) q.n * cfg_.rs);
                    for (uint32_t done = 0; done < q.n; ) {
                        const uint32_t pos = q.pos0 + done, pi = pos / PAGE, off = pos % PAGE;
                        const uint32_t take = std::min(q.n - done, PAGE - off);
                        memcpy(out.data() + (size_t) done * cfg_.rs, (q.which ? L[pi].v : L[pi].k) + (size_t) off * cfg_.rs, (size_t) take * cfg_.rs);
                        done += take;
                    }
                    ok = send_msg(fd, OK, out.data(), out.size());
                } break;
                case ATTN: if (ane_) ane_->wake(); ok = attn(fd, buf); break;
                case ATTN_BIG: ok = attn_big(fd, buf); break;
                case STATS: {
                    std::string s;
                    { std::lock_guard<std::mutex> lk(st_->mu);
                      char b[1024]; snprintf(b, sizeof b, "state=%s attn_calls=%llu last_ms=%.3f (gpu %.3f sme %.3f ane %.3f on %u keys) mean_ms=%.3f held=%llu appended=%llu",
                          st_->state.c_str(), (unsigned long long) st_->attn_calls, st_->last_attn_ms, st_->last_gpu_ms, st_->last_sme_ms,
                          st_->last_ane_ms, st_->last_ane_keys,
                          st_->attn_calls ? st_->sum_attn_ms / st_->attn_calls : 0.0,
                          (unsigned long long) st_->held_keys, (unsigned long long) st_->appended); s = b; }
                    if (eng_) s += " " + eng_->describe();
                    if (ane_) s += " " + ane_->describe();
                    if (st_->thermal) s += " thermal=" + std::to_string(st_->thermal());
                    { std::lock_guard<std::mutex> lk(st_->mu); if (!st_->last_big.empty()) s += " | " + st_->last_big;
                      if (!st_->last_dec.empty()) s += " | " + st_->last_dec; }
                    ok = send_msg(fd, OK, s.data(), s.size());
                } break;
                case PING: {
                    uint32_t n = 0; if (h.len >= 4) memcpy(&n, buf.data(), 4);
                    std::vector<uint8_t> r(n, 0x5a);
                    ok = send_msg(fd, OK, r.data(), r.size());
                } break;
                case BYE: return;
                default: ok = err(fd, "unknown message " + std::to_string(h.type));
            }
            if (!ok) return;
            hot_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(hot_ms);
            attn_hot = h.type == ATTN || h.type == ATTN_BIG;
            last_warm = std::chrono::steady_clock::now();
        }
    }

    // SME partial over one page's keys, all heads
    void sme_page(const float * Q, const page & p, uint32_t nk, float scale, float * O, float * m, float * l) {
        const config_req sc = store_cfg();
        const int nkv = (int) cfg_.n_head_kv;
        if (cfg_.sme_helpers > 0) {
            const int nw = std::max(1, std::min((int) cfg_.sme_workers, nkv)), nh = (int) cfg_.sme_helpers;
            if (pipe_nh_ != nh) { pipes_.clear(); pipe_nh_ = nh; }   // (old pipes leak: reconfiguration is rare)
            while ((int) pipes_.size() < nw) pipes_.push_back(sme_pipe_new(nh));
            sme_attn_pipe(pipes_.data(), nw, nh, Q, p.k, p.v, sc.rs, sc.hb, (int) sc.is_q8, nkv, (int) nk, scale, O, m, l);
            return;
        }
        const int nt = std::max(1, std::min(cfg_.sme_workers ? (int) cfg_.sme_workers : n_thr_, nkv));
        while ((int) ws_.size() < nt) ws_.push_back(sme_ws_new());
        auto run = [&](int t) {
            for (int h = t * nkv / nt; h < (t + 1) * nkv / nt; h++) {
                sme_attn_partial(ws_[t], Q + (size_t) h * NR * HD, HD, p.k + (size_t) h * sc.hb, p.v + (size_t) h * sc.hb, sc.rs,
                                 (int) sc.is_q8, (int) nk, scale, 512, O + (size_t) h * NR * HD, m + (size_t) h * NR, l + (size_t) h * NR);
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; t++) th.emplace_back(run, t);
        run(0);
        for (auto & x : th) x.join();
    }

    // ng 48-row groups (8 tokens x GQA 6 each) per KV head against the first nk held keys of `layer`: qh f16 [ng][nkv][48][256]
    // in, Oh f16 [ng][nkv][48][256] (normalized) + lse [ng][nkv][48] out. ng > 1 (ATTN_BIG, a prefill ubatch): the GPU takes all
    // groups in one command buffer while the ANE runs its pages group by group, and the ANE/GPU split comes from their measured
    // speeds (PA_BIG_ANE=0: no ANE in big calls). Returns "" or an error.
    std::string attn_core(uint32_t layer, uint32_t nk, float scale, const uint16_t * qh, int ng, uint16_t * Oh, float * lse_out,
                          double & gpu_ms, double & sme_ms, int & gp_out, int & np_out) {
        const int nkv = (int) cfg_.n_head_kv;
        const size_t qn = (size_t) nkv * NR * HD, rows = (size_t) nkv * NR, QN = (size_t) ng * qn, ROWS = (size_t) ng * rows;
        // PA_NULL=1 (pa-tool serve only: a timing probe): answer at once with an empty partial (lse -inf, the Mac's own keys
        // decide the output), so an e2e run shows what prefill/decode cost with the phone's attention made free
        static const bool null_attn = getenv("PA_NULL") && atoi(getenv("PA_NULL"));
        if (null_attn) {
            std::fill(Oh, Oh + QN, (uint16_t) 0); std::fill(lse_out, lse_out + ROWS, -INFINITY);
            gp_out = 0; np_out = 0; return "";
        }
        std::vector<float> Q(QN), O(QN, 0.0f), m(ROWS, -INFINITY), l(ROWS, 0.0f);
        for (size_t i = 0; i < QN; i++) Q[i] = h2f(qh[i]);

        // keys [0, a0*PAGE) to the ANE (ready page models, oldest first), then storage pages [a0, a0 + gp) to the GPU and
        // [a0 + gp, np) to SME
        const auto & L = pages_[layer];
        const int np_all = (int) ((nk + PAGE - 1) / PAGE);
        int na = 0, a0 = 0, dec_t = 0;
        const auto tcall = std::chrono::steady_clock::now();
        struct use_guard { page_engine * e; ~use_guard() { if (e) e->end_use(); } } ug { ane_ };
        static const bool big_ane = !getenv("PA_BIG_ANE") || atoi(getenv("PA_BIG_ANE")) != 0;
        if (ane_ && (ng == 1 || big_ane)) {
            ane_->begin_use();
            const uint32_t pk = ane_->page_keys();
            uint32_t want;
            if (ng == 1) {
                // the ANE's share of the keys (nearest whole page) by the call's tokens: its cost per page doesn't depend on them,
                // the GPU's (dense-row kernel, real rows only) grows with them. A19 Pro, 140k (75.5k keys): 1 token best at 1 page
                // (131.1 ms/token; 2 pages 147.2), 8-token rounds at 2 (232.3 ms; 1 page 251.9). PA_ANE_SHARE1 / PA_ANE_SHARE8
                // (permille, default 220 / 430; linear between), PA_ANE_SHARE sets both. Learning it online failed: a sometimes-
                // used ANE clocks down, so its trial calls ran 2-3x slow and the GPU-only arm won (rounds 288 ms)
                static const int s1 = getenv("PA_ANE_SHARE") ? atoi(getenv("PA_ANE_SHARE")) : getenv("PA_ANE_SHARE1") ? atoi(getenv("PA_ANE_SHARE1")) : 220;
                static const int s8 = getenv("PA_ANE_SHARE") ? atoi(getenv("PA_ANE_SHARE")) : getenv("PA_ANE_SHARE8") ? atoi(getenv("PA_ANE_SHARE8")) : 430;
                dec_t = cur_ntok_;
                const uint64_t share = (uint64_t) (s1 + (s8 - s1) * (dec_t - 1) / 7);
                want = (uint32_t) ((nk * share / 1000 + pk / 2) / pk);
            } else {
                // big: the number of pages that best balances ng x pages x (ANE ms per page call) against the GPU's
                // ng x (its keys) x (GPU ms per group x 1k keys), both from the last big calls (EMA)
                want = 0;
                double best = 1e30;
                for (uint32_t k = 0; k <= nk / pk; k++) {
                    const double t = std::max(ng * k * big_ane_ms_, ng * ((double) nk - (double) k * pk) / 1000.0 * big_gpu_ms_);
                    if (t < best - 1e-9) { best = t; want = k; }
                }
            }
            na = (int) ane_->ready(layer, want);
            a0 = na * (int) (pk / PAGE);
        }
        const int np = np_all - a0;   // storage pages left for the GPU / SME
        const page * Lr = L.data() + a0;
        std::vector<uint32_t> nkeys(np);
        for (int i = 0; i < np; i++) nkeys[i] = std::min(PAGE, nk - (uint32_t) (a0 + i) * PAGE);
        int gp = 0;
        if (eng_ && cfg_.gpu_permille) gp = std::min(np, (int) ((np * cfg_.gpu_permille + 500) / 1000));
        // The GPU pays a wake-up each call when calls are a few ms apart (a real decode): A19 Pro, 5 ms gaps, 8k keys: SME only
        // 2.97 ms vs GPU split 4.88; 32k keys: 10.09 vs 6.03. So the GPU joins only above PA_GPU_MIN_KEYS (default 16384).
        static const uint32_t gpu_min = getenv("PA_GPU_MIN_KEYS") ? (uint32_t) atoi(getenv("PA_GPU_MIN_KEYS")) : 16384;
        if (nk < gpu_min && store_cfg().is_q8 != 2 && ng == 1) gp = 0;
        if (ng > 1 && eng_) gp = np;                    // big: the GPU amortizes its launch over the groups, SME stays out
        if (gp < np && !sme2_available()) gp = eng_ ? np : 0;
        if (store_cfg().is_q8 == 2) {                   // q4_0 pages: the SME kernel has no q4_0 path, the GPU takes all
            if (!eng_) return "q4_0 pages need the GPU engine";
            gp = np;
        }
        if (gp < np && !sme2_available()) return "no SME2 and no GPU engine on this device";
        bool gpu_on = false;
        std::vector<uint16_t> Qs;
        if (gp > 0 || na > 0) {
            Qs.resize(QN);
            for (size_t i = 0; i < QN; i++) Qs[i] = f2h(Q[i] * scale);
        }
        const auto tg = std::chrono::steady_clock::now();
        if (gp > 0) {
            eng_->ntok_hint = ng == 1 ? cur_ntok_ : 8;
            gpu_on = eng_->begin(Lr, nkeys.data(), gp, Qs.data(), store_cfg(), ng);
            if (!gpu_on) return "GPU engine failed to start";
        }
        std::vector<float> O2(QN), m2(ROWS), l2(ROWS);
        if (na > 0) {   // the ANE's pages while the GPU works
            const auto ta = std::chrono::steady_clock::now();
            for (int g = 0; g < ng; g++) {
                if (!ane_->run(layer, (uint32_t) na, Qs.data() + g * qn, O2.data() + g * qn, m2.data() + g * rows, l2.data() + g * rows)) {
                    if (gpu_on) eng_->end(O2.data(), m2.data(), l2.data());
                    return "ANE page engine failed";
                }
            }
            merge_partial(O.data(), m.data(), l.data(), O2.data(), m2.data(), l2.data(), ROWS);
            const double ane_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
            if (ng > 1) big_ane_ms_ = 0.7 * big_ane_ms_ + 0.3 * ane_ms / (ng * na);
            if (st_) { std::lock_guard<std::mutex> lk(st_->mu);
                       st_->last_ane_ms = ane_ms;
                       st_->last_ane_keys = (uint32_t) a0 * PAGE; }
        }
        const auto ts = std::chrono::steady_clock::now();
        for (int i = gp; i < np; i++) {   // (ng == 1 only: big calls give every storage page to the GPU when there is one)
            for (int g = 0; g < ng; g++) {
                sme_page(Q.data() + g * qn, Lr[i], nkeys[i], scale, O2.data() + g * qn, m2.data() + g * rows, l2.data() + g * rows);
            }
            merge_partial(O.data(), m.data(), l.data(), O2.data(), m2.data(), l2.data(), ROWS);
        }
        sme_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        if (gpu_on) {
            const double gms = eng_->end(O2.data(), m2.data(), l2.data());
            gpu_ms += gms;
            if (ng == 1) {   // the next call is most likely the next layer with this split: armed once the reply is out
                arm_next_.valid = true; arm_next_.layer = (layer + 1) % (uint32_t) cfg_.n_layer; arm_next_.a0 = a0; arm_next_.gp = gp;
                arm_next_.nkeys.assign(nkeys.begin(), nkeys.begin() + gp); arm_next_.ntok = cur_ntok_;
            }
            merge_partial(O.data(), m.data(), l.data(), O2.data(), m2.data(), l2.data(), ROWS);
            uint32_t gk = 0; for (int i = 0; i < gp; i++) gk += nkeys[i];
            // GPU ms per (group x 1k keys): its own execution time when known, else commit -> done
            const double ex = eng_->exec_ms() > 0 ? eng_->exec_ms() : gms;   // end()'s wall time includes waiting on the ANE loop
            if (ng > 1 && gk > 0) big_gpu_ms_ = 0.7 * big_gpu_ms_ + 0.3 * ex / (ng * gk / 1000.0);
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count();
            if (ng > 1) { big_n_++; big_wall_ += wall; big_na_ += na; big_ng_ += ng; }
            if (ng > 1 && st_) { std::lock_guard<std::mutex> lk(st_->mu); st_->last_big = "big totals: " + std::to_string(big_n_) + " calls, " +
                std::to_string(big_wall_ / big_n_).substr(0, 6) + " ms mean, " + std::to_string((double) big_na_ / big_n_).substr(0, 4) +
                " ANE pages mean, " + std::to_string((double) big_ng_ / big_n_).substr(0, 4) + " groups mean | last big: " + std::to_string(ng) + " groups, " +
                std::to_string(na) + " ANE pages / " + std::to_string(gk) + " GPU keys, ane " + std::to_string(big_ane_ms_).substr(0, 5) +
                " ms/page-call, gpu " + std::to_string(big_gpu_ms_).substr(0, 5) + " ms/(group x 1k), gpu exec " + std::to_string(ex).substr(0, 6) + " ms, wall " +
                std::to_string(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count()).substr(0, 6) + " ms"; }
        }
        for (size_t r = 0; r < ROWS; r++) {
            const float inv = l[r] > 0 ? 1.0f / l[r] : 0.0f;
            for (int d = 0; d < HD; d++) Oh[r * HD + d] = f2h(O[r * HD + d] * inv);
            lse_out[r] = l[r] > 0 ? m[r] + logf(l[r]) : -INFINITY;
        }
        gp_out = gp; np_out = np_all;
        if (dec_t > 0) dec_update(dec_t, na, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tcall).count());
        return "";
    }

    // one-group calls: ms per call by (tokens, ANE pages), a diagnostic in STATS
    static constexpr int DEC_K = 8;
    struct dec_arm { double ms = 0; int n = 0; };
    dec_arm dec_[9][DEC_K];
    uint64_t dec_upd_ = 0;
    void dec_update(int t, int k, double ms) {
        if (k < 0 || k >= DEC_K) return;
        dec_arm & a = dec_[t][k];
        a.ms = a.n ? 0.9 * a.ms + 0.1 * ms : ms; a.n++;
        if (!st_ || (++dec_upd_ % 8) != 0) return;
        std::string d = "decode pages (ms/call by tokens):";
        for (int tt = 1; tt <= 8; tt++) {
            std::string row;
            for (int kk = 0; kk < DEC_K; kk++) if (dec_[tt][kk].n) {
                char b[32]; snprintf(b, sizeof b, " %d=%.2f", kk, dec_[tt][kk].ms); row += b;
            }
            if (!row.empty()) d += " t" + std::to_string(tt) + ":" + row;
        }
        std::lock_guard<std::mutex> lk(st_->mu); st_->last_dec = d;
    }
    // big-call balance (EMA): ANE ms per page call (A19: ~1.5), GPU ms per group x 1k keys (A19 q4_0 hot: ~0.14; the dense-row kernel ~0.076)
    double big_ane_ms_ = 1.6, big_gpu_ms_ = 0.14;
    uint64_t big_n_ = 0, big_na_ = 0, big_ng_ = 0;   // ATTN_BIG totals since launch (STATS)
    double big_wall_ = 0;
    int cur_ntok_ = 8;   // the current one-group call's tokens (engine ntok_hint)
    struct { bool valid = false; uint32_t layer = 0; int a0 = 0, gp = 0, ntok = 8; std::vector<uint32_t> nkeys; } arm_next_;
    void arm_after_reply() {
        if (!arm_next_.valid || !eng_) return;
        arm_next_.valid = false;
        if (arm_next_.layer >= pages_.size()) return;
        const auto & L = pages_[arm_next_.layer];
        if ((int) L.size() < arm_next_.a0 + arm_next_.gp) return;
        for (int i = 0; i < arm_next_.gp; i++) {   // the next layer must hold the same keys in those pages
            if (std::min(PAGE, n_[arm_next_.layer] - std::min(n_[arm_next_.layer], (uint32_t) (arm_next_.a0 + i) * PAGE)) != arm_next_.nkeys[i]) return;
        }
        eng_->arm(L.data() + arm_next_.a0, arm_next_.nkeys.data(), arm_next_.gp, store_cfg(), arm_next_.ntok);
    }

    // ATTN (one group) and ATTN_BIG (ng groups, v3): same reply layout, groups back to back
    bool attn_groups(int fd, const std::vector<uint8_t> & buf, bool big) {
        attn_req q; if (buf.size() < sizeof q) return err(fd, "short ATTN");
        memcpy(&q, buf.data(), sizeof q);
        const int nkv = (int) cfg_.n_head_kv;
        const size_t qn = (size_t) nkv * NR * HD, rows = (size_t) nkv * NR;
        const int ng = big ? (int) ((q.n_tok + 7) / 8) : 1;
        if (ng < 1 || ng > MAX_GROUPS) return err(fd, "ATTN_BIG n_tok " + std::to_string(q.n_tok));
        if (q.layer >= n_.size() || buf.size() != sizeof q + (size_t) ng * qn * 2) return err(fd, "bad ATTN");
        const uint32_t nk = q.nk ? q.nk : n_[q.layer];
        if (nk > n_[q.layer] || nk % KEY_ALIGN) return err(fd, "ATTN nk " + std::to_string(nk) + " (held " + std::to_string(n_[q.layer]) + ", must be a multiple of 64)");

        const auto t0 = std::chrono::steady_clock::now();
        const uint16_t * qh = (const uint16_t *) (buf.data() + sizeof q);
        std::vector<uint16_t> Oh((size_t) ng * qn);
        std::vector<float> lse((size_t) ng * rows);
        double gpu_ms = 0, sme_ms = 0;
        int gp = 0, np = 0;
        static const bool big_loop = getenv("PA_BIG_LOOP") && atoi(getenv("PA_BIG_LOOP"));   // A/B: the old group-by-group path
        cur_ntok_ = big ? 8 : (int) std::max(1u, std::min(8u, q.n_tok));
        if (big_loop || ng == 1) {
            for (int g = 0; g < ng; g++) {
                const std::string e = attn_core(q.layer, nk, q.scale, qh + (size_t) g * qn, 1, Oh.data() + (size_t) g * qn,
                                                lse.data() + (size_t) g * rows, gpu_ms, sme_ms, gp, np);
                if (!e.empty()) return err(fd, e);
            }
        } else {
            const std::string e = attn_core(q.layer, nk, q.scale, qh, ng, Oh.data(), lse.data(), gpu_ms, sme_ms, gp, np);
            if (!e.empty()) return err(fd, e);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (st_) { std::lock_guard<std::mutex> lk(st_->mu); st_->attn_calls++; st_->last_attn_ms = ms; st_->sum_attn_ms += ms;
                   st_->last_gpu_ms = gpu_ms; st_->last_sme_ms = sme_ms; }
        attn_rep r = { nk, (float) ms, (float) gpu_ms, (float) sme_ms, (uint32_t) gp, (uint32_t) np };
        const bool ok = send_msg(fd, ATTN_OK, &r, sizeof r, Oh.data(), Oh.size() * 2, lse.data(), lse.size() * 4);
        arm_after_reply();   // the Mac works on this reply; meanwhile the next call's GPU work gets committed
        return ok;
    }
    bool attn(int fd, const std::vector<uint8_t> & buf) { return attn_groups(fd, buf, false); }
    bool attn_big(int fd, const std::vector<uint8_t> & buf) { return attn_groups(fd, buf, true); }
};

// ---------------------------------------------------------------- client (Mac side)
class client {
public:
    ~client() { if (fd_ >= 0) { send_msg(fd_, BYE, nullptr, 0); ::close(fd_); } }
    // end the connection from another thread: a send/recv blocked in it fails at once (shutdown at exit, a stalled phone)
    void abort_io() { if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR); }
    std::string last_err;

    bool connect(const std::string & host, int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t) port);
        if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) { last_err = "bad host"; return false; }
        if (::connect(fd_, (sockaddr *) &a, sizeof a) != 0) { last_err = strerror(errno); return false; }
        host_ = host;
        tune_socket(fd_);
        return true;
    }
    bool hello(hello_rep & r) { return call(HELLO, nullptr, 0, nullptr, 0, HELLO_OK) && take(&r, sizeof r); }
    bool config(const config_req & c) { return call(CONFIG, &c, sizeof c, nullptr, 0); }
    bool append(uint32_t layer, uint32_t pos0, uint32_t n, const void * K, const void * V, size_t rs) {
        append_req q = { layer, pos0, n };
        hdr h = { MAGIC, APPEND, sizeof q + 2 * (size_t) n * rs };
        if (!send_all(fd_, &h, sizeof h) || !send_all(fd_, &q, sizeof q) || !send_all(fd_, K, n * rs) || !send_all(fd_, V, n * rs)) return false;
        return reply(OK);
    }
    bool truncate(uint32_t n) { return call(TRUNCATE, &n, 4, nullptr, 0); }
    // v4: n rows of keys (which 0) or values (which 1) of one layer, from position pos0 of this phone's store
    bool fetch(uint32_t layer, uint32_t pos0, uint32_t n, uint32_t which, void * out, size_t rs) {
        fetch_req q = { layer, pos0, n, which };
        return call(FETCH, &q, sizeof q, nullptr, 0) && take(out, (size_t) n * rs);
    }
    // Q f16 [nkv][48][256] in, O f16 [nkv][48][256] + lse [nkv][48] out
    bool attn(uint32_t layer, uint32_t n_tok, uint32_t nk, float scale, const uint16_t * Q, size_t qn,
              uint16_t * O, float * lse, attn_rep & rep) {
        attn_req q = { layer, n_tok, nk, scale };
        if (!call(ATTN, &q, sizeof q, Q, qn * 2, ATTN_OK)) return false;
        return take(&rep, sizeof rep) && take(O, qn * 2) && take(lse, qn / HD * 4);
    }
    // v3: ng = ceil(n_tok/8) groups, Q f16 [ng][nkv][48][256] in, O f16 same + lse [ng][nkv][48] out
    bool attn_big(uint32_t layer, uint32_t n_tok, uint32_t nk, float scale, const uint16_t * Q, size_t qn_all,
                  uint16_t * O, float * lse, attn_rep & rep) {
        attn_req q = { layer, n_tok, nk, scale };
        if (!call(ATTN_BIG, &q, sizeof q, Q, qn_all * 2, ATTN_OK)) return false;
        return take(&rep, sizeof rep) && take(O, qn_all * 2) && take(lse, qn_all / HD * 4);
    }
    bool ping(uint32_t up, uint32_t down) {
        std::vector<uint8_t> p(4 + up, 0); memcpy(p.data(), &down, 4);
        return call(PING, p.data(), p.size(), nullptr, 0);
    }
    std::string stats() { return call(STATS, nullptr, 0, nullptr, 0) ? std::string(rbuf_.begin(), rbuf_.end()) : last_err; }

private:
    int fd_ = -1;
    std::string host_;

    // Wait for the phone's reply. A phone can stop answering mid-call (Backburner went to the background, the screen locked,
    // the cable came out) and keep the connection open, so a plain recv() would wait forever. On the Mac the GPU work that
    // waits for this reply is killed by macOS after a few seconds anyway (kIOGPUCommandBufferCallbackErrorTimeout), and the
    // server can't compute again until it restarts, so say what is going on every 5 s and give up after PA_REPLY_TIMEOUT_S
    // seconds (default 15; 0 = wait forever).
    bool wait_reply() {
        static const int limit_s = getenv("PA_REPLY_TIMEOUT_S") ? atoi(getenv("PA_REPLY_TIMEOUT_S")) : 15;
        for (int waited = 0;; ) {
            pollfd p = { fd_, POLLIN, 0 };
            const int r = ::poll(&p, 1, 5000);
            if (r > 0) return true;
            if (r < 0) {
                if (errno == EINTR) continue;
                last_err = strerror(errno);
                return false;
            }
            waited += 5;
            if (limit_s > 0 && waited >= limit_s) {
                last_err = "the phone at " + host_ + " did not answer for " + std::to_string(waited) + " s";
                return false;
            }
            fprintf(stderr, "phone-attn: the phone at %s hasn't answered for %d s: is Backburner open and in front on it?\n",
                    host_.c_str(), waited);
        }
    }
    std::vector<uint8_t> rbuf_;
    size_t roff_ = 0;
    bool call(uint32_t type, const void * a, size_t na, const void * b, size_t nb, uint32_t want = OK) {
        return send_msg(fd_, type, a, na, b, nb) && reply(want);
    }
    bool reply(uint32_t want) {
        hdr h;
        if (!wait_reply()) return false;
        if (!recv_all(fd_, &h, sizeof h) || h.magic != MAGIC) { last_err = "link closed"; return false; }
        rbuf_.resize(h.len); roff_ = 0;
        if (h.len && !recv_all(fd_, rbuf_.data(), h.len)) { last_err = "link closed"; return false; }
        if (h.type == ERR) { last_err = std::string(rbuf_.begin(), rbuf_.end()); return false; }
        if (h.type != want) { last_err = "unexpected reply " + std::to_string(h.type); return false; }
        return true;
    }
    bool take(void * p, size_t n) {
        if (roff_ + n > rbuf_.size()) { last_err = "short reply"; return false; }
        memcpy(p, rbuf_.data() + roff_, n); roff_ += n; return true;
    }
};

} // namespace pa

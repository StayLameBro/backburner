// pa-ane.mm - phone-attn's Neural Engine page engine (docs/ANE.md): the OLDEST keys of each full-attention
// layer run as CoreML models whose conv weights are the keys and values themselves.
//
// One page model = one layer's 16,384 keys, all KV heads: q [1, nkv*256, 1, 48] (f16, scale folded in) and c [1, nkv, 1, 48]
// (centering: the previous call's row max for this page) in; o = 256 O/l [1, nkv*256, 1, 48] and ml = (m - c, l) per head
// [1, 2*nkv, 1, 48] out. fp16 weights, P normalized before P.V, centered scores: the accurate variant (phone-attn/ane-kv/
// check_real.py; the 51k full-model gate matched the exact SME engine, 33/33 tokens).
//
// Pages are made on the device by patching a TEMPLATE model (built once on the Mac: phone-attn/ane-kv/build.py
// --keys 16384 --rows 48 --wq fp16 --pfix --center input): the template .mlmodelc is cloned (APFS copy-on-write) and its
// weights/weight.bin rewritten at the 2*nkv conv blobs (K' = [K | 1] [N][257] and V^T [256][N] per head, raw fp16 after a
// 64-byte blob header). Bit-identical to a fresh coremltools build (M1, patch_check.py). Then MLModel loads it on the ANE
// (~0.7 s the first time: the on-device ANE compile), one warm-up prediction, ready. All on the builder thread.
//
// Built into Sidecar (RPCBridge.mm includes this file) and pa-tool (macOS: the Mac ANE computes these models bit-identically
// to the A18's, so the accuracy gate runs on the Mac).
#pragma once

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "phone-attn.h"

#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>

namespace pa {

class ane_engine : public page_engine {
public:
    std::string error;
    // true while the engine holds or wants page models (keys queued), false once everything is dropped: the host can let
    // other wired memory go idle meanwhile (Sidecar: the prefill tail's residency is released only while this is true)
    std::function<void(bool)> on_want_memory;

    // tmpl: the template .mlmodelc; cache: a directory for page models (emptied here); budget_mb: most page-model bytes loaded
    bool init(const std::string & tmpl, const std::string & cache, size_t budget_mb) {
        tmpl_ = tmpl; cache_ = cache; budget_ = budget_mb << 20;
        NSFileManager * fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:@(cache_.c_str()) error:nil];
        [fm createDirectoryAtPath:@(cache_.c_str()) withIntermediateDirectories:YES attributes:nil error:nil];
        // blob offsets from model.mil: conv_{2h}_weight_0 = K' of head h, conv_{2h+1}_weight_0 = V^T
        NSString * mil = [NSString stringWithContentsOfFile:@((tmpl_ + "/model.mil").c_str()) encoding:NSUTF8StringEncoding error:nil];
        if (!mil) { error = "no model.mil in " + tmpl_; return false; }
        NSRegularExpression * re = [NSRegularExpression regularExpressionWithPattern:
            @"conv_(\\d+)_weight_0 = const\\(\\)\\[[^\\n]*?offset = uint64\\((\\d+)\\)" options:0 error:nil];
        std::vector<std::pair<int, uint64_t>> blobs;
        for (NSTextCheckingResult * r in [re matchesInString:mil options:0 range:NSMakeRange(0, mil.length)]) {
            blobs.push_back({ [[mil substringWithRange:[r rangeAtIndex:1]] intValue],
                              (uint64_t) [[mil substringWithRange:[r rangeAtIndex:2]] longLongValue] });
        }
        std::sort(blobs.begin(), blobs.end());
        nkv_ = (int) blobs.size() / 2;
        if (nkv_ < 1 || blobs.size() % 2 || blobs.front().first != 0) { error = "template: unexpected conv weights"; return false; }
        const int fd = open((tmpl_ + "/weights/weight.bin").c_str(), O_RDONLY);
        if (fd < 0) { error = "template: no weights/weight.bin"; return false; }
        for (auto & b : blobs) {
            struct { uint32_t sentinel, dtype; uint64_t size, data; } hb;
            if (pread(fd, &hb, sizeof hb, (off_t) b.second) != sizeof hb || hb.sentinel != 0xDEADBEEF || hb.dtype != 1) {
                close(fd); error = "template: bad blob header"; return false;
            }
            data_off_.push_back(hb.data); blob_bytes_.push_back(hb.size);
        }
        close(fd);
        pk_ = (uint32_t) (blob_bytes_[1] / 2 / HD);   // V^T is [256][N] fp16
        if (pk_ == 0 || pk_ % PAGE || blob_bytes_[0] != (uint64_t) pk_ * (HD + 1) * 2) { error = "template: unexpected blob sizes"; return false; }
        struct stat st; model_bytes_ = stat((tmpl_ + "/weights/weight.bin").c_str(), &st) == 0 ? (size_t) st.st_size : 0;
        // the inputs, reused by every call
        q_ = [[MLMultiArray alloc] initWithShape:@[@1, @(nkv_ * HD), @1, @(NR)] dataType:MLMultiArrayDataTypeFloat16 error:nil];
        c_ = [[MLMultiArray alloc] initWithShape:@[@1, @(nkv_), @1, @(NR)] dataType:MLMultiArrayDataTypeFloat16 error:nil];
        if (!q_ || !c_) { error = "MLMultiArray alloc failed"; return false; }
        // PA_ANE_BUILDERS (default 1): parallel builds. 2 made each CoreML load (the ANE compile) 3x slower: 3.6 vs 1.1 s per page, 140k 116 vs 82 s
        const int nb = std::max(1, getenv("PA_ANE_BUILDERS") ? atoi(getenv("PA_ANE_BUILDERS")) : 1);
        for (int i = 0; i < nb; i++) builders_.emplace_back([this] { build_loop(); });
        return true;
    }

    ~ane_engine() override {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto & t : builders_) if (t.joinable()) t.join();
    }

    uint32_t page_keys() const override { return pk_; }

    void begin_use() override { use_mu_.lock(); }
    void end_use() override { use_mu_.unlock(); }

    // The prefill tail is about to run (Sidecar's tail server, on_work): unload every page model (their compiled dirs stay) so
    // the tail's weights can be wired again. A build or reload already running (an on-device ANE compile and load, ~1 s) is
    // waited for first (<= 3 s): it used to carry on while the tail wired its weights, the two together past the phone's
    // memory (GitHub #14: A20 Pro, 112k). Then waits (<= 3 s) only while wired + need_mb (the tail) would pass the wired ceiling:
    // the page models' memory is often NOT wired (A19, 2026-09-28: 57 pages loaded, 1.9 GB system wired), and waiting for a
    // drop that never comes cost every prefill the full 3 s. Later calls only note the time. The builder reloads the pages on
    // the first decode ATTN (wake) or once no work was seen for PA_ANE_RESUME_S (default 30). Keys of suspended pages run on the
    // GPU meanwhile.
    void suspend(double need_mb = 0) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            last_work_ = std::chrono::steady_clock::now();
            wake_ = false;
            if (suspended_) return;
        }
        int released = 0;
        const double w0 = wired_mb();
        {
            std::lock_guard<std::mutex> use(use_mu_);   // an ATTN between ready() and run() finishes first
            std::unique_lock<std::mutex> lk(mu_);
            suspended_ = true;   // no build starts from here on; one in flight drops its model when it ends (build: "suspended")
            const auto tb = std::chrono::steady_clock::now();
            if (!building_.empty()) {
                cv_.wait_for(lk, std::chrono::seconds(3), [&] { return building_.empty(); });
                n_suspend_waits_++;
                suspend_build_wait_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
            }
            for (auto & L : slots_) for (auto & x : L) {
                if (x->state == READY) { x->state = SUSPENDED; x->model = nil; released++; }
            }
            n_loaded_ = 0; n_suspends_++;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const double ceil_mb = wired_max_mb();   // ~1 GB under the kill line (phone-attn.h)
        while (released > 0 && need_mb > 0 && wired_mb() + need_mb > ceil_mb && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::lock_guard<std::mutex> lk(mu_);
        suspend_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        suspend_freed_mb_ = w0 - wired_mb();
    }

    // the first decode ATTN after a prefill: reload the suspended pages now (not PA_ANE_RESUME_S after the tail's last work);
    // on_wake lets the host free the tail's wired memory for them right away
    std::function<void()> on_wake;
    void wake() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!suspended_ || wake_) return;
            wake_ = true; n_wakes_++;
        }
        if (on_wake) on_wake();
        cv_.notify_all();
    }

    void configure(const config_req & sc) override {
        drop_from(0);
        std::lock_guard<std::mutex> lk(mu_);
        if (on_want_memory && wanting_) { wanting_ = false; on_want_memory(false); }
        per_layer_cap_ = SIZE_MAX;
        sc_ = sc;
        bad_cfg_ = sc.n_head_kv != 0 && (int) sc.n_head_kv != nkv_;
        error = bad_cfg_ ? "template has " + std::to_string(nkv_) + " KV heads, CONFIG " + std::to_string(sc.n_head_kv) : "";
    }

    void queue(uint32_t layer, uint32_t p, const std::vector<page> & src) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (bad_cfg_) return;
        if (slots_.size() <= layer) slots_.resize(layer + 1);
        auto & L = slots_[layer];
        if (L.size() != p) return;   // pages are queued in order; anything else (a gap) is ignored, those keys stay on GPU/SME
        // the budget is split evenly over the layers (every layer pays its own attention; filling the first layers' pages
        // first would leave the rest with none)
        const size_t per_layer = std::min(per_layer_cap_, sc_.n_layer ? budget_ / model_bytes_ / sc_.n_layer : 0);
        if (p >= per_layer || (n_loaded_ + n_pending() + n_susp() + 1) * model_bytes_ > budget_) { n_over_budget_++; return; }
        auto s = std::make_shared<slot>();
        s->layer = layer; s->p = p; s->src = src; s->c.assign((size_t) nkv_ * NR, 0);
        L.push_back(s);
        q_jobs_.push_back(s);
        cv_.notify_all();
        if (on_want_memory && !wanting_) { wanting_ = true; on_want_memory(true); }
    }
    // PA_ANE_WAIT=1 (gates only): APPEND returns once its pages are built, so the ANE serves every later call
    bool wait_builds() const { static const bool w = getenv("PA_ANE_WAIT") && atoi(getenv("PA_ANE_WAIT")); return w; }
    void drain() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (q_jobs_.empty() && deferred_.empty() && building_.empty()) || suspended_; });
    }

    void truncate(uint32_t n) override { drop_from(n); }

    bool shed() override {
        std::shared_ptr<slot> victim;
        {
            std::lock_guard<std::mutex> lk(mu_);
            size_t best = 0; int bl = -1;
            for (size_t l = 0; l < slots_.size(); l++) {
                size_t nr = 0; for (auto & x : slots_[l]) nr += x->state == READY;
                if (nr > best && slots_[l].back()->state == READY) { best = nr; bl = (int) l; }
            }
            if (bl < 0) return false;
            victim = slots_[bl].back(); slots_[bl].pop_back();
            victim->cancel = true; n_loaded_--; n_shed_++;
            per_layer_cap_ = std::min<size_t>(per_layer_cap_, slots_[bl].size());   // don't rebuild what memory can't hold
        }
        victim->model = nil;
        if (!victim->dir.empty()) [[NSFileManager defaultManager] removeItemAtPath:@(victim->dir.c_str()) error:nil];
        return true;
    }

    uint32_t ready(uint32_t layer, uint32_t max_pages) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (suspended_ || layer >= slots_.size()) return 0;
        uint32_t k = 0;
        for (auto & s : slots_[layer]) { if (k >= max_pages || s->state != READY) break; k++; }
        return k;
    }

    bool run(uint32_t layer, uint32_t n_pages, const uint16_t * Qs, float * O, float * m, float * l) override {
        std::vector<std::shared_ptr<slot>> ps;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (uint32_t p = 0; p < n_pages; p++) ps.push_back(slots_[layer][p]);
        }
        const size_t rows = (size_t) nkv_ * NR;
        std::fill(O, O + rows * HD, 0.0f); std::fill(m, m + rows, -INFINITY); std::fill(l, l + rows, 0.0f);
        std::vector<float> O2(rows * HD), m2(rows), l2(rows);
        const auto t0 = std::chrono::steady_clock::now();
        @autoreleasepool {
            // q: channel h*256 + d, last axis the row
            {
                const NSArray<NSNumber *> * st = q_.strides; const long s1 = st[1].longValue, s3 = st[3].longValue;
                __fp16 * q = (__fp16 *) q_.dataPointer;
                for (int h = 0; h < nkv_; h++) for (int r = 0; r < NR; r++) {
                    const uint16_t * src = Qs + ((size_t) h * NR + r) * HD;
                    for (int d = 0; d < HD; d++) memcpy(&q[(long) (h * HD + d) * s1 + r * s3], &src[d], 2);
                }
            }
            MLFeatureValue * fq = [MLFeatureValue featureValueWithMultiArray:q_];
            for (auto & s : ps) {
                {
                    const NSArray<NSNumber *> * st = c_.strides; const long s1 = st[1].longValue, s3 = st[3].longValue;
                    __fp16 * c = (__fp16 *) c_.dataPointer;
                    for (int h = 0; h < nkv_; h++) for (int r = 0; r < NR; r++) memcpy(&c[h * s1 + r * s3], &s->c[(size_t) h * NR + r], 2);
                }
                NSError * err = nil;
                MLDictionaryFeatureProvider * in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{ @"q": fq, @"c": [MLFeatureValue featureValueWithMultiArray:c_] } error:&err];
                id<MLFeatureProvider> out = in ? [s->model predictionFromFeatures:in error:&err] : nil;
                MLMultiArray * o = [out featureValueForName:@"o"].multiArrayValue, * ml = [out featureValueForName:@"ml"].multiArrayValue;
                if (!o || !ml || o.dataType != MLMultiArrayDataTypeFloat16 || ml.dataType != MLMultiArrayDataTypeFloat16) {
                    std::lock_guard<std::mutex> lk(mu_);
                    error = std::string("ANE prediction failed: ") + (err ? err.localizedDescription.UTF8String : "no outputs");
                    return false;
                }
                const long o1 = o.strides[1].longValue, o3 = o.strides[3].longValue, m1 = ml.strides[1].longValue, m3 = ml.strides[3].longValue;
                const __fp16 * op = (const __fp16 *) o.dataPointer, * mp = (const __fp16 *) ml.dataPointer;
                for (int h = 0; h < nkv_; h++) for (int r = 0; r < NR; r++) {
                    const size_t row = (size_t) h * NR + r;
                    const float c = (float) *(const __fp16 *) &s->c[row];
                    const float mm = (float) mp[(2 * h) * m1 + r * m3] + c, ll = (float) mp[(2 * h + 1) * m1 + r * m3];
                    m2[row] = mm; l2[row] = ll;
                    const __fp16 cn = (__fp16) mm; memcpy(&s->c[row], &cn, 2);   // the next call's centering for this page
                    const float f = ll / 256.0f;
                    for (int d = 0; d < HD; d++) O2[row * HD + d] = (float) op[(long) (h * HD + d) * o1 + r * o3] * f;
                }
                merge_partial(O, m, l, O2.data(), m2.data(), l2.data(), rows);
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        n_calls_++; n_page_calls_ += ps.size();
        ms_sum_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }

    std::string describe() override {
        std::lock_guard<std::mutex> lk(mu_);
        char b[400];
        snprintf(b, sizeof b, "ane: page %u keys, %d loaded (%.0f MB), %d queued, %d failed (%d wired), %d over budget, %d shed, "
                 "wired %.0f MB, %.1f ms/page build, %.3f ms/page call (%llu calls)%s%s", pk_, n_loaded_, n_loaded_ * model_bytes_ / 1048576.0,
                 n_pending(), n_failed_, n_wired_skip_, n_over_budget_, n_shed_, wired_mb(), n_built_ ? build_ms_ / n_built_ : 0.0,
                 n_page_calls_ ? ms_sum_ / n_page_calls_ : 0.0,
                 (unsigned long long) n_calls_, error.empty() ? "" : " | ", error.c_str());
        char b2[320];
        snprintf(b2, sizeof b2, " | %s, %d suspends (last freed %.0f MB wired in %.0f ms; %d waited for a build, last %.0f ms), %d wakes, "
                 "%d reloads | system free %.0f MB now, lowest at a build %.0f MB, %d builds put off for it | ceiling %.0f MB",
                 suspended_ ? "SUSPENDED" : "active", n_suspends_, suspend_freed_mb_, suspend_ms_, n_suspend_waits_, suspend_build_wait_ms_,
                 n_wakes_, n_reloads_, sys_free_mb(), min_sys_free_mb_, n_sysfree_skip_, wired_max_mb());
        char b3[160];
        const int nl = n_built_ + n_reloads_;
        snprintf(b3, sizeof b3, " | build phases: weights %.0f, load %.0f, warm-up %.0f ms/page", n_built_ ? weights_ms_ / n_built_ : 0.0,
                 nl ? load_ms_ / nl : 0.0, nl ? warm_ms_ / nl : 0.0);
        return std::string(b) + b2 + b3;
    }

private:
    enum { QUEUED = 0, READY = 1, FAILED = 2, SUSPENDED = 3 };
    struct slot {
        uint32_t layer = 0, p = 0;
        std::atomic<int> state { QUEUED };
        std::atomic<bool> cancel { false };
        std::vector<page> src;
        std::vector<uint16_t> c;    // f16 centering per row (the previous call's max)
        MLModel * model = nil;
        std::string dir;
    };

    std::string tmpl_, cache_;
    size_t budget_ = 0, model_bytes_ = 0;
    uint32_t pk_ = 0;
    int nkv_ = 0;
    std::vector<uint64_t> data_off_, blob_bytes_;
    MLMultiArray * q_ = nil, * c_ = nil;

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> builders_;
    bool stop_ = false, bad_cfg_ = false;
    config_req sc_ = {};
    std::vector<std::vector<std::shared_ptr<slot>>> slots_;   // [layer][page]
    std::deque<std::shared_ptr<slot>> q_jobs_;
    std::vector<std::shared_ptr<slot>> building_;   // builds in flight (one per builder thread)
    int n_loaded_ = 0, n_failed_ = 0, n_over_budget_ = 0, n_built_ = 0, n_shed_ = 0, n_wired_skip_ = 0, n_suspends_ = 0, n_reloads_ = 0;
    std::mutex use_mu_;
    bool wanting_ = false;
    bool suspended_ = false, wake_ = false;
    int n_wakes_ = 0;
    int n_suspend_waits_ = 0, n_sysfree_skip_ = 0;
    double suspend_build_wait_ms_ = 0, min_sys_free_mb_ = 0;
    std::chrono::steady_clock::time_point last_work_, retry_at_;
    std::deque<std::shared_ptr<slot>> deferred_;   // builds / reloads waiting for wired headroom
    double suspend_ms_ = 0, suspend_freed_mb_ = 0;
    int n_pending() const { return (int) (q_jobs_.size() + deferred_.size()) + (int) building_.size(); }
    int n_susp() const { int n = 0; for (auto & L : slots_) for (auto & x : L) n += x->state == SUSPENDED; return n; }
    size_t per_layer_cap_ = SIZE_MAX;
    double build_ms_ = 0, ms_sum_ = 0, weights_ms_ = 0, load_ms_ = 0, warm_ms_ = 0;   // build phases (sums; loads include reloads)
    uint64_t n_calls_ = 0, n_page_calls_ = 0;

    // drop every page with keys past n: queued ones leave the queue, a build in flight is waited for, loaded ones unload
    void drop_from(uint32_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        std::vector<std::shared_ptr<slot>> gone;
        for (auto & L : slots_) {
            while (!L.empty() && (uint64_t) (L.back()->p + 1) * pk_ > n) { L.back()->cancel = true; gone.push_back(L.back()); L.pop_back(); }
        }
        for (auto * q : { &q_jobs_, &deferred_ }) {
            for (auto it = q->begin(); it != q->end(); ) { if ((*it)->cancel) it = q->erase(it); else ++it; }
        }
        cv_.wait(lk, [&] { for (auto & b : building_) if (b->cancel) return false; return true; });
        for (auto & s : gone) {
            if (s->state == READY) n_loaded_--;
            s->model = nil;
            if (!s->dir.empty()) [[NSFileManager defaultManager] removeItemAtPath:@(s->dir.c_str()) error:nil];
        }
    }

    // one row's head h (256 values) of storage format sc_ -> f16
    void deq_head(const uint8_t * row, int h, __fp16 * out) const {
        const uint8_t * x = row + (size_t) h * sc_.hb;
        if (sc_.is_q8 == 0) { memcpy(out, x, HD * 2); return; }
        if (sc_.is_q8 == 1) {
            for (int b = 0; b < HD / 32; b++) {
                const uint8_t * blk = x + b * 34; uint16_t dh; memcpy(&dh, blk, 2); const float d = h2f(dh);
                for (int i = 0; i < 32; i++) out[b * 32 + i] = (__fp16) (d * (float) (int8_t) blk[2 + i]);
            }
            return;
        }
        for (int b = 0; b < HD / 32; b++) {   // q4_0
            const uint8_t * blk = x + b * 18; uint16_t dh; memcpy(&dh, blk, 2); const float d = h2f(dh);
            for (int i = 0; i < 16; i++) {
                out[b * 32 + i]      = (__fp16) (d * (float) ((blk[2 + i] & 15) - 8));
                out[b * 32 + i + 16] = (__fp16) (d * (float) ((blk[2 + i] >> 4) - 8));
            }
        }
    }

    // 0 built, 1 failed, 2 deferred (wired memory: retried later)
    int build(slot & s, std::string & why) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool reload = s.state == SUSPENDED;
#if TARGET_OS_IPHONE
        // a loaded page model counts 1:1 against the app's memory limit (M2, docs/ANE.md)
        static const size_t reserve = (size_t) (getenv("PA_ANE_RESERVE_MB") ? atoi(getenv("PA_ANE_RESERVE_MB")) : 1024) << 20;
        if (os_proc_available_memory() < model_bytes_ + reserve) { why = "app memory"; return 2; }
#endif
        // the compile and load run partly in other processes (ANECompilerService): not while the whole phone is short of memory
        // (PA_ANE_SYS_FREE_MB, default 256, free + purgeable). The lowest value seen is in describe(), for tuning on devices.
        {
            static const double sys_min = getenv("PA_ANE_SYS_FREE_MB") ? atof(getenv("PA_ANE_SYS_FREE_MB")) : 256;
            const double f = sys_free_mb();
            std::lock_guard<std::mutex> lk(mu_);
            if (f > 0 && (min_sys_free_mb_ == 0 || f < min_sys_free_mb_)) min_sys_free_mb_ = f;
            if (TARGET_OS_IPHONE && f > 0 && f < sys_min) { n_sysfree_skip_++; why = "system memory"; return 2; }
        }
        // the ANE's model memory is wired (not in the app footprint): stay PA_ANE_WIRED_MARGIN_MB (default 800) under the ceiling,
        // leaving room for the KV still to come
        static const double margin = getenv("PA_ANE_WIRED_MARGIN_MB") ? atof(getenv("PA_ANE_WIRED_MARGIN_MB")) : 800;
        if (TARGET_OS_IPHONE && wired_mb() + model_bytes_ / 1048576.0 + margin > wired_max_mb()) {
            std::lock_guard<std::mutex> lk(mu_); n_wired_skip_++; why = "wired memory"; return 2;
        }
        NSError * err = nil;
        if (reload) goto load;
        char name[64]; snprintf(name, sizeof name, "/L%u_P%u.mlmodelc", s.layer, s.p);
        s.dir = cache_ + name;
        {
        NSFileManager * fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:@(s.dir.c_str()) error:nil];
        if (![fm copyItemAtPath:@(tmpl_.c_str()) toPath:@(s.dir.c_str()) error:&err]) { why = "copy template"; return 1; }
        const int fd = open((s.dir + "/weights/weight.bin").c_str(), O_RDWR);
        if (fd < 0) { why = "open weight.bin"; return 1; }
        const size_t N = pk_;
        std::vector<__fp16> Kp(N * (HD + 1)), Vt((size_t) HD * N), tmp((size_t) 64 * HD);
        bool ok = true;
        const auto tw = std::chrono::steady_clock::now();
        for (int h = 0; h < nkv_ && ok && !s.cancel; h++) {
            for (size_t j0 = 0; j0 < N; j0 += 64) {   // V^T in 64-key tiles: a row-at-a-time transpose wrote 32 KB apart
                const size_t nj = std::min<size_t>(64, N - j0);
                for (size_t jj = 0; jj < nj; jj++) {
                    const size_t j = j0 + jj;
                    const page & pg = s.src[j / PAGE];
                    const size_t off = (j % PAGE) * sc_.rs;
                    deq_head(pg.k + off, h, &Kp[j * (HD + 1)]);
                    Kp[j * (HD + 1) + HD] = (__fp16) 1.0f;
                    deq_head(pg.v + off, h, &tmp[jj * HD]);
                }
                for (int d = 0; d < HD; d++) {
                    __fp16 * dst = &Vt[(size_t) d * N + j0];
                    for (size_t jj = 0; jj < nj; jj++) dst[jj] = tmp[jj * HD + d];
                }
            }
            ok = pwrite(fd, Kp.data(), Kp.size() * 2, (off_t) data_off_[2 * h]) == (ssize_t) (Kp.size() * 2) &&
                 pwrite(fd, Vt.data(), Vt.size() * 2, (off_t) data_off_[2 * h + 1]) == (ssize_t) (Vt.size() * 2);
        }
        close(fd);
        { std::lock_guard<std::mutex> lk(mu_); weights_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count(); }
        if (!ok) { why = "write weights"; return 1; }
        if (s.cancel) { why = "cancelled"; return 1; }
        }
    load:
        MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const auto tl = std::chrono::steady_clock::now();
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(s.dir.c_str())] configuration:cfg error:&err];
        if (!model) { why = std::string("load: ") + (err ? err.localizedDescription.UTF8String : "?"); return 1; }
        const auto tu = std::chrono::steady_clock::now();
        { std::lock_guard<std::mutex> lk(mu_); load_ms_ += std::chrono::duration<double, std::milli>(tu - tl).count(); }
        {   // warm-up (the first prediction pays the ANE's program load)
            MLMultiArray * q = [[MLMultiArray alloc] initWithShape:@[@1, @(nkv_ * HD), @1, @(NR)] dataType:MLMultiArrayDataTypeFloat16 error:nil];
            MLMultiArray * c = [[MLMultiArray alloc] initWithShape:@[@1, @(nkv_), @1, @(NR)] dataType:MLMultiArrayDataTypeFloat16 error:nil];
            memset(q.dataPointer, 0, (size_t) q.count * 2); memset(c.dataPointer, 0, (size_t) c.count * 2);
            MLDictionaryFeatureProvider * in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{ @"q": [MLFeatureValue featureValueWithMultiArray:q], @"c": [MLFeatureValue featureValueWithMultiArray:c] } error:nil];
            if (![model predictionFromFeatures:in error:&err]) { why = "warm-up prediction"; return 1; }
        }
        std::lock_guard<std::mutex> lk(mu_);
        warm_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tu).count();
        if (suspended_) { why = "suspended"; return 2; }   // the tail started meanwhile: don't hold wired memory now
        s.model = model;
        if (reload) n_reloads_++;
        else { n_built_++; build_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
        return 0;
    }

    void build_loop() {
        // background work: never in the way of ATTN. PA_ANE_BUILD_QOS=1: user-initiated (P cores; builds finish sooner)
        static const bool fast = getenv("PA_ANE_BUILD_QOS") && atoi(getenv("PA_ANE_BUILD_QOS"));
        pthread_set_qos_class_self_np(fast ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY, 0);
        // 30 s (was 8): a long prompt arrives as batches with gaps between them, and reloading every page in a gap only to unload
        // them for the next batch was a load storm meeting the tail's wiring (#14). Decoding wakes them at once anyway (wake()).
        static const double resume_s = getenv("PA_ANE_RESUME_S") ? atof(getenv("PA_ANE_RESUME_S")) : 30;
        for (;;) {
            std::shared_ptr<slot> s;
            {
                std::unique_lock<std::mutex> lk(mu_);
                for (;;) {
                    if (stop_) return;
                    const auto now = std::chrono::steady_clock::now();
                    if (suspended_ && (wake_ || std::chrono::duration<double>(now - last_work_).count() > resume_s)) {
                        suspended_ = false; wake_ = false;   // the tail has been idle: reload the suspended pages, oldest first
                        for (size_t p = 0; ; p++) {
                            bool any = false;
                            for (auto & L : slots_) if (p < L.size()) { any = true; if (L[p]->state == SUSPENDED) deferred_.push_back(L[p]); }
                            if (!any) break;
                        }
                        retry_at_ = now;
                    }
                    if (!suspended_ && !deferred_.empty() && now >= retry_at_) {
                        while (!deferred_.empty()) { q_jobs_.push_front(deferred_.back()); deferred_.pop_back(); }
                    }
                    if (!suspended_ && !q_jobs_.empty()) {
                        // page-major: every layer's page 0, then every layer's page 1, ... A restore appends layer by layer, and
                        // building in that order left the last layers with no ANE page for minutes (140k: 78 pages x 1.9 s), so
                        // their prefill calls ran all keys on the GPU (416 ms vs ~110 per layer)
                        auto best = q_jobs_.begin();
                        for (auto it = q_jobs_.begin(); it != q_jobs_.end(); ++it)
                            if ((*it)->p < (*best)->p || ((*it)->p == (*best)->p && (*it)->layer < (*best)->layer)) best = it;
                        s = *best; q_jobs_.erase(best); building_.push_back(s); break;
                    }
                    cv_.wait_for(lk, std::chrono::milliseconds(500));
                }
            }
            std::string why;
            int rc = 1;
            @autoreleasepool { rc = build(*s, why); }
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (rc == 0 && !s->cancel) { s->state = READY; n_loaded_++; }
                else if (rc == 2 && !s->cancel) {
                    s->model = nil;   // stays QUEUED / SUSPENDED: retried once there is room (or the tail is idle again)
                    deferred_.push_back(s);
                    retry_at_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                } else {
                    s->state = FAILED; s->model = nil;
                    if (!s->cancel) { n_failed_++; error = "L" + std::to_string(s->layer) + " P" + std::to_string(s->p) + ": " + why; }
                    if (!s->dir.empty()) [[NSFileManager defaultManager] removeItemAtPath:@(s->dir.c_str()) error:nil];
                }
                building_.erase(std::find(building_.begin(), building_.end(), s));
            }
            cv_.notify_all();
        }
    }
};

}  // namespace pa

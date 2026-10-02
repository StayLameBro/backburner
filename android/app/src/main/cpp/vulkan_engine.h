// vulkan_engine.h - Android port of phone-attn/pa-metal.mm.
//
// pa-metal.mm runs partial attention over phone-held q8_0/q4_0/f16 pages on the
// Apple GPU's matrix/neural-accelerator units (pa_attn_q8, pa_attn_na*,
// pa_attn_big, pa_merge*). The pa::engine interface (phone-attn/phone-attn.h)
// is backend-agnostic, so Android implements the same interface on Vulkan
// compute (or OpenCL) without touching the protocol or the merge math.
//
// v1 STATUS: stub. begin() returns false, so the server in phone-attn.h runs
// every key on the CPU/SME path (exact, slower). This keeps the port
// wire-compatible and testable (pa-tool test PASS) before the shader work.
// Porting order when enabling:
//   1. pa_attn_q8 (decode, q8_0 pages) -> GLSL compute, one workgroup per
//      (head, key-chunk), online softmax, unnormalised (O, m, l) out.
//   2. pa_attn_big (prefill, dense rows, ATTN_BIG ng groups in one dispatch).
//   3. f16 direct-read path (pa_attn_na) for store_f16 pages.
//   4. q4_0 dequant path (pa_attn_naq / a.q==2).
// Tile sizes (PA_BIG_CFG default 32x64x8) must be retuned per Adreno/Mali/Immortalis.
#pragma once

#include <string>

#include "phone-attn.h"

namespace pa {

class vulkan_engine : public engine {
public:
    // v1: no GPU pages. Returning false makes attn_core fall back to SME/CPU
    // for the whole call (see the `gpu_on` branch in phone-attn.h).
    bool alloc_page(size_t bytes, page& p) override {
        (void)bytes; (void)p;
        return false;
    }
    void free_page(page& p) override { p = page(); }
    bool begin(const page* pages, const uint32_t* nkeys, int n_pages, const uint16_t* Qs,
               const config_req& cfg, int ng = 1) override {
        (void)pages; (void)nkeys; (void)n_pages; (void)Qs; (void)cfg; (void)ng;
        return false;  // stub: CPU handles all keys
    }
    double end(float* O, float* m, float* l) override {
        (void)O; (void)m; (void)l;
        return 0;
    }
    std::string describe() override { return "gpu off (vulkan stub: CPU/SME only)"; }
};

}  // namespace pa

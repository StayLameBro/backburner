// nnapi_page_engine.h - Android port of phone-attn/pa-ane.mm.
//
// pa-ane.mm compiles each 16,384-key page of a layer into a CoreML/ANE model
// with K/V as weights (docs/ANE.md: 279 -> 176 ms/token at 140k). The
// pa::page_engine interface (phone-attn/phone-attn.h) is delegate-agnostic:
// configure/queue/truncate/ready/run/shed/begin_use/end_use/wake.
//
// v1 STATUS: stub holding no pages (ready() == 0). Every key then runs on the
// CPU/SME (or Vulkan once enabled) path: exact, no background builds, no
// extra memory. Implement later with NNAPI (or QNN on Snapdragon) by mapping:
//   page_keys() = 16384 (keep the wire-compatible page size)
//   queue()     = build the page model off the ATTN thread (like build_loop)
//   run()       = burst execution over pages [0, n_pages), same (O, m, l) math
// Accuracy gate first: fp16 page weights matched the exact path 33/33 tokens
// on the iOS build; int8 did not and must not be used.
#pragma once

#include <cstdint>
#include <string>

#include "phone-attn.h"

namespace pa {

class nnapi_page_engine : public page_engine {
public:
    uint32_t page_keys() const override { return 16384; }
    void configure(const config_req& sc) override { (void)sc; }
    void queue(uint32_t layer, uint32_t p, const std::vector<page>& src) override {
        (void)layer; (void)p; (void)src;
    }
    void truncate(uint32_t n) override { (void)n; }
    uint32_t ready(uint32_t layer, uint32_t max_pages) override {
        (void)layer; (void)max_pages;
        return 0;  // stub: no ANE pages, GPU/SME handles all keys
    }
    bool run(uint32_t layer, uint32_t n_pages, const uint16_t* Qs, float* O, float* m,
             float* l) override {
        (void)layer; (void)n_pages; (void)Qs; (void)O; (void)m; (void)l;
        return false;
    }
    std::string describe() override { return "ane: off (nnapi stub)"; }
};

}  // namespace pa

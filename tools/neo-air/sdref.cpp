// sdref: greedy / teacher-forced next-token logits dump, through the normal llama_decode path (split decode or not).
// Takes llama's common CLI args (same as llama-server: -m -ngl -fa -c -ctk -ctv -ub -ot --no-repack ...). Env:
//   SDREF_PROMPT  text file (special tokens parsed)       SDREF_N        steps (64)
//   SDREF_FORCE   token-id file (one per line): feed these instead of the argmax (teacher forcing)
//   SDREF_OUT     prefix: writes <prefix>.tok (fed tokens), <prefix>.argmax, <prefix>.logits (N x n_vocab f32)
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "sdref: init failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream pf(getenv("SDREF_PROMPT"));
    std::stringstream ss; ss << pf.rdbuf();
    const auto prompt = common_tokenize(ctx, ss.str(), true, true);
    const int N = getenv("SDREF_N") ? atoi(getenv("SDREF_N")) : 64;
    std::vector<llama_token> force;
    if (const char * f = getenv("SDREF_FORCE")) { std::ifstream ff(f); llama_token t; while (ff >> t) force.push_back(t); }
    const std::string out = getenv("SDREF_OUT");
    fprintf(stderr, "sdref: prompt %zu tokens, %d steps, %s\n", prompt.size(), N, force.empty() ? "greedy" : "teacher-forced");

    llama_memory_clear(llama_get_memory(ctx), true);
    const int n_batch = llama_n_batch(ctx);
    double t0 = now_ms();
    for (size_t i = 0; i < prompt.size(); i += n_batch) {
        const int n = (int) std::min<size_t>(n_batch, prompt.size() - i);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int j = 0; j < n; j++) common_batch_add(b, prompt[i + j], (llama_pos) (i + j), { 0 }, i + j == prompt.size() - 1);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc) { fprintf(stderr, "sdref: prompt decode failed rc=%d\n", rc); return 3; }
    }
    const double t_pp = now_ms() - t0;

    FILE * fl = fopen((out + ".logits").c_str(), "wb");
    std::ofstream ft(out + ".tok"), fa(out + ".argmax");
    t0 = now_ms();
    for (int s = 0; s < N; s++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), n_vocab, fl);
        int am = 0;
        for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[am]) am = v;
        const llama_token tok = force.empty() ? am : force.at(s);
        fa << am << "\n"; ft << tok << "\n";
        if (s == N - 1) break;
        llama_batch b = llama_batch_init(1, 0, 1);
        common_batch_add(b, tok, (llama_pos) (prompt.size() + s), { 0 }, true);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc) { fprintf(stderr, "sdref: decode failed at step %d rc=%d\n", s, rc); return 4; }
    }
    fclose(fl);
    fprintf(stderr, "sdref: prompt %.0f ms (%.1f tok/s), %d steps %.0f ms (%.2f tok/s)\n", t_pp, prompt.size() * 1000.0 / t_pp, N,
            now_ms() - t0, (N - 1) * 1000.0 / (now_ms() - t0));
    return 0;
}

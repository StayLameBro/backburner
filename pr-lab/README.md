# llama.cpp PR mass-merge lab

Goal: merge 50+ (eventually 1k+) open upstream PRs for performance, 2–3 at a
time, with fully automated build/smoke/bench gates and self-healing (conflict /
build / runtime failures auto-quarantine the bad PR and keep going).

## Why this design (for 1k PRs)

Manual merging is impossible: PRs step on each other (same `ggml/src/*`,
`src/*` files), go stale, or break the build. So:

1. **Triage first, merge second.** `fetch_prs.py` scores all open PRs by
   perf keywords + perf paths, penalizes huge diffs, skips drafts /
   `do-not-merge`. Only the top-N get expensive detail API calls.
2. **Merge via `pull/N/head` refspec, not fork remotes.** `git fetch origin
   pull/N/head:pr/N` needs no token, no per-fork remote (unlike
   `scripts/pr2wt.sh`, which is for interactive single-PR worktrees).
3. **Small batches (2–3).** Limits blast radius; a bad batch bisects to
   singles. Pairs are never retried (combinatorial explosion).
4. **Gates:** configure+build → `llama-cli` smoke on TinyLlama → `llama-bench`
   pp/tg on 7B Q4_K_M. >5% tg regression quarantines the batch.
5. **Self-heal:** merge conflict / build fail / smoke fail / regression all
   quarantine the culprit(s) to `quarantined.json` and continue. Nothing stops
   the loop. Full log in `lab-log.jsonl`.

## Layout

- Lab checkout (fast Linux fs, NOT /mnt/c): `~/llama-pr-lab/llama.cpp`
- Models: `~/llama-pr-lab/models/{tinyllama.gguf,qwen2.5-7b.gguf}`
- This dir (`pr-lab/` in LLMDroid, mirrored to WSL) holds the automation.
- State (in WSL lab dir or here): `lab-state.json`, `lab-log.jsonl`,
  `quarantined.json`, `candidates.json`

## Run (all in WSL Ubuntu)

```bash
# 0. toolchain (once)
sudo apt-get update && sudo apt-get install -y cmake ninja-build gcc g++ curl jq pkg-config libcurl4-openssl-dev ccache
pip3 install huggingface_hub

# 1. clone + models
git clone https://github.com/ggml-org/llama.cpp.git ~/llama-pr-lab/llama.cpp
bash pr-lab/fetch_models.sh   # tiny (~0.7GB) + 7B Q4_K_M (~4.7GB)

# 2. triage (set GH_TOKEN to avoid 60/hr limit)
export GH_TOKEN=xxx
python3 pr-lab/fetch_prs.py --limit 1000 --top 100 --out candidates.json

# 3. plan only
python3 pr-lab/merge_lab.py --candidates candidates.json --dry-run --batch 3 --max-prs 50

# 4. baseline build + bench (proves harness before any PR)
cmake -S ~/llama-pr-lab/llama.cpp -B ~/llama-pr-lab/llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama-pr-lab/llama.cpp/build --target llama-cli llama-bench -j4
~/llama-pr-lab/llama.cpp/build/bin/llama-cli -m ~/llama-pr-lab/models/tinyllama.gguf -p "Hello" -n 20 --temp 0
~/llama-pr-lab/llama.cpp/build/bin/llama-bench -m ~/llama-pr-lab/models/qwen2.5-7b.gguf -p 128 -n 128 -o json

# 5. auto-merge loop (2-3 at a time, up to 50)
python3 pr-lab/merge_lab.py --candidates candidates.json --batch 3 --max-prs 50
```

Resume: re-run step 5; it loads `lab-state.json` and skips merged/quarantined.

## What to expect

- ~17 builds for 50 PRs in batches of 3, ~5–10 min each on 4 cores → a few hours.
  Use `ccache` (auto-detected) to cut rebuilds.
- Most of 1k+ PRs will NOT be perf PRs and many conflict; expect high quarantine
  rate. That's normal — the log tells you which ones and why.
- Interaction conflicts (A+B fail but each passes alone) are quarantined as a
  pair for manual review — not retried automatically.

## Rolling fixes (learned the hard way)

- **v1 bug — stacked uncommitted merges are impossible.** `git merge --no-commit`
  twice in a row fails with "MERGE_HEAD exists"; the 2nd `merge --abort` wipes
  the 1st PR too, and an unchecked `git commit` on the clean tree silently
  records a phantom merge. Fixed in v3: **one commit per PR**
  (`merge_one_committed`, verified 2-parent merge commit), batch = N commits.
  Revert-one = `git reset --hard HEAD~1` (local branch, never pushed: safe).
- **Always check `git commit` rc + verify HEAD advanced.** State must never
  record uncommitted work. `--doctor` reconciles state vs repo and requeues
  quarantines whose detail contains "MERGE_HEAD exists".
- **Stream logs unbuffered** (`python3 -u`, log file + `tail -f`, never
  `| tail -30` on a live process — it buffers until exit and the run looks dead).
- **llama-cli 0.5.0-dev is silent non-interactively** (server-backend refactor);
  smoke gate uses `llama-bench -p 32 -n 32` on TinyLlama instead.
- **Models: curl-direct, not HF API.** `huggingface_hub` hit 401 for public
  repos; `curl -L <repo>/resolve/main/<file>?download=true` works. Qwen2.5-7B
  Q4_K_M is **split** (`...-00001-of-00002` + `...-00002-of-00002`); point
  llama tools at part 1. TinyLlama: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
- **WSL `~/.wslconfig` duplicate-key warnings** (`wsl2.memory`/`processors`)
  are harmless noise from the user's config; ignore.

## Results (2026-10-02, WSL Ubuntu, 4 vCPU, CPU backend, threads=2)

22/30 triaged PRs merged (`pr-lab/base-bed0a856` atop `bed0a8566`), 8
quarantined with genuine content conflicts, `EXIT:0`, state==repo verified.

| model | clean base `bed0a8566` | 22-merge HEAD `84244bf6d` | verdict |
|---|---|---|---|
| TinyLlama-1.1B Q4_K_M pp32 | 164.39 ± 5.56 | 160.39 ± 3.83 | parity (-2.4%, in noise) |
| TinyLlama-1.1B Q4_K_M tg32 | 42.22 ± 0.20 | 38.05 ± 1.54 | parity (-10%, tg noisy on this box) |
| Qwen2.5-7B Q4_K_M pp16/tg16 | — | 15.79 / 3.71 | loads + runs (regression reference: tg32 1.93) |

No CPU regression (expected: merged diffs are metal/cuda/sycl backends).
CPU gains not expected from this batch; GPU-backend gains (metal multi-column
mat-vec #29110, CUDA top-k #28671, cooperative softmax #28982, SYCL FA #29171)
materialize on those backends. Early "baseline" 121/21 numbers were
machine-load noise (downloads+builds concurrent) — always compare same-state.

## Next scale-up

- Full 1k triage needs `GH_TOKEN` (unauthenticated API caps at 60 req/hr;
  1k PRs need ~2000 detail calls). Then: `fetch_prs.py --limit 1000 --top 200`.
- `candidates.json` (30) is exhausted: 22 merged + 8 quarantined. Next run needs
  fresh triage output.
- Keep lab worktrees under `~/llama-pr-lab/` (persistent); WSL `/tmp` is wiped
  on restart (lost one base comparison to this).

- Add CI status check (`combinedStatus`) to skip red PRs before building.
- Add `llama-perplexity` gate on WikiText-2 sample for correctness.
- Nightly rebase: `git fetch origin master`, re-triage (PRs close/merge daily).

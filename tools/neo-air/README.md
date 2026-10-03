# neo-air: the split-decode quality harness

Checks that split decode gives the same answers as the Mac alone ([docs/SPLIT-DECODE.md](../../docs/SPLIT-DECODE.md)).

- `sdref` runs a prompt through the normal `llama_decode` path (llama-server's CLI arguments) and dumps the full logits row
  at each of N steps: greedy, or teacher-forced on a token file.
- `cmp.py` compares two dumps position by position: top-1 agreement and KL(ref ‖ test) over the whole vocabulary, in f64.

```bash
tools/neo-air/build.sh        # -> tools/neo-air/bin/sdref (BUILD=<dir> for another llama.cpp build)

# 1. Mac-only reference (no split engine). On 8 GB the whole model doesn't fit the GPU, so the CPU computes layers >= 20
#    from the mapped file: ~10 s per token, ~20 min per prompt. On a bigger Mac drop the -ot / -t arguments.
OT=(-ot '^blk\.(20|[1-9][0-9][0-9]+|[3-9][0-9]|2[1-9])\.=CPU' -ot '^output=CPU' --no-repack)
for p in short code text; do tools/neo-air/run-ref.sh $p "${OT[@]}" -t 6 -tb 6; done

# 2. Split decode at L=20, teacher-forced on the reference's tokens (phone up, the tail with its head loaded)
for p in short code text; do TAIL=169.254.x.y:50060 tools/neo-air/run-split.sh $p force; done

# 3. Compare (needs numpy)
for p in short code text; do uv run --with numpy tools/neo-air/cmp.py tools/neo-air/out/ref_$p tools/neo-air/out/sd_force_$p; done
```

- Output goes to `tools/neo-air/out/` (or `OUT=`); `bin/` and `out/` are git-ignored. `MODEL` defaults to
  `~/Models/Qwen3.8-27B-IQ2_XS.gguf`, `LLAMA_SPLIT_L` to 20.
- Prompts: `prompts/p_short.txt` (a 19-token chat), `p_code.txt` (1,059 tokens: `scripts/split-gguf.py`), `p_text.txt`
  (5,275 tokens: README + docs/ANE.md).
- `run-split.sh NAME greedy` is the free-running greedy output, for checking two builds give identical tokens
  (`cmp out/a_greedy_short.tok out/b_greedy_short.tok`, with `TAG=a` / `TAG=b`).

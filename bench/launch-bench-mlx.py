#!/usr/bin/env python3
"""launch-bench-mlx.py - the MLX row of the published speed table: mlx-lm with Qwen3.8-27B 4-bit (MLX's own format, group 64),
the same prompt as bench/launch-bench.py (bench/prompt.txt), the same steps.

  1. prefill depth curve: --step appends up to --max, each pushed through the model in mlx-lm's default 2048-token chunks
     against one prompt cache (prefill speed at that depth; the running sum is the cold read time);
  2. decode at --max: append the same question, generate --gen tokens greedily (mlx_lm.stream_generate);
  3. decode at 8192: a fresh cache read to 8192, then the same (the hybrid model's recurrent cache can't be rewound).
Rows go to bench/results/results.jsonl with config "mlx" and the prompt's sha256. Needs mlx-lm (any venv):
PY=... bench/launch-bench-mlx.py

  $VENV/bin/python bench/launch-bench-mlx.py
"""
import argparse, hashlib, json, os, time
import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTION = '\n\n/* Review of the code above: what each function does, in order, one short paragraph each. */\n'
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.expanduser('~/Models/qwen38-27b-mlx-4bit'))
ap.add_argument('--max', type=int, default=32768)
ap.add_argument('--step', type=int, default=4096)
ap.add_argument('--chunk', type=int, default=2048, help="mlx-lm's prefill step size (its default)")
ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--kv-bits', type=int, default=8, help='quantized KV cache for the attention layers (0 = mlx default f16); 8 matches q8_0')
ap.add_argument('--cache-limit-gb', type=float, default=1.0, help="cap on MLX's buffer cache (freed buffers kept for reuse); "
                'the default (no cap) grew into swap on a 24 GB Mac')
ap.add_argument('--note', default='')
ap.add_argument('--prompt', default=os.path.join(ROOT, 'bench', 'prompt.txt'))
a = ap.parse_args()
OUT = os.path.join(ROOT, 'bench', 'results')
PROMPT = open(a.prompt, 'rb').read()
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)
base = dict(config='mlx', rep=1, version=f'mlx-lm {mlx_lm.__version__}, mlx {mx.__version__}', model=os.path.basename(a.model),
            kv=f'{a.kv_bits}-bit g64' if a.kv_bits else 'f16 (mlx default)', cache_limit_gb=a.cache_limit_gb, note=a.note,
            prompt_sha256=hashlib.sha256(PROMPT).hexdigest())
if a.cache_limit_gb > 0:
    mx.set_cache_limit(int(a.cache_limit_gb * (1 << 30)))


def new_cache():
    cache = make_prompt_cache(model)
    if a.kv_bits:   # only the full-attention layers have a KV cache; the GDN layers keep their recurrent state
        cache = [QuantizedKVCache(group_size=64, bits=a.kv_bits) if type(c) is KVCache else c for c in cache]
    return cache


def record(kind, **kw):
    with open(os.path.join(OUT, 'results.jsonl'), 'a') as f:
        f.write(json.dumps(dict(kind=kind, time=time.strftime('%Y-%m-%d %H:%M:%S'), **base, **kw)) + '\n')


model, tok = load(a.model)
toks = tok.encode(PROMPT.decode('utf-8'))
q = tok.encode(QUESTION)
say(f'{base["version"]}: {len(toks)} prompt tokens available')


def prefill(cache, ids):
    for i in range(0, len(ids), a.chunk):
        model(mx.array(ids[i:i + a.chunk])[None], cache=cache)
        mx.eval([c.state for c in cache])


def decode(cache, depth):
    last = None
    for r in stream_generate(model, tok, q, max_tokens=a.gen, sampler=make_sampler(temp=0.0), prompt_cache=cache):
        last = r
    say(f'  decode at {depth:6d}: {last.generation_tokens} tok at {last.generation_tps:5.1f} tok/s')
    record('decode', depth=depth, predicted_n=last.generation_tokens, tok_s=last.generation_tps, peak_gb=last.peak_memory)


cache = new_cache()
total, n = 0.0, 0
while n < a.max:
    n1 = min(a.max, n + a.step)
    t0 = time.time()
    prefill(cache, toks[n:n1])
    dt = time.time() - t0
    total += dt
    say(f'  prefill {n:6d} -> {n1:6d}: {n1 - n} tok in {dt:6.1f} s = {(n1 - n) / dt:6.1f} tok/s, cumulative {total:.0f} s')
    record('prefill', depth0=n, depth1=n1, prompt_n=n1 - n, prompt_ms=dt * 1000, cum_ms=total * 1000)
    n = n1
record('prefill_total', depth1=a.max, cum_ms=total * 1000, tok_s=a.max / total)
say(f'  cold read of {a.max} tokens: {total:.0f} s, {a.max / total:.1f} tok/s; peak {mx.get_peak_memory() / 1e9:.1f} GB')
decode(cache, a.max)
del cache
cache = new_cache()
prefill(cache, toks[:8192])
decode(cache, 8192)

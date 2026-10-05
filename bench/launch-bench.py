#!/usr/bin/env python3
"""launch-bench.py - the published speed table: stock llama.cpp vs this fork (Mac only) vs this fork + iPhone, same model,
same KV type, same prompts.

One server load per config, then:
  1. Prefill depth curve: real source code (bench/prompt.txt: llama.cpp's src/*.cpp at a pinned commit, written by
     bench/make-prompt.py, so every checkout reads the same prompt) fed in --step-token appends up to --max
     tokens. Each append extends the cached prompt, so its timing is "prefill speed at that depth"; the running sum is the
     cold time to read the whole prompt. At each --save depth the slot is saved.
  2. Decode at depth: restore each saved slot, append a fixed question, generate --gen tokens greedily. Reports tok/s, the
     generated token ids and the top-10 log-probs of the first 64 (bench/launch-quality.py compares configs with them).
Every measurement is one JSON line in bench/results/runs.jsonl (untracked; --out to change) with the config, versions,
settings, the run's run_id and the prompt's sha256. bench/results/results.jsonl holds the published rows and is not written.

  bench/launch-bench.py --config stock
  bench/launch-bench.py --config fork-mac
  bench/launch-bench.py --config fork-phone
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser('~/Models/Qwen3.8-27B-IQ4_XS.gguf')
STOCK = '/opt/homebrew/bin/llama-server'
QUESTION = '\n\n/* Review of the code above: what each function does, in order, one short paragraph each. */\n'

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True, choices=['stock', 'stock-spec', 'fork-nospec', 'fork-mac', 'fork-phone'])
ap.add_argument('--max', type=int, default=61440)
ap.add_argument('--step', type=int, default=4096)
ap.add_argument('--save', default='8192,32768,61440')
ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--rep', type=int, default=1, help='label only: which repeat this is')
ap.add_argument('--ctx', type=int, default=65536)
ap.add_argument('--port', type=int, default=8097)
ap.add_argument('--note', default='')
ap.add_argument('--prompt', default=os.path.join(ROOT, 'bench', 'prompt.txt'))
ap.add_argument('--out', default=os.path.join(ROOT, 'bench', 'results', 'runs.jsonl'))
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
OUT = os.path.join(ROOT, 'bench', 'results')   # server logs and saved slots
PROMPT = open(a.prompt, 'rb').read()
PROMPT_SHA = hashlib.sha256(PROMPT).hexdigest()
RUN_ID = time.strftime('%Y%m%d-%H%M%S-') + a.config + '-' + os.urandom(2).hex()
SLOTS = os.path.join(OUT, 'slots', a.config)
os.makedirs(SLOTS, exist_ok=True)
saves = sorted(int(x) for x in a.save.split(','))
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


def post(path, body, timeout=7200):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'}),
                               timeout=timeout)
    return json.loads(r.read())


def git(*args):
    try:
        return subprocess.run(['git', '-C', *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


def record(kind, **kw):
    row = dict(kind=kind, config=a.config, rep=a.rep, time=time.strftime('%Y-%m-%d %H:%M:%S'), note=a.note, run_id=RUN_ID,
               prompt_sha256=PROMPT_SHA, **kw)
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')


# ---- server
log_path = os.path.join(OUT, f'server-{a.config}-rep{a.rep}.log')
log = open(log_path, 'w')
DRAFT = os.path.expanduser('~/Models/dflash2-v2-q4km-self16.gguf')
if a.config.startswith('stock'):
    cmd = [STOCK, '-m', MODEL, '-ngl', '999', '-fa', 'on', '-c', str(a.ctx), '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '--slot-save-path', SLOTS + '/', '--host', '127.0.0.1', '--port', str(a.port), '--jinja']
    if a.config == 'stock-spec':   # stock's own speculative decoding with the same drafter and draft settings as the fork
        # STOCK_SPEC overrides (stock can't load the fork's DFlash2 files: 81 tensors, it knows 58)
        cmd += os.environ.get('STOCK_SPEC', '--spec-type ngram-simple,draft-dflash -md ' + os.path.expanduser('~/Models/dflash2-v2-q4km-self16.gguf') + ' '
                              '-ngld 999 --spec-draft-n-max 7').split()
    version = subprocess.run([STOCK, '--version'], capture_output=True, text=True).stderr.strip().splitlines()[0]
    env = dict(os.environ)
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
else:
    env = dict(os.environ, PROXY='0', PORT=str(a.port), CTX=str(a.ctx), KV='q8_0', CACHE_DIR=SLOTS)
    if a.config in ('fork-mac', 'fork-nospec'):
        env['PHONE'] = '0'
    if a.config == 'fork-nospec':   # the fork's kernels without speculative decoding
        env['SPEC_TYPE'] = 'none'
    cmd = [f'{ROOT}/scripts/serve.sh']
    version = 'fork ' + git(f'{ROOT}/llama.cpp', 'rev-parse', '--short', 'HEAD') + ' (integration ' + git(ROOT, 'rev-parse', '--short', 'HEAD') + ')'
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
say(f'{a.config}: {version}; log {log_path}')
say(f'run {RUN_ID}, prompt {a.prompt} sha256 {PROMPT_SHA[:16]}, rows to {a.out}')
try:
    for _ in range(600):
        try:
            if urllib.request.urlopen(URL + '/health', timeout=2).status == 200:
                break
        except Exception:
            pass
        if srv.poll() is not None:
            sys.exit(f'server exited, see {log_path}')
        time.sleep(1)
    phone = ''
    if a.config == 'fork-phone':
        phone = next((l.strip() for l in open(log_path) if 'serve:' in l), '')
        if 'split prefill on' not in phone:
            sys.exit(f'fork-phone: the phone is not in use: {phone!r}')
    say('server up', phone)
    base = dict(version=version, model=os.path.basename(MODEL), kv='q8_0', ctx=a.ctx, mac=platform.node(),
                phone=phone)

    # ---- prompt tokens: the pinned prompt file
    toks = post('/tokenize', {'content': PROMPT.decode('utf-8')})['tokens']
    if len(toks) < a.max:
        sys.exit(f'only {len(toks)} tokens in {a.prompt}, need {a.max}')
    q = post('/tokenize', {'content': QUESTION})['tokens']

    # ---- 1. prefill depth curve
    total_ms, n = 0.0, 0
    while n < a.max:
        n1 = min(a.max, n + a.step)
        t0 = time.time()
        r = post('/completion', {'prompt': toks[:n1], 'n_predict': 0, 'cache_prompt': True, 'temperature': 0})
        wall = time.time() - t0
        tm = r.get('timings', {})
        pn, pms = tm.get('prompt_n', 0), tm.get('prompt_ms', 0)
        if pn < n1 - n - 8:
            say(f'  warning: expected ~{n1 - n} new tokens, server read {pn} (cache_n {tm.get("cache_n")})')
        total_ms += pms
        say(f'  prefill {n:6d} -> {n1:6d}: {pn} tok in {pms/1000:6.1f} s = {pn/(pms/1000):6.1f} tok/s (wall {wall:.1f} s), cumulative {total_ms/1000:.0f} s')
        record('prefill', depth0=n, depth1=n1, prompt_n=pn, prompt_ms=pms, wall_s=wall, cum_ms=total_ms, **base)
        n = n1
        if n in saves:
            post(f'/slots/0?action=save', {'filename': f'd{n}.bin'})
    record('prefill_total', depth1=a.max, cum_ms=total_ms, tok_s=a.max / (total_ms / 1000), **base)
    say(f'  cold read of {a.max} tokens: {total_ms/1000:.0f} s, {a.max/(total_ms/1000):.1f} tok/s')

    # ---- 2. decode at depth
    for d in saves:
        post(f'/slots/0?action=restore', {'filename': f'd{d}.bin'})
        t0 = time.time()
        r = post('/completion', {'prompt': toks[:d] + q, 'n_predict': a.gen, 'cache_prompt': True, 'temperature': 0,
                                 'ignore_eos': True, 'n_probs': 10, 'return_tokens': True})
        tm = r.get('timings', {})
        gn, gms = tm.get('predicted_n', 0), tm.get('predicted_ms', 0)
        dn, da = tm.get('draft_n', 0), tm.get('draft_n_accepted', 0)
        say(f'  decode at {d:6d}: {gn} tok in {gms/1000:.1f} s = {gn/(gms/1000):5.1f} tok/s' + (f' (drafts {da}/{dn})' if dn else ''))
        record('decode', depth=d, predicted_n=gn, predicted_ms=gms, tok_s=gn / (gms / 1000), draft_n=dn, draft_accepted=da,
               text_head=r.get('content', '')[:160], tokens=r.get('tokens', []),
               probs=[[(p.get('id'), round(p.get('logprob', 0), 5)) for p in (c.get('top_logprobs') or [])]
                      for c in (r.get('completion_probabilities') or [])[:64]], **base)
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
    if not a.config.startswith('stock'):   # serve.sh leaves its llama-server behind on TERM only if it was mid-trap
        subprocess.run(['pkill', '-f', f'llama-server .*--port {a.port}'])

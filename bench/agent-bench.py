#!/usr/bin/env python3
"""agent-bench.py - the everyday case: a coding-agent session from a cold start, timed the way a user feels it.

  1. a new session: a ~17.5k-token preamble (system prompt + project docs, like omp's) and a short first message,
  2. then --turns turns of: a ~1.5k-token tool result (real source files) + a one-line instruction, --gen tokens answered.
Prompts are token arrays extended exactly (each turn = previous prompt + the model's own answer + the new turn), so every
engine reuses its cache the same way. Greedy, fixed answer length. Reports per turn: wait before the first token (prefill),
answer speed, and the whole session's wall time. Rows go to bench/results/agent.jsonl.

  bench/agent-bench.py --config stock|fork-mac|fork-phone
"""
import argparse, glob, json, os, subprocess, sys, time, urllib.request
from scrub import scrub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser('~/Models/Qwen3.8-27B-IQ4_XS.gguf')
STOCK = '/opt/homebrew/bin/llama-server'

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True, choices=['stock', 'stock-spec', 'fork-mac', 'fork-phone'])
ap.add_argument('--preamble', type=int, default=17500)
ap.add_argument('--turns', type=int, default=6)
ap.add_argument('--tool', type=int, default=1500)
ap.add_argument('--gen', type=int, default=150)
ap.add_argument('--rep', type=int, default=1)
ap.add_argument('--port', type=int, default=8096)
ap.add_argument('--note', default='')
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
OUT = os.path.join(ROOT, 'bench', 'results')
os.makedirs(OUT, exist_ok=True)
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


def post(path, body, timeout=7200):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'}),
                               timeout=timeout)
    return json.loads(r.read())


def git(*args):
    return subprocess.run(['git', '-C', *args], capture_output=True, text=True).stdout.strip()


log_path = os.path.join(OUT, f'agent-server-{a.config}-rep{a.rep}.log')
log = open(log_path, 'w')
cache = os.path.join(OUT, 'agent-cache', a.config)
os.makedirs(cache, exist_ok=True)
if a.config.startswith('stock'):
    version = subprocess.run([STOCK, '--version'], capture_output=True, text=True).stderr.strip().splitlines()[0]
    cmd = [STOCK, '-m', MODEL, '-ngl', '999', '-fa', 'on', '-c', '65536', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '--host', '127.0.0.1', '--port', str(a.port), '--jinja']
    if a.config == 'stock-spec':   # stock's own speculative decoding, same drafter and settings as the fork
        # STOCK_SPEC overrides (stock can't load the fork's DFlash2 files: 81 tensors, it knows 58)
        cmd += os.environ.get('STOCK_SPEC', '--spec-type ngram-simple,draft-dflash -md ' + os.path.expanduser('~/Models/dflash2-v2-q4km-self16.gguf') + ' '
                              '-ngld 999 --spec-draft-n-max 7').split()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
else:
    env = dict(os.environ, PROXY='0', PORT=str(a.port), CTX='65536', KV='q8_0', CACHE_DIR=cache)
    if a.config == 'fork-mac':
        env['PHONE'] = '0'
    version = 'fork ' + git(f'{ROOT}/llama.cpp', 'rev-parse', '--short', 'HEAD') + ' (integration ' + git(ROOT, 'rev-parse', '--short', 'HEAD') + ')'
    srv = subprocess.Popen([f'{ROOT}/scripts/serve.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
say(f'{a.config}: {version}')
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
    phone = scrub(next((l.strip() for l in open(log_path) if 'serve:' in l), '')) if not a.config.startswith('stock') else ''
    if a.config == 'fork-phone' and 'split prefill on' not in phone:
        sys.exit(f'fork-phone: the phone is not in use: {phone!r}')

    tok = lambda s: post('/tokenize', {'content': s})['tokens']
    docs = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob(f'{ROOT}/llama.cpp/docs/*.md')))
    pre = tok('<|im_start|>system\nYou are a coding agent working in this repository. Project notes follow.\n\n' + docs)[:a.preamble]
    srcs = sorted(glob.glob(f'{ROOT}/llama.cpp/tools/server/*.cpp') + glob.glob(f'{ROOT}/llama.cpp/common/*.cpp'))
    tools = [tok(open(f, errors='replace').read())[:a.tool] for f in srcs[:a.turns]]
    asks = ['What does this file do? Answer briefly.', 'Find one bug risk in it.', 'Which function would you test first, and why?',
            'Suggest a smaller name for the longest function.', 'What does it depend on?', 'Summarize what we learned so far.']

    prompt = pre + tok('<|im_end|>\n<|im_start|>user\nLet us review some files.<|im_end|>\n<|im_start|>assistant\n')
    t_session = time.time()
    rows = []
    for turn in range(a.turns + 1):
        if turn > 0:
            prompt += tok('<|im_end|>\n<|im_start|>user\n<tool_result>\n') + tools[turn - 1] + \
                      tok(f'\n</tool_result>\n{asks[(turn - 1) % len(asks)]}<|im_end|>\n<|im_start|>assistant\n')
        r = post('/completion', {'prompt': prompt, 'n_predict': a.gen, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True,
                                 'return_tokens': True})
        tm = r.get('timings', {})
        pn, pms, gn, gms = tm.get('prompt_n', 0), tm.get('prompt_ms', 0), tm.get('predicted_n', 0), tm.get('predicted_ms', 0)
        row = dict(kind='agent_turn', config=a.config, rep=a.rep, turn=turn, depth=len(prompt), prompt_n=pn, prompt_ms=pms,
                   predicted_n=gn, predicted_ms=gms, gen_tok_s=gn / (gms / 1000) if gms else 0, version=version, phone=phone,
                   note=a.note, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        rows.append(row)
        with open(os.path.join(OUT, 'agent.jsonl'), 'a') as f:
            f.write(json.dumps(row) + '\n')
        say(f'  turn {turn}: depth {len(prompt):6d}, read {pn:5d} tok in {pms/1000:6.1f} s ({pn/(pms/1000) if pms else 0:6.1f} tok/s), '
            f'wrote {gn} at {row["gen_tok_s"]:5.1f} tok/s')
        prompt += r.get('tokens') or tok(r.get('content', ''))
    wall = time.time() - t_session
    total = dict(kind='agent_session', config=a.config, rep=a.rep, wall_s=wall, first_wait_s=rows[0]['prompt_ms'] / 1000,
                 turn_wait_s=sum(r['prompt_ms'] for r in rows[1:]) / 1000 / max(1, len(rows) - 1),
                 gen_tok_s=sum(r['predicted_n'] for r in rows) / (sum(r['predicted_ms'] for r in rows) / 1000),
                 version=version, phone=phone, note=a.note, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(OUT, 'agent.jsonl'), 'a') as f:
        f.write(json.dumps(total) + '\n')
    say(f'  session: {wall:.0f} s; first answer after {total["first_wait_s"]:.0f} s; later turns wait {total["turn_wait_s"]:.1f} s; '
        f'writing {total["gen_tok_s"]:.1f} tok/s')
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
    if not a.config.startswith('stock'):
        subprocess.run(['pkill', '-f', f'llama-server .*--port {a.port}'])

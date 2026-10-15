#!/usr/bin/env python3
"""long-bench.py - the one long run: past the Mac's own 64k cells with the iPhone holding the oldest KV, at full quality (q8_0).

A fact is planted near the start ("needle"), then real source code (llama.cpp src/*.cpp + common/*.cpp) is read in --step
appends to --max tokens (prefill speed at each depth). Then the model is asked for the fact (recall at full depth) and writes
--gen tokens (decode speed at depth). Rows go to bench/results/long.jsonl.

  bench/long-bench.py --max 131072
"""
import argparse, glob, json, os, subprocess, sys, time, urllib.request
from scrub import scrub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# three facts at different depths: at 128k with the phone, the first two sit in the part the phone holds (the oldest 64k)
NEEDLES = [(1500, 'Note for later: the release codename chosen by the team is "violet harbor 4417".\n', 'violet harbor 4417'),
           (40000, 'Note for later: the backup server is named "copper finch 2093".\n', 'copper finch 2093'),
           (100000, 'Note for later: the test account password hint is "silent maple 7718".\n', 'silent maple 7718')]
ASK = ('\n\n<|im_end|>\n<|im_start|>user\nThree notes for later appear in the text above. List the release codename, the backup '
       'server name and the password hint, one per line, nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
ap = argparse.ArgumentParser()
ap.add_argument('--max', type=int, default=131072)
ap.add_argument('--step', type=int, default=8192)
ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--port', type=int, default=8095)
ap.add_argument('--config', default='phone', choices=['phone', 'mac-q4'],
                help='phone: Mac keeps 64k at q8_0, the iPhone the rest; mac-q4: Mac alone, q4_0 KV for the whole context '
                     '(the only way a 24 GB Mac fits 128k)')
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
OUT = os.path.join(ROOT, 'bench', 'results')
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


def post(path, body, timeout=7200):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'}),
                               timeout=timeout)
    return json.loads(r.read())


def record(**kw):
    with open(os.path.join(OUT, 'long.jsonl'), 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)) + '\n')


log_path = os.path.join(OUT, 'server-long.log')
env = dict(os.environ, PROXY='0', PORT=str(a.port), CTX='65536', KV='q8_0', CACHE_DIR=os.path.join(OUT, 'slots', 'long'),
           CTX_TOTAL=str(max(a.max + 4096, 65536)))
if a.config == 'mac-q4':
    env.update(PHONE='0', CTX=str(a.max + 4096), KV='q4_0')
    env.pop('CTX_TOTAL')
os.makedirs(env['CACHE_DIR'], exist_ok=True)
srv = subprocess.Popen([f'{ROOT}/scripts/serve.sh'], cwd=ROOT, env=env, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
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
    phone = scrub(next((l.strip() for l in open(log_path) if 'serve:' in l), ''))
    say('server up', phone)
    if a.config == 'phone' and 'remote KV on' not in phone:
        sys.exit('the phone is not holding KV: ' + phone)
    text = ''
    for f in sorted(glob.glob(f'{ROOT}/llama.cpp/src/*.cpp') + glob.glob(f'{ROOT}/llama.cpp/common/*.cpp') +
                    glob.glob(f'{ROOT}/llama.cpp/ggml/src/*.c')):
        text += f'\n// ===== {os.path.basename(f)} =====\n' + open(f, errors='replace').read()
        if len(text) > a.max * 5:
            break
    head = post('/tokenize', {'content': '<|im_start|>user\n'})['tokens']
    toks = post('/tokenize', {'content': text})['tokens']
    for at, sentence, _ in reversed(NEEDLES):   # from the deepest, so earlier insertions don't shift the later positions
        toks = toks[:at] + post('/tokenize', {'content': sentence})['tokens'] + toks[at:]
    toks = head + toks
    if len(toks) < a.max:
        sys.exit(f'only {len(toks)} tokens of text')
    total, n = 0.0, 0
    while n < a.max:
        n1 = min(a.max, n + a.step)
        r = post('/completion', {'prompt': toks[:n1], 'n_predict': 0, 'cache_prompt': True, 'temperature': 0})
        tm = r.get('timings', {})
        pn, pms = tm.get('prompt_n', 0), tm.get('prompt_ms', 0)
        total += pms
        say(f'  prefill {n:6d} -> {n1:6d}: {pn} tok in {pms/1000:6.1f} s = {pn/(pms/1000):6.1f} tok/s, cumulative {total/1000:.0f} s')
        record(kind='prefill', config=a.config, depth0=n, depth1=n1, prompt_n=pn, prompt_ms=pms, cum_ms=total, phone=phone)
        n = n1
    say(f'  cold read of {a.max}: {total/1000:.0f} s ({a.max/(total/1000):.1f} tok/s)')
    q = post('/tokenize', {'content': ASK})['tokens']
    r = post('/completion', {'prompt': toks[:a.max] + q, 'n_predict': 64, 'cache_prompt': True, 'temperature': 0,
                             'return_tokens': True})
    tm = r.get('timings', {})
    ans = r.get('content', '')
    found = [key in ans.lower() for _, _, key in NEEDLES]
    say(f'  recall at {a.max}: {sum(found)} of {len(found)} ({", ".join(f"{at}: {"yes" if f else "no"}" for (at, _, _), f in zip(NEEDLES, found))}): {ans[:160]!r}')
    record(kind='recall', config=a.config, depth=a.max, found=sum(found), of=len(found),
           where=[at for at, _, _ in NEEDLES], hits=found, answer=ans[:400], cold_read_s=total / 1000, phone=phone)
    # writing speed at full depth: continue the same conversation (an exact extension of the cache), fixed length
    q2 = post('/tokenize', {'content': '<|im_end|>\n<|im_start|>user\nNow explain what the last file above does.<|im_end|>\n'
                                       '<|im_start|>assistant\n<think>\n\n</think>\n\n'})['tokens']
    r2 = post('/completion', {'prompt': toks[:a.max] + q + (r.get('tokens') or []) + q2, 'n_predict': a.gen, 'cache_prompt': True,
                              'temperature': 0, 'ignore_eos': True})
    t2 = r2.get('timings', {})
    say(f'  decode at {a.max}: {t2.get("predicted_n")} tok at {t2.get("predicted_per_second", 0):.1f} tok/s (read {t2.get("prompt_n")} new)')
    record(kind='decode', config=a.config, depth=a.max, predicted_n=t2.get('predicted_n'), tok_s=t2.get('predicted_per_second'), prompt_n=t2.get('prompt_n'),
           text_head=r2.get('content', '')[:200], phone=phone)
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
    subprocess.run(['pkill', '-f', f'llama-server .*--port {a.port}'])

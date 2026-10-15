#!/usr/bin/env python3
"""session-bench.py - a coding-agent session sent exactly the way omp sends it, measured the way a user feels it.

Requests: /v1/chat/completions, streamed, with omp's own system prompt, its 12 tool definitions and its request fields
(bench/omp-request.json, captured from omp 18.4.9). Thinking on at medium effort (chat_template_kwargs.reasoning_effort, what
scripts/proxy.py now adds for omp). No temperature is sent, like omp: the server's default sampling applies
(llama-server: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05). A fixed seed only makes reruns repeatable.

Session: the system prompt + project notes (~20k tokens with the tools, the size of a real omp preamble), then --turns turns.
Turn 0 asks for a plan; each later turn brings a ~1.5k-token tool result (real source files) and one instruction. Every
config gets the same inputs; the answers differ (sampling), so per-token rates are the comparison, not identical text.

Per turn: wait until the first token, tokens read and the reading rate, thinking and answer tokens, writing rate, the gaps
between streamed tokens (median, 95th percentile, longest), draft acceptance. Once a second, in the background: server
memory and CPU, swap in use, memory macOS still has free, GPU power (tools/gpu-pstate, no sudo), and for fork-phone the
iPhone's thermal state and app memory. Rows: bench/results/session.jsonl.

  bench/session-bench.py --config stock|fork-mac|fork-phone
"""
import argparse, glob, json, os, re, statistics, subprocess, sys, threading, time, urllib.request
from scrub import scrub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser('~/Models/Qwen3.8-27B-IQ4_XS.gguf')
STOCK = '/opt/homebrew/bin/llama-server'

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True, choices=['stock', 'fork-mac', 'fork-phone'])
ap.add_argument('--turns', type=int, default=4, help='turn 0 (plan) + turns-1 tool results')
ap.add_argument('--notes-chars', type=int, default=60000, help='project notes in the system prompt')
ap.add_argument('--tool-chars', type=int, default=6000, help='each tool result (~1.5k tokens)')
ap.add_argument('--max-tokens', type=int, default=1024, help='cap per answer (thinking + answer)')
ap.add_argument('--effort', default='medium', choices=['low', 'medium', 'xhigh'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--mac-ctx', type=int, default=65536, help='KV cells on the Mac; with fork-phone the iPhone holds the rest')
ap.add_argument('--port', type=int, default=8095)
ap.add_argument('--rep', type=int, default=1)
ap.add_argument('--note', default='')
a = ap.parse_args()
URL = f'http://127.0.0.1:{a.port}'
OUT = os.path.join(ROOT, 'bench', 'results')
os.makedirs(OUT, exist_ok=True)
say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)


def git(*args):
    return subprocess.run(['git', '-C', *args], capture_output=True, text=True).stdout.strip()


def record(kind, **kw):
    row = dict(kind=kind, config=a.config, rep=a.rep, effort=a.effort, mac_ctx=a.mac_ctx, note=a.note, time=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)
    with open(os.path.join(OUT, 'session.jsonl'), 'a') as f:
        f.write(json.dumps(row) + '\n')
    return row


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


# ---- resource sampler: one row per second while the session runs
class Sampler(threading.Thread):
    def __init__(self, phone_ip):
        super().__init__(daemon=True)
        self.rows, self.stop, self.phone_ip, self.gpu_w = [], False, phone_ip, 0.0
        self.pstate = subprocess.Popen([f'{ROOT}/tools/gpu-pstate/gpu-pstate', '1000'], stdout=subprocess.PIPE, text=True,
                                       stderr=subprocess.DEVNULL) if os.path.exists(f'{ROOT}/tools/gpu-pstate/gpu-pstate') else None
        if self.pstate:
            threading.Thread(target=self.read_pstate, daemon=True).start()

    def read_pstate(self):
        for line in self.pstate.stdout:
            m = re.search(r'([\d.]+) W', line)
            if m:
                self.gpu_w = float(m.group(1))

    def run(self):
        page = int(subprocess.run(['sysctl', '-n', 'hw.pagesize'], capture_output=True, text=True).stdout or 16384)
        k = 0
        while not self.stop:
            row = dict(t=time.time(), gpu_w=self.gpu_w)
            pid = subprocess.run(['pgrep', '-f', f'llama-server .*--port {SPORT}'], capture_output=True, text=True).stdout.split()
            if pid:
                ps = subprocess.run(['ps', '-o', 'rss=,%cpu=', '-p', pid[0]], capture_output=True, text=True).stdout.split()
                if len(ps) == 2:
                    row['server_gb'], row['server_cpu'] = int(ps[0]) / 1048576, float(ps[1])
            sw = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
            m = re.search(r'used = ([\d.]+)M', sw)
            row['swap_gb'] = float(m.group(1)) / 1024 if m else None
            vm = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
            pages = {n: int(v) for n, v in re.findall(r'Pages (free|inactive|speculative):\s+(\d+)', vm)}
            row['free_gb'] = sum(pages.values()) * page / 1e9
            if self.phone_ip and k % 5 == 0:
                try:
                    out = subprocess.run(['nc', '-G', '1', '-w', '2', self.phone_ip, '50061'], input='mem\n', capture_output=True,
                                         text=True, timeout=4).stdout
                    d = json.loads(out.strip().splitlines()[0])
                    row['phone_thermal'], row['phone_app_mb'] = d.get('thermal'), d.get('footprint_mb')
                except Exception:
                    pass
            self.rows.append(row)
            k += 1
            time.sleep(1)
        if self.pstate:
            self.pstate.terminate()


def chat(messages, fx):
    """One streamed request; returns timing, token counts and the reply."""
    body = dict(fx['request_fields'], messages=messages, tools=fx['tools'], max_completion_tokens=a.max_tokens, seed=a.seed,
                timings_per_token=False)
    body['chat_template_kwargs'] = dict(body.get('chat_template_kwargs', {}), reasoning_effort=a.effort)
    req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    first, last, gaps, n_think, n_ans, text, think, timings, finish, calls = None, None, [], 0, 0, '', '', {}, None, []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            d = json.loads(line[6:])
            timings = d.get('timings', timings)
            for ch in d.get('choices', []):
                delta = ch.get('delta') or {}
                finish = ch.get('finish_reason') or finish
                got = False
                if delta.get('reasoning_content'):
                    think += delta['reasoning_content']; n_think += 1; got = True
                if delta.get('content'):
                    text += delta['content']; n_ans += 1; got = True
                for tc in delta.get('tool_calls') or []:
                    calls.append(tc); got = True
                if got:
                    now = time.time()
                    if first is None:
                        first = now
                    else:
                        gaps.append((now - last) * 1000)
                    last = now
    return dict(wall_s=time.time() - t0, first_token_s=(first or time.time()) - t0, gaps=gaps, think_chunks=n_think,
                answer_chunks=n_ans, text=text, think=think, timings=timings, finish=finish, tool_calls=len(calls))


# ---- server
log_path = os.path.join(OUT, f'session-server-{a.config}-rep{a.rep}.log')
log = open(log_path, 'w')
SPORT = a.port
if a.config == 'stock':
    version = subprocess.run([STOCK, '--version'], capture_output=True, text=True).stderr.strip().splitlines()[0]
    cmd = [STOCK, '-m', MODEL, '-ngl', '999', '-fa', 'on', '-c', '65536', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
           '--host', '127.0.0.1', '--port', str(a.port), '--jinja']
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
else:
    cache = os.path.join(OUT, 'session-cache', a.config)
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, PROXY='0', PORT=str(a.port), CTX=str(a.mac_ctx), KV='q8_0', CACHE_DIR=cache)
    if a.config == 'fork-mac':
        env['PHONE'] = '0'
    version = 'fork ' + git(f'{ROOT}/llama.cpp', 'rev-parse', '--short', 'HEAD') + ' (integration ' + git(ROOT, 'rev-parse', '--short', 'HEAD') + ')'
    srv = subprocess.Popen([f'{ROOT}/scripts/serve.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
say(f'{a.config}: {version}; log {log_path}')
sampler = None
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
    phone, phone_ip = '', ''
    if a.config != 'stock':
        phone = next((l.strip() for l in open(log_path) if 'serve:' in l), '')
    if a.config == 'fork-phone':
        m = re.search(r'at (169\.254\.\d+\.\d+)', phone)
        if 'split prefill on' not in phone or not m:
            sys.exit(f'fork-phone: the phone is not in use: {phone!r}')
        phone_ip = m.group(1)
    phone = scrub(phone)   # rows keep no address; phone_ip is only for the live samples

    fx = json.load(open(f'{ROOT}/bench/omp-request.json'))
    notes = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob(f'{ROOT}/llama.cpp/docs/*.md')))[:a.notes_chars]
    messages = [{'role': 'system', 'content': fx['system'] + '\n\n<project-notes>\n' + notes + '\n</project-notes>'}]
    srcs = sorted(glob.glob(f'{ROOT}/llama.cpp/tools/server/*.cpp') + glob.glob(f'{ROOT}/llama.cpp/common/*.cpp'))
    asks = ['What does this file do? Answer briefly.', 'Find one bug risk in it.', 'Which function would you test first, and why?',
            'What does it depend on?', 'Summarize what we learned so far.']

    sampler = Sampler(phone_ip)
    sampler.start()
    t_session, rows = time.time(), []
    for turn in range(a.turns):
        if turn == 0:
            messages.append({'role': 'user', 'content': 'We are going to review a few source files of this project. Give a short plan.'})
        else:
            src = srcs[turn - 1]
            messages.append({'role': 'user', 'content': f'<tool-result name="read" path="{os.path.relpath(src, ROOT)}">\n'
                             + open(src, errors='replace').read()[:a.tool_chars] + f'\n</tool-result>\n{asks[(turn - 1) % len(asks)]}'})
        t_turn = time.time()
        r = chat(messages, fx)
        tm = r['timings']
        pn, pms, gn, gms = tm.get('prompt_n', 0), tm.get('prompt_ms', 0), tm.get('predicted_n', 0), tm.get('predicted_ms', 0)
        res = [s for s in sampler.rows if s['t'] >= t_turn]
        row = record('session_turn', turn=turn, prompt_n=pn, prompt_ms=pms, cache_n=tm.get('cache_n'),
                     first_token_s=r['first_token_s'], wall_s=r['wall_s'], predicted_n=gn, predicted_ms=gms,
                     gen_tok_s=gn / (gms / 1000) if gms else 0, think_chunks=r['think_chunks'], answer_chunks=r['answer_chunks'],
                     gap_p50_ms=pct(r['gaps'], .5), gap_p95_ms=pct(r['gaps'], .95), gap_max_ms=max(r['gaps'] or [0]),
                     draft_n=tm.get('draft_n'), draft_accepted=tm.get('draft_n_accepted'), finish=r['finish'],
                     tool_calls=r['tool_calls'], gpu_j=sum(s['gpu_w'] for s in res),
                     answer_head=r['text'][:200], version=version, phone=phone)
        rows.append(row)
        say(f'  turn {turn}: read {pn:5d} tok in {pms/1000:5.1f} s, first token after {r["first_token_s"]:5.1f} s, wrote {gn} '
            f'({r["think_chunks"]} thinking) at {row["gen_tok_s"]:5.1f} tok/s, gaps p50 {row["gap_p50_ms"]:.0f} / p95 '
            f'{row["gap_p95_ms"]:.0f} / max {row["gap_max_ms"]:.0f} ms, {r["finish"]}')
        # the next turn sees the answer the way omp keeps it (preserve_thinking: the reasoning stays in the history)
        messages.append({'role': 'assistant', 'content': r['text'] or '(tool call)', 'reasoning_content': r['think']})
    wall = time.time() - t_session
    sampler.stop = True
    time.sleep(1.5)
    s = sampler.rows
    vals = lambda k: [x[k] for x in s if x.get(k) is not None]
    gen = sum(r['predicted_n'] for r in rows)
    total = record('session', wall_s=wall, first_wait_s=rows[0]['first_token_s'],
                   turn_wait_s=statistics.mean(r['first_token_s'] for r in rows[1:]) if len(rows) > 1 else None,
                   read_tok_s=sum(r['prompt_n'] for r in rows) / (sum(r['prompt_ms'] for r in rows) / 1000),
                   gen_tok_s=gen / (sum(r['predicted_ms'] for r in rows) / 1000), generated=gen,
                   gap_p50_ms=statistics.median(r['gap_p50_ms'] for r in rows), gap_p95_ms=max(r['gap_p95_ms'] for r in rows),
                   draft_accept=(sum(r['draft_accepted'] or 0 for r in rows) / max(1, sum(r['draft_n'] or 0 for r in rows)))
                   if any(r['draft_n'] for r in rows) else None,
                   server_gb_peak=max(vals('server_gb') or [0]), server_cpu_mean=statistics.mean(vals('server_cpu') or [0]),
                   swap_gb_peak=max(vals('swap_gb') or [0]), swap_gb_start=(vals('swap_gb') or [0])[0],
                   free_gb_min=min(vals('free_gb') or [0]), gpu_w_mean=statistics.mean(vals('gpu_w') or [0]),
                   gpu_j=sum(vals('gpu_w')), gpu_j_per_token=sum(vals('gpu_w')) / max(1, gen),
                   phone_thermal_max=max(vals('phone_thermal') or [None]) if vals('phone_thermal') else None,
                   phone_app_mb_peak=max(vals('phone_app_mb') or [0]) if vals('phone_app_mb') else None,
                   version=version, phone=phone)
    with open(os.path.join(OUT, f'session-samples-{a.config}-rep{a.rep}.jsonl'), 'w') as f:
        f.writelines(json.dumps(x) + '\n' for x in s)
    say(f'  session {wall:.0f} s: first answer after {total["first_wait_s"]:.0f} s, later turns wait {total["turn_wait_s"] or 0:.1f} s, '
        f'writing {total["gen_tok_s"]:.1f} tok/s, server {total["server_gb_peak"]:.1f} GB, swap {total["swap_gb_start"]:.1f} -> '
        f'{total["swap_gb_peak"]:.1f} GB, free min {total["free_gb_min"]:.1f} GB, GPU {total["gpu_w_mean"]:.1f} W, '
        f'{total["gpu_j_per_token"]:.2f} J/token, phone thermal max {total["phone_thermal_max"]}')
finally:
    if sampler:
        sampler.stop = True
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
    if a.config != 'stock':
        subprocess.run(['pkill', '-f', f'llama-server .*--port {a.port}'])

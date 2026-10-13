#!/usr/bin/env python3
"""proxy.py - prompt cache on the SSD in front of llama-server (one slot, hybrid/recurrent models).

Why: a hybrid model (GDN + attention) can only reuse the server's cache when a request *extends* what is in the slot.
Every new conversation (new omp session, server restart, a side request with another system prompt) therefore re-reads
the whole preamble: system prompt + AGENTS.md + tool descriptions, ~14-23k tokens, ~2-3.5 minutes at ~110 tok/s.
This proxy keeps that work on the SSD:
  - preamble seen for the first time: prefill just the preamble, save the slot (pre-<hash>.bin), then answer
    (the user waits the same cold time once, ever, per distinct preamble);
  - new conversation with a known preamble: restore it (~0.1-0.3 s) before forwarding;
  - switching conversations: the one leaving the slot is saved (conv-*.bin, newest few kept) and restored if it comes back;
  - continuing the conversation in the slot: plain passthrough (the server's own cache, ~1 s TTFT measured).
It never changes a response and, by default, never a request; generation speed is untouched.
BACKBURNER_KEEP_REASONING=1 (opt-in): put the model's own reasoning_content back into assistant turns the client sent without
it (omp drops it). Qwen3.8's chat template keeps past thinking by default (preserve_thinking undefined -> kept), so the
rendered prompt then equals both the template's default and the tokens already in the slot: no re-read of the model's last
turn on every agent step (+37-53% prefill per step measured by round-cost). Cost: past thinking stays in the context. Every decision is logged with the server's own
prompt_n / cache_n, so cold and cached numbers stay separate.

  scripts/proxy.py --listen 8080 --upstream 8180 --cache DIR   (DIR = the server's --slot-save-path)
"""
import argparse, hashlib, http.client, http.server, json, os, socket, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument('--listen', type=int, default=8080)
ap.add_argument('--upstream', type=int, default=8180)
ap.add_argument('--cache', required=True)
ap.add_argument('--keep-conv', type=int, default=2, help='saved conversations to keep')
ap.add_argument('--max-gb', type=float, default=8.0, help='cap on the cache directory; oldest files go first')
ap.add_argument('--effort', default=os.environ.get('THINK_EFFORT', 'medium'),
                help="thinking effort when the client sends none (omp 18.4 sends enable_thinking only, so the template's default, "
                     "xhigh, applied to every level); 'keep' = don't add one")
ap.add_argument('--phone', default='', help="the iPhone's USB address: tell its Backburner screen what the Mac is doing")
args = ap.parse_args()
os.makedirs(args.cache, exist_ok=True)
INDEX = os.path.join(args.cache, 'proxy-index.json')
LOCK = threading.Lock()


def log(*a):
    line = ' '.join(str(x) for x in (time.strftime('%H:%M:%S'), 'proxy:') + a)
    print(line, file=sys.stderr, flush=True)
    try:                     # also a file next to the cache, so a session can be read back later
        with open(os.path.join(args.cache, 'proxy.log'), 'a') as f:
            f.write(time.strftime('%Y-%m-%d ') + line + '\n')
    except OSError:
        pass


def up(method, path, body=None, timeout=3600):
    c = http.client.HTTPConnection('127.0.0.1', args.upstream, timeout=timeout)
    c.request(method, path, json.dumps(body).encode() if body is not None else None,
              {'Content-Type': 'application/json'} if body is not None else {})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, {'raw': data[:200].decode(errors='replace')}


class Phone:
    """Tells the Backburner screen what the Mac is doing: "mac PHASE N1 N2 CTX" to :50061, one line per short connection (that port
    serves one connection at a time). PHASE is reading (N1 = tokens read so far when known), thinking / writing (N1 = tokens
    written, N2 = tokens a second), done (N1 = prompt tokens read, N2 = tokens written, CTX = tokens in the conversation). Latest value only, sent
    from its own thread with a 1 s timeout: a slow or missing phone never delays a request."""
    def __init__(self, ip):
        self.ip, self.msg, self.cv = ip, None, threading.Condition()
        if ip:
            threading.Thread(target=self.run, daemon=True).start()

    def note(self, *parts):
        if self.ip:
            with self.cv:
                self.msg = ' '.join(str(int(p)) if isinstance(p, float) else str(p) for p in parts)
                self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while self.msg is None:
                    self.cv.wait()
                m, self.msg = self.msg, None
            try:
                with socket.create_connection((self.ip, 50061), timeout=1) as c:
                    c.sendall(f'mac {m}\n'.encode())
                    c.recv(8192)
            except OSError:
                pass
            time.sleep(0.2)


PHONE = Phone(args.phone)


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_index():
    try:
        return json.load(open(INDEX))
    except (OSError, ValueError):
        return {'conv': []}


def save_index(ix):
    tmp = INDEX + '.tmp'
    json.dump(ix, open(tmp, 'w'))
    os.replace(tmp, INDEX)


def rm_state(name):
    for ext in ('', '.dft'):
        try:
            os.remove(os.path.join(args.cache, name + ext))
        except OSError:
            pass


def trim_cache():
    files = []
    for f in os.listdir(args.cache):
        p = os.path.join(args.cache, f)
        if f.endswith('.bin') and os.path.isfile(p):
            sz = os.path.getsize(p) + (os.path.getsize(p + '.dft') if os.path.exists(p + '.dft') else 0)
            files.append((os.path.getatime(p), sz, f))
    total = sum(s for _, s, _ in files)
    for _, s, f in sorted(files):
        if total <= args.max_gb * 1e9:
            break
        rm_state(f)
        total -= s
        log(f'cache over {args.max_gb} GB: removed {f}')


def slot(action, name):
    t0 = time.time()
    st, d = up('POST', f'/slots/0?action={action}', {'filename': name})
    ok = st == 200 and 'error' not in d
    n = d.get('n_restored', d.get('n_saved'))
    log(f'{action} {name}: {"ok" if ok else d} {n} tokens {1000*(time.time()-t0):.0f} ms')
    return ok


KEEP_REASONING = os.environ.get('BACKBURNER_KEEP_REASONING', '0') not in ('', '0')
REASONING = {}           # key of an assistant reply (visible content + tool calls) -> the reasoning_content it came with


def reply_key(content, tool_calls):
    def norm_args(a):
        try:
            return json.dumps(json.loads(a) if isinstance(a, str) else a, sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            return str(a or '').strip()
    if isinstance(content, list):
        content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
    calls = [((c.get('function') or {}).get('name', ''), norm_args((c.get('function') or {}).get('arguments', '')))
             for c in (tool_calls or [])]
    return hashlib.sha1(json.dumps([(content or '').strip(), calls], ensure_ascii=False).encode()).hexdigest()


def restore_reasoning(msgs):
    n = 0
    for m in msgs:
        if m.get('role') == 'assistant' and not m.get('reasoning_content'):
            rc = REASONING.get(reply_key(m.get('content'), m.get('tool_calls')))
            if rc:
                m['reasoning_content'] = rc
                n += 1
    return n


class State:
    msgs = None          # messages of the request now in the slot (None = unknown)
    reply = None         # what the model answered to it (content, reasoning_content), as streamed back


def same_reply(m, reply):
    if not reply or m.get('role') != 'assistant':
        return False
    def norm(x):
        if isinstance(x, list):
            x = ''.join(p.get('text', '') for p in x if isinstance(p, dict))
        return (x or '').strip()
    if norm(m.get('content')) != norm(reply.get('content')):
        return False
    rc = m.get('reasoning_content')
    return rc is None or norm(rc) == norm(reply.get('reasoning_content'))


def parse_reply(raw):
    # streamed (SSE) or plain JSON chat completion -> {'content', 'reasoning_content'}
    content, reasoning, calls = [], [], {}
    text = raw.decode(errors='replace')
    if text.lstrip().startswith('{'):
        try:
            m = json.loads(text)['choices'][0]['message']
            return {'content': m.get('content') or '', 'reasoning_content': m.get('reasoning_content') or '',
                    'tool_calls': m.get('tool_calls') or []}
        except (ValueError, KeyError, IndexError):
            return None
    for line in text.splitlines():
        if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
            continue
        try:
            d = json.loads(line[6:])
        except ValueError:
            continue
        for c in d.get('choices', []):
            dl = c.get('delta', {})
            content.append(dl.get('content') or '')
            reasoning.append(dl.get('reasoning_content') or '')
            for tc in dl.get('tool_calls') or []:
                c = calls.setdefault(tc.get('index', 0), {'function': {'name': '', 'arguments': ''}})
                fn = tc.get('function') or {}
                c['function']['name'] += fn.get('name') or ''
                c['function']['arguments'] += fn.get('arguments') or ''
    return {'content': ''.join(content), 'reasoning_content': ''.join(reasoning),
            'tool_calls': [calls[i] for i in sorted(calls)]}


def preamble_prefix(body):
    """Text of the chat-template render that every conversation with this system prompt + tools shares.
    Rendered twice with different user turns; the common prefix, cut before its last '<|im_start|>' so the boundary is a
    special token (no BPE merge across it)."""
    sys_msgs = [m for m in body['messages'] if m.get('role') in ('system', 'developer')]
    base = {k: v for k, v in body.items() if k not in ('messages', 'stream', 'stream_options', 'max_tokens',
                                                        'max_completion_tokens', 'n_predict')}
    renders = []
    for u in ('⁣a', '⁣b'):
        st, d = up('POST', '/apply-template', dict(base, messages=sys_msgs + [{'role': 'user', 'content': u}]))
        if st != 200 or 'prompt' not in d:
            return None
        renders.append(d['prompt'])
    a, b = renders
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    pre = a[:n]
    cut = pre.rfind('<|im_start|>')
    return pre[:cut] if cut > 0 else pre


def prepare(body):
    """Make the slot hold the longest saved state this request extends. Returns a short label for the log."""
    msgs = body.get('messages') or []
    if State.msgs is not None and len(msgs) >= len(State.msgs) and msgs[:len(State.msgs)] == State.msgs:
        return 'continue'
    ix = load_index()
    # the conversation leaving the slot: save it so a later request that continues it restores instead of re-reading
    if State.msgs and len(State.msgs) > 1:
        name = f'conv-{h(State.msgs)}.bin'
        if slot('save', name):
            ix['conv'] = [c for c in ix['conv'] if c['file'] != name] + [{'file': name, 'n': len(State.msgs), 'hash': h(State.msgs),
                                                                          'reply': State.reply}]
            while len(ix['conv']) > args.keep_conv:
                rm_state(ix['conv'].pop(0)['file'])
            save_index(ix)
    # a saved conversation this request continues (longest first)
    for c in sorted(ix['conv'], key=lambda c: -c['n']):
        if len(msgs) >= c['n'] and h(msgs[:c['n']]) == c['hash'] and os.path.exists(os.path.join(args.cache, c['file'])):
            # the saved state ends after the model's own reply; a history that rewrote that reply can't reuse it (a recurrent
            # model has no mid-sequence rollback), so only restore when the next assistant message is that reply
            if len(msgs) > c['n'] and not same_reply(msgs[c['n']], c.get('reply')):
                log(f'conversation {c["file"]} matches but its last reply was rewritten: using the preamble instead')
                continue
            if slot('restore', c['file']):
                return 'restored conversation'
    # the preamble: restore, or build it once. Keyed by the EXACT rendered text the server will see (2026-09-26 fix): the old
    # key (system messages + tools + chat_template_kwargs) ignored top-level fields such as reasoning_effort, so a medium-effort
    # session restored the xhigh preamble ("Reasoning effort is set to xhigh" is its first line), the server found no common
    # prefix (cache_n 0) and re-read the whole prompt on every new session.
    pre = preamble_prefix(body)
    if not pre:
        return 'no preamble (template render failed): cold'
    key = hashlib.sha256(pre.encode()).hexdigest()[:16]
    name = f'pre-{key}.bin'
    if os.path.exists(os.path.join(args.cache, name)) and slot('restore', name):
        return 'restored preamble'
    t0 = time.time()
    st, d = up('POST', '/completion', {'prompt': pre, 'n_predict': 0, 'cache_prompt': True, 'temperature': 0})
    tm = d.get('timings', {}) if st == 200 else {}
    log(f'built preamble {key}: {tm.get("prompt_n")} tokens (cache_n {tm.get("cache_n")}) in {time.time()-t0:.1f} s')
    slot('save', name)
    trim_cache()
    return 'built preamble (cold, once)'


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def relay(self, body_bytes, watch=False, strip_progress=False):
        """strip_progress: the proxy asked the server for prompt_progress events (the client didn't): they feed the phone's
        end-to-end reading speed and are taken out of the stream the client sees."""
        c = http.client.HTTPConnection('127.0.0.1', args.upstream, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ('host', 'content-length', 'connection')}
        if body_bytes is not None:
            hdrs['Content-Length'] = str(len(body_bytes))
        c.request(self.command, self.path, body_bytes, hdrs)
        r = c.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                self.send_header(k, v)
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        tail = bytearray()   # in-place extend: avoids the O(n^2) realloc that `bytes += chunk` does per chunk
        buf, phase, n, t_note, t_first = b'', 'reading', 0, 0.0, 0.0
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            out = chunk
            if watch and PHONE.ip:   # SSE events: the first thinking / answer token ends the prompt read
                buf += chunk
                out = b''
                while b'\n\n' in buf:
                    ev, buf = buf.split(b'\n\n', 1)
                    if not ev.startswith(b'data: {'):
                        out += ev + b'\n\n'
                        continue
                    try:
                        obj = json.loads(ev[6:])
                    except ValueError:
                        out += ev + b'\n\n'
                        continue
                    pp = obj.get('prompt_progress')
                    if pp:
                        # the whole prompt read, end to end: new tokens over the server's time so far (the phone shows this,
                        # not its own layers' rate)
                        new, ms = pp.get('processed', 0) - pp.get('cache', 0), pp.get('time_ms', 0)
                        if new > 0 and ms > 0:
                            PHONE.note('reading', new, new * 1000.0 / ms)
                    d = (obj.get('choices') or [{}])[0].get('delta') or {}
                    if pp and strip_progress:
                        del obj['prompt_progress']
                        ch = (obj.get('choices') or [{}])[0]
                        if not (d or ch.get('finish_reason') or obj.get('timings')):
                            continue   # a progress-only event: the client never asked for it
                        ev = b'data: ' + json.dumps(obj, ensure_ascii=False).encode()
                    out += ev + b'\n\n'
                    p = 'thinking' if d.get('reasoning_content') else 'writing' if d.get('content') or d.get('tool_calls') else None
                    if p:
                        n += 1
                        now = time.time()
                        t_first = t_first or now
                        if p != phase or now - t_note > 0.5:
                            phase, t_note = p, now
                            PHONE.note(phase, n, (n - 1) / (now - t_first) if now > t_first else 0)
            if len(tail) < 8 << 20:
                tail += chunk
            if out:
                self.wfile.write(b'%x\r\n%s\r\n' % (len(out), out))
                self.wfile.flush()
        if buf:   # a last event without its blank line
            self.wfile.write(b'%x\r\n%s\r\n' % (len(buf), buf))
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()
        c.close()
        return bytes(tail)

    def do_GET(self):
        self.relay(None)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        if not self.path.split('?')[0].endswith('/chat/completions'):
            self.relay(raw)
            return
        try:
            body = json.loads(raw)
        except ValueError:
            self.relay(raw)
            return
        # the effort the client asked for, or --effort: Qwen3.8's template reads reasoning_effort (low / medium / xhigh) and
        # falls back to xhigh, the longest thinking, when it is missing; thinking off (enable_thinking false) is left alone
        kw = body.setdefault('chat_template_kwargs', {}) if isinstance(body.get('chat_template_kwargs', {}), dict) else None
        add_effort = (args.effort != 'keep' and kw is not None and kw.get('enable_thinking') is not False
                      and 'reasoning_effort' not in body and 'reasoning_effort' not in kw)
        if add_effort:
            kw['reasoning_effort'] = args.effort
        elif kw == {}:
            del body['chat_template_kwargs']
        with LOCK:
            kept = restore_reasoning(body.get('messages') or []) if KEEP_REASONING else 0
            # the phone shows the end-to-end reading speed from the server's prompt_progress events; ask for them when the
            # client didn't, and strip them from what it gets back
            want_progress = bool(PHONE.ip and body.get('stream') and not body.get('return_progress'))
            if want_progress:
                body['return_progress'] = True
            if kept or add_effort or want_progress:
                raw = json.dumps(body).encode()
            t0 = time.time()
            PHONE.note('reading', 0)
            try:
                what = prepare(body)
            except Exception as e:  # the cache must never break a request: fall through cold
                what = f'cache error, cold: {e!r}'
            t_prep = time.time() - t0
            tail = self.relay(raw, watch=True, strip_progress=want_progress)
            State.msgs = body.get('messages')
            State.reply = parse_reply(tail)
            if KEEP_REASONING and State.reply and State.reply.get('reasoning_content'):
                REASONING[reply_key(State.reply['content'], State.reply['tool_calls'])] = State.reply['reasoning_content']
                while len(REASONING) > 4096:
                    REASONING.pop(next(iter(REASONING)))
            # the server's own counts for this request (last timings object in the stream / body)
            s = tail.decode(errors='replace')
            i = s.rfind('"timings"')
            tm = ''
            if i >= 0:
                try:
                    tm = json.JSONDecoder().raw_decode(s[s.index('{', i):])[0]
                    PHONE.note('done', tm.get('prompt_n', 0), tm.get('predicted_n', 0),
                               tm.get('prompt_n', 0) + tm.get('cache_n', 0) + tm.get('predicted_n', 0))
                    tm = f'prompt_n {tm.get("prompt_n")} cache_n {tm.get("cache_n")} ' \
                         f'prompt {tm.get("prompt_ms", 0)/1000:.1f} s gen {tm.get("predicted_per_second", 0):.1f} tok/s'
                except (ValueError, IndexError):
                    tm = ''
            if not tm:
                PHONE.note('done', 0, 0)
            log(f'{what} (prep {t_prep:.2f} s) {tm}' + (f' reasoning restored in {kept} turns' if kept else ''))


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


log(f'listening on 127.0.0.1:{args.listen} -> 127.0.0.1:{args.upstream}, cache {args.cache}, thinking effort {args.effort} unless the client sends one')
Server(('127.0.0.1', args.listen), H).serve_forever()

#!/usr/bin/env python3
"""webgpu_bridge.py - host-side bridge for the wireless WebGPU node.

Makes a browser tab running web/webgpu-phone look like a phone-attn server
(:50062) to the inference host, with no cable and no adb:

    browser (WebGPU kernels) <--WebSocket--> bridge <--TCP--> llama-server host

The bridge is a dumb authenticated pipe: PATN framing passes through
untouched, so strict request->reply ordering is preserved. It additionally
answers the :50061 command port locally (`mem` for serve sizing, `mac PHASE`
notes forwarded to the tab as text frames), because serve/proxy phone_note
expect it.

    python3 scripts/webgpu_bridge.py --listen 0.0.0.0 --token SECRET
    # tab:  http://<host-lan>:8000  (served separately, secure context needed)
    #       connect ws://<host-lan>:50063?token=SECRET
    # host: PHONE_KV=127.0.0.1:50062 scripts/serve.ps1  (no split tail: TAIL 0)

Split prefill (:50060 tail) is NOT served wirelessly in v1 (24 tail layers
don't fit the kernel set); the wireless node does phone-held attention only.
Scope: past-64k old-key attention + all ATTN/ATTN_BIG verification via
webgpu-kernels MatMul + Softmax. Stdlib only (Windows/Mac/Linux).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import socket
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
PATN_HELLO = struct.pack("<IIQ", 0x4E544150, 1, 0)


class Pipe:
    """One TCP host client <-> one WS browser, plus a local cmd port."""

    def __init__(self, avail_mb: int):
        self.avail_mb = avail_mb
        self.lock = threading.Lock()
        self.tcp = None       # socket to llama host
        self.ws = None        # socket to browser
        self.notes = []       # pending text notes for the browser
        self.notes_cv = threading.Condition(self.lock)

    def set_tcp(self, s):
        with self.lock:
            if self.tcp is not None:
                try:
                    s.close()
                except OSError:
                    pass
                return False
            self.tcp = s
            return True

    def set_ws(self, s):
        with self.lock:
            if self.ws is not None:
                try:
                    s.close()
                except OSError:
                    pass
                return False
            self.ws = s
            self.notes_cv.notify_all()
            return True

    def clear(self, s):
        with self.lock:
            if self.tcp is s:
                self.tcp = None
            if self.ws is s:
                self.ws = None


# ---------------- minimal WebSocket server (RFC 6455, stdlib) ----------------

def ws_accept(s, token: str) -> bool:
    """HTTP Upgrade handshake. Returns True if token + upgrade check out."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            return False
        data += chunk
        if len(data) > 65536:
            return False
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin1")
    lines = head.split("\r\n")
    target = lines[0].split(" ")[1] if len(lines[0].split(" ")) > 1 else "/"
    hdr = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    query = target.split("?", 1)[1] if "?" in target else ""
    got = ""
    for part in query.split("&"):
        if part.startswith("token="):
            got = part[6:]
    import urllib.parse
    if urllib.parse.unquote(got) != token:
        s.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        return False
    if hdr.get("upgrade", "").lower() != "websocket":
        s.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        return False
    key = hdr.get("sec-websocket-key", "")
    acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    s.sendall(
        ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
         "Connection: Upgrade\r\nSec-WebSocket-Accept: " + acc + "\r\n\r\n").encode()
    )
    return True


def ws_recv(s) -> tuple[int, bytes] | None:
    """One reassembled message. Returns (opcode, payload) or None on close."""
    chunks, op = [], None
    while True:
        hdr = b""
        while len(hdr) < 2:
            c = s.recv(2 - len(hdr))
            if not c:
                return None
            hdr += c
        fin = hdr[0] & 0x80
        cur = hdr[0] & 0x0F
        masked = hdr[1] & 0x80
        ln = hdr[1] & 0x7F
        if ln == 126:
            ext = b""
            while len(ext) < 2:
                ext += s.recv(2 - len(ext))
            ln = struct.unpack(">H", ext)[0]
        elif ln == 127:
            ext = b""
            while len(ext) < 8:
                ext += s.recv(8 - len(ext))
            ln = struct.unpack(">Q", ext)[0]
        mask = b""
        if masked:
            while len(mask) < 4:
                mask += s.recv(4 - len(mask))
        payload = b""
        while len(payload) < ln:
            c = s.recv(min(65536, ln - len(payload)))
            if not c:
                return None
            payload += c
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if cur == 0x8:
            return None
        if cur == 0x9:  # ping -> pong
            ws_send(s, 0xA, payload)
            continue
        if cur == 0xA:
            continue
        if cur in (0x1, 0x2):
            if op is None:
                op = cur
            chunks.append(payload)
            if fin:
                return op, b"".join(chunks)
        # continuation frames (0x0) accumulate above
        elif cur == 0x0:
            chunks.append(payload)
            if fin:
                return op or 0x2, b"".join(chunks)


def ws_send(s, opcode: int, payload: bytes) -> bool:
    try:
        h = bytes([0x80 | opcode])
        n = len(payload)
        if n < 126:
            h += bytes([n])
        elif n < 65536:
            h += bytes([126]) + struct.pack(">H", n)
        else:
            h += bytes([127]) + struct.pack(">Q", n)
        s.sendall(h + payload)
        return True
    except OSError:
        return False


# ---------------- bridge ----------------

def tune(s):
    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    except OSError:
        pass


def forward_ws_to_tcp(pipe: Pipe, ws):
    try:
        while True:
            msg = ws_recv(ws)
            if msg is None:
                break
            op, payload = msg
            if op == 0x1:
                continue  # browser never sends notes; ignore stray text
            with pipe.lock:
                tcp = pipe.tcp
            if tcp is None:
                continue  # host not connected yet; drop (host retries HELLO)
            try:
                tcp.sendall(payload)
            except OSError:
                break
    finally:
        pipe.clear(ws)
        try:
            ws.close()
        except OSError:
            pass


def forward_tcp_to_ws(pipe: Pipe, tcp):
    try:
        while True:
            data = tcp.recv(1 << 20)
            if not data:
                break
            with pipe.lock:
                ws = pipe.ws
            if ws is None:
                # No tab (yet): the host's HELLO gets no reply and it retries.
                # Anything else would corrupt request->reply sync, so drop the
                # connection and let the host reconnect once the tab is back.
                break
            if not ws_send(ws, 0x2, data):
                break
            # flush one queued host note behind the reply (text frame)
            with pipe.lock:
                note = pipe.notes.pop(0) if pipe.notes else None
            if note is not None:
                ws_send(ws, 0x1, note.encode())
    finally:
        pipe.clear(tcp)
        try:
            tcp.close()
        except OSError:
            pass


def serve_ws(pipe: Pipe, listen: str, port: int, token: str):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen, port))
    srv.listen(4)
    print(f"webgpu-bridge: ws on {listen}:{port} (token set)", flush=True)
    while True:
        s, _ = srv.accept()
        tune(s)
        try:
            if not ws_accept(s, token):
                s.close()
                continue
        except OSError:
            try:
                s.close()
            except OSError:
                pass
            continue
        if not pipe.set_ws(s):
            print("webgpu-bridge: extra tab rejected (one at a time)", flush=True)
            continue
        print("webgpu-bridge: tab connected", flush=True)
        threading.Thread(target=forward_ws_to_tcp, args=(pipe, s), daemon=True).start()


def serve_tcp(pipe: Pipe, listen: str, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen, port))
    srv.listen(4)
    print(f"webgpu-bridge: phone-attn TCP on {listen}:{port}", flush=True)
    while True:
        s, _ = srv.accept()
        tune(s)
        if not pipe.set_tcp(s):
            print("webgpu-bridge: extra host client rejected", flush=True)
            continue
        threading.Thread(target=forward_tcp_to_ws, args=(pipe, s), daemon=True).start()


def serve_cmd(pipe: Pipe, listen: str, port: int):
    """Local :50061: mem for serve sizing + mac notes for the tab."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen, port))
    srv.listen(4)
    print(f"webgpu-bridge: cmd on {listen}:{port}", flush=True)
    while True:
        s, _ = srv.accept()
        threading.Thread(target=handle_cmd, args=(pipe, s), daemon=True).start()


def handle_cmd(pipe: Pipe, s):
    try:
        f = s.makefile("rw")
        for line in f:
            line = line.strip()
            if line == "mem":
                # pa_ane:false (no page engine wirelessly); avail sizes CTX_TOTAL.
                f.write(json.dumps({"avail_mb": pipe.avail_mb,
                                    "sys_wired_mb": pipe.avail_mb,
                                    "pa_ane": False}) + "\n")
                f.flush()
            elif line.startswith("mac "):
                with pipe.lock:
                    pipe.notes.append(line)
                    pipe.notes = pipe.notes[-8:]
                # no reply expected by phone_note (it reads; give it bytes)
                try:
                    f.write("\n")
                    f.flush()
                except OSError:
                    pass
            elif line.startswith("fetch "):
                f.write(json.dumps({"error": "webgpu node has no filesystem (no tail over wireless v1)"}) + "\n")
                f.flush()
            else:
                try:
                    f.write("\n")
                    f.flush()
                except OSError:
                    pass
    except OSError:
        pass
    finally:
        try:
            s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1", help="WS/TCP bind (0.0.0.0 for Wi-Fi)")
    ap.add_argument("--tcp-port", type=int, default=50062)
    ap.add_argument("--ws-port", type=int, default=50063)
    ap.add_argument("--cmd-port", type=int, default=50061)
    ap.add_argument("--token", default="", help="bearer token (generated if empty)")
    ap.add_argument("--avail-mb", type=int, default=2048, help="mem avail_mb for serve sizing")
    args = ap.parse_args()

    token = args.token or secrets.token_hex(16)
    if not args.token:
        print(f"webgpu-bridge: generated token: {token}", flush=True)
    pipe = Pipe(args.avail_mb)
    for target, fn in (("ws", serve_ws), ("tcp", serve_tcp), ("cmd", serve_cmd)):
        if target == "ws":
            t = threading.Thread(target=fn, args=(pipe, args.listen, args.ws_port, token), daemon=True)
        else:
            port = args.tcp_port if target == "tcp" else args.cmd_port
            t = threading.Thread(target=fn, args=(pipe, args.listen, port), daemon=True)
        t.start()
    print("webgpu-bridge: serving (Ctrl-C to stop)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

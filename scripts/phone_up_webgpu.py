#!/usr/bin/env python3
"""phone_up_webgpu.py - wireless probe for the WebGPU bridge.

Same stdout contract as phone-up.sh / phone_up_android.py:

    IP TAIL_UP ATTN_VERSION AVAIL_MB WIRED_MB NAME

so serve wrappers keep working with PHONE_KV=IP:50062. TAIL is always 0
(split prefill is not served wirelessly in v1); HELLO is forwarded through
the bridge to the tab, so this fails until the tab is connected.

Usage:
    python3 scripts/phone_up_webgpu.py [--host 127.0.0.1] [--tcp-port 50062]
                                       [--cmd-port 50061]
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys

PATN = 0x4E544150


def say(msg: str) -> None:
    print(f"phone-up-webgpu: {msg}", file=sys.stderr, flush=True)


def hello(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=5) as s:
            s.sendall(struct.pack("<IIQ", PATN, 1, 0))
            h = b""
            while len(h) < 16:
                chunk = s.recv(16 - len(h))
                if not chunk:
                    return ""
                h += chunk
            n = struct.unpack("<IIQ", h)[2]
            body = b""
            while len(body) < n:
                chunk = s.recv(n - len(body))
                if not chunk:
                    return ""
                body += chunk
            ver = struct.unpack("<I", body[:4])[0]
            try:
                s.sendall(struct.pack("<IIQ", PATN, 11, 0))
            except OSError:
                pass
            return str(ver)
    except OSError:
        return ""


def mem(host: str, port: int) -> tuple[int, int]:
    try:
        with socket.create_connection((host, port), timeout=5) as s:
            s.sendall(b"mem\n")
            line = s.makefile().readline()
        m = json.loads(line)
        return int(m["avail_mb"]), int(m["sys_wired_mb"])
    except Exception:
        return 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=50062)
    ap.add_argument("--cmd-port", type=int, default=50061)
    args = ap.parse_args()

    ver = hello(args.host, args.tcp_port)
    if not ver:
        say("no HELLO from the wireless node: start webgpu_bridge.py and connect the tab, then retry")
        sys.exit(1)
    avail, wired = mem(args.host, args.cmd_port)
    say(f"webgpu-wireless at {args.host}: phone-attn v{ver}, prefill tail down (wireless v1), "
        f"{wired} MiB wired-ish, {avail} MiB tab budget")
    print(f"{args.host} 0 {ver} {avail} {wired} webgpu-wireless", flush=True)


if __name__ == "__main__":
    main()

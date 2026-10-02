#!/usr/bin/env python3
"""phone_note.py - cross-platform replacement for `printf ... | nc` phone notes.

Sends one `mac PHASE ...` line to the Backburner command port (:50061),
which is what scripts/serve.sh does with `nc -G 1 -w 2 IP 50061` and what
scripts/proxy.py does with a raw socket. `nc` flags differ between BSD/GNU
and `nc` is usually absent on Windows, so serve.ps1 and automation call this.

Usage:
    python3 scripts/phone_note.py 127.0.0.1 starting
    python3 scripts/phone_note.py 127.0.0.1 ready
    python3 scripts/phone_note.py 127.0.0.1 stopped
    python3 scripts/phone_note.py --port 50061 127.0.0.1 reading 2048 157
Never fails the caller: errors are swallowed (exit 0) like `... ; true` in serve.sh.
"""
from __future__ import annotations

import argparse
import socket

ap = argparse.ArgumentParser()
ap.add_argument("ip", nargs="?", default="127.0.0.1")
ap.add_argument("phase", nargs="?", default="ready")
ap.add_argument("rest", nargs="*")
ap.add_argument("--port", type=int, default=50061)
ap.add_argument("--timeout", type=float, default=2.0)
args = ap.parse_args()

try:
    line = "mac " + " ".join([args.phase, *args.rest]) + "\n"
    with socket.create_connection((args.ip, args.port), timeout=args.timeout) as s:
        s.settimeout(args.timeout)
        s.sendall(line.encode())
        try:
            s.recv(4096)
        except OSError:
            pass
except OSError:
    pass

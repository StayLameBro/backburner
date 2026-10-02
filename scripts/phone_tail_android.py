#!/usr/bin/env python3
"""phone_tail_android.py - cross-platform port of scripts/phone-tail-android.sh.

Show which split-prefill tail the phone has, or push a new one via adb.

    python3 scripts/phone_tail_android.py                 # show L + :50061 mem
    python3 scripts/phone_tail_android.py L40             # ~/Models/tail-iq4xs-L40-nohead.gguf
    python3 scripts/phone_tail_android.py /path/tail.gguf

Tails are the same head-less splits as iOS (scripts/split-gguf.py -L N).
Delivery is `adb push` to /sdcard/Download + in-app import (no devicectl,
no relaunch dance: the tail server picks up tail.gguf on next connect).
Assumes `adb reverse` is up (run phone_up_android.py first); the tail HELLO
is probed on 127.0.0.1:50060.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys

TAIL_HELLO = struct.pack("<IIQ8I", 0x4C545053, 1, 32, 2, 1, 0, 0, 0, 0, 1, 1)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def loaded() -> str:
    try:
        with socket.create_connection(("127.0.0.1", 50060), timeout=15) as s:
            s.sendall(TAIL_HELLO)
            h = b""
            while len(h) < 16:
                h += s.recv(16 - len(h))
            n = struct.unpack("<IIQ", h)[2]
            body = b""
            while len(body) < n:
                body += s.recv(n - len(body))
            m = re.search(rb"phone has layers \[(\d+),", body)
            return m.group(1).decode() if m else "?"
    except Exception:
        return "down"


def mem() -> str:
    try:
        with socket.create_connection(("127.0.0.1", 50061), timeout=2) as s:
            s.sendall(b"mem\n")
            return s.makefile().read().strip()
    except Exception as e:
        return f"mem unavailable: {e}"


def main() -> None:
    tail = sys.argv[1] if len(sys.argv) > 1 else ""
    if re.fullmatch(r"L\d+", tail or ""):
        tail = os.path.join(os.path.expanduser("~"), "Models", f"tail-iq4xs-{tail}-nohead.gguf")
    adb = shutil.which("adb")
    if not adb:
        print("adb not on PATH", file=sys.stderr)
        sys.exit(1)
    dev = ""
    p = run([adb, "devices"])
    for line in p.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            dev = parts[0]
            break
    if not dev:
        print("no adb device", file=sys.stderr)
        sys.exit(1)

    if not tail:
        print(f"phone 127.0.0.1 (adb {dev}): tail L={loaded()}")
        print(mem())
        return

    if not os.path.isfile(tail):
        print(f"missing tail: {tail}", file=sys.stderr)
        sys.exit(1)
    m = re.search(r"-L(\d+)-", os.path.basename(tail))
    want = m.group(1) if m else "?"
    have = loaded()
    if have == want and not os.environ.get("FORCE"):
        print(f"phone already has L={want}")
        print(mem())
        return
    size_mb = os.path.getsize(tail) / 1e6
    print(f"phone: tail L={have} -> L={want} ({size_mb:.0f} MB)")
    p = run([adb, "-s", dev, "push", tail, "/sdcard/Download/tail.gguf"])
    if p.returncode != 0:
        print((p.stderr or p.stdout).strip(), file=sys.stderr)
        sys.exit(1)
    print("pushed to /sdcard/Download/tail.gguf; in Backburner: Import tail.gguf.")
    try:
        print(json.dumps(json.loads(mem()), indent=1))
    except Exception:
        print(mem())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""phone_up_android.py - cross-platform port of scripts/phone-up-android.sh.

Finds the wired Android phone via adb, forwards the Backburner ports with
`adb reverse` (no IP hunting, no Wi-Fi, works on Windows/Mac/Linux), probes
phone-attn HELLO (:50062), the split-prefill tail (:50060) and :50061 mem,
then prints the same line as phone-up.sh:

    IP TAIL_UP ATTN_VERSION AVAIL_MB WIRED_MB NAME

Exit 1 (reason on stderr) if no usable phone.

Usage:
    python3 scripts/phone_up_android.py [--tail-wait 45] [--device SERIAL] [--no-reverse]
    TAIL_WAIT=45 python3 scripts/phone_up_android.py   # env also honored
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time

PORTS = (50060, 50061, 50062, 50052)
PATN = 0x4E544150
HELLO, HELLO_OK, BYE = 1, 2, 11


def say(msg: str) -> None:
    print(f"phone-up-android: {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def find_device(prefer: str = "") -> tuple[str, str]:
    adb = shutil.which("adb")
    if not adb:
        say("adb not on PATH (install Android platform-tools)")
        sys.exit(1)
    if prefer:
        return adb, prefer
    p = run([adb, "devices"])
    dev = ""
    for line in p.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            dev = parts[0]
            break
    if not dev:
        say("no Android device on adb (enable USB debugging, unlock it)")
        sys.exit(1)
    return adb, dev


def device_name(adb: str, dev: str) -> str:
    p = run([adb, "-s", dev, "shell", "getprop", "ro.product.model"])
    name = (p.stdout or "").strip().replace("\r", "")
    return name or "wired Android"


def do_reverse(adb: str, dev: str) -> None:
    for port in PORTS:
        p = run([adb, "-s", dev, "reverse", f"tcp:{port}", f"tcp:{port}"])
        if p.returncode != 0:
            say(f"adb reverse tcp:{port} failed: {(p.stderr or p.stdout).strip()}")
            sys.exit(1)


def hello(ip: str, port: int = 50062, timeout: float = 2.0) -> str:
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.sendall(struct.pack("<IIQ", PATN, HELLO, 0))
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
                s.sendall(struct.pack("<IIQ", PATN, BYE, 0))
            except OSError:
                pass
            return str(ver)
    except OSError:
        return ""


def tail_loaded(ip: str = "127.0.0.1", timeout: float = 15.0) -> str:
    """First tail layer from the tail HELLO reply, '?' or 'down'."""
    try:
        with socket.create_connection((ip, 50060), timeout=timeout) as s:
            s.sendall(struct.pack("<IIQ8I", 0x4C545053, 1, 32, 2, 1, 0, 0, 0, 0, 1, 1))
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


def mem_mb(ip: str = "127.0.0.1", timeout: float = 2.0) -> tuple[int, int]:
    try:
        with socket.create_connection((ip, 50061), timeout=timeout) as s:
            s.sendall(b"mem\n")
            line = s.makefile().readline()
        m = json.loads(line)
        return int(m["avail_mb"]), int(m["sys_wired_mb"])
    except Exception:
        return 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-wait", type=int, default=int(os.environ.get("TAIL_WAIT", "45")))
    ap.add_argument("--device", default=os.environ.get("ANDROID_SERIAL", ""))
    ap.add_argument("--no-reverse", action="store_true", help="skip adb reverse (ports already forwarded)")
    args = ap.parse_args()

    adb, dev = find_device(args.device)
    name = device_name(adb, dev)
    if not args.no_reverse:
        do_reverse(adb, dev)
    ip = "127.0.0.1"

    ver = hello(ip)
    if not ver:
        say(f"Backburner HELLO did not answer on {name}: open the app and keep it foreground, then retry")
        sys.exit(1)

    tail = 0
    for _ in range(max(1, args.tail_wait)):
        try:
            with socket.create_connection((ip, 50060), timeout=1):
                tail = 1
                break
        except OSError:
            time.sleep(1)
    if not tail:
        say("the prefill tail (:50060) isn't up (no tail.gguf on the phone?): split prefill stays off")

    avail, wired = mem_mb(ip)
    say(f"{name} at {ip} (adb {dev}): phone-attn v{ver}, prefill tail "
        f"{'up' if tail else 'down'}, {wired} MiB wired-ish, {avail} MiB app budget")
    print(f"{ip} {tail} {ver} {avail} {wired} {name}", flush=True)


if __name__ == "__main__":
    main()

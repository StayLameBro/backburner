#!/usr/bin/env python3
"""phone_push_android.py - Windows-safe file push for the Android app.

Why not just phone-push.py: that script serves the file over HTTP on the
host's LAN address and tells the phone to `fetch <host-LAN-url>`. Behind
`adb reverse` the phone is reached as 127.0.0.1, and a random HTTP port has
no reverse rule, so the phone's fetch of http://127.0.0.1:RANDOM hits the
phone itself. This script fixes both:

  auto (default): single files (tail GGUF) go via `adb push` to
      /sdcard/Download/<basename> -- no network at all, works on Windows.
      Then import it in the app (or pass --fetch to also run the :50061
      fetch path below).
  fetch: serves the tree over HTTP on 127.0.0.1 and adds
      `adb reverse tcp:HTTPPORT tcp:HTTPPORT` first, so the phone's
      `fetch http://127.0.0.1:HTTPPORT/...` reaches this host. Cross-platform
      (Windows/Mac/Linux), no `nc`, no broadcast ping.

Usage:
    python3 scripts/phone_push_android.py 127.0.0.1 ~/Models/tail.gguf [tail.gguf]
    python3 scripts/phone_push_android.py --via adb --device SERIAL 127.0.0.1 file.gguf
    python3 scripts/phone_push_android.py --via fetch 127.0.0.1 ./dir remote-name
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

CMD_PORT = 50061


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def pick_device(explicit: str = "") -> tuple[str, str]:
    adb = shutil.which("adb")
    if not adb:
        print("adb not on PATH", file=sys.stderr)
        sys.exit(1)
    if explicit:
        return adb, explicit
    p = sh([adb, "devices"])
    for line in p.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return adb, parts[0]
    print("no adb device", file=sys.stderr)
    sys.exit(1)
    raise AssertionError


def adb_push(adb: str, dev: str, local: str) -> None:
    dst = "/sdcard/Download/" + os.path.basename(local)
    p = sh([adb, "-s", dev, "push", local, dst])
    if p.returncode != 0:
        print((p.stderr or p.stdout).strip(), file=sys.stderr)
        sys.exit(1)
    size = os.path.getsize(local)
    print(f"pushed {size / 1e6:.0f} MB to {dst} (adb push). In Backburner: Import tail.gguf.")


def fetch_tree(ip: str, local: str, remote: str, adb: str, dev: str) -> None:
    root = os.path.dirname(os.path.abspath(local))
    base = os.path.basename(local)

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=root, **k)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    # The phone fetches http://127.0.0.1:port/... : that 127.0.0.1 is the
    # PHONE's loopback, so reverse it to this host (this is the step the
    # original phone-push.py misses behind adb reverse).
    p = sh([adb, "-s", dev, "reverse", f"tcp:{port}", f"tcp:{port}"])
    if p.returncode != 0:
        print(f"adb reverse tcp:{port} failed", file=sys.stderr)
        sys.exit(1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    files = [local] if os.path.isfile(local) else [
        os.path.join(d, f) for d, _, fs in os.walk(local) for f in fs
    ]
    c = socket.create_connection((ip, CMD_PORT), timeout=900)
    f = c.makefile("rw")
    t0, total = time.time(), 0
    try:
        for path in sorted(files):
            rel = os.path.relpath(path, root)
            dst = remote + rel[len(base):]
            url = f"http://127.0.0.1:{port}/" + urllib.parse.quote(rel)
            f.write(f"fetch {url} {dst}\n")
            f.flush()
            r = json.loads(f.readline())
            if r.get("error") or r.get("bytes") != os.path.getsize(path):
                sys.exit(f"FAILED {rel}: {r}")
            total += r["bytes"]
    finally:
        try:
            sh([adb, "-s", dev, "reverse", "--remove", f"tcp:{port}"])
        except Exception:
            pass
        srv.shutdown()
    dt = time.time() - t0
    print(f"pushed {len(files)} files, {total / 1e6:.0f} MB to files/{remote} "
          f"in {dt:.1f} s ({total / 1e6 / max(dt, 1e-9):.0f} MB/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ip", nargs="?", default="127.0.0.1")
    ap.add_argument("local")
    ap.add_argument("remote", nargs="?")
    ap.add_argument("--via", choices=("auto", "adb", "fetch"), default="auto")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    local = os.path.abspath(args.local)
    if not os.path.exists(local):
        print(f"missing: {local}", file=sys.stderr)
        sys.exit(1)
    remote = args.remote or os.path.basename(local)
    adb, dev = pick_device(args.device)

    if args.via == "fetch" or (args.via == "auto" and os.path.isdir(local)):
        fetch_tree(args.ip, local, remote, adb, dev)
    else:
        if os.path.isdir(local):
            print("directories need --via fetch", file=sys.stderr)
            sys.exit(1)
        adb_push(adb, dev, local)


if __name__ == "__main__":
    main()

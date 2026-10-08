#!/usr/bin/env python3
"""device-probe.py - describe this setup as a device profile: the Mac, and every phone or iPad on the cable with Backburner open.

    bench/device-probe.py            # JSON on stdout: paste it into a "[Results]" issue
    bench/device-probe.py --no-phone # the Mac only

Reads only: no model, no GPU work, a few short requests to each device. The profile holds hardware facts (chip, RAM, cores,
SME2, OS version, the app's memory budget, link round trip) and nothing that identifies you or a device: no UDID, serial,
device name, IP address, user name or path. The planner (docs/ROADMAP.md, "any device") will learn from these profiles.
"""
import argparse
import json
import os
import platform
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PA_MAGIC, PA_PING, PA_HELLO, PA_BYE = 0x4E544150, 12, 1, 11


def sh(*cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def sysctl(name, cast=str):
    v = sh("/usr/sbin/sysctl", "-n", name)
    try:
        return cast(v) if v else None
    except ValueError:
        return None


def mac_profile():
    if platform.system() != "Darwin":
        return {"os": platform.system(), "os_version": platform.release(), "machine": platform.machine(),
                "cpu_count": sysctl("hw.ncpu", int)}
    gpu_cores = None
    try:
        d = json.loads(sh("/usr/sbin/system_profiler", "SPDisplaysDataType", "-json", timeout=30) or "{}")
        for g in d.get("SPDisplaysDataType", []):
            if g.get("sppci_cores"):
                gpu_cores = int(g["sppci_cores"])
    except (ValueError, TypeError):
        pass
    mem = sysctl("hw.memsize", int)
    return {
        "os": "macOS",
        "os_version": sh("/usr/bin/sw_vers", "-productVersion"),
        "model": sysctl("hw.model"),                      # e.g. Mac16,7: a product type, not a serial
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_gb": round(mem / 2**30) if mem else None,
        "p_cores": sysctl("hw.perflevel0.physicalcpu", int),
        "e_cores": sysctl("hw.perflevel1.physicalcpu", int),
        "gpu_cores": gpu_cores,
        "sme2": sysctl("hw.optional.arm.FEAT_SME2", int) == 1,
        "gpu_wired_limit_mb": sysctl("iogpu.wired_limit_mb", int),   # 0 = the macOS default
    }


def phone_ips():
    """Addresses of the devices on the cable with Backburner open (scripts/phone-up.sh finds them; nothing is relaunched)."""
    out = subprocess.run([str(ROOT / "scripts/phone-up.sh")], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PHONES_ALL": "1", "TAIL_WAIT": "1"}).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def ctl(ip, cmd):
    try:
        with socket.create_connection((ip, 50061), timeout=3) as s:
            s.sendall((cmd + "\n").encode())
            s.settimeout(3)
            b = b""
            try:
                while not b.endswith(b"\n") and len(b) < 65536:
                    c = s.recv(4096)
                    if not c:
                        break
                    b += c
            except socket.timeout:
                pass
        return json.loads(b.decode(errors="replace").strip() or "{}")
    except (OSError, ValueError):
        return {}


def rtt_us(ip, n=50):
    """Median request/reply time of an empty phone-attn PING: what one hop across the cable costs."""
    try:
        with socket.create_connection((ip, 50062), timeout=3) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = []
            for _ in range(n):
                t0 = time.perf_counter()
                s.sendall(struct.pack("<IIQI", PA_MAGIC, PA_PING, 4, 0))
                h = b""
                while len(h) < 16:
                    h += s.recv(16 - len(h))
                t.append((time.perf_counter() - t0) * 1e6)
            s.sendall(struct.pack("<IIQ", PA_MAGIC, PA_BYE, 0))
            return round(statistics.median(t))
    except OSError:
        return None


def phone_profile(ip):
    m = ctl(ip, "mem")
    machine = m.get("machine")
    return {
        "kind": "iPad" if str(machine).startswith("iPad") else "iPhone" if machine else "unknown (app before 0.0.5)",
        "machine": machine,                               # e.g. iPhone18,2: a product type, not a serial
        "ram_mb": round(m["phys_mb"]) if "phys_mb" in m else None,
        "app_budget_mb": round(m["avail_mb"]) if "avail_mb" in m else None,
        "system_wired_mb": round(m["sys_wired_mb"]) if "sys_wired_mb" in m else None,
        "thermal": m.get("thermal"),
        "link_rtt_us": rtt_us(ip),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-phone", action="store_true", help="describe the Mac only")
    a = ap.parse_args()
    prof = {"profile_version": 1, "host": mac_profile(), "devices": []}
    if not a.no_phone and platform.system() == "Darwin":
        try:
            prof["devices"] = [phone_profile(ip) for ip in phone_ips()]
        except Exception as e:   # no phone is a valid profile too
            print(f"device-probe: no phone ({e})", file=sys.stderr)
    json.dump(prof, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()

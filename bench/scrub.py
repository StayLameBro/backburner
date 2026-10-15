"""scrub.py - keep machine-identifying text out of the rows the bench scripts write (bench/results/*.jsonl are committed).

  scrub(text)   the captured serve: log line and similar: link-local / LAN addresses become 169.254.x.y style placeholders,
                fe80:: addresses fe80::x, home folders ~/, Bonjour host names <host>.local
  mac_name()    "Apple M4 Pro 24 GB": chip and memory, like the hand-written rows, never the host name (platform.node() is usually "Firstname-MacBook-Pro")
"""
import platform, re, subprocess

_RULES = [
    (re.compile(r'\b169\.254\.\d{1,3}\.\d{1,3}\b'), '169.254.x.y'),
    (re.compile(r'\b192\.168\.\d{1,3}\.\d{1,3}\b'), '192.168.x.y'),
    (re.compile(r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), '10.x.y.z'),
    (re.compile(r'\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b'), '172.16.x.y'),
    (re.compile(r'\bfe80::[0-9a-fA-F:]*(?:%\w+)?', re.I), 'fe80::x'),
    (re.compile(r'(/Users|/home)/[^/\s"\']+/'), '~/'),
    (re.compile(r'\b[A-Za-z0-9][\w-]*\.local\b'), '<host>.local'),
]


def scrub(text):
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text


def mac_name():
    def sysctl(name):
        try:
            return subprocess.run(['/usr/sbin/sysctl', '-n', name], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''
    if platform.system() != 'Darwin':
        return f'{platform.system()} {platform.machine()}'
    chip = sysctl('machdep.cpu.brand_string') or platform.machine()
    mem = sysctl('hw.memsize')
    return f'{chip} {round(int(mem) / 2**30)} GB' if mem.isdigit() else chip

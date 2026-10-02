#!/bin/bash
# phone-tail-android.sh - Android port of scripts/phone-tail.sh.
# Put a split-prefill tail on the wired Android phone and confirm which tail it loaded.
#   scripts/phone-tail-android.sh                      # show loaded tail + memory, change nothing
#   scripts/phone-tail-android.sh L40                  # ~/Models/tail-iq4xs-L40-nohead.gguf
#   scripts/phone-tail-android.sh /path/tail-...-L44-nohead.gguf
# Tails are the same head-less IQ4_XS splits as iOS (scripts/split-gguf.py).
# Delivery is `adb push` to the app's files (no relaunch needed: the tail
# server picks up tail.gguf on next connect; force-stop + reopen to be sure).
set -u
cd "$(dirname "$0")/.."
TAIL=${1:-}
case "$TAIL" in
  L[0-9]*) TAIL=$HOME/Models/tail-iq4xs-$TAIL-nohead.gguf ;;
esac
command -v adb >/dev/null || { echo "adb not on PATH"; exit 1; }
DEV=$(adb devices | awk '$2=="device"{print $1; exit}')
[ -n "$DEV" ] || { echo "no adb device"; exit 1; }

loaded() {
  python3 - <<'PY' 2>/dev/null
import re, socket, struct
try:
    with socket.create_connection(('127.0.0.1', 50060), timeout=15) as s:
        s.sendall(struct.pack('<IIQ8I', 0x4C545053, 1, 32, 2, 1, 0, 0, 0, 0, 1, 1))
        b = b''
        while len(b) < 16: b += s.recv(16 - len(b))
        magic, typ, n = struct.unpack('<IIQ', b)
        body = b''
        while len(body) < n: body += s.recv(n - len(body))
        m = re.search(r'phone has layers \[(\d+),', body.decode(errors='replace'))
        print(m.group(1) if m else '?')
except Exception:
    print('down')
PY
}
mem() { printf 'mem\n' | python3 -c "import socket; s=socket.create_connection(('127.0.0.1',50061),timeout=2); s.sendall(b'mem\n'); print(s.makefile().read())" 2>/dev/null; echo; }

if [ -z "$TAIL" ]; then
  echo "phone 127.0.0.1 (adb $DEV): tail L=$(loaded)"; mem; exit 0
fi
test -s "$TAIL" || { echo "missing tail: $TAIL"; exit 1; }
WANT=$(basename "$TAIL" | sed -n 's/.*-L\([0-9]*\)-.*/\1/p')
HAVE=$(loaded)
if [ "$HAVE" = "$WANT" ] && [ -z "${FORCE:-}" ]; then echo "phone already has L=$WANT"; mem; exit 0; fi
echo "phone: tail L=$HAVE -> L=$WANT ($(du -h "$TAIL" | cut -f1))"
# Prefer the in-app importer path: push beside the app, then `fetch` it into filesDir
# over the same :50061 protocol as iOS phone-push.py (works after adb reverse).
adb -s "$DEV" push "$TAIL" /sdcard/Download/tail.gguf
echo "pushed to /sdcard/Download/tail.gguf; in Backburner use Import tail.gguf."
echo "Or: python3 scripts/phone-push.py 127.0.0.1 \"$TAIL\" tail.gguf  (after phone-up-android.sh)"

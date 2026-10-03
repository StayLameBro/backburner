#!/bin/bash
# phone-tail.sh - put a split-prefill tail on the wired iPhone, restart Backburner, confirm which tail it loaded.
#   scripts/phone-tail.sh                      # show the loaded tail + phone memory, change nothing
#   scripts/phone-tail.sh L40                  # ~/Models/tail-iq4xs-L40-nohead.gguf (made by scripts/split-gguf.py --no-head, 5.1 GB)
#   scripts/phone-tail.sh /path/tail-...-L44-nohead.gguf
# Tails (head-less, IQ4_XS): A19 L40 (safe max; L36 hits ~9.7 GB wired, jetsam ~10.4), L44; A18 L52, L56.
set -u
cd "$(dirname "$0")/.."
TAIL=${1:-}
case "$TAIL" in
  L[0-9]*) TAIL=$HOME/Models/tail-iq4xs-$TAIL-nohead.gguf ;;
esac
PUP=$(TAIL_WAIT=${TAIL_WAIT:-5} scripts/phone-up.sh) || exit 1
read -r IP _ _ _ _ _ <<< "$PUP"

loaded() {   # the tail's first layer, from its HELLO reply ("phone has layers [L, ...")
  python3 - "$IP" <<'PY'
import re, socket, struct, sys
try:
    with socket.create_connection((sys.argv[1], 50060), timeout=15) as s:
        s.sendall(struct.pack('<IIQ8I', 0x4C545053, 1, 32, 2, 1, 0, 0, 0, 0, 1, 1))
        b = b''
        while len(b) < 16: b += s.recv(16 - len(b))
        magic, typ, n = struct.unpack('<IIQ', b)
        body = b''
        while len(body) < n: body += s.recv(n - len(body))
        m = re.search(r'phone has layers \[(\d+),', body.decode(errors='replace'))
        print(m.group(1) if m else '?')
except Exception as e:
    print('down')
PY
}
mem() { printf 'mem\n' | nc -G 2 "$IP" 50061 2>/dev/null; echo; }

if [ -z "$TAIL" ]; then
  echo "phone $IP: tail L=$(loaded)"; mem; exit 0
fi
test -s "$TAIL" || { echo "missing tail: $TAIL"; exit 1; }
WANT=$(basename "$TAIL" | sed -n 's/.*-L\([0-9]*\)-.*/\1/p')
HAVE=$(loaded)
if [ "$HAVE" = "$WANT" ] && [ -z "${FORCE:-}" ]; then echo "phone $IP already has L=$WANT"; mem; exit 0; fi
echo "phone $IP: tail L=$HAVE -> L=$WANT ($(du -h "$TAIL" | cut -f1))"
python3 scripts/phone-push.py "$IP" "$TAIL" tail.gguf || exit 1
UDID=$(ioreg -p IOUSB -w 0 -l | sed -n 's/.*"kUSBSerialNumberString" = "\(00008[0-9A-F]*\)".*/\1/p' | head -1)
UDID=${UDID:0:8}-${UDID:8}
BID=$(scripts/bundle-id.sh "$UDID") || exit 2
perl -e 'alarm 30; exec @ARGV' xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BID" >/dev/null 2>&1 \
  || { echo "relaunch failed: unlock the phone, reopen Backburner, then run: scripts/phone-tail.sh (no args) to check"; exit 2; }
for _ in $(seq 1 60); do nc -z -G 1 "$IP" 50060 >/dev/null 2>&1 && break; sleep 2; done
sleep 3
HAVE=$(loaded)
echo "phone $IP: tail L=$HAVE"; mem
[ "$HAVE" = "$WANT" ]

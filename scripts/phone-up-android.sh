#!/bin/bash
# phone-up-android.sh - Android port of scripts/phone-up.sh.
# Find the wired Android phone (adb), forward the Backburner ports over the
# cable (adb reverse, no IP hunting, no Wi-Fi), and print the same line:
#   "IP TAIL_UP ATTN_VERSION AVAIL_MB WIRED_MB NAME"
# so serve.sh wrappers keep working with PHONE_KV=IP:50062.
#
# Why adb reverse instead of broadcast ping: Android USB gives no stable
# link-local discovery story across OEMs (USB tethering subnets vary, Wi-Fi
# must never be used), while `adb reverse` forwards localhost on the Mac to
# localhost on the phone over the one cable. After this script,
# 127.0.0.1:50060/50061/50062 on the Mac IS the phone.
set -u
TAIL_WAIT=${TAIL_WAIT:-45}

say() { echo "phone-up-android: $*" >&2; }
command -v adb >/dev/null || { say "adb not on PATH"; exit 1; }

DEV=$(adb devices | awk '$2=="device"{print $1; exit}')
[ -n "$DEV" ] || { say "no Android device on adb (enable USB debugging, unlock it)"; exit 1; }
NAME=$(adb -s "$DEV" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
[ -n "$NAME" ] || NAME="wired Android"

for p in 50060 50061 50062 50052; do
  adb -s "$DEV" reverse "tcp:$p" "tcp:$p" >/dev/null 2>&1 \
    || { say "adb reverse tcp:$p failed"; exit 1; }
done
IP=127.0.0.1

hello() {
  python3 - "$1" <<'EOF' 2>/dev/null
import socket, struct, sys
try:
    s = socket.create_connection((sys.argv[1], 50062), timeout=2)
    s.sendall(struct.pack('<IIQ', 0x4E544150, 1, 0))
    h = b''
    while len(h) < 16: h += s.recv(16 - len(h))
    n = struct.unpack('<IIQ', h)[2]; b = b''
    while len(b) < n: b += s.recv(n - len(b))
    print(struct.unpack('<I', b[:4])[0])
    s.sendall(struct.pack('<IIQ', 0x4E544150, 11, 0)); s.close()
except Exception:
    pass
EOF
}

VER=$(hello "$IP")
if [ -z "$VER" ]; then
  say "Backburner HELLO did not answer on $NAME: open the app and keep it foreground, then retry"
  exit 1
fi
TAIL=0
for _ in $(seq 1 "$TAIL_WAIT"); do
  (echo >/dev/tcp/127.0.0.1/50060) >/dev/null 2>&1 && { TAIL=1; break; }
  sleep 1
done
[ $TAIL = 1 ] || say "the prefill tail (:50060) isn't up (no tail.gguf on the phone?): split prefill stays off"
# Same JSON as iOS :50061 mem (avail_mb, sys_wired_mb). sys_wired has no Android
# equivalent; the app reports MemAvailable for both (conservative cap math).
MEM=$(printf 'mem\n' | python3 -c "import socket,sys; s=socket.create_connection(('127.0.0.1',50061),timeout=2); s.sendall(b'mem\n'); print(s.makefile().readline())" 2>/dev/null \
  | python3 -c "import json,sys; m=json.loads(sys.stdin.read()); print(int(m['avail_mb']), int(m['sys_wired_mb']))" 2>/dev/null)
read -r AVAIL WIRED <<< "${MEM:-0 0}"
say "$NAME at $IP (adb $DEV): phone-attn v$VER, prefill tail $([ $TAIL = 1 ] && echo up || echo down), ${WIRED} MiB wired-ish, ${AVAIL} MiB app budget"
echo "$IP $TAIL $VER $AVAIL $WIRED $NAME"

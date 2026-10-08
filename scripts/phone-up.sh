#!/bin/bash
# phone-up.sh - find the iPhone on the USB cable, make sure Backburner answers, print "IP TAIL_UP ATTN_VERSION AVAIL_MB WIRED_MB NAME".
# Exit 1 (and a reason on stderr) if no usable phone. Used by serve.sh (PHONE=auto, the default).
#
# Why: the phone's link-local address changes after every replug/reboot and iOS stops Backburner when the phone locks, so a
# hardcoded IP or a "just open the app" instruction silently left the Mac running alone.
#   1. the wired phone's UDID (USB device tree; CoreDevice can time out while the cable still works)
#   2. its USB address: broadcast ping on each link-local interface, first reply that isn't this Mac
#   3. Backburner's phone-attn (:50062) answering HELLO; if not: relaunch Backburner (the phone must be unlocked) and wait
#   4. the split-prefill tail (:50060), up to TAIL_WAIT seconds (it loads a ~5 GB model)
set -u
TAIL_WAIT=${TAIL_WAIT:-45}
say() { echo "phone-up: $*" >&2; }

UDID=${UDID:-}
if [ -z "$UDID" ]; then
  # ioreg sees the USB device even when CoreDeviceService is stuck during a replug.
  raw=$(ioreg -p IOUSB -w 0 -l 2>/dev/null | sed -n 's/.*"kUSBSerialNumberString" = "\(00008[0-9A-F]*\)".*/\1/p' | head -1)
  if [ ${#raw} -eq 24 ]; then UDID=${raw:0:8}-${raw:8}; fi
fi
[ -n "$UDID" ] || { say "no iPhone or iPad on the USB cable (one on Wi-Fi doesn't count)"; exit 1; }
NAME="wired iPhone"

find_ip() {
  local ifs ip
  ifs=$(/sbin/ifconfig | awk '/^[a-z]/{i=$1} /inet 169\.254\./{print i}' | tr -d :)
  for i in $ifs; do
    local own; own=$(/sbin/ifconfig "$i" | awk '/inet 169\.254\./{print $2}')
    ip=$(/sbin/ping -b "$i" -c 2 -t 2 169.254.255.255 2>/dev/null | awk '/bytes from/{sub(":","",$4); print $4}' | grep -v "^$own$" | head -1)
    [ -n "$ip" ] && { echo "$ip"; return 0; }
  done
  return 1
}

all_ips() {   # every other device answering on the link-local interfaces (more than one phone: one line each)
  local ifs
  ifs=$(/sbin/ifconfig | awk '/^[a-z]/{i=$1} /inet 169\.254\./{print i}' | tr -d :)
  for i in $ifs; do
    local own; own=$(/sbin/ifconfig "$i" | awk '/inet 169\.254\./{print $2}')
    /sbin/ping -b "$i" -c 2 -t 2 169.254.255.255 2>/dev/null | awk '/bytes from/{sub(":","",$4); print $4}' | grep -v "^$own$"
  done | sort -u
}

hello() {   # phone-attn protocol version, or empty
  python3 - "$1" <<'EOF' 2>/dev/null
import errno, socket, struct, sys
try:
    s = socket.create_connection((sys.argv[1], 50062), timeout=2)
    s.sendall(struct.pack('<IIQ', 0x4E544150, 1, 0))
    h = b''
    while len(h) < 16: h += s.recv(16 - len(h))
    n = struct.unpack('<IIQ', h)[2]; b = b''
    while len(b) < n: b += s.recv(n - len(b))
    print(struct.unpack('<I', b[:4])[0])
    s.sendall(struct.pack('<IIQ', 0x4E544150, 11, 0)); s.close()
except OSError as e:
    if e.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EACCES, errno.EPERM):
        print('ROUTE_ERROR')
except Exception:
    pass
EOF
}

IP=$(find_ip) || { say "$NAME ($UDID) is wired but doesn't answer on the USB network (unlock it, or replug the cable)"; exit 1; }
VER=$(hello "$IP")
if [ "$VER" = ROUTE_ERROR ]; then
  say "$NAME ($UDID) answers on USB at $IP, but this process cannot route a direct connection to it"
  exit 1
fi
if [ -z "$VER" ]; then
  say "Backburner HELLO did not answer on $NAME: trying an app relaunch"
  BID=$("$(dirname "$0")/bundle-id.sh" "$UDID") || exit 1   # Xcode build (team id) or AltStore install
  out=$(xcrun devicectl device process launch --device "$UDID" --terminate-existing "$BID" 2>&1)
  echo "$out" | grep -qiE 'locked' && { say "the phone is locked: unlock it and run again"; exit 1; }
  for _ in $(seq 1 20); do sleep 1; VER=$(hello "$IP"); [ -n "$VER" ] && break; done
  [ -n "$VER" ] || { say "Backburner didn't come up on $IP after a relaunch"; exit 1; }
fi
TAIL=0
for _ in $(seq 1 "$TAIL_WAIT"); do /usr/bin/nc -z -G 1 "$IP" 50060 >/dev/null 2>&1 && { TAIL=1; break; }; sleep 1; done
if [ $TAIL = 1 ]; then
  # the port is open as soon as the app runs; ask the tail whether a model is loaded (the v1 HELLO probe phone-tail.sh uses:
  # its error names the loaded layers, or says there is no tail model)
  TSTATE=$(python3 - "$IP" <<'PY' 2>/dev/null
import socket, struct, sys
try:
    with socket.create_connection((sys.argv[1], 50060), timeout=15) as s:
        s.sendall(struct.pack('<IIQ8I', 0x4C545053, 1, 32, 2, 1, 0, 0, 0, 0, 1, 1))
        b = b''
        while len(b) < 16: b += s.recv(16 - len(b))
        n = struct.unpack('<IIQ', b)[2]; body = b''
        while len(body) < min(n, 4096): body += s.recv(min(n, 4096) - len(body))
        t = body.decode(errors='replace')
        print('none' if ('no tail model' in t or 'not a split TAIL' in t or 'failed to load' in t) else 'loaded')
except Exception:
    print('unknown')
PY
)
  [ "$TSTATE" = none ] && { TAIL=0; say "the prefill tail answers but has no model loaded: push one with scripts/phone-tail.sh (split prefill stays off)"; }
else
  say "the prefill tail (:50060) isn't up (no tail.gguf on the phone?): split prefill stays off"
fi
# Metal KV and tail weights increase system wired memory, not the process footprint.
MEM=$(printf 'mem\n' | /usr/bin/nc -G 2 "$IP" 50061 2>/dev/null | python3 -c "import json,sys; m=json.loads(sys.stdin.read()); print(int(m['avail_mb']), int(m['sys_wired_mb']), m.get('machine', '?'))" 2>/dev/null)
read -r AVAIL WIRED MACHINE <<< "${MEM:-0 0 ?}"
case "${MACHINE:-?}" in iPad*) NAME="wired iPad" ;; esac   # apps before 0.0.5 don't report it: they stay "iPhone"
say "$NAME at $IP: phone-attn v$VER, prefill tail $([ $TAIL = 1 ] && echo up || echo down), ${WIRED} MiB system wired, ${AVAIL} MiB app budget"
echo "$IP $TAIL $VER $AVAIL $WIRED $NAME"
# PHONES_ALL=1: one more line per extra phone with Backburner open (old-KV share only, no prefill tail; no relaunch)
if [ "${PHONES_ALL:-0}" = 1 ]; then
  for ip2 in $(all_ips); do
    [ "$ip2" = "$IP" ] && continue
    v2=$(hello "$ip2"); { [ -n "$v2" ] && [ "$v2" != ROUTE_ERROR ]; } || continue
    m2=$(printf 'mem\n' | /usr/bin/nc -G 2 "$ip2" 50061 2>/dev/null | python3 -c "import json,sys; m=json.loads(sys.stdin.read()); print(int(m['avail_mb']), int(m['sys_wired_mb']))" 2>/dev/null)
    read -r a2 w2 <<< "${m2:-0 0}"
    say "another device at $ip2: phone-attn v$v2, ${w2} MiB system wired, ${a2} MiB app budget"
    echo "$ip2 0 $v2 $a2 $w2 iPhone"
  done
fi

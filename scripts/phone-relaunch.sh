#!/bin/bash
# phone-relaunch.sh IP: relaunch the Backburner app on the wired iPhone (env UDID to pick one) and wait for the prefill tail (:50060)
set -eu
IP=$1
# the wired phone's UDID from the USB device tree, as in phone-up.sh: Xcode 26's `devicectl list devices` shows CoreDevice
# UUIDs, not UDIDs, so grepping its table for a UDID found nothing and the relaunch silently went to an empty --device
UDID=${UDID:-}
if [ -z "$UDID" ]; then
  raw=$(ioreg -p IOUSB -w 0 -l 2>/dev/null | sed -n 's/.*"kUSBSerialNumberString" = "\(00008[0-9A-F]*\)".*/\1/p' | head -1)
  if [ ${#raw} -eq 24 ]; then UDID=${raw:0:8}-${raw:8}; fi
fi
[ -n "$UDID" ] || { echo "phone-relaunch: no iPhone on the USB cable" >&2; exit 1; }
BID=$("$(dirname "$0")/bundle-id.sh" "$UDID")
xcrun devicectl device process launch --terminate-existing --device "$UDID" "$BID" 2>&1 | grep -v 'provisioning paramter' | grep -E 'Launched|Locked|rror' | head -1
for i in $(seq 1 30); do nc -z -G1 "$IP" 50060 2>/dev/null && { echo "tail up after ~$((i*2)) s"; exit 0; }; sleep 2; done
echo "tail not up after 60 s" >&2; exit 1

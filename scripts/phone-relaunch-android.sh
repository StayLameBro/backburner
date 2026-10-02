#!/bin/bash
# phone-relaunch-android.sh - Android port of scripts/phone-relaunch.sh.
# Relaunch Backburner on the wired phone (it must be unlocked).
set -u
DEV=${1:-$(adb devices | awk '$2=="device"{print $1; exit}')}
[ -n "$DEV" ] || { echo "no adb device"; exit 1; }
adb -s "$DEV" shell am force-stop app.backburner >/dev/null 2>&1
adb -s "$DEV" shell monkey -p app.backburner -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
echo "relaunched app.backburner on $DEV (unlock the phone if it did not come up)"

# phone-up-android.ps1 - Windows wrapper: adb reverse + HELLO/tail/mem probes.
# Same stdout contract as phone-up-android.sh: "IP TAIL_UP VER AVAIL WIRED NAME".
param([int]$TailWait = 45, [string]$Device = "", [switch]$NoReverse)
$ErrorActionPreference = 'Stop'
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$args = @("$Root\scripts\phone_up_android.py", '--tail-wait', "$TailWait")
if ($Device) { $args += @('--device', $Device) }
if ($NoReverse) { $args += '--no-reverse' }
& python @args
exit $LASTEXITCODE

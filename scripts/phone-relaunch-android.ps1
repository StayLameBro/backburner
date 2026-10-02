# phone-relaunch-android.ps1 - Windows port of scripts/phone-relaunch-android.sh.
param([string]$Device = "")
$ErrorActionPreference = 'Stop'
if (-not $Device) {
  $Device = (adb devices | Select-String '^\S+\s+device$' | ForEach-Object { ($_ -split '\s+')[0] } | Select-Object -First 1)
}
if (-not $Device) { throw 'no adb device' }
adb -s $Device shell am force-stop app.backburner | Out-Null
adb -s $Device shell monkey -p app.backburner -c android.intent.category.LAUNCHER 1 | Out-Null
Write-Host "relaunched app.backburner on $Device (unlock the phone if it did not come up)"

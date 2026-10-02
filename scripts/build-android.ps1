# build-android.ps1 - Windows host port of scripts/build-android.sh.
# Builds the Backburner Android app (arm64-v8a) and installs it with adb.
# Needs: ANDROID_SDK_ROOT (or ANDROID_HOME) + NDK r26+, JDK 17, adb + gradle on PATH.
param([switch]$Release)
$ErrorActionPreference = 'Stop'
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$AndRoot = Join-Path $Root 'android'
$Mode = if ($Release) { 'assembleRelease' } else { 'assembleDebug' }

$Sdk = $env:ANDROID_SDK_ROOT; if (-not $Sdk) { $Sdk = $env:ANDROID_HOME }
if (-not $Sdk) { throw 'set ANDROID_SDK_ROOT to your Android SDK' }
if (-not (Get-Command adb -ErrorAction SilentlyContinue)) { throw 'adb not on PATH (platform-tools)' }
if (-not (Get-Command java -ErrorAction SilentlyContinue)) { throw 'JDK 17 required' }
$LlamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { Join-Path $Root 'llama.cpp' }
if (-not (Test-Path (Join-Path $LlamaDir 'include\llama.h'))) { throw "missing $LlamaDir (git submodule update --init --recursive)" }

Set-Location $AndRoot
$gradle = if (Test-Path '.\gradlew.bat') { '.\gradlew.bat' } else { 'gradle' }
& $gradle $Mode
if ($LASTEXITCODE -ne 0) { throw "gradle $Mode failed" }
$apk = Get-ChildItem -Recurse -Filter *.apk 'app\build\outputs\apk' | Select-Object -First 1
if (-not $apk) { throw 'no APK produced' }
Write-Host "apk: $($apk.FullName)"
try {
  adb get-state 2>$null | Out-Null
  adb install -r $apk.FullName
  Write-Host 'installed. Open Backburner, keep it foreground, then:'
  Write-Host '  adb reverse tcp:50060 tcp:50060; adb reverse tcp:50061 tcp:50061; adb reverse tcp:50062 tcp:50062'
} catch {
  Write-Host 'no adb device: install the APK by hand.'
}

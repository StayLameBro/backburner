# phone-push-android.ps1 - Windows wrapper for scripts/phone_push_android.py.
#   scripts\phone-push-android.ps1 127.0.0.1 $HOME\Models\tail.gguf tail.gguf
param([string]$Ip = "127.0.0.1", [Parameter(Mandatory=$true)][string]$Local, [string]$Remote = "", [ValidateSet('auto','adb','fetch')][string]$Via = 'auto')
$ErrorActionPreference = 'Stop'
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$args = @("$Root\scripts\phone_push_android.py", '--via', $Via, $Ip, $Local)
if ($Remote) { $args += $Remote }
& python @args
exit $LASTEXITCODE

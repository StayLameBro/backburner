# phone-tail-android.ps1 - Windows wrapper: show or push the split-prefill tail.
#   scripts\phone-tail-android.ps1            # show loaded tail + memory
#   scripts\phone-tail-android.ps1 L40        # $HOME\Models\tail-iq4xs-L40-nohead.gguf via adb push
param([string]$Tail = "")
$ErrorActionPreference = 'Stop'
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
if ($Tail) { & python "$Root\scripts\phone_tail_android.py" $Tail } else { & python "$Root\scripts\phone_tail_android.py" }
exit $LASTEXITCODE

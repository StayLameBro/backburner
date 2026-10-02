# serve.ps1 - Windows host port of scripts/serve.sh for Windows <> Android.
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
#   powershell -File scripts\serve.ps1 -Ctx 98304 -Port 8081
#   powershell -File scripts\serve.ps1 -NoPhone          # Windows alone
#   $env:LLAMA_SPLIT_TAIL='127.0.0.1:50060'; $env:PHONE_KV='127.0.0.1:50062'; scripts\serve.ps1
#
# What changed vs serve.sh (Mac):
#  - No bash/nc/curl/sysctl. Phone discovery is scripts/phone_up_android.py
#    (adb reverse + pure-Python HELLO/tail/mem probes); phone notes go through
#    scripts/phone_note.py (raw TCP, no nc flags to get wrong on Windows).
#  - No iogpu.wired_limit_mb gate, no -lm none default, no GGML_METAL_* /
#    GGML_METAL_FA_SME / GGML_METAL_MM_SME exports: those are Apple-Metal and
#    M4-SME tunables. The lossless spec tunables (LLAMA_SPEC_*) are kept.
#  - GPU backend is whatever your llama-server.exe was built with (CUDA or
#    Vulkan). No `-dev MTL0` default; set $env:PHONE_DRAFT yourself only if you
#    know the device name in your build.
#  - proxy.py is unchanged (stdlib only, Windows-safe) and still owns PORT
#    while llama-server runs on PORT+100.
param(
  [int]$Ctx = 0,
  [int]$Port = 8080,
  [string]$Model = "",
  [string]$Draft = "",
  [string]$Kv = "",
  [string]$PhoneKv = "",
  [string]$SplitTail = "",
  [string]$Bin = "",
  [switch]$NoPhone,
  [switch]$NoProxy,
  [string]$CacheDir = "",
  [string]$ServerArgs = ""
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
Set-Location $Root
$Scripts = Join-Path $Root 'scripts'
function PhoneNote([string]$msg) {
  if ($script:PhoneIp) {
    & python "$Scripts\phone_note.py" $script:PhoneIp $msg 2>$null | Out-Null
  }
}

# --- defaults (same model files as serve.sh, Windows home) ---
if (-not $Bin)   { $Bin = $env:BIN; if (-not $Bin) { $Bin = Join-Path $Root 'llama.cpp\build-windows\bin' } }
if (-not $Model) { $Model = $env:MODEL; if (-not $Model) { $Model = Join-Path $HOME 'Models\Qwen3.8-27B-IQ4_XS.gguf' } }
if (-not $Draft) { $Draft = $env:DRAFT; if (-not $Draft) { $Draft = Join-Path $HOME 'Models\dflash2-v2-q4km-self16.gguf' } }
if ($Ctx -eq 0)  { $Ctx = if ($env:CTX) { [int]$env:CTX } else { 65536 } }
if (-not $Kv)    { $Kv = if ($env:KV) { $env:KV } else { if ($Ctx -gt 65536) { 'q4_0' } else { 'q8_0' } } }
if (-not $PhoneKv)   { $PhoneKv = $env:PHONE_KV }
if (-not $SplitTail) { $SplitTail = $env:LLAMA_SPLIT_TAIL }
$Proxy = -not $NoProxy -and ($env:PROXY ?? '1') -ne '0'

$ServerExe = Join-Path $Bin 'llama-server.exe'
if (-not (Test-Path $ServerExe)) { $ServerExe = Join-Path $Bin 'llama-server' }
if (-not (Test-Path $ServerExe)) { throw "llama-server not found in $Bin (build llama.cpp for Windows first; see docs/ANDROID.md)" }
if (-not (Test-Path $Model)) { throw "missing model: $Model" }

# --- phone auto-detect (Android over adb reverse; explicit env wins) ---
$PhoneIp = ''
$CtxTotal = $env:CTX_TOTAL
if (-not $NoPhone -and -not $PhoneKv -and -not $SplitTail -and ($env:PHONE ?? 'auto') -ne '0') {
  try {
    $line = & python "$Scripts\phone_up_android.py" 2> (Join-Path ([IO.Path]::GetTempPath()) 'phone-up-android.log') | Select-Object -Last 1
    if ($line -match '^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$') {
      $pip, $ptail, $pver, $pavail, $pwired = $Matches[1], $Matches[2], $Matches[3], [int]$Matches[4], [int]$Matches[5]
      $PhoneIp = $pip
      if ($ptail -eq '1') { $SplitTail = "$pip`:50060" }
      # Android v1: no ANE page template (nnapi stub reports pa_ane:false).
      $pane = 'off'
      $wiredMax = if ($env:PHONE_WIRED_MAX_MB) { [int]$env:PHONE_WIRED_MAX_MB } else { 9400 }
      $appReserve = if ($env:PHONE_APP_RESERVE_MB) { [int]$env:PHONE_APP_RESERVE_MB } else { 512 }
      if ($pwired -le 0 -or $pwired -ge $wiredMax -or $pavail -le $appReserve) {
        Write-Warning 'serve: phone memory unknown or too low; remote KV disabled'
      } else {
        $PhoneKv = "$pip`:50062"
        $bpt = switch ($Kv) { 'q4_0' { 18432 } 'f16' { 65536 } default { 34816 } }
        $capW = $Ctx + [int](($wiredMax - $pwired) * 1048576 / $bpt / 4096) * 4096
        $capA = $Ctx + [int](($pavail - $appReserve) * 1048576 / $bpt / 4096) * 4096
        $cap = [Math]::Min($capW, $capA)
        $cap = [Math]::Min($cap, 262144)
        if ($CtxTotal -and [int]$CtxTotal -gt $cap) { throw "serve: requested CTX_TOTAL=$CtxTotal exceeds phone safe $Kv cap $cap" }
        if (-not $CtxTotal) { $CtxTotal = "$cap" }
      }
      Write-Warning "serve: phone at $pip : split prefill $(if ($ptail -eq '1') {'on'} else {'off'}); context up to $($CtxTotal ?? $Ctx) tokens (remote KV $(if ($PhoneKv) {'on'} else {'off'}), ANE pages $pane, v$pver)"
    } else {
      Write-Warning 'serve: no phone: Windows alone'
    }
  } catch {
    Write-Warning "serve: no phone: Windows alone ($($_.Exception.Message))"
  }
}
if ($PhoneKv -and -not $PhoneIp) { $PhoneIp = ($PhoneKv -split ',')[0] -split ':' | Select-Object -First 1 }

# --- remote-KV env (same contract as serve.sh) ---
if ($PhoneKv) {
  $env:LLAMA_KV_REMOTE = $PhoneKv
  $env:LLAMA_KV_REMOTE_CTX = if ($CtxTotal) { $CtxTotal } else { '262144' }
  if (-not $env:LLAMA_REMOTE_PIPE) { $env:LLAMA_REMOTE_PIPE = '1' }
  if (-not $env:LLAMA_UBATCH_REMOTE) { $env:LLAMA_UBATCH_REMOTE = '512' }
}
$Total = $Ctx
if ($PhoneKv -and $CtxTotal) { $Total = [int]$CtxTotal }
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'cache') | Out-Null
Set-Content -NoNewline -Path (Join-Path $Root 'cache\server-context') -Value "$Total"

# --- portable spec tunables (kept from serve.sh; Metal/SME ones dropped) ---
$env:SPEC_DRAFT_UBATCH = '64'
if (-not $env:LLAMA_SPEC_SAMPLE) { $env:LLAMA_SPEC_SAMPLE = '1' }
if ($SplitTail) {
  if (-not $env:LLAMA_SPLIT_MIN) { $env:LLAMA_SPLIT_MIN = '512' }
  if (-not $env:LLAMA_SPLIT_ONE_BATCH) { $env:LLAMA_SPLIT_ONE_BATCH = '1' }
  if (-not $env:LLAMA_SPEC_DEPTH_CAP) { $env:LLAMA_SPEC_DEPTH_CAP = '65537:4' }
} else {
  if (-not $env:LLAMA_SPEC_DEPTH_CAP) { $env:LLAMA_SPEC_DEPTH_CAP = '65537:4' }
}
if (-not $env:LLAMA_SPEC_BLOCK) { $env:LLAMA_SPEC_BLOCK = '1' }
if (-not $env:LLAMA_SPEC_Q_CONF) { $env:LLAMA_SPEC_Q_CONF = '1' }
if (-not $env:LLAMA_LAZY_EMBD) { $env:LLAMA_LAZY_EMBD = '1' }

if (-not $CacheDir) { $CacheDir = $env:CACHE_DIR; if (-not $CacheDir) { $CacheDir = Join-Path $Root "cache\$Kv" } }
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$Sport = $Port
if ($Proxy) { $Sport = $Port + 100 }

$extra = @()
$big = $Ctx -gt 65536
if ($big) { $extra += '-lm', 'none', '-ot', 'token_embd.weight=CPU', '-ub', '256' }
if ($env:PHONE_DRAFT) { $extra += '--rpc', $env:PHONE_DRAFT }
if ($SplitTail) { $env:LLAMA_SPLIT_TAIL = $SplitTail; if (-not $big) { $extra += '-ub', ($env:SPLIT_UB ?? '256') } }
if ($ServerArgs) { $extra += $ServerArgs -split '\s+' }

$serverArgs = @('-m', $Model, '-ngl', '999', '-fa', 'on', '-c', "$Ctx", '-np', '1',
  '-ctk', $Kv, '-ctv', $Kv, '-t', '2', '-tb', '2',
  '--spec-type', ($env:SPEC_TYPE ?? 'ngram-simple,draft-dflash'), '-md', $Draft,
  '-ngld', '999', '--spec-draft-n-max', '7', '--spec-gdn-replay', '8',
  '--slot-save-path', "$CacheDir\", '--jinja', '--host', '127.0.0.1', '--port', "$Sport",
  '--cache-ram', ($env:CACHE_RAM ?? '0'), '--ctx-checkpoints', ($env:CTX_CHECKPOINTS ?? '3')) + $extra

PhoneNote 'starting'
Write-Warning "serve: $ServerExe $($serverArgs -join ' ')"
$srv = Start-Process -FilePath $ServerExe -ArgumentList $serverArgs -PassThru -NoNewWindow
$prx = $null
try {
  if ($Proxy) {
    $pargs = @('scripts/proxy.py', '--listen', "$Port", '--upstream', "$Sport", '--cache', $CacheDir)
    if ($PhoneIp) { $pargs += @('--phone', $PhoneIp) }
    $prx = Start-Process -FilePath 'python' -ArgumentList $pargs -PassThru -NoNewWindow
  }
  if ($PhoneIp) {
    # Signal `ready` once /health is 200 (same as the serve.sh background loop).
    $job = Start-Job -ScriptBlock {
      param($p) for ($i = 0; $i -lt 600; $i++) {
        try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$p/health" -TimeoutSec 2).StatusCode -eq 200) { break } } catch {}
        Start-Sleep 1
      }
    } -ArgumentList $Sport
    Wait-Job $job -Timeout 600 | Out-Null
    PhoneNote 'ready'
  }
  Wait-Process -Id $srv.Id
} finally {
  if ($prx -and -not $prx.HasExited) { Stop-Process -Id $prx.Id -Force -ErrorAction SilentlyContinue }
  if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue }
  PhoneNote 'stopped'
}

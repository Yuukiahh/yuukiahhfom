<#
.SYNOPSIS
    Installiert die AI Content Factory unter Windows 11 (idempotent – beliebig oft ausführbar).

.DESCRIPTION
    - prüft/installiert per winget: Python 3.12, Git, FFmpeg (Gyan.FFmpeg), uv, optional Ollama
    - erstellt .venv im content-factory-Ordner und installiert requirements.txt
      (onnxruntime-gpu mit automatischem Rückfall auf onnxruntime/CPU)
    - lädt die Kokoro-TTS-Modelle nach models\
    - klont optional MoneyPrinterTurbo nach ..\MoneyPrinterTurbo und führt "uv sync" aus
    - prüft GPU (nvidia-smi), FFmpeg h264_nvenc, Claude Code CLI und Ollama
    - gibt am Ende eine Zusammenfassung aus

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows\install.ps1
    powershell -ExecutionPolicy Bypass -File install\windows\install.ps1 -WithOllama -WithWhisper
    powershell -ExecutionPolicy Bypass -File install\windows\install.ps1 -WithMoneyPrinterTurbo:$false
#>
[CmdletBinding()]
param(
    # MoneyPrinterTurbo als optionales Render-Backend mitinstallieren (Standard: an)
    [switch]$WithMoneyPrinterTurbo = $true,
    # Ollama (lokales LLM) per winget installieren, falls nicht vorhanden
    [switch]$WithOllama,
    # faster-whisper (requirements-gpu.txt) zusätzlich installieren
    [switch]$WithWhisper,
    # GPU-Paket für onnxruntime überspringen und direkt die CPU-Variante nutzen
    [switch]$CpuOnly
)

# 'Continue' statt 'Stop': native Programme (pip, git, winget) schreiben Warnungen nach stderr,
# was in Windows PowerShell 5.1 sonst als Abbruchfehler gewertet würde. Fehler werden explizit geprüft.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'   # macht Invoke-WebRequest deutlich schneller

# --------------------------------------------------------------------------- Pfade
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path      # ...\content-factory
$Parent  = Split-Path $Root -Parent
$Venv    = Join-Path $Root '.venv'
$VenvPy  = Join-Path $Venv 'Scripts\python.exe'
$Models  = Join-Path $Root 'models'
$MptDir  = Join-Path $Parent 'MoneyPrinterTurbo'

$Summary = [System.Collections.Generic.List[object]]::new()
function Add-Result([string]$Name, [string]$Status, [string]$Detail = '') {
    $Summary.Add([pscustomobject]@{ Komponente = $Name; Status = $Status; Details = $Detail })
}
function Write-Step([string]$Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text)   { Write-Host "    [OK] $Text" -ForegroundColor Green }
function Write-Warn2([string]$Text){ Write-Host "    [!]  $Text" -ForegroundColor Yellow }

# PATH aus der Registry neu laden (nach winget-Installationen nötig)
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

function Test-Command([string]$Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

# winget-Paket installieren, wenn der Befehl noch nicht existiert (idempotent)
function Install-WingetPackage {
    param([string]$Id, [string]$Command, [string]$Label)
    if ($Command -and (Test-Command $Command)) {
        Write-Ok "$Label bereits vorhanden"
        Add-Result $Label 'vorhanden'
        return $true
    }
    if (-not (Test-Command 'winget')) {
        Write-Warn2 "winget fehlt – bitte '$Label' manuell installieren (App Installer aus dem Microsoft Store)."
        Add-Result $Label 'FEHLT' 'winget nicht verfügbar'
        return $false
    }
    Write-Host "    Installiere $Label ($Id) per winget ..."
    & winget install --id $Id -e --silent --accept-source-agreements --accept-package-agreements | Out-Host
    # winget liefert != 0 auch bei "bereits installiert" – danach einfach erneut prüfen
    Update-SessionPath
    if (-not $Command -or (Test-Command $Command)) {
        Write-Ok "$Label installiert"
        Add-Result $Label 'installiert'
        return $true
    }
    Write-Warn2 "$Label wurde installiert, ist aber in dieser Sitzung noch nicht im PATH. Ggf. neues Terminal öffnen."
    Add-Result $Label 'prüfen' 'nicht im PATH – Terminal neu öffnen'
    return $false
}

# Wert aus config/local.toml (Vorrang) bzw. config/default.toml lesen (einfacher Regex, reicht für Strings)
function Get-ConfigValue([string]$Key) {
    foreach ($f in @('config\local.toml', 'config\default.toml')) {
        $p = Join-Path $Root $f
        if (Test-Path $p) {
            $m = Select-String -Path $p -Pattern "^\s*$Key\s*=\s*`"([^`"]+)`"" | Select-Object -First 1
            if ($m) { return $m.Matches[0].Groups[1].Value }
        }
    }
    return $null
}

Write-Host "AI Content Factory – Installation" -ForegroundColor White
Write-Host "Ordner: $Root"

# --------------------------------------------------------------------------- 1. Systempakete
Write-Step 'Systempakete (winget)'
$pyLauncher = Test-Command 'py'
$hasPy312 = $false
if ($pyLauncher) {
    & py -3.12 --version *> $null
    $hasPy312 = ($LASTEXITCODE -eq 0)
}
if ($hasPy312) {
    Write-Ok 'Python 3.12 vorhanden'
    Add-Result 'Python 3.12' 'vorhanden'
} else {
    Install-WingetPackage -Id 'Python.Python.3.12' -Command '' -Label 'Python 3.12' | Out-Null
    Update-SessionPath
}
Install-WingetPackage -Id 'Git.Git'        -Command 'git'    -Label 'Git'    | Out-Null
Install-WingetPackage -Id 'Gyan.FFmpeg'    -Command 'ffmpeg' -Label 'FFmpeg' | Out-Null
Install-WingetPackage -Id 'astral-sh.uv'   -Command 'uv'     -Label 'uv'     | Out-Null
if ($WithOllama) {
    Install-WingetPackage -Id 'Ollama.Ollama' -Command 'ollama' -Label 'Ollama' | Out-Null
}

# --------------------------------------------------------------------------- 2. venv + Python-Pakete
Write-Step 'Python-Umgebung (.venv)'
if (-not (Test-Path $VenvPy)) {
    if (Test-Command 'py') { & py -3.12 -m venv $Venv }
    elseif (Test-Command 'python') { & python -m venv $Venv }
    else { throw 'Python wurde nicht gefunden. Bitte Terminal neu öffnen und Installer erneut starten.' }
    if ($LASTEXITCODE -ne 0) { throw 'venv konnte nicht erstellt werden.' }
    Write-Ok '.venv erstellt'
} else {
    Write-Ok '.venv vorhanden'
}
$pyVer = (& $VenvPy --version) 2>&1
Add-Result '.venv' 'ok' "$pyVer"

& $VenvPy -m pip install --upgrade pip wheel --quiet
$req = Join-Path $Root 'requirements.txt'

# Requirements ohne onnxruntime-gpu installieren, onnxruntime separat behandeln
$tmpReq = Join-Path $env:TEMP 'cf-requirements-base.txt'
Get-Content $req | Where-Object { $_ -notmatch '^\s*onnxruntime' } | Set-Content -Encoding UTF8 $tmpReq
& $VenvPy -m pip install -r $tmpReq
if ($LASTEXITCODE -ne 0) { throw 'pip install -r requirements.txt ist fehlgeschlagen.' }
Write-Ok 'Basis-Pakete installiert'

# onnxruntime: GPU versuchen, sonst CPU (beide gleichzeitig führen zu Konflikten)
$ortStatus = 'CPU'
$hasNvidia = Test-Command 'nvidia-smi'
if ($hasNvidia -and -not $CpuOnly) {
    Write-Host '    Installiere onnxruntime-gpu (CUDA 12 + cuDNN als pip-Pakete) ...'
    & $VenvPy -m pip uninstall -y onnxruntime *> $null
    & $VenvPy -m pip install --upgrade 'onnxruntime-gpu[cuda,cudnn]'
    if ($LASTEXITCODE -ne 0) { & $VenvPy -m pip install --upgrade onnxruntime-gpu }
    $providers = & $VenvPy -c "import onnxruntime as o; getattr(o,'preload_dlls',lambda:None)(); print(','.join(o.get_available_providers()))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $providers -match 'CUDAExecutionProvider') {
        $ortStatus = 'GPU (CUDA)'
        Write-Ok "onnxruntime-gpu aktiv: $providers"
    } else {
        Write-Warn2 'onnxruntime-gpu nicht nutzbar – Rückfall auf onnxruntime (CPU).'
        & $VenvPy -m pip uninstall -y onnxruntime-gpu *> $null
    }
}
if ($ortStatus -eq 'CPU') {
    & $VenvPy -m pip install --upgrade onnxruntime | Out-Host
}
Add-Result 'onnxruntime' 'ok' $ortStatus

if ($WithWhisper) {
    & $VenvPy -m pip install -r (Join-Path $Root 'requirements-gpu.txt') |
        Out-Host
    Add-Result 'faster-whisper' $(if ($LASTEXITCODE -eq 0) { 'installiert' } else { 'FEHLER' })
}

# --------------------------------------------------------------------------- 3. Kokoro-Modelle
Write-Step 'Kokoro-TTS-Modelle'
New-Item -ItemType Directory -Force -Path $Models | Out-Null
$base = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0'
foreach ($file in @('kokoro-v1.0.onnx', 'voices-v1.0.bin')) {
    $dest = Join-Path $Models $file
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 1MB)) {
        Write-Ok "$file vorhanden"
        continue
    }
    Write-Host "    Lade $file ..."
    try {
        Invoke-WebRequest -Uri "$base/$file" -OutFile "$dest.part" -UseBasicParsing
        Move-Item -Force "$dest.part" $dest
        Write-Ok "$file geladen"
    } catch {
        Write-Warn2 "Download fehlgeschlagen: $file ($($_.Exception.Message))"
        Remove-Item -ErrorAction SilentlyContinue "$dest.part"
    }
}
$kok = (Test-Path (Join-Path $Models 'kokoro-v1.0.onnx')) -and (Test-Path (Join-Path $Models 'voices-v1.0.bin'))
Add-Result 'Kokoro-Modelle' $(if ($kok) { 'ok' } else { 'FEHLT' }) $Models

# --------------------------------------------------------------------------- 4. Schrift (optional)
Write-Step 'Schrift Inter (optional)'
$fontDirs = @("$env:WINDIR\Fonts", "$env:LOCALAPPDATA\Microsoft\Windows\Fonts")
$inter = $fontDirs | Where-Object { Test-Path $_ } | ForEach-Object { Get-ChildItem $_ -Filter 'Inter*' -ErrorAction SilentlyContinue } | Select-Object -First 1
if ($inter) {
    Write-Ok "Inter gefunden ($($inter.Name))"
    Add-Result 'Schrift Inter' 'vorhanden'
} else {
    Write-Warn2 'Inter nicht installiert. Optional: https://rsms.me/inter/ herunterladen, "Inter Display" installieren.'
    Write-Warn2 'Ohne Inter nutzen die Untertitel automatisch eine Ersatzschrift.'
    Add-Result 'Schrift Inter' 'optional' 'Fallback-Schrift aktiv'
}

# --------------------------------------------------------------------------- 5. MoneyPrinterTurbo (optional)
if ($WithMoneyPrinterTurbo) {
    Write-Step 'MoneyPrinterTurbo (optionales Backend)'
    if (-not (Test-Command 'git')) {
        Write-Warn2 'Git fehlt – MoneyPrinterTurbo übersprungen.'
        Add-Result 'MoneyPrinterTurbo' 'übersprungen' 'Git fehlt'
    } else {
        if (Test-Path (Join-Path $MptDir '.git')) {
            Write-Host '    Aktualisiere vorhandenen Klon ...'
            & git -C $MptDir pull --ff-only | Out-Host
        } else {
            & git clone --depth 1 https://github.com/harry0703/MoneyPrinterTurbo $MptDir | Out-Host
        }
        $mptStatus = if (Test-Path $MptDir) { 'geklont' } else { 'FEHLER' }
        if ((Test-Path $MptDir) -and (Test-Command 'uv')) {
            Push-Location $MptDir
            try {
                if (Test-Path 'pyproject.toml') {
                    & uv sync | Out-Host
                } elseif (Test-Path 'requirements.txt') {
                    & uv venv --python 3.11 | Out-Host
                    & uv pip install -r requirements.txt | Out-Host
                }
                $mptStatus = if ($LASTEXITCODE -eq 0) { 'installiert (uv)' } else { 'uv-Fehler' }
            } finally { Pop-Location }
        } elseif (Test-Path $MptDir) {
            Write-Warn2 'uv nicht gefunden – Abhängigkeiten von MoneyPrinterTurbo nicht installiert.'
        }
        Add-Result 'MoneyPrinterTurbo' $mptStatus $MptDir
    }
}

# --------------------------------------------------------------------------- 6. Prüfungen
Write-Step 'Systemprüfungen'
if ($hasNvidia) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) 2>$null
    Write-Ok "GPU: $gpu"
    Add-Result 'NVIDIA GPU' 'ok' "$gpu"
} else {
    Write-Warn2 'nvidia-smi nicht gefunden – NVIDIA-Treiber installieren/aktualisieren.'
    Add-Result 'NVIDIA GPU' 'FEHLT' 'Treiber prüfen'
}

if (Test-Command 'ffmpeg') {
    $enc = (& ffmpeg -hide_banner -encoders 2>$null) -join "`n"
    if ($enc -match 'h264_nvenc') {
        # echter Kurztest: kann NVENC tatsächlich kodieren?
        & ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=256x256:d=0.5 -c:v h264_nvenc -f null - 2>$null
        if ($LASTEXITCODE -eq 0) { Write-Ok 'FFmpeg h264_nvenc funktioniert'; Add-Result 'FFmpeg NVENC' 'ok' }
        else { Write-Warn2 'h264_nvenc vorhanden, Test fehlgeschlagen (Treiber?) – Fallback libx264.'; Add-Result 'FFmpeg NVENC' 'Fallback' 'libx264' }
    } else {
        Write-Warn2 'FFmpeg ohne h264_nvenc – Fallback libx264.'
        Add-Result 'FFmpeg NVENC' 'Fallback' 'libx264'
    }
} else {
    Add-Result 'FFmpeg NVENC' 'FEHLT' 'ffmpeg nicht im PATH'
}

if (Test-Command 'claude') {
    $cv = (& claude --version) 2>$null
    Write-Ok "Claude Code CLI: $cv"
    Add-Result 'Claude Code CLI' 'ok' "$cv"
} else {
    Write-Warn2 'Claude Code CLI nicht gefunden (optional, kostenloses LLM über dein Abo): https://claude.com/claude-code'
    Add-Result 'Claude Code CLI' 'fehlt' 'optional'
}

if (Test-Command 'ollama') {
    $ov = (& ollama --version) 2>$null
    $model = Get-ConfigValue 'ollama_model'
    Add-Result 'Ollama' 'ok' "$ov"
    if ($model) {
        $have = ((& ollama list) 2>$null) -join "`n"
        if ($have -match [regex]::Escape($model)) {
            Write-Ok "Ollama-Modell $model vorhanden"
            Add-Result "Ollama-Modell" 'ok' $model
        } else {
            # Große Downloads (mehrere GB) nicht automatisch starten – nur Befehl anzeigen
            Write-Warn2 "Ollama-Modell '$model' fehlt. Laden mit:  ollama pull $model   (mehrere GB)"
            Add-Result "Ollama-Modell" 'fehlt' "ollama pull $model"
        }
    }
} else {
    Add-Result 'Ollama' 'fehlt' 'optional: -WithOllama'
}

# Import-Test der wichtigsten Module
& $VenvPy -c "import numpy, PIL, soundfile, requests, fastapi, keyring, cryptography, kokoro_onnx, edge_tts; print('ok')" *> $null
Add-Result 'Python-Importe' $(if ($LASTEXITCODE -eq 0) { 'ok' } else { 'FEHLER' })

# --------------------------------------------------------------------------- 7. Zusammenfassung
Write-Step 'Zusammenfassung'
$Summary | Format-Table -AutoSize | Out-Host
Write-Host @"
Nächste Schritte:
  1. Dashboard starten:   install\windows\start-dashboard.ps1
  2. Worker starten:      install\windows\start-worker.ps1
  3. Autostart (optional): install\windows\register-autostart.ps1
  4. API-Schlüssel sicher hinterlegen:  .venv\Scripts\python -m factory secrets set <name>
  Veröffentlichung bleibt AUS, bis du sie ausdrücklich aktivierst.
"@

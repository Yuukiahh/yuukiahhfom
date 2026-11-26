<#
.SYNOPSIS  Starts Docker Desktop, Ollama and the SupoClip stack (after a reboot, for example).
#>
param([string]$InstallRoot = (Join-Path $env:USERPROFILE 'AutoClipper'))
$ErrorActionPreference = 'Continue'   # see Install-AutoClipper.ps1: native stderr must not throw in PS 5.1

function Up([string]$url) {
    try { (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5).StatusCode -lt 500 } catch { $false }
}

docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Start-Process (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe')
    $deadline = (Get-Date).AddMinutes(5)
    do { Start-Sleep 5; docker info 2>$null | Out-Null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
}
if (-not (Up 'http://127.0.0.1:11434/api/version')) {
    Start-Process (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama app.exe')
}
Push-Location (Join-Path $InstallRoot 'supoclip')
docker compose up -d
Pop-Location
$deadline = (Get-Date).AddMinutes(5)
while (-not (Up 'http://127.0.0.1:8000/health') -and (Get-Date) -lt $deadline) { Start-Sleep 5 }
autoclipper doctor

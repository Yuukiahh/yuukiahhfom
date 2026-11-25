<#
.SYNOPSIS
    Startet den Worker der AI Content Factory (python -m factory worker).

.PARAMETER Log
    Ausgabe zusätzlich nach data\logs\worker-<Datum>.log schreiben
    (wird vom Autostart-Task verwendet, da dort kein Fenster sichtbar ist).
#>
param(
    [switch]$Log
)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

$Activate = Join-Path $Root '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $Activate)) {
    Write-Host 'Keine .venv gefunden – bitte zuerst install\windows\install.ps1 ausführen.' -ForegroundColor Red
    exit 1
}
. $Activate
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

if ($Log) {
    $LogDir = Join-Path $Root 'data\logs'
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $LogFile = Join-Path $LogDir ("worker-{0:yyyy-MM-dd}.log" -f (Get-Date))
    Add-Content -Path $LogFile -Value ("`n===== Worker-Start {0:yyyy-MM-dd HH:mm:ss} =====" -f (Get-Date))
    # cmd /c leitet stdout+stderr des Python-Prozesses unverändert in die Logdatei um
    cmd /c "python -m factory worker >> `"$LogFile`" 2>&1"
} else {
    Write-Host 'Starte Worker ... (Beenden mit Strg+C)' -ForegroundColor Cyan
    python -m factory worker
}
# Exit-Code weitergeben, damit die Aufgabenplanung bei Fehlern neu startet
exit $LASTEXITCODE

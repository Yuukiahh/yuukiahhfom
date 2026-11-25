<#
.SYNOPSIS
    Startet das Dashboard der AI Content Factory (python -m factory dashboard).
#>
param(
    # zusätzliche Argumente werden an "python -m factory dashboard" durchgereicht
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
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
Write-Host 'Starte Dashboard ... (Beenden mit Strg+C)' -ForegroundColor Cyan
python -m factory dashboard @Rest
exit $LASTEXITCODE

<#
.SYNOPSIS
    Entfernt die geplante Aufgabe "AI Content Factory Worker" (idempotent).
#>
$ErrorActionPreference = 'Stop'
$TaskName = 'AI Content Factory Worker'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Host "Aufgabe '$TaskName' ist nicht registriert – nichts zu tun."
    exit 0
}
if ($task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $TaskName
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Aufgabe '$TaskName' entfernt." -ForegroundColor Green

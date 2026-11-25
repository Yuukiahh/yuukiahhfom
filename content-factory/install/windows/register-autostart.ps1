<#
.SYNOPSIS
    Registriert den Worker als geplante Aufgabe "AI Content Factory Worker" (Start bei Anmeldung).

.DESCRIPTION
    - Trigger: Anmeldung des aktuellen Benutzers
    - läuft unsichtbar (PowerShell mit -WindowStyle Hidden), Ausgabe in data\logs\
    - Neustart bei Fehler: 3x im Abstand von 5 Minuten
    - läuft auch im Akkubetrieb (Netzbetrieb NICHT erforderlich)
    - keine Zeitbegrenzung, keine parallelen Instanzen
    Die Aufgabe läuft im Kontext des angemeldeten Benutzers (Interactive), damit der
    Worker Zugriff auf die Windows-Anmeldeinformationsverwaltung (API-Schlüssel) und die GPU hat.
    Erneutes Ausführen aktualisiert die vorhandene Aufgabe (idempotent).
#>
$ErrorActionPreference = 'Stop'
$TaskName = 'AI Content Factory Worker'
$Worker   = Join-Path $PSScriptRoot 'start-worker.ps1'
$Root     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if (-not (Test-Path $Worker)) { throw "start-worker.ps1 nicht gefunden: $Worker" }

$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$Action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Worker`" -Log" `
    -WorkingDirectory $Root

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
# kurze Verzögerung, damit Netzwerk/Treiber nach der Anmeldung bereit sind
$Trigger.Delay = 'PT1M'

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Principal $Principal -Description 'AI Content Factory – Hintergrund-Worker (Videoproduktion). Veröffentlichung nur nach Freigabe.' `
    -Force | Out-Null

Write-Host "Aufgabe '$TaskName' registriert (Start bei Anmeldung von $User)." -ForegroundColor Green
Write-Host "Sofort starten:   Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs:             $Root\data\logs\"
Write-Host "Entfernen:        install\windows\unregister-autostart.ps1"

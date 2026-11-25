@echo off
chcp 65001 >nul
title AI Content Factory - Video erstellen
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Bitte zuerst install\windows\install.ps1 ausfuehren.
  pause & exit /b 1
)
set /p THEMA=Thema fuer das Video: 
set /p FORMAT=Format [short = 25-45 s / long = 60-90 s] (Enter = short): 
if "%FORMAT%"=="" set FORMAT=short
".venv\Scripts\python.exe" -m factory video "%THEMA%" --format %FORMAT%
echo.
echo Fertig. Das Video liegt im Ordner output\
explorer output
pause

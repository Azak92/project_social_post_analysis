@echo off
powershell.exe -NoProfile -File "%~dp0start-lab.ps1"
if errorlevel 1 (
  echo The lab could not start. Please keep this message for troubleshooting.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8000/"

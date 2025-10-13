@echo off
REM Run the PowerShell image renamer
echo Running image renamer...
powershell -ExecutionPolicy Bypass -File rename_images.ps1
pause

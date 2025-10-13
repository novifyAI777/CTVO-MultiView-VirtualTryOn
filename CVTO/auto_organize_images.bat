@echo off
REM Auto-Organize Images Script
REM This script automatically organizes renamed images into the correct directories

echo ========================================
echo CTVO Auto-Image Organizer
echo ========================================

REM Check if temp_organization folder exists
if not exist "temp_organization" (
    echo ERROR: temp_organization folder not found!
    echo Please run organize_images_helper.bat first
    pause
    exit /b 1
)

REM Check if there are any images
dir temp_organization\*.jpg >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: No JPG images found in temp_organization folder!
    echo Please copy your images there first
    pause
    exit /b 1
)

echo Found images in temp_organization folder
echo.

REM Create the auto-organize PowerShell script
echo Creating auto-organize script...
(
echo $images = Get-ChildItem "temp_organization\*.jpg" ^| Sort-Object Name
echo Write-Host "Processing $($images.Count) images..."
echo.
echo for ($i=0; $i -lt $images.Count; $i++) {
echo     $person = [math]::Floor($i/8) + 1
echo     $view = ($i %% 8) + 1
echo     $personDir = "person_{0:D3}" -f $person
echo     $viewDir = "view_{0:D2}" -f $view
echo     $targetDir = "data\multiview_dataset\persons\$personDir\$viewDir"
echo     
echo     if (Test-Path $targetDir) {
echo         $targetFile = Join-Path $targetDir $images[$i].Name
echo         Copy-Item $images[$i].FullName $targetFile
echo         Write-Host "Copied: $($images[$i].Name) → $personDir\$viewDir"
echo     } else {
echo         Write-Host "ERROR: Directory not found: $targetDir"
echo     }
echo }
echo.
echo Write-Host "Organization complete!"
echo Write-Host "You can now delete the temp_organization folder"
) > auto_organize.ps1

echo Running auto-organize script...
powershell -ExecutionPolicy Bypass -File auto_organize.ps1

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo ✅ SUCCESS: Images organized!
    echo ========================================
    echo.
    echo Your images are now in the correct directories:
    echo data\multiview_dataset\persons\person_001\view_01\
    echo data\multiview_dataset\persons\person_001\view_02\
    echo ... and so on for all 12 people
    echo.
    echo You can now delete the temp_organization folder
    echo and run: test_setup.bat
) else (
    echo.
    echo ========================================
    echo ❌ ERROR: Organization failed!
    echo ========================================
    echo Check the error messages above
)

echo.
pause

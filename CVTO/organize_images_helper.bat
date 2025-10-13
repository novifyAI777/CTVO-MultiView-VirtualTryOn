@echo off
REM Quick Image Organization Helper
REM This script helps you organize your 96 images (12 people x 8 views) efficiently

echo ========================================
echo CTVO Image Organization Helper
echo ========================================
echo.
echo You have 12 people with 8 views each = 96 images total
echo.

REM Create a temporary organization folder
if not exist "temp_organization" mkdir temp_organization

echo 📁 Created temp_organization folder
echo.
echo 📋 ORGANIZATION INSTRUCTIONS:
echo.
echo 1. Copy ALL your 96 images to: temp_organization\
echo 2. Rename them systematically:
echo    - Person 1: person001_view01.jpg, person001_view02.jpg, ... person001_view08.jpg
echo    - Person 2: person002_view01.jpg, person002_view02.jpg, ... person002_view08.jpg
echo    - Person 3: person003_view01.jpg, person003_view03.jpg, ... person003_view08.jpg
echo    - ... and so on for all 12 people
echo.
echo 3. Run the auto-organize script after renaming
echo.

echo 🚀 QUICK RENAME HELPER:
echo.
echo If your images are named like: IMG_001.jpg, IMG_002.jpg, etc.
echo Use this PowerShell command to rename them:
echo.
echo $images = Get-ChildItem "temp_organization\*.jpg" ^| Sort-Object Name
echo for ($i=0; $i -lt $images.Count; $i++) {
echo     $person = [math]::Floor($i/8) + 1
echo     $view = ($i %% 8) + 1
echo     $newName = "person{0:D3}_view{1:D2}.jpg" -f $person, $view
echo     Rename-Item $images[$i].FullName $newName
echo }
echo.

echo 📝 MANUAL ORGANIZATION:
echo.
echo For each person, copy their 8 images to the correct folders:
echo.
echo Person 001:
echo   Copy image 1 → data\multiview_dataset\persons\person_001\view_01\
echo   Copy image 2 → data\multiview_dataset\persons\person_001\view_02\
echo   Copy image 3 → data\multiview_dataset\persons\person_001\view_03\
echo   Copy image 4 → data\multiview_dataset\persons\person_001\view_04\
echo   Copy image 5 → data\multiview_dataset\persons\person_001\view_05\
echo   Copy image 6 → data\multiview_dataset\persons\person_001\view_06\
echo   Copy image 7 → data\multiview_dataset\persons\person_001\view_07\
echo   Copy image 8 → data\multiview_dataset\persons\person_001\view_08\
echo.
echo Repeat for person_002 through person_012
echo.

echo ⚡ AUTO-ORGANIZE (if you renamed files):
echo.
echo After renaming your files, run:
echo auto_organize_images.bat
echo.

pause

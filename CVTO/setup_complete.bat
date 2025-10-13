@echo off
REM CTVO Complete Setup Summary
REM This script shows you the complete setup and next steps

echo ========================================
echo CTVO Multi-View Dataset Setup Complete!
echo ========================================
echo.
echo 📊 DATASET SUMMARY:
echo   • 12 people total
echo   • 8 views per person
echo   • 96 images total (12 × 8)
echo   • 5 cloth images
echo   • Total combinations: 96 × 5 = 480 try-on combinations
echo.

echo 📁 DIRECTORY STRUCTURE CREATED:
echo   data\multiview_dataset\persons\
echo   ├── person_001\ (view_01 to view_08)
echo   ├── person_002\ (view_01 to view_08)
echo   ├── person_003\ (view_01 to view_08)
echo   ├── person_004\ (view_01 to view_08)
echo   ├── person_005\ (view_01 to view_08)
echo   ├── person_006\ (view_01 to view_08)
echo   ├── person_007\ (view_01 to view_08)
echo   ├── person_008\ (view_01 to view_08)
echo   ├── person_009\ (view_01 to view_08)
echo   ├── person_010\ (view_01 to view_08)
echo   ├── person_011\ (view_01 to view_08)
echo   └── person_012\ (view_01 to view_08)
echo.

echo 🚀 QUICK ORGANIZATION OPTIONS:
echo.
echo OPTION 1 - AUTOMATIC (Recommended):
echo   1. Create folder: temp_organization
echo   2. Copy ALL 96 images to temp_organization\
echo   3. Run: rename_images.bat
echo   4. Run: auto_organize_images.bat
echo   5. Delete temp_organization folder
echo.
echo OPTION 2 - MANUAL:
echo   1. Copy images one by one to correct directories
echo   2. Use organize_images_helper.bat for guidance
echo.

echo 📋 NEXT STEPS AFTER ORGANIZING IMAGES:
echo.
echo 1. Add cloth images to: data\multiview_dataset\clothes\
echo    - cloth_01.jpg
echo    - cloth_02.jpg
echo    - cloth_03.jpg
echo    - cloth_04.jpg
echo    - cloth_05.jpg
echo.
echo 2. Test setup: test_setup.bat
echo.
echo 3. Run pipeline: run_multiview_batch.bat
echo.

echo 🎯 EXPECTED RESULTS:
echo   • Stage 1: Human parsing + pose for all 96 images
echo   • Stage 2: Warped cloth for all 480 combinations
echo   • Stage 3: Fusion generation with multi-view training
echo   • Stage 4: NeRF multi-view rendering
echo.

echo ⚡ QUICK START COMMANDS:
echo   organize_images_helper.bat    ← Get organization help
echo   rename_images.bat            ← Auto-rename images
echo   auto_organize_images.bat     ← Auto-organize images
echo   test_setup.bat               ← Test your setup
echo   run_multiview_batch.bat      ← Run complete pipeline
echo.

echo ========================================
echo 🎉 READY TO ORGANIZE YOUR 96 IMAGES!
echo ========================================
echo.

pause

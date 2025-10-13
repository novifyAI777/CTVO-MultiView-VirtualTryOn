@echo off
REM Directory Structure Verification Script
REM This script shows you exactly where your view folders are

echo ========================================
echo CTVO Directory Structure Verification
echo ========================================
echo.

echo 📁 Current working directory:
cd
echo.

echo 📁 Checking data\multiview_dataset\persons\ directory:
if exist "data\multiview_dataset\persons" (
    echo ✅ persons directory exists
    echo.
    echo 📋 Person directories found:
    dir data\multiview_dataset\persons /b
    echo.
    
    echo 📋 Checking view folders in person_001:
    if exist "data\multiview_dataset\persons\person_001" (
        echo ✅ person_001 directory exists
        echo.
        echo View folders in person_001:
        dir data\multiview_dataset\persons\person_001 /b
        echo.
        
        echo 📋 Checking view folders in person_012:
        if exist "data\multiview_dataset\persons\person_012" (
            echo ✅ person_012 directory exists
            echo.
            echo View folders in person_012:
            dir data\multiview_dataset\persons\person_012 /b
            echo.
            
            echo ========================================
            echo ✅ ALL DIRECTORIES EXIST!
            echo ========================================
            echo.
            echo Your complete structure:
            echo data\multiview_dataset\persons\
            echo ├── person_001\ (view_01 to view_08)
            echo ├── person_002\ (view_01 to view_08)
            echo ├── person_003\ (view_01 to view_08)
            echo ├── person_004\ (view_01 to view_08)
            echo ├── person_005\ (view_01 to view_08)
            echo ├── person_006\ (view_01 to view_08)
            echo ├── person_007\ (view_01 to view_08)
            echo ├── person_008\ (view_01 to view_08)
            echo ├── person_009\ (view_01 to view_08)
            echo ├── person_010\ (view_01 to view_08)
            echo ├── person_011\ (view_01 to view_08)
            echo └── person_012\ (view_01 to view_08)
            echo.
            echo 🎯 READY TO ORGANIZE YOUR 96 IMAGES!
            echo.
        ) else (
            echo ❌ person_012 directory not found
        )
    ) else (
        echo ❌ person_001 directory not found
    )
) else (
    echo ❌ persons directory not found
    echo Please run the setup script first
)

echo.
echo 💡 TIP: If you can't see the folders in Windows Explorer:
echo 1. Make sure you're in the right directory: C:\Users\DELL\OneDrive\Desktop\CVTO
echo 2. Navigate to: data\multiview_dataset\persons\
echo 3. Double-click on any person_xxx folder to see the view_xx folders
echo.

pause

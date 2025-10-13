@echo off
REM CTVO Setup Test Script
REM This script tests if everything is set up correctly

echo ========================================
echo CTVO Setup Test
echo ========================================

REM Test Python
echo Testing Python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found
    python --version
) else (
    echo ❌ Python not found
    echo Please install Python or add it to PATH
    goto :end
)

echo.

REM Test dataset structure
echo Testing dataset structure...
if exist "data\multiview_dataset\persons" (
    echo ✅ Dataset directory exists
) else (
    echo ❌ Dataset directory not found
    echo Please run: mkdir data\multiview_dataset\persons
    goto :end
)

echo.

REM Test Python imports
echo Testing Python imports...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ PyTorch not installed
    echo Please run: pip install torch torchvision
    goto :end
)

python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ OpenCV not installed
    echo Please run: pip install opencv-python
    goto :end
)

python -c "import PIL; print('✅ Pillow:', PIL.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Pillow not installed
    echo Please run: pip install Pillow
    goto :end
)

echo.

REM Test project structure
echo Testing project structure...
if exist "ctvo_core" (
    echo ✅ ctvo_core package found
) else (
    echo ❌ ctvo_core package not found
    goto :end
)

if exist "scripts\run_stage1.py" (
    echo ✅ Stage 1 script found
) else (
    echo ❌ Stage 1 script not found
    goto :end
)

if exist "scripts\run_stage2.py" (
    echo ✅ Stage 2 script found
) else (
    echo ❌ Stage 2 script not found
    goto :end
)

echo.

REM Test pretrained models
echo Testing pretrained models...
if exist "ctvo_core\stage2_cloth_warping\pretrained_weights\unet_wrap.pth" (
    echo ✅ Stage 2 model found
) else (
    echo ⚠️  Stage 2 model not found (will use random weights)
)

echo.

echo ========================================
echo ✅ ALL TESTS PASSED!
echo ========================================
echo Your CTVO setup is ready!
echo.
echo Next steps:
echo 1. Organize your 80 images in data\multiview_dataset\persons\
echo 2. Add cloth images to data\multiview_dataset\clothes\
echo 3. Run: run_multiview_batch.bat
echo.

:end
pause


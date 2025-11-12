@echo off
REM Batch script to run Stage 1 processing on all images

echo ============================================================
echo STAGE 1 BATCH PROCESSING
echo ============================================================
echo.

REM Check if models exist
if not exist "ctvo_core\stage1_parsing_pose\pretrained_models\body_pose_model.pth" (
    echo [ERROR] Pose model not found!
    echo Please download it first by running: scripts\download_stage1_models.bat
    pause
    exit /b 1
)

if not exist "ctvo_core\stage1_parsing_pose\pretrained_models\parsing_lip.onnx" (
    echo [WARNING] Parsing model not found!
    echo Please download it manually. See STAGE1_SETUP.md for instructions.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
)

echo.
echo Processing all images in: data\multiview_dataset\images\train
echo Output will be saved to: data\multiview_dataset\stage1_outputs
echo.

REM Activate virtual environment if it exists
if exist "ctvo_env\Scripts\activate.bat" (
    call ctvo_env\Scripts\activate.bat
)

REM Run the batch processing script
python scripts\run_stage1_batch.py --input_dir "data\multiview_dataset\images\train" --output_dir "data\multiview_dataset\stage1_outputs"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Processing completed successfully!
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo Processing failed. Check the error messages above.
    echo ============================================================
)

pause


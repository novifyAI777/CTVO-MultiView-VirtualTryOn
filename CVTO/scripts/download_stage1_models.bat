@echo off
REM Download Stage 1 Pretrained Models for Windows
REM This script downloads the pose estimation model and provides instructions for parsing model

echo ============================================================
echo STAGE 1 PRETRAINED MODELS DOWNLOADER
echo ============================================================
echo.

REM Create models directory if it doesn't exist
if not exist "ctvo_core\stage1_parsing_pose\pretrained_models" (
    mkdir "ctvo_core\stage1_parsing_pose\pretrained_models"
    echo Created models directory
)

echo [1/2] Downloading Pose Estimation Model...
echo.

REM Download pose model
set POSE_URL=https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
set POSE_OUTPUT=ctvo_core\stage1_parsing_pose\pretrained_models\body_pose_model.pth

if exist "%POSE_OUTPUT%" (
    echo Pose model already exists: %POSE_OUTPUT%
) else (
    echo Downloading from: %POSE_URL%
    echo Saving to: %POSE_OUTPUT%
    echo.
    powershell -Command "Invoke-WebRequest -Uri '%POSE_URL%' -OutFile '%POSE_OUTPUT%'"
    if exist "%POSE_OUTPUT%" (
        echo [OK] Pose model downloaded successfully!
    ) else (
        echo [ERROR] Failed to download pose model
    )
)

echo.
echo ============================================================
echo [2/2] Human Parsing Model
echo ============================================================
echo.
echo The human parsing model needs to be downloaded manually.
echo.
echo Recommended sources:
echo   1. Self-Correction-Human-Parsing: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
echo   2. ONNX Model Zoo: https://github.com/onnx/models
echo   3. HuggingFace: https://huggingface.co/models (search for "human parsing")
echo.
echo Save the ONNX model as:
echo   ctvo_core\stage1_parsing_pose\pretrained_models\parsing_lip.onnx
echo.
echo See STAGE1_SETUP.md for detailed instructions.
echo.

REM Check if parsing model exists
set PARSING_OUTPUT=ctvo_core\stage1_parsing_pose\pretrained_models\parsing_lip.onnx
if exist "%PARSING_OUTPUT%" (
    echo [OK] Parsing model found: %PARSING_OUTPUT%
) else (
    echo [WARNING] Parsing model not found. Please download manually.
)

echo.
echo ============================================================
echo Download Summary
echo ============================================================
echo.

if exist "%POSE_OUTPUT%" (
    echo [OK] Pose model: %POSE_OUTPUT%
) else (
    echo [MISSING] Pose model: %POSE_OUTPUT%
)

if exist "%PARSING_OUTPUT%" (
    echo [OK] Parsing model: %PARSING_OUTPUT%
) else (
    echo [MISSING] Parsing model: %PARSING_OUTPUT%
)

echo.
echo ============================================================
pause


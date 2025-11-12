@echo off
REM Stage 2 Batch Processing Script
REM This script runs Stage 2 (Cloth Warping) on all person-cloth combinations

echo ========================================
echo STAGE 2 BATCH PROCESSOR
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "ctvo_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ctvo_env\Scripts\activate.bat
)

REM Change to CVTO directory
cd CVTO

REM Run Stage 2 batch processing
echo Running Stage 2 batch processing...
echo.

python scripts\run_stage2_batch.py ^
    --person_images_dir "data\multiview_dataset\images\train" ^
    --cloth_images_dir "data\multiview_dataset\clothes" ^
    --stage1_parsing_dir "data\multiview_dataset\stage1_outputs\parsing_maps" ^
    --stage1_pose_dir "data\multiview_dataset\stage1_outputs\pose_heatmaps" ^
    --output_dir "data\multiview_dataset\stage2_outputs" ^
    --model_checkpoint "ctvo_core\stage2_cloth_warping\pretrained_weights\unet_wrap.pth" ^
    --model_type unet ^
    --device cpu ^
    --skip_existing

echo.
echo ========================================
echo Stage 2 batch processing completed!
echo ========================================
echo.
echo Check results in: data\multiview_dataset\stage2_outputs
echo.

pause


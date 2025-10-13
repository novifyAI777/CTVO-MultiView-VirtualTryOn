@echo off
REM CTVO Multi-View Dataset Processing Batch File
REM This file helps you run the CTVO pipeline with proper Python paths

echo ========================================
echo CTVO Multi-View Dataset Processor
echo ========================================

REM Try different Python commands
set PYTHON_CMD=

REM Try python first
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    goto :found_python
)

REM Try python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python3
    goto :found_python
)

REM Try py (Windows Python launcher)
py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    goto :found_python
)

REM Try common Python installations
if exist "C:\Python39\python.exe" (
    set PYTHON_CMD=C:\Python39\python.exe
    goto :found_python
)

if exist "C:\Python38\python.exe" (
    set PYTHON_CMD=C:\Python38\python.exe
    goto :found_python
)

if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
    goto :found_python
)

REM Try Anaconda
if exist "C:\Users\%USERNAME%\Anaconda3\python.exe" (
    set PYTHON_CMD=C:\Users\%USERNAME%\Anaconda3\python.exe
    goto :found_python
)

if exist "C:\Users\%USERNAME%\Miniconda3\python.exe" (
    set PYTHON_CMD=C:\Users\%USERNAME%\Miniconda3\python.exe
    goto :found_python
)

echo ERROR: Python not found!
echo Please install Python or add it to your PATH
echo Download from: https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_CMD%
echo.

REM Check if dataset is organized
if not exist "data\multiview_dataset\persons" (
    echo ERROR: Dataset not organized!
    echo Please organize your 80 images first.
    echo See: DATASET_SETUP_INSTRUCTIONS.md
    pause
    exit /b 1
)

echo Dataset structure found. Starting processing...
echo.

REM Run the batch processor
echo Running multi-view batch processor...
%PYTHON_CMD% scripts\process_multiview_batch.py --dataset_dir "data\multiview_dataset" --output_dir "results\multiview_batch"

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo SUCCESS: Batch processing completed!
    echo ========================================
    echo Check results in: results\multiview_batch
) else (
    echo.
    echo ========================================
    echo ERROR: Batch processing failed!
    echo ========================================
    echo Check the error messages above.
)

echo.
pause


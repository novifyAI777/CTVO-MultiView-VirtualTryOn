#!/bin/bash
# Training Script for All Stages
# This script trains all stages of the CTVO pipeline sequentially

set -e  # Exit on any error

echo "Starting CTVO Pipeline Training"
echo "==============================="

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
RESULTS_DIR="$BASE_DIR/results"

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

# Function to run a stage
run_stage() {
    local stage=$1
    local script=$2
    local config=$3
    
    echo ""
    echo "Starting Stage $stage Training"
    echo "-------------------------------"
    
    if [ -f "$script" ]; then
        python "$script" --config "$config" --mode train
        if [ $? -eq 0 ]; then
            echo "Stage $stage training completed successfully"
        else
            echo "Stage $stage training failed"
            exit 1
        fi
    else
        echo "Script not found: $script"
        exit 1
    fi
}

# Check if we're in the right directory
if [ ! -f "scripts/run_stage1.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Stage 1: Human Parsing & Pose Estimation
echo "Note: Stage 1 uses pre-trained models, no training required"

# Stage 2: Cloth Warping
echo "Note: Stage 2 uses pre-trained models, no training required"

# Stage 3: Fusion Generation
run_stage "3" "scripts/run_stage3.py" "configs/stage3_fusion.yaml"

# Stage 4: NeRF Multi-view Generation
run_stage "4" "scripts/run_stage4.py" "configs/stage4_nerf.yaml"

echo ""
echo "All Stages Training Completed Successfully!"
echo "==========================================="
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "You can now run evaluation or inference using the individual stage scripts."

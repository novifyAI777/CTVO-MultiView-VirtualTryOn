#!/usr/bin/env python3
"""
Download Stage 1 Pretrained Models

This script downloads the required pretrained models for Stage 1:
1. Human Parsing Model (LIP/ATR) - ONNX format
2. Pose Estimation Model (MobileNet OpenPose) - PyTorch format
"""

import os
import sys
import urllib.request
from pathlib import Path
import argparse

# Project root
project_root = Path(__file__).parent.parent
models_dir = project_root / "ctvo_core" / "stage1_parsing_pose" / "pretrained_models"
models_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, description: str):
    """Download a file with progress indication"""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
        print("\n✓ Download completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {str(e)}")
        return False


def download_pose_model():
    """Download pose estimation model"""
    url = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"
    output_path = models_dir / "body_pose_model.pth"
    
    if output_path.exists():
        print(f"✓ Pose model already exists: {output_path}")
        return True
    
    return download_file(url, output_path, "Pose Estimation Model (MobileNet OpenPose)")


def download_parsing_model():
    """Download human parsing model - provides multiple options"""
    print("\n" + "="*60)
    print("HUMAN PARSING MODEL DOWNLOAD")
    print("="*60)
    print("\nThe human parsing model (LIP/ATR) needs to be downloaded manually.")
    print("Here are the recommended sources:\n")
    print("Option 1: Self-Correction-Human-Parsing (Recommended)")
    print("  - Repository: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing")
    print("  - Convert the PyTorch model to ONNX format")
    print("  - Or download pre-converted ONNX from model repositories\n")
    print("Option 2: ONNX Model Zoo")
    print("  - Check: https://github.com/onnx/models")
    print("  - Search for 'human parsing' or 'semantic segmentation'\n")
    print("Option 3: Direct Download Links (if available):")
    print("  - LIP model: Check HuggingFace or other model hubs")
    print("  - ATR model: Check original paper repositories\n")
    print("="*60)
    print("\nFor now, you can:")
    print("1. Download from Self-Correction-Human-Parsing repo")
    print("2. Convert PyTorch model to ONNX using:")
    print("   python -c \"import torch; model = torch.load('model.pth'); ...\"")
    print("3. Place the ONNX file at:")
    print(f"   {models_dir / 'parsing_lip.onnx'}\n")
    
    # Check if file exists
    output_path = models_dir / "parsing_lip.onnx"
    if output_path.exists():
        print(f"✓ Parsing model already exists: {output_path}")
        return True
    
    # Try to download from a common source (if available)
    # Note: These URLs may need to be updated with actual working links
    alternative_urls = [
        # Add known working URLs here if available
    ]
    
    for url in alternative_urls:
        if download_file(url, output_path, "Human Parsing Model (LIP)"):
            return True
    
    print("\n⚠ Please download the parsing model manually and place it at:")
    print(f"   {output_path}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Stage 1 pretrained models")
    parser.add_argument("--pose-only", action="store_true", 
                       help="Download only pose estimation model")
    parser.add_argument("--parsing-only", action="store_true",
                       help="Download only parsing model (shows instructions)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STAGE 1 PRETRAINED MODELS DOWNLOADER")
    print("="*60)
    
    success = True
    
    if not args.parsing_only:
        print("\n[1/2] Downloading Pose Estimation Model...")
        if not download_pose_model():
            success = False
    
    if not args.pose_only:
        print("\n[2/2] Downloading Human Parsing Model...")
        if not download_parsing_model():
            success = False
    
    print("\n" + "="*60)
    if success:
        print("✓ All models downloaded successfully!")
        print(f"\nModels are located at: {models_dir}")
    else:
        print("⚠ Some models may need to be downloaded manually.")
        print("Please check the instructions above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


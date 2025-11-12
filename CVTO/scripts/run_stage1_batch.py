#!/usr/bin/env python3
"""
Stage 1 Batch Processing Script

This script processes all images in your dataset through Stage 1
(Human Parsing & Pose Estimation) and saves the results.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ctvo_core.stage1_parsing_pose import Stage1Processor


def find_all_images(base_dir: Path, extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[Path]:
    """Find all images in directory tree"""
    images = []
    for ext in extensions:
        images.extend(base_dir.rglob(f'*{ext}'))
        images.extend(base_dir.rglob(f'*{ext.upper()}'))
    return sorted(images)


def process_image(processor: Stage1Processor, image_path: Path, 
                  output_base: Path, relative_path: Path) -> Dict:
    """Process a single image through Stage 1"""
    # Create output directory structure matching input structure
    output_dir = output_base / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    image_name = image_path.stem
    parsing_output = output_dir / f"{image_name}_parsing.png"
    pose_output = output_dir / f"{image_name}_pose.json"
    
    try:
        # Process image
        parsing_result, pose_result = processor.process_image(
            str(image_path),
            parsing_output_path=str(parsing_output),
            pose_output_path=str(pose_output)
        )
        
        return {
            "status": "success",
            "input": str(image_path),
            "parsing_output": str(parsing_output),
            "pose_output": str(pose_output),
            "parsing_shape": list(parsing_result.shape) if hasattr(parsing_result, 'shape') else None,
            "num_keypoints": len(pose_result.get('people', [{}])[0].get('pose_keypoints_2d', [])) if pose_result.get('people') else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "input": str(image_path),
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Batch process images through Stage 1")
    parser.add_argument("--input_dir", type=str, 
                       default="data/multiview_dataset/images/train",
                       help="Input directory containing images (Men/Women/Tier structure)")
    parser.add_argument("--output_dir", type=str,
                       default="data/multiview_dataset/stage1_outputs",
                       help="Output directory for Stage 1 results")
    parser.add_argument("--parsing_model", type=str,
                       default="ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx",
                       help="Path to parsing model")
    parser.add_argument("--pose_model", type=str,
                       default="ctvo_core/stage1_parsing_pose/pretrained_models/body_pose_model.pth",
                       help="Path to pose model")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--gender", type=str, default="all",
                       choices=["all", "Men", "Women"],
                       help="Process specific gender or all")
    parser.add_argument("--tier", type=str, default="all",
                       choices=["all", "Tier 1", "Tier 2", "Tier 3", "Tier1", "Tier2", "Tier3"],
                       help="Process specific tier or all")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip images that already have outputs")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    parsing_model_path = Path(args.parsing_model)
    pose_model_path = Path(args.pose_model)
    
    # Validate inputs
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    if not parsing_model_path.exists():
        print(f"❌ Parsing model not found: {parsing_model_path}")
        print("   Please download it first. See STAGE1_SETUP.md for instructions.")
        return
    
    if not pose_model_path.exists():
        print(f"❌ Pose model not found: {pose_model_path}")
        print("   Please download it first. See STAGE1_SETUP.md for instructions.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    print("\n" + "="*60)
    print("STAGE 1 BATCH PROCESSOR")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Gender filter: {args.gender}")
    print(f"Tier filter: {args.tier}")
    print("="*60 + "\n")
    
    print("Initializing Stage 1 processor...")
    try:
        processor = Stage1Processor(
            str(parsing_model_path),
            str(pose_model_path),
            device=args.device
        )
        print("✓ Processor initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return
    
    # Find all images
    print("Scanning for images...")
    all_images = find_all_images(input_dir)
    
    # Filter by gender and tier
    filtered_images = []
    for img_path in all_images:
        # Get relative path from input_dir
        try:
            rel_path = img_path.relative_to(input_dir)
            parts = rel_path.parts
            
            # Check gender filter
            if args.gender != "all":
                if args.gender not in parts:
                    continue
            
            # Check tier filter
            if args.tier != "all":
                # Handle both "Tier 1" and "Tier1" formats
                tier_found = False
                for part in parts:
                    if args.tier in part or part.replace(" ", "") == args.tier.replace(" ", ""):
                        tier_found = True
                        break
                if not tier_found:
                    continue
            
            filtered_images.append((img_path, rel_path))
        except ValueError:
            # Image not in input_dir, skip
            continue
    
    if not filtered_images:
        print(f"❌ No images found matching filters")
        print(f"   Gender: {args.gender}, Tier: {args.tier}")
        return
    
    print(f"Found {len(filtered_images)} images to process\n")
    
    # Process images
    results = {
        "success": [],
        "errors": [],
        "skipped": []
    }
    
    print("Processing images...")
    for image_path, rel_path in tqdm(filtered_images, desc="Processing"):
        # Check if should skip existing
        if args.skip_existing:
            image_name = image_path.stem
            output_subdir = output_dir / rel_path.parent
            parsing_output = output_subdir / f"{image_name}_parsing.png"
            pose_output = output_subdir / f"{image_name}_pose.json"
            
            if parsing_output.exists() and pose_output.exists():
                results["skipped"].append(str(image_path))
                continue
        
        # Process image
        result = process_image(processor, image_path, output_dir, rel_path)
        
        if result["status"] == "success":
            results["success"].append(result)
        else:
            results["errors"].append(result)
    
    # Save results summary
    summary = {
        "total_images": len(filtered_images),
        "successful": len(results["success"]),
        "errors": len(results["errors"]),
        "skipped": len(results["skipped"]),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "device": args.device,
        "filters": {
            "gender": args.gender,
            "tier": args.tier
        }
    }
    
    summary_path = output_dir / "stage1_batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {len(filtered_images)}")
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ Errors: {len(results['errors'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if results["errors"]:
        print(f"\n⚠ {len(results['errors'])} images failed to process:")
        for error in results["errors"][:5]:  # Show first 5 errors
            print(f"   - {error['input']}: {error.get('error', 'Unknown error')}")
        if len(results["errors"]) > 5:
            print(f"   ... and {len(results['errors']) - 5} more")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


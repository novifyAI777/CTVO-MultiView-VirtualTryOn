#!/usr/bin/env python3
"""
Stage 2 Batch Processing Script

This script processes all person-cloth combinations through Stage 2
(Cloth Warping) and saves the results.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc="", **kwargs):
        print(desc)
        return iterable
    TQDM_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ctvo_core.stage2_cloth_warping import Stage2Inference


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped based on name"""
    name_lower = file_path.name.lower()
    skip_keywords = ['mask', 'pose', 'parsing', 'warp', 'bg']
    return any(keyword in name_lower for keyword in skip_keywords)


def find_all_images(base_dir: Path, extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[Path]:
    """Find all images in directory tree, excluding unwanted files"""
    images = []
    for ext in extensions:
        images.extend(base_dir.rglob(f'*{ext}'))
        images.extend(base_dir.rglob(f'*{ext.upper()}'))
    # Filter out unwanted files
    images = [img for img in images if not should_skip_file(img)]
    return sorted(images)


def extract_person_path_info(person_path: Path, person_images_dir: Path) -> Tuple[str, str, str]:
    """
    Extract gender, tier, and garment from person image path.
    
    Expected format: images/train/{gender}/{tier}/{garment}/{view}.png
    Or: images/{gender}/{tier}/{garment}/{view}.png
    Example: images/train/Men/Tier 1/Blazer with dress pants/back_view.png
    
    Returns:
        (gender, tier, garment) or (None, None, None) if parsing fails
    """
    try:
        rel_path = person_path.relative_to(person_images_dir)
        parts = rel_path.parts
        
        # Handle case where 'train' is in the path
        if len(parts) > 0 and parts[0] == 'train':
            parts = parts[1:]  # Remove 'train' from parts
        
        # Expected structure: [gender, tier, garment, filename]
        if len(parts) >= 4:
            gender = parts[0]  # "Men" or "Women"
            tier = parts[1]   # "Tier 1", "Tier 2", etc.
            garment = parts[2]  # "Blazer with dress pants", etc.
            return gender, tier, garment
        elif len(parts) >= 3:
            # Try alternative structure
            gender = parts[0] if parts[0] in ['Men', 'Women'] else None
            tier = parts[1] if 'Tier' in parts[1] else None
            garment = parts[2] if gender and tier else None
            return gender, tier, garment
    except (ValueError, IndexError):
        pass
    
    return None, None, None


def find_matching_cloth(person_path: Path, cloth_dir: Path, person_images_dir: Path) -> Path:
    """
    Find the matching cloth image for a person image based on path structure.
    
    Person path: images/train/{gender}/{tier}/{garment}/{view}.png
    Cloth path: clothes/{gender}/{tier}/{garment}/{garment}.png
    
    Returns:
        Path to matching cloth image, or None if not found
    """
    gender, tier, garment = extract_person_path_info(person_path, person_images_dir)
    
    if not gender or not tier or not garment:
        return None
    
    # Actual structure: clothes/{gender}/{tier}/{garment}/{garment}.png
    # Example: clothes/Men/Tier 1/Blazer with dress pants/Blazer with dress pants.png
    
    # Construct cloth path - direct match
    cloth_path = cloth_dir / gender / tier / garment / f"{garment}.png"
    
    # Check if cloth exists
    if cloth_path.exists():
        return cloth_path
    
    # Try alternative: look for any PNG file in the garment directory
    garment_dir = cloth_dir / gender / tier / garment
    if garment_dir.exists():
        cloth_files = list(garment_dir.glob("*.png"))
        if cloth_files:
            # Prefer exact match, otherwise take first PNG
            for f in cloth_files:
                if f.name == f"{garment}.png":
                    return f
            return cloth_files[0]
    
    # Try with tier variations (Tier 1 vs Tier1 vs Tier_1)
    tier_variations = [
        tier,  # Original: "Tier 1"
        tier.replace(" ", ""),  # "Tier1"
        tier.replace(" ", "_"),  # "Tier_1"
    ]
    
    for tier_var in tier_variations:
        if tier_var == tier:
            continue  # Already tried above
        
        garment_dir = cloth_dir / gender / tier_var / garment
        if garment_dir.exists():
            cloth_files = list(garment_dir.glob("*.png"))
            if cloth_files:
                for f in cloth_files:
                    if f.name == f"{garment}.png":
                        return f
                return cloth_files[0]
    
    return None


def get_stage1_outputs(person_path: Path, stage1_parsing_dir: Path, 
                       stage1_pose_dir: Path, person_images_dir: Path) -> Tuple[Path, Path]:
    """
    Get corresponding Stage 1 outputs for a person image.
    
    Note: Stage 1 outputs pose heatmaps as .pt or .pth files (PyTorch tensors).
    Expected naming: back_view.pt, casual_view.pt, front_view.pt, etc.
    These .pt/.pth files are loaded directly using torch.load() - NOT opened as images.
    
    Person image: images/train/{gender}/{tier}/{garment}/{view}.png
    Pose heatmap: stage1_outputs/pose_heatmaps/{gender}/{tier}/{garment}/{view}.pt or .pth
    
    Returns (parsing_map_path, pose_tensor_path)
    """
    # Get relative path from images directory
    # Handle both images/ and images/train/ structures
    rel_path = None
    
    try:
        # Get relative path from person_images_dir
        rel_path = person_path.relative_to(person_images_dir)
        
        # Remove 'train' if present
        if rel_path.parts and rel_path.parts[0] == 'train':
            rel_path = Path(*rel_path.parts[1:])
    except (ValueError, IndexError):
        # Fallback: try to find by matching filename
        image_name = person_path.stem
        # Search for matching files - check both .pt and .pth extensions
        parsing_files = list(stage1_parsing_dir.rglob(f"{image_name}*.png"))
        pose_files_pt = list(stage1_pose_dir.rglob(f"{image_name}*.pt"))
        pose_files_pth = list(stage1_pose_dir.rglob(f"{image_name}*.pth"))
        pose_files = pose_files_pt + pose_files_pth
        
        if parsing_files and pose_files:
            return parsing_files[0], pose_files[0]
        return None, None
    
    # Construct paths
    parsing_path = stage1_parsing_dir / rel_path
    
    # Try .pt first (PyTorch standard), then .pth
    pose_path = stage1_pose_dir / rel_path.with_suffix('.pt')
    if not pose_path.exists():
        pose_path = stage1_pose_dir / rel_path.with_suffix('.pth')
    
    # If exact match doesn't exist, try to find by name
    if not parsing_path.exists():
        parsing_files = list(stage1_parsing_dir.rglob(f"{person_path.stem}*.png"))
        if parsing_files:
            parsing_path = parsing_files[0]
    
    if not pose_path.exists():
        # Search for both .pt and .pth files
        pose_files_pt = list(stage1_pose_dir.rglob(f"{person_path.stem}*.pt"))
        pose_files_pth = list(stage1_pose_dir.rglob(f"{person_path.stem}*.pth"))
        pose_files = pose_files_pt + pose_files_pth
        if pose_files:
            pose_path = pose_files[0]
    
    return parsing_path, pose_path


def process_combination(processor: Stage2Inference, 
                       person_img: Path,
                       cloth_img: Path,
                       parsing_map: Path,
                       pose_tensor: Path,
                       output_dir: Path,
                       person_images_dir: Path) -> Dict:
    """Process a single person-cloth combination through Stage 2"""
    # Extract path info for output structure
    gender, tier, garment = extract_person_path_info(person_img, person_images_dir)
    
    if gender and tier and garment:
        # Output structure: stage2_outputs/{gender}/{tier}/{garment}/{person_stem}_warped.png
        output_subdir = output_dir / gender / tier / garment
        output_subdir.mkdir(parents=True, exist_ok=True)
        person_name = person_img.stem
        output_filename = f"{person_name}_warped.png"
        output_path = output_subdir / output_filename
    else:
        # Fallback: use relative path structure
        rel_path = person_img.relative_to(person_images_dir)
        # Remove 'train' if present
        if rel_path.parts and rel_path.parts[0] == 'train':
            rel_path = Path(*rel_path.parts[1:])
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        person_name = person_img.stem
        output_filename = f"{person_name}_warped.png"
        output_path = output_subdir / output_filename
    
    # Skip if already exists
    if output_path.exists():
        return {
            "status": "skipped",
            "person": str(person_img),
            "cloth": str(cloth_img),
            "output": str(output_path)
        }
    
    try:
        # Check if inputs exist
        if not parsing_map.exists():
            return {
                "status": "error",
                "person": str(person_img),
                "cloth": str(cloth_img),
                "error": f"Parsing map not found: {parsing_map}"
            }
        
        if not pose_tensor.exists():
            return {
                "status": "error",
                "person": str(person_img),
                "cloth": str(cloth_img),
                "error": f"Pose tensor not found: {pose_tensor}"
            }
        
        # Run Stage 2
        # Note: pose_tensor is a .pt or .pth file that will be loaded directly as a PyTorch tensor
        # using torch.load() in the preprocessing pipeline (not opened as an image)
        processor.warp_cloth(
            person_img_path=str(person_img),
            cloth_img_path=str(cloth_img),
            parsing_map_path=str(parsing_map),
            pose_tensor_path=str(pose_tensor),  # .pt or .pth file - loaded as tensor via torch.load()
            output_path=str(output_path)
        )
        
        return {
            "status": "success",
            "person": str(person_img),
            "cloth": str(cloth_img),
            "output": str(output_path)
        }
    except Exception as e:
        return {
            "status": "error",
            "person": str(person_img),
            "cloth": str(cloth_img),
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Batch process images through Stage 2")
    parser.add_argument("--person_images_dir", type=str,
                       default="data/multiview_dataset/images/train",
                       help="Directory containing person images")
    parser.add_argument("--cloth_images_dir", type=str,
                       default="data/multiview_dataset/clothes",
                       help="Directory containing cloth images")
    parser.add_argument("--stage1_parsing_dir", type=str,
                       default="data/multiview_dataset/stage1_outputs/parsing_maps",
                       help="Directory containing Stage 1 parsing maps")
    parser.add_argument("--stage1_pose_dir", type=str,
                       default="data/multiview_dataset/stage1_outputs/pose_heatmaps",
                       help="Directory containing Stage 1 pose tensors (.pt or .pth files)")
    parser.add_argument("--output_dir", type=str,
                       default="data/multiview_dataset/stage2_outputs",
                       help="Output directory for Stage 2 results")
    parser.add_argument("--model_checkpoint", type=str,
                       default="ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth",
                       help="Path to Stage 2 model checkpoint")
    parser.add_argument("--model_type", type=str, default="gmm",
                       choices=["unet", "gmm"],
                       help="Type of model to use (default: gmm for unet_wrap.pth checkpoint)")
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
                       help="Skip combinations that already have outputs")
    parser.add_argument("--max_combinations", type=int, default=None,
                       help="Maximum number of combinations to process (for testing)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    person_images_dir = Path(args.person_images_dir)
    cloth_images_dir = Path(args.cloth_images_dir)
    stage1_parsing_dir = Path(args.stage1_parsing_dir)
    stage1_pose_dir = Path(args.stage1_pose_dir)
    output_dir = Path(args.output_dir)
    model_checkpoint = Path(args.model_checkpoint)
    
    # Validate inputs
    if not person_images_dir.exists():
        print(f"[ERROR] Person images directory not found: {person_images_dir}")
        return
    
    if not cloth_images_dir.exists():
        print(f"[ERROR] Cloth images directory not found: {cloth_images_dir}")
        return
    
    if not stage1_parsing_dir.exists():
        print(f"[ERROR] Stage 1 parsing directory not found: {stage1_parsing_dir}")
        print("   Please run Stage 1 first.")
        return
    
    if not stage1_pose_dir.exists():
        print(f"[ERROR] Stage 1 pose directory not found: {stage1_pose_dir}")
        print("   Please run Stage 1 first.")
        return
    
    if not model_checkpoint.exists():
        print(f"[ERROR] Model checkpoint not found: {model_checkpoint}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    print("\n" + "="*60)
    print("STAGE 2 BATCH PROCESSOR")
    print("="*60)
    print(f"Person images: {person_images_dir}")
    print(f"Cloth images: {cloth_images_dir}")
    print(f"Stage 1 parsing: {stage1_parsing_dir}")
    print(f"Stage 1 pose: {stage1_pose_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Gender filter: {args.gender}")
    print(f"Tier filter: {args.tier}")
    print("="*60 + "\n")
    
    print("Initializing Stage 2 processor...")
    try:
        processor = Stage2Inference(
            checkpoint_path=str(model_checkpoint),
            device=args.device
        )
        print("[OK] Processor initialized successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize processor: {e}")
        return
    
    # Find all person images
    print("Scanning for person images...")
    all_person_images = find_all_images(person_images_dir)
    
    # Filter by gender and tier
    filtered_person_images = []
    for img_path in all_person_images:
        try:
            gender, tier, garment = extract_person_path_info(img_path, person_images_dir)
            
            if not gender or not tier or not garment:
                continue  # Skip if we can't parse the path
            
            # Check gender filter
            if args.gender != "all":
                if args.gender != gender:
                    continue
            
            # Check tier filter
            if args.tier != "all":
                # Handle both "Tier 1" and "Tier1" formats
                tier_normalized = tier.replace(" ", "")
                filter_normalized = args.tier.replace(" ", "")
                if tier_normalized != filter_normalized and tier != args.tier:
                    continue
            
            filtered_person_images.append(img_path)
        except (ValueError, Exception):
            continue
    
    if not filtered_person_images:
        print(f"[ERROR] No person images found matching filters")
        print(f"   Gender: {args.gender}, Tier: {args.tier}")
        return
    
    print(f"Found {len(filtered_person_images)} person images\n")
    
    # Generate one-to-one combinations
    print("Generating one-to-one combinations...")
    combinations = []
    missing_cloths = []
    missing_stage1 = []
    
    for person_img in filtered_person_images:
        # Find matching cloth (one-to-one)
        cloth_img = find_matching_cloth(person_img, cloth_images_dir, person_images_dir)
        
        if cloth_img is None:
            gender, tier, garment = extract_person_path_info(person_img, person_images_dir)
            missing_cloths.append(f"{person_img.name} (expected: {gender}/{tier}/{garment})")
            continue
        
        # Get Stage 1 outputs
        parsing_map, pose_tensor = get_stage1_outputs(
            person_img, stage1_parsing_dir, stage1_pose_dir, person_images_dir
        )
        
        if parsing_map is None or pose_tensor is None:
            missing_stage1.append(person_img.name)
            continue
        
        # Create one-to-one pair
        combinations.append((person_img, cloth_img, parsing_map, pose_tensor))
    
    if missing_cloths:
        print(f"[WARNING] {len(missing_cloths)} person images without matching cloth:")
        for msg in missing_cloths[:5]:
            print(f"   - {msg}")
        if len(missing_cloths) > 5:
            print(f"   ... and {len(missing_cloths) - 5} more")
        print()
    
    if missing_stage1:
        print(f"[WARNING] {len(missing_stage1)} person images without Stage 1 outputs:")
        for name in missing_stage1[:5]:
            print(f"   - {name}")
        if len(missing_stage1) > 5:
            print(f"   ... and {len(missing_stage1) - 5} more")
        print()
    
    if args.max_combinations:
        combinations = combinations[:args.max_combinations]
        print(f"Limiting to {args.max_combinations} combinations for testing")
    
    print(f"Generated {len(combinations)} combinations to process\n")
    
    # Process combinations
    results = {
        "success": [],
        "errors": [],
        "skipped": []
    }
    
    print("Processing combinations...")
    
    # Configure tqdm for cleaner progress display
    if TQDM_AVAILABLE:
        # Custom format: percentage | bar | count | ETA
        tqdm_kwargs = {
            "desc": "Stage 2",
            "total": len(combinations),
            "unit": "img",
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] | ETA: {remaining}",
            "miniters": max(1, len(combinations) // 50),  # Update at most 50 times total
            "mininterval": 2.0,  # Update at most once per 2 seconds
            "maxinterval": 10.0,  # Force update every 10 seconds
            "smoothing": 0.1  # Smooth the ETA calculation
        }
        progress_bar = tqdm(combinations, **tqdm_kwargs)
    else:
        progress_bar = combinations
        print(f"Processing {len(combinations)} combinations...")
    
    for person_img, cloth_img, parsing_map, pose_tensor in progress_bar:
        try:
            result = process_combination(
                processor, person_img, cloth_img, parsing_map, pose_tensor, 
                output_dir, person_images_dir
            )
            
            if result["status"] == "success":
                results["success"].append(result)
            elif result["status"] == "skipped":
                results["skipped"].append(result)
            else:
                results["errors"].append(result)
        except Exception as e:
            # Catch any unexpected errors to prevent early termination
            error_result = {
                "status": "error",
                "person": str(person_img),
                "cloth": str(cloth_img),
                "error": f"Unexpected error: {str(e)}"
            }
            results["errors"].append(error_result)
            print(f"\n[ERROR] Unexpected error processing {person_img.name}: {e}")
            import traceback
            traceback.print_exc()
            # Continue processing remaining items
            continue
    
    if TQDM_AVAILABLE:
        progress_bar.close()
    
    # Save results summary
    summary = {
        "total_combinations": len(combinations),
        "successful": len(results["success"]),
        "errors": len(results["errors"]),
        "skipped": len(results["skipped"]),
        "person_images_dir": str(person_images_dir),
        "cloth_images_dir": str(cloth_images_dir),
        "output_dir": str(output_dir),
        "model_type": args.model_type,
        "device": args.device,
        "filters": {
            "gender": args.gender,
            "tier": args.tier
        }
    }
    
    summary_path = output_dir / "stage2_batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total combinations: {len(combinations)}")
    print(f"[OK] Successful: {len(results['success'])}")
    print(f"[ERROR] Errors: {len(results['errors'])}")
    print(f"[SKIP] Skipped: {len(results['skipped'])}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if results["errors"]:
        print(f"\n[WARNING] {len(results['errors'])} combinations failed:")
        for error in results["errors"][:5]:  # Show first 5 errors
            print(f"   - {Path(error['person']).name} + {Path(error['cloth']).name}: {error.get('error', 'Unknown error')}")
        if len(results["errors"]) > 5:
            print(f"   ... and {len(results['errors']) - 5} more")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Diagnostic script to check for missing Stage 1 outputs
"""

import os
from pathlib import Path
from collections import defaultdict

def find_all_images(base_dir: Path, extensions=None):
    """Find all images in directory tree"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in extensions:
        images.extend(base_dir.rglob(f'*{ext}'))
        images.extend(base_dir.rglob(f'*{ext.upper()}'))
    # Filter out unwanted files
    skip_keywords = ['mask', 'pose', 'parsing', 'warp', 'bg']
    images = [img for img in images if not any(kw in img.name.lower() for kw in skip_keywords)]
    return sorted(images)

def get_stage1_outputs(person_path: Path, stage1_parsing_dir: Path, stage1_pose_dir: Path):
    """Get corresponding Stage 1 outputs for a person image"""
    rel_path = None
    for parent in person_path.parents:
        if 'images' in parent.parts or 'train' in parent.parts:
            try:
                rel_path = person_path.relative_to(parent)
                break
            except ValueError:
                continue
    
    if rel_path is None:
        image_name = person_path.stem
        parsing_files = list(stage1_parsing_dir.rglob(f"{image_name}*.png"))
        pose_files = list(stage1_pose_dir.rglob(f"{image_name}*.pt"))
        if parsing_files and pose_files:
            return parsing_files[0], pose_files[0]
        return None, None
    
    parsing_path = stage1_parsing_dir / rel_path
    pose_path = stage1_pose_dir / rel_path.with_suffix('.pt')
    
    if not parsing_path.exists():
        parsing_files = list(stage1_parsing_dir.rglob(f"{person_path.stem}*.png"))
        if parsing_files:
            parsing_path = parsing_files[0]
        else:
            parsing_path = None
    
    if not pose_path.exists():
        pose_files = list(stage1_pose_dir.rglob(f"{person_path.stem}*.pt"))
        if pose_files:
            pose_path = pose_files[0]
        else:
            pose_path = None
    
    return parsing_path, pose_path

def main():
    # Default paths
    person_images_dir = Path("data/multiview_dataset/images/train")
    stage1_parsing_dir = Path("data/multiview_dataset/stage1_outputs/parsing_maps")
    stage1_pose_dir = Path("data/multiview_dataset/stage1_outputs/pose_heatmaps")
    
    print("="*60)
    print("STAGE 1 OUTPUTS DIAGNOSTIC")
    print("="*60)
    print(f"Person images: {person_images_dir}")
    print(f"Parsing maps: {stage1_parsing_dir}")
    print(f"Pose tensors: {stage1_pose_dir}")
    print("="*60 + "\n")
    
    # Check if directories exist
    if not person_images_dir.exists():
        print(f"[ERROR] Person images directory not found: {person_images_dir}")
        return
    
    if not stage1_parsing_dir.exists():
        print(f"[ERROR] Stage 1 parsing directory not found: {stage1_parsing_dir}")
        return
    
    if not stage1_pose_dir.exists():
        print(f"[ERROR] Stage 1 pose directory not found: {stage1_pose_dir}")
        return
    
    # Find all person images
    print("Scanning for person images...")
    all_person_images = find_all_images(person_images_dir)
    print(f"Found {len(all_person_images)} person images\n")
    
    # Debug: Check for duplicates and Stage 1 outputs in person images directory
    print("="*60)
    print("DEBUGGING: Checking for duplicate counting...")
    print("="*60)
    
    # Check if Stage 1 outputs are being counted
    parsing_in_person_dir = [img for img in all_person_images if 'parsing' in str(img).lower() or 'stage1' in str(img).lower()]
    pose_in_person_dir = [img for img in all_person_images if 'pose' in str(img).lower()]
    
    print(f"\n1. Checking for Stage 1 output files in person images directory:")
    print(f"   Files with 'parsing' or 'stage1' in path: {len(parsing_in_person_dir)}")
    print(f"   Files with 'pose' in path: {len(pose_in_person_dir)}")
    
    if parsing_in_person_dir or pose_in_person_dir:
        print(f"\n   ⚠️  WARNING: Found Stage 1 output files in person images directory!")
        print(f"   Sample files:")
        for img in (parsing_in_person_dir + pose_in_person_dir)[:5]:
            print(f"      - {img}")
    else:
        print(f"   ✓ No Stage 1 output files found in person images directory")
    
    # Check for duplicate filenames (same name, different extension or location)
    print(f"\n2. Checking for duplicate filenames:")
    filename_counts = defaultdict(list)
    for img in all_person_images:
        filename_counts[img.stem.lower()].append(img)
    
    duplicates = {name: paths for name, paths in filename_counts.items() if len(paths) > 1}
    print(f"   Unique filenames: {len(filename_counts)}")
    print(f"   Filenames with multiple files: {len(duplicates)}")
    
    if duplicates:
        print(f"\n   ⚠️  WARNING: Found {len(duplicates)} filenames with multiple files!")
        print(f"   Sample duplicates:")
        for name, paths in list(duplicates.items())[:5]:
            print(f"      - {name}: {len(paths)} files")
            for p in paths:
                print(f"        {p.relative_to(person_images_dir)}")
    
    # Check directory structure
    print(f"\n3. Checking directory structure:")
    print(f"   Person images dir: {person_images_dir}")
    print(f"   Stage 1 parsing dir: {stage1_parsing_dir}")
    print(f"   Stage 1 pose dir: {stage1_pose_dir}")
    
    try:
        rel_parsing = stage1_parsing_dir.relative_to(person_images_dir)
        print(f"   ⚠️  WARNING: Stage 1 parsing dir is INSIDE person images dir!")
        print(f"      Relative path: {rel_parsing}")
    except ValueError:
        print(f"   ✓ Stage 1 parsing dir is OUTSIDE person images dir")
    
    try:
        rel_pose = stage1_pose_dir.relative_to(person_images_dir)
        print(f"   ⚠️  WARNING: Stage 1 pose dir is INSIDE person images dir!")
        print(f"      Relative path: {rel_pose}")
    except ValueError:
        print(f"   ✓ Stage 1 pose dir is OUTSIDE person images dir")
    
    # Check for both .png and .jpg versions of same image
    print(f"\n4. Checking for same image with different extensions:")
    ext_duplicates = []
    for name, paths in filename_counts.items():
        extensions = [p.suffix.lower() for p in paths]
        if len(set(extensions)) > 1:
            ext_duplicates.append((name, paths))
    
    print(f"   Images with multiple extensions: {len(ext_duplicates)}")
    if ext_duplicates:
        print(f"   Sample:")
        for name, paths in ext_duplicates[:3]:
            print(f"      - {name}: {[p.suffix for p in paths]}")
    
    print("="*60 + "\n")
    
    # Count Stage 1 outputs
    all_parsing_maps = list(stage1_parsing_dir.rglob("*.png"))
    all_pose_tensors = list(stage1_pose_dir.rglob("*.pt"))
    
    print(f"Found {len(all_parsing_maps)} parsing maps")
    print(f"Found {len(all_pose_tensors)} pose tensors\n")
    
    # Check each person image
    print("Checking Stage 1 outputs for each person image...")
    stats = {
        "total": len(all_person_images),
        "has_both": 0,
        "missing_parsing": 0,
        "missing_pose": 0,
        "missing_both": 0
    }
    
    missing_parsing_list = []
    missing_pose_list = []
    missing_both_list = []
    
    for person_img in all_person_images:
        parsing_path, pose_path = get_stage1_outputs(person_img, stage1_parsing_dir, stage1_pose_dir)
        
        has_parsing = parsing_path is not None and parsing_path.exists()
        has_pose = pose_path is not None and pose_path.exists()
        
        if has_parsing and has_pose:
            stats["has_both"] += 1
        elif has_parsing and not has_pose:
            stats["missing_pose"] += 1
            missing_pose_list.append(person_img)
        elif not has_parsing and has_pose:
            stats["missing_parsing"] += 1
            missing_parsing_list.append(person_img)
        else:
            stats["missing_both"] += 1
            missing_both_list.append(person_img)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total person images: {stats['total']}")
    print(f"✅ Has both outputs: {stats['has_both']}")
    print(f"❌ Missing parsing only: {stats['missing_parsing']}")
    print(f"❌ Missing pose only: {stats['missing_pose']}")
    print(f"❌ Missing both: {stats['missing_both']}")
    print("="*60 + "\n")
    
    # Print details
    if missing_parsing_list:
        print(f"\n[WARNING] {len(missing_parsing_list)} images missing parsing maps:")
        for img in missing_parsing_list[:10]:
            print(f"   - {img.relative_to(person_images_dir)}")
        if len(missing_parsing_list) > 10:
            print(f"   ... and {len(missing_parsing_list) - 10} more")
    
    if missing_pose_list:
        print(f"\n[WARNING] {len(missing_pose_list)} images missing pose tensors:")
        for img in missing_pose_list[:10]:
            print(f"   - {img.relative_to(person_images_dir)}")
        if len(missing_pose_list) > 10:
            print(f"   ... and {len(missing_pose_list) - 10} more")
    
    if missing_both_list:
        print(f"\n[WARNING] {len(missing_both_list)} images missing both outputs:")
        for img in missing_both_list[:10]:
            print(f"   - {img.relative_to(person_images_dir)}")
        if len(missing_both_list) > 10:
            print(f"   ... and {len(missing_both_list) - 10} more")
    
    # Expected vs actual
    print("\n" + "="*60)
    print("EXPECTED vs ACTUAL")
    print("="*60)
    expected = 1310  # 163.75 garments × 8 views (some with 6 views)
    print(f"Expected person images: {expected}")
    print(f"Actual person images found: {stats['total']}")
    print(f"Difference: {stats['total'] - expected}")
    
    if stats['total'] == expected * 2:
        print(f"\n⚠️  WARNING: Found exactly 2x expected count ({expected * 2})!")
        print(f"   This suggests:")
        print(f"   - Stage 1 outputs might be counted as person images, OR")
        print(f"   - Images are duplicated in the directory structure, OR")
        print(f"   - Same images exist with different extensions (.png and .jpg)")
    
    print(f"\nExpected Stage 1 outputs: {expected}")
    print(f"Actual Stage 1 outputs (both): {stats['has_both']}")
    if stats['has_both'] > expected:
        print(f"⚠️  More outputs than expected - likely counting duplicates")
    elif stats['has_both'] < expected:
        print(f"Missing Stage 1 outputs: {expected - stats['has_both']}")
    else:
        print(f"✓ Perfect match!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
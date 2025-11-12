#!/usr/bin/env python3
"""
Diagnostic script to check why Stage 2 has only 960 combinations instead of 1310
"""

from pathlib import Path
from collections import defaultdict

def find_all_images(base_dir: Path, extensions=None):
    """Find all images in directory tree, avoiding duplicates"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    images = []
    seen_stems = set()  # Track unique filenames (without extension)
    
    for ext in extensions:
        for pattern in [f'*{ext}', f'*{ext.upper()}']:
            for img_path in base_dir.rglob(pattern):
                skip_keywords = ['mask', 'pose', 'parsing', 'warp', 'bg']
                if any(kw in img_path.name.lower() for kw in skip_keywords):
                    continue
                
                # Use stem (filename without extension) to avoid .png/.jpg duplicates
                stem_key = (img_path.stem.lower(), str(img_path.parent))
                if stem_key not in seen_stems:
                    seen_stems.add(stem_key)
                    images.append(img_path)
    
    return sorted(images)

def extract_person_path_info(person_path: Path, person_images_dir: Path):
    """Extract gender, tier, and garment from person image path"""
    try:
        rel_path = person_path.relative_to(person_images_dir)
        parts = rel_path.parts
        
        if len(parts) >= 4:
            gender = parts[0]
            tier = parts[1]
            garment = parts[2]
            return gender, tier, garment
    except (ValueError, IndexError):
        pass
    
    return None, None, None

def find_matching_cloth(person_path: Path, cloth_dir: Path, person_images_dir: Path):
    """Find matching cloth image"""
    gender, tier, garment = extract_person_path_info(person_path, person_images_dir)
    
    if not gender or not tier or not garment:
        return None
    
    cloth_path = cloth_dir / gender / tier / garment / f"{garment}.png"
    if cloth_path.exists():
        return cloth_path
    
    garment_dir = cloth_dir / gender / tier / garment
    if garment_dir.exists():
        cloth_files = list(garment_dir.glob("*.png"))
        if cloth_files:
            return cloth_files[0]
    
    return None

def get_stage1_outputs(person_path: Path, stage1_parsing_dir: Path, stage1_pose_dir: Path):
    """Get Stage 1 outputs for a person image"""
    rel_path = None
    for parent in person_path.parents:
        if 'images' in parent.parts or 'train' in parent.parts:
            try:
                rel_path = person_path.relative_to(parent)
                break
            except ValueError:
                continue
    
    if rel_path is None:
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
    person_images_dir = Path("data/multiview_dataset/images/train")
    cloth_images_dir = Path("data/multiview_dataset/clothes")
    stage1_parsing_dir = Path("data/multiview_dataset/stage1_outputs/parsing_maps")
    stage1_pose_dir = Path("data/multiview_dataset/stage1_outputs/pose_heatmaps")
    
    print("="*60)
    print("STAGE 2 COMBINATIONS DIAGNOSTIC")
    print("="*60)
    print(f"Person images: {person_images_dir}")
    print(f"Cloth images: {cloth_images_dir}")
    print(f"Stage 1 parsing: {stage1_parsing_dir}")
    print(f"Stage 1 pose: {stage1_pose_dir}")
    print("="*60 + "\n")
    
    # Find unique person images (avoiding .png/.jpg duplicates)
    print("Finding unique person images...")
    all_person_images = find_all_images(person_images_dir)
    print(f"Found {len(all_person_images)} unique person images\n")
    
    # Analyze each person image
    stats = {
        "total": len(all_person_images),
        "has_cloth": 0,
        "has_stage1": 0,
        "has_both": 0,
        "has_all": 0,
        "missing_cloth": 0,
        "missing_stage1": 0,
        "missing_both": 0
    }
    
    missing_cloth_list = []
    missing_stage1_list = []
    missing_both_list = []
    valid_combinations = []
    
    print("Analyzing combinations...")
    for person_img in all_person_images:
        # Check cloth
        cloth_img = find_matching_cloth(person_img, cloth_images_dir, person_images_dir)
        has_cloth = cloth_img is not None
        
        # Check Stage 1 outputs
        parsing_map, pose_tensor = get_stage1_outputs(
            person_img, stage1_parsing_dir, stage1_pose_dir
        )
        has_parsing = parsing_map is not None and parsing_map.exists()
        has_pose = pose_tensor is not None and pose_tensor.exists()
        has_stage1 = has_parsing and has_pose
        
        # Update stats
        if has_cloth:
            stats["has_cloth"] += 1
        else:
            stats["missing_cloth"] += 1
            missing_cloth_list.append(person_img)
        
        if has_stage1:
            stats["has_stage1"] += 1
        else:
            stats["missing_stage1"] += 1
            missing_stage1_list.append(person_img)
        
        if has_cloth and has_stage1:
            stats["has_all"] += 1
            valid_combinations.append(person_img)
        elif not has_cloth and not has_stage1:
            stats["missing_both"] += 1
            missing_both_list.append(person_img)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total unique person images: {stats['total']}")
    print(f"✅ Have matching cloth: {stats['has_cloth']}")
    print(f"✅ Have Stage 1 outputs: {stats['has_stage1']}")
    print(f"✅ Have ALL (cloth + Stage 1): {stats['has_all']}")
    print(f"❌ Missing cloth only: {stats['missing_cloth']}")
    print(f"❌ Missing Stage 1 only: {stats['missing_stage1']}")
    print(f"❌ Missing both: {stats['missing_both']}")
    print("="*60 + "\n")
    
    # Expected vs actual
    expected = 1310
    print("="*60)
    print("EXPECTED vs ACTUAL")
    print("="*60)
    print(f"Expected person images: {expected}")
    print(f"Actual unique person images: {stats['total']}")
    print(f"Difference: {stats['total'] - expected}")
    print(f"\nExpected combinations: {expected}")
    print(f"Actual valid combinations: {stats['has_all']}")
    print(f"Missing combinations: {expected - stats['has_all']}")
    print("="*60 + "\n")
    
    # Breakdown by reason
    if missing_cloth_list:
        print(f"\n[INFO] {len(missing_cloth_list)} images missing cloth:")
        # Group by garment
        by_garment = defaultdict(list)
        for img in missing_cloth_list:
            gender, tier, garment = extract_person_path_info(img, person_images_dir)
            if garment:
                by_garment[f"{gender}/{tier}/{garment}"].append(img)
        
        print(f"   Affects {len(by_garment)} unique garments:")
        for garment_path, imgs in list(by_garment.items())[:10]:
            print(f"      - {garment_path}: {len(imgs)} images")
        if len(by_garment) > 10:
            print(f"      ... and {len(by_garment) - 10} more garments")
    
    if missing_stage1_list:
        print(f"\n[INFO] {len(missing_stage1_list)} images missing Stage 1 outputs:")
        for img in missing_stage1_list[:10]:
            print(f"   - {img.relative_to(person_images_dir)}")
        if len(missing_stage1_list) > 10:
            print(f"   ... and {len(missing_stage1_list) - 10} more")

if __name__ == "__main__":
    main()
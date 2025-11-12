import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

from ctvo_core.stage1_parsing_pose.model_parsing import load_parsing_model, run_parsing
from ctvo_core.stage1_parsing_pose.model_pose import load_pose_model, run_pose


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Process multiview images through Stage 1 (Parsing & Pose Estimation). "
                    "Preserves directory structure from data_root in output directories."
    )
    parser.add_argument("--data_root", required=True,
                       help="Root directory containing images (e.g., data/multiview_dataset/images/train)")
    parser.add_argument("--out_parsing", required=True,
                       help="Output directory for parsing maps (will preserve input directory structure)")
    parser.add_argument("--out_pose", required=True,
                       help="Output directory for pose heatmaps (will preserve input directory structure)")
    parser.add_argument("--parse_weights", required=True,
                       help="Path to parsing model weights")
    parser.add_argument("--pose_weights", required=True,
                       help="Path to pose model weights")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip images that already have both parsing and pose outputs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running on device: {device}")

    # --- Load models ---
    print("Loading models...")
    parsing_model = load_parsing_model(args.parse_weights, device)
    pose_model = load_pose_model(args.pose_weights, device)
    print("Models loaded successfully!")

    # Find all images
    data_root = Path(args.data_root)
    image_paths = sorted(list(data_root.rglob("*.png")) + list(data_root.rglob("*.jpg")))
    print(f"\nFound {len(image_paths)} images under {data_root}")
    
    if len(image_paths) == 0:
        print("No images found! Please check your data_root path.")
        return

    # Count images by folder structure for reporting
    folder_counts = defaultdict(int)
    for img_path in image_paths:
        rel_path = img_path.relative_to(data_root)
        folder = str(rel_path.parent)
        folder_counts[folder] += 1
    
    print(f"Found {len(folder_counts)} unique folders")
    print(f"Sample folders: {list(folder_counts.keys())[:3]}...")

    # Process images
    stats = {
        "total": len(image_paths),
        "processed": 0,
        "skipped": 0,
        "parsing_errors": 0,
        "pose_errors": 0,
        "both_errors": 0
    }

    print("\nProcessing images...")
    for img_path in tqdm(image_paths, desc="Stage 1 Processing"):
        rel_path = img_path.relative_to(data_root)

        # PARSING OUTPUT PATH (preserves directory structure)
        parsing_out_path = Path(args.out_parsing) / rel_path
        ensure_dir(parsing_out_path.parent)

        # POSE OUTPUT PATH (preserves directory structure)
        pose_out_path = Path(args.out_pose) / rel_path.with_suffix(".pt")
        ensure_dir(pose_out_path.parent)

        # Skip if both outputs exist
        if args.skip_existing and parsing_out_path.exists() and pose_out_path.exists():
            stats["skipped"] += 1
            continue

        parsing_success = False
        pose_success = False

        # --- Run Parsing ---
        if not (args.skip_existing and parsing_out_path.exists()):
            try:
                parsing_map = run_parsing(parsing_model, str(img_path), device)
                Image.fromarray(parsing_map.astype(np.uint8)).save(parsing_out_path)
                parsing_success = True
            except Exception as e:
                stats["parsing_errors"] += 1
                print(f"\n[WARN] Parsing failed for {img_path}: {e}")
        else:
            parsing_success = True

        # --- Run Pose ---
        if not (args.skip_existing and pose_out_path.exists()):
            try:
                pose_tensor = run_pose(pose_model, str(img_path), device)
                torch.save(pose_tensor, pose_out_path)
                pose_success = True
            except Exception as e:
                stats["pose_errors"] += 1
                print(f"\n[WARN] Pose failed for {img_path}: {e}")
        else:
            pose_success = True

        if parsing_success and pose_success:
            stats["processed"] += 1
        elif not parsing_success and not pose_success:
            stats["both_errors"] += 1

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {stats['total']}")
    print(f"✓ Successfully processed: {stats['processed']}")
    print(f"⊘ Skipped (already exist): {stats['skipped']}")
    print(f"✗ Parsing errors: {stats['parsing_errors']}")
    print(f"✗ Pose errors: {stats['pose_errors']}")
    print(f"✗ Both failed: {stats['both_errors']}")
    print(f"\nOutput structure preserved from: {data_root}")
    print(f"Parsing maps saved to: {args.out_parsing}")
    print(f"Pose heatmaps saved to: {args.out_pose}")
    print("="*60)


if __name__ == "__main__":
    main()

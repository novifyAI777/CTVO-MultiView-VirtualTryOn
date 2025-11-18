"""
Stage 2 Garment Mask Generation
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
from PIL import Image
import cv2
from sklearn.cluster import KMeans


def generate_garment_mask(parsing_map: np.ndarray,
                         clothing_labels: Optional[list] = None) -> np.ndarray:
    """
    Generate binary garment mask from parsing map.
    
    Args:
        parsing_map: parsing map array [H, W] with label indices
        clothing_labels: list of clothing label indices to include in mask
                        Default: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        
    Returns:
        binary mask [H, W] where 1 indicates garment region
    """
    if clothing_labels is None:
        # Default clothing labels (LIP dataset format)
        clothing_labels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    mask = np.zeros_like(parsing_map, dtype=np.float32)
    
    for label in clothing_labels:
        mask[parsing_map == label] = 1.0
    
    return mask


def generate_garment_mask_from_tensor(parsing_tensor: torch.Tensor,
                                     clothing_labels: Optional[list] = None) -> torch.Tensor:
    """
    Generate binary garment mask from parsing tensor.
    
    Args:
        parsing_tensor: parsing map tensor [C, H, W] or [H, W]
        clothing_labels: list of clothing label indices
        
    Returns:
        binary mask tensor [1, H, W]
    """
    # Convert to numpy if needed
    if isinstance(parsing_tensor, torch.Tensor):
        if parsing_tensor.dim() == 3:
            # [C, H, W] -> take argmax or first channel
            if parsing_tensor.shape[0] > 1:
                parsing_map = torch.argmax(parsing_tensor, dim=0).cpu().numpy()
            else:
                parsing_map = parsing_tensor[0].cpu().numpy()
        elif parsing_tensor.dim() == 2:
            parsing_map = parsing_tensor.cpu().numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {parsing_tensor.shape}")
    else:
        parsing_map = np.array(parsing_tensor)
    
    # Generate mask
    mask = generate_garment_mask(parsing_map, clothing_labels)
    
    # Convert back to tensor
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]
    
    return mask_tensor


def refine_mask_with_morphology(mask: np.ndarray,
                               kernel_size: int = 5,
                               iterations: int = 2) -> np.ndarray:
    """
    Refine mask using morphological operations.
    
    Args:
        mask: binary mask [H, W]
        kernel_size: size of morphological kernel
        iterations: number of iterations
        
    Returns:
        refined mask [H, W]
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Closing: fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Opening: remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    return mask


def apply_mask_to_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply binary mask to image.
    
    Args:
        image: image tensor [C, H, W] or [B, C, H, W]
        mask: mask tensor [1, H, W] or [B, 1, H, W]
        
    Returns:
        masked image tensor
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Ensure mask matches image batch size
    if mask.shape[0] != image.shape[0]:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    
    # Apply mask
    masked_image = image * mask
    
    return masked_image.squeeze(0) if image.dim() == 4 and image.shape[0] == 1 else masked_image


def generate_garment_mask_from_file(parsing_path: Path,
                                   output_path: Path,
                                   person_image_path: Optional[Path] = None,
                                   k_clusters: int = 3,
                                   refine: bool = True) -> bool:
    """
    Generate garment mask from parsing map file.
    
    Uses K-Means on the original person image (if provided) or falls back to
    label-based extraction from parsing map if it's a label-index image.
    
    Args:
        parsing_path: path to parsing map image
        output_path: path to save mask
        person_image_path: optional path to original person image (for K-Means)
        k_clusters: number of clusters for K-Means (default: 3)
        refine: whether to apply morphological refinement
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load parsing map to check if it's label-index or RGB
        parsing_img = Image.open(parsing_path)
        parsing_array = np.array(parsing_img)
        
        # Check if parsing map is label-index (grayscale with discrete values 0-18)
        is_label_index = False
        if len(parsing_array.shape) == 2:
            is_label_index = True
        elif len(parsing_array.shape) == 3:
            # Check if all channels are identical (grayscale saved as RGB)
            if np.allclose(parsing_array[:, :, 0], parsing_array[:, :, 1]) and \
               np.allclose(parsing_array[:, :, 0], parsing_array[:, :, 2]):
                is_label_index = True
                parsing_array = parsing_array[:, :, 0]  # Use single channel
        
        # Prefer K-Means on person image if available (best results)
        if person_image_path and person_image_path.exists():
            # Use K-Means on original person image (like cloth masks)
            mask = generate_mask_kmeans(person_image_path, k_clusters=k_clusters, refine=False)
        elif is_label_index:
            # Fallback: use label-based extraction from parsing map
            mask = generate_garment_mask(parsing_array, clothing_labels=None)
        else:
            # Last resort: try K-Means on parsing map (if it's RGB visualization)
            img_bgr = cv2.imread(str(parsing_path))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {parsing_path}")
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W, C = img_rgb.shape
            
            # Reshape to (H*W, 3) for K-Means
            pixels = img_rgb.reshape(-1, 3).astype(np.float32)
            
            # Run K-Means clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Calculate brightness for each cluster
            cluster_brightness = []
            for i in range(k_clusters):
                cluster_pixels = pixels[labels == i]
                luminance = np.mean(cluster_pixels @ np.array([0.299, 0.587, 0.114]))
                cluster_brightness.append(luminance)
            
            # Select cluster with lowest brightness (clothing, not background)
            cloth_cluster = np.argmin(cluster_brightness)
            
            # Create binary mask
            mask = (labels == cloth_cluster).astype(np.uint8) * 255
            mask = mask.reshape(H, W)
        
        # Apply morphological refinement if requested
        if refine:
            mask = refine_mask_with_morphology(mask.astype(np.float32) / 255.0, 
                                              kernel_size=5, iterations=2)
            mask = (mask * 255).astype(np.uint8)
        
        # Save mask
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = Image.fromarray(mask)
        mask_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {parsing_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_mask_kmeans(image_path: Union[str, Path],
                         k_clusters: int = 2,
                         refine: bool = True) -> np.ndarray:
    """
    Generate binary mask from flat cloth image using K-Means clustering.
    
    This method works for any color cloth (blue, black, white, etc.) by clustering
    pixels and selecting the cloth cluster based on lowest brightness mean.
    
    Args:
        image_path: path to cloth image
        k_clusters: number of clusters for K-Means (2 or 3, default: 2)
        refine: whether to apply morphological refinement
        
    Returns:
        binary mask as numpy array [H, W] where 255 = cloth, 0 = background
    """
    # Load image using OpenCV (BGR format)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, C = img_rgb.shape
    
    # Reshape to (H*W, 3) for K-Means
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Calculate brightness (luminance) for each cluster
    cluster_brightness = []
    for i in range(k_clusters):
        cluster_pixels = pixels[labels == i]
        # Calculate luminance: 0.299*R + 0.587*G + 0.114*B
        luminance = np.mean(cluster_pixels @ np.array([0.299, 0.587, 0.114]))
        cluster_brightness.append(luminance)
    
    # Select cluster with lowest brightness (usually the cloth, background is brighter)
    cloth_cluster = np.argmin(cluster_brightness)
    
    # Create binary mask: 255 for cloth cluster, 0 for background
    mask = (labels == cloth_cluster).astype(np.uint8) * 255
    mask = mask.reshape(H, W)
    
    # Apply morphological refinement if requested
    if refine:
        mask = refine_mask_with_morphology(mask.astype(np.float32) / 255.0, 
                                          kernel_size=5, iterations=2)
        mask = (mask * 255).astype(np.uint8)
    
    return mask


def generate_cloth_mask_from_file(cloth_path: Path,
                                  output_path: Path,
                                  k_clusters: int = 2,
                                  refine: bool = True) -> bool:
    """
    Generate cloth mask from flat cloth image using K-Means clustering.
    
    This replaces the old thresholding method and works for any color cloth.
    
    Args:
        cloth_path: path to cloth image
        output_path: path to save mask
        k_clusters: number of clusters for K-Means (2 or 3, default: 2)
        refine: whether to apply morphological refinement
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate mask using K-Means
        mask = generate_mask_kmeans(cloth_path, k_clusters=k_clusters, refine=refine)
        
        # Save mask
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = Image.fromarray(mask)
        mask_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {cloth_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_parsing_directory(parsing_dir: Path,
                              mask_output_dir: Path,
                              person_images_dir: Optional[Path] = None,
                              k_clusters: int = 3,
                              refine: bool = True,
                              skip_existing: bool = True) -> dict:
    """
    Process all parsing maps in a directory to generate garment masks.
    
    Uses K-Means on person images (if provided) or label-based extraction from parsing maps.
    
    Args:
        parsing_dir: directory containing parsing maps (for finding corresponding person images)
        mask_output_dir: directory to save masks
        person_images_dir: directory containing original person images (required for K-Means)
        k_clusters: number of clusters for K-Means (default: 3)
        refine: whether to refine masks
        skip_existing: whether to skip existing masks
        
    Returns:
        dictionary with processing statistics
    """
    from tqdm import tqdm
    
    # Find all parsing maps
    parsing_files = list(parsing_dir.rglob("*.png")) + list(parsing_dir.rglob("*.jpg"))
    
    if not parsing_files:
        print(f"No parsing maps found in {parsing_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    if person_images_dir is None:
        print("Warning: person_images_dir not provided. Using label-based extraction from parsing maps.")
        print("For better results, provide --person_images_dir to use K-Means on person images.")
    
    stats = {"total": len(parsing_files), "processed": 0, "skipped": 0, "errors": 0}
    
    for parsing_path in tqdm(parsing_files, desc="Generating garment masks"):
        # Get relative path
        rel_path = parsing_path.relative_to(parsing_dir)
        mask_output_path = mask_output_dir / rel_path
        
        # Skip if exists
        if skip_existing and mask_output_path.exists():
            stats["skipped"] += 1
            continue
        
        # Try to find corresponding person image
        person_image_path = None
        if person_images_dir:
            person_image_path = person_images_dir / rel_path
            if not person_image_path.exists():
                # Try alternative extensions
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    alt_path = person_images_dir / rel_path.with_suffix(ext)
                    if alt_path.exists():
                        person_image_path = alt_path
                        break
        
        # Generate mask using K-Means on person image (preferred) or label-based from parsing map
        if generate_garment_mask_from_file(parsing_path, mask_output_path,
                                          person_image_path=person_image_path,
                                          k_clusters=k_clusters, refine=refine):
            stats["processed"] += 1
        else:
            stats["errors"] += 1
    
    return stats


def generate_masks_for_folder(dir_path: Union[str, Path],
                               output_dir: Optional[Union[str, Path]] = None,
                               k_clusters: int = 2,
                               refine: bool = True,
                               skip_existing: bool = True) -> dict:
    """
    Generate masks for all cloth images in a folder using K-Means.
    
    If input is cloth_rgb.png, output will be cloth_mask.png (or in output_dir).
    
    Args:
        dir_path: directory containing cloth images
        output_dir: optional output directory (if None, saves alongside input)
        k_clusters: number of clusters for K-Means (2 or 3)
        refine: whether to apply morphological refinement
        skip_existing: whether to skip files that already have masks
        
    Returns:
        dictionary with processing statistics
    """
    from tqdm import tqdm
    
    dir_path = Path(dir_path)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    cloth_files = []
    for ext in image_extensions:
        cloth_files.extend(list(dir_path.rglob(f"*{ext}")))
    
    # Filter out mask files (files with "mask" in name)
    cloth_files = [f for f in cloth_files if 'mask' not in f.name.lower()]
    
    if not cloth_files:
        print(f"No cloth images found in {dir_path}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    stats = {"total": len(cloth_files), "processed": 0, "skipped": 0, "errors": 0}
    
    for cloth_path in tqdm(cloth_files, desc="Generating cloth masks"):
        # Determine output path
        if output_dir:
            # Use specified output directory, preserve relative structure
            rel_path = cloth_path.relative_to(dir_path)
            mask_output_path = Path(output_dir) / rel_path.parent / f"{cloth_path.stem}_mask.png"
        else:
            # Save alongside input: cloth_rgb.png -> cloth_mask.png
            mask_output_path = cloth_path.parent / f"{cloth_path.stem}_mask.png"
        
        # Skip if mask already exists
        if skip_existing and mask_output_path.exists():
            stats["skipped"] += 1
            continue
        
        # Generate mask using K-Means
        if generate_cloth_mask_from_file(cloth_path, mask_output_path, 
                                        k_clusters=k_clusters, refine=refine):
            stats["processed"] += 1
        else:
            stats["errors"] += 1
    
    return stats


def process_cloth_directory(cloth_dir: Path,
                            mask_output_dir: Path,
                            k_clusters: int = 2,
                            refine: bool = True,
                            skip_existing: bool = True) -> dict:
    """
    Process all cloth images in a directory to generate cloth masks using K-Means.
    
    Args:
        cloth_dir: directory containing cloth images
        mask_output_dir: directory to save masks
        k_clusters: number of clusters for K-Means (2 or 3, default: 2)
        refine: whether to refine masks
        skip_existing: whether to skip existing masks
        
    Returns:
        dictionary with processing statistics
    """
    from tqdm import tqdm
    
    # Find all cloth images
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    cloth_files = []
    for ext in image_extensions:
        cloth_files.extend(list(cloth_dir.rglob(f"*{ext}")))
    
    # Filter out mask files
    cloth_files = [f for f in cloth_files if 'mask' not in f.name.lower()]
    
    if not cloth_files:
        print(f"No cloth images found in {cloth_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    stats = {"total": len(cloth_files), "processed": 0, "skipped": 0, "errors": 0}
    
    for cloth_path in tqdm(cloth_files, desc="Generating cloth masks"):
        # Get relative path
        rel_path = cloth_path.relative_to(cloth_dir)
        mask_output_path = mask_output_dir / rel_path.with_suffix('.png')
        
        # Skip if exists
        if skip_existing and mask_output_path.exists():
            stats["skipped"] += 1
            continue
        
        # Generate mask using K-Means
        if generate_cloth_mask_from_file(cloth_path, mask_output_path, 
                                        k_clusters=k_clusters, refine=refine):
            stats["processed"] += 1
        else:
            stats["errors"] += 1
    
    return stats


def main():
    """Command-line interface for mask generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 masks from parsing maps and cloth images"
    )
    
    parser.add_argument(
        "--parsing_dir",
        type=str,
        default="data/multiview_dataset/stage1_outputs/parsing_maps",
        help="Directory containing parsing maps from Stage 1"
    )
    
    parser.add_argument(
        "--person_images_dir",
        type=str,
        default="data/multiview_dataset/images/train",
        help="Directory containing original person images (required for K-Means garment mask generation)"
    )
    
    parser.add_argument(
        "--mask_output_dir",
        type=str,
        default="data/multiview_dataset/stage2_inputs/garment_masks",
        help="Directory to save garment masks"
    )
    
    parser.add_argument(
        "--cloth_dir",
        type=str,
        default=None,
        help="Directory containing flat cloth images (optional)"
    )
    
    parser.add_argument(
        "--cloth_mask_output_dir",
        type=str,
        default="data/multiview_dataset/stage2_inputs/cloth_masks",
        help="Directory to save cloth masks (optional)"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Process a single garment directory (alternative to --cloth_dir)"
    )
    
    parser.add_argument(
        "--k_clusters",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Number of K-Means clusters for garment masks (2-4, default: 3). For cloth masks, typically use 2."
    )
    
    parser.add_argument(
        "--cloth_k_clusters",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of K-Means clusters for cloth masks (2 or 3, default: 2)"
    )
    
    
    parser.add_argument(
        "--no_refine",
        action="store_true",
        help="Disable morphological refinement"
    )
    
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Regenerate masks even if they already exist"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    parsing_dir = Path(args.parsing_dir)
    mask_output_dir = Path(args.mask_output_dir)
    
    # Validate inputs
    if not parsing_dir.exists():
        print(f"Error: Parsing directory not found: {parsing_dir}")
        return
    
    print("="*60)
    print("STAGE 2 MASK GENERATION")
    print("="*60)
    print(f"Parsing maps directory: {parsing_dir}")
    print(f"Garment masks output: {mask_output_dir}")
    print(f"K-Means clusters (garment): {args.k_clusters}")
    print(f"K-Means clusters (cloth): {args.cloth_k_clusters}")
    print(f"Refine masks: {not args.no_refine}")
    print(f"Skip existing: {not args.no_skip_existing}")
    print("="*60)
    
    # Handle --dir option (process single garment folder)
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return
        
        print("="*60)
        print("GENERATING CLOTH MASKS (K-Means)")
        print("="*60)
        print(f"Input directory: {dir_path}")
        print(f"K-Means clusters: {args.k_clusters}")
        print(f"Refine masks: {not args.no_refine}")
        print(f"Skip existing: {not args.no_skip_existing}")
        print("="*60)
        
        stats = generate_masks_for_folder(
            dir_path=dir_path,
            k_clusters=args.cloth_k_clusters,
            refine=not args.no_refine,
            skip_existing=not args.no_skip_existing
        )
        
        print(f"\n✓ Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
        print("\n✓ Mask generation completed!")
        return
    
    # Generate garment masks from parsing maps
    person_images_dir = None
    if args.person_images_dir:
        person_images_dir = Path(args.person_images_dir)
    
    print("\n[1/2] Generating garment masks from parsing maps...")
    stats = process_parsing_directory(
        parsing_dir=parsing_dir,
        mask_output_dir=mask_output_dir,
        person_images_dir=person_images_dir,
        k_clusters=args.k_clusters,
        refine=not args.no_refine,
        skip_existing=not args.no_skip_existing
    )
    
    print(f"\n✓ Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    
    # Generate cloth masks if cloth directory provided
    if args.cloth_dir:
        cloth_dir = Path(args.cloth_dir)
        cloth_mask_output_dir = Path(args.cloth_mask_output_dir)
        
        if cloth_dir.exists():
            print("\n[2/2] Generating cloth masks from flat cloth images (K-Means)...")
            cloth_stats = process_cloth_directory(
                cloth_dir=cloth_dir,
                mask_output_dir=cloth_mask_output_dir,
                k_clusters=args.cloth_k_clusters,
                refine=not args.no_refine,
                skip_existing=not args.no_skip_existing
            )
            print(f"\n✓ Processed: {cloth_stats['processed']}, Skipped: {cloth_stats['skipped']}, Errors: {cloth_stats['errors']}")
        else:
            print(f"\nWarning: Cloth directory not found: {cloth_dir}")
    
    print("\n✓ Mask generation completed!")
    print("\nStage 2 inputs are now ready:")
    print("  - Person images: data/multiview_dataset/images/train/")
    print("  - Parsing maps: data/multiview_dataset/stage1_outputs/parsing_maps/")
    print("  - Pose heatmaps: data/multiview_dataset/stage1_outputs/pose_heatmaps/")
    print("  - Garment masks: data/multiview_dataset/stage2_inputs/garment_masks/")


if __name__ == "__main__":
    main()

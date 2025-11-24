"""
Stage 2 Dataset for Cloth Warping

Loads data from multiview_dataset structure:
- 8-view RGB images from images/train/<Gender>/<Tier>/<sample_id>/<view>.png
- Parsing maps from stage1_outputs/parsing_maps/...
- Pose heatmaps from stage1_outputs/pose_heatmaps/...
- Flat cloth RGB and cloth_mask from clothes/<Gender>/<Tier>/<sample_id>/
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 cloth warping.
    
    Loads:
    - Person RGB images (8 views per sample)
    - Parsing maps (PNG or .pth, converted to K channels)
    - Pose heatmaps (.pth or .pt files, P channels)
    - Cloth RGB images
    - Cloth masks
    """
    
    def __init__(self,
                 data_dir: str,
                 image_size: Tuple[int, int] = (256, 192),
                 is_train: bool = True,
                 load_targets: bool = False,
                 num_parsing_classes: int = 19,
                 num_pose_keypoints: int = 18):
        """
        Initialize dataset.
        
        Args:
            data_dir: root directory containing multiview_dataset
            image_size: target image size (H, W)
            is_train: whether this is training dataset
            load_targets: whether to load target warped cloth (for supervised training)
            num_parsing_classes: number of parsing classes (K)
            num_pose_keypoints: number of pose keypoints (P)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.load_targets = load_targets
        self.num_parsing_classes = num_parsing_classes
        self.num_pose_keypoints = num_pose_keypoints
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        
        # Mask transform (no normalization)
        self.mask_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
        
        # Load all samples
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            print(f"Warning: No samples found in {data_dir}")
        else:
            print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all data samples from directory structure"""
        samples = []
        
        # Expected structure:
        # data_dir/
        #   images/train/<Gender>/<Tier>/<sample_id>/<view>.png
        #   stage1_outputs/parsing_maps/<Gender>/<Tier>/<sample_id>/<view>.png (or .pth)
        #   stage1_outputs/pose_heatmaps/<Gender>/<Tier>/<sample_id>/<view>.pth (or .pt)
        #   clothes/<Gender>/<Tier>/<sample_id>/cloth.png
        #   stage2_inputs/cloth_masks/<Gender>/<Tier>/<sample_id>/cloth_mask.png
        
        images_dir = self.data_dir / "images" / ("train" if self.is_train else "test")
        parsing_dir = self.data_dir / "stage1_outputs" / "parsing_maps"
        pose_dir = self.data_dir / "stage1_outputs" / "pose_heatmaps"
        clothes_dir = self.data_dir / "clothes"
        cloth_masks_dir = self.data_dir / "stage2_inputs" / "cloth_masks"
        
        if not images_dir.exists():
            print(f"Warning: Images directory not found: {images_dir}")
            return []
        
        # Find all person images
        person_images = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpg"))
        
        for person_img_path in person_images:
            # Get relative path from images/train
            try:
                rel_path = person_img_path.relative_to(images_dir)
            except ValueError:
                continue
            
            # Extract components: Gender/Tier/sample_id/view.png
            parts = rel_path.parts
            if len(parts) < 4:
                continue
            
            gender = parts[0]
            tier = parts[1]
            sample_id = parts[2]
            view_name = rel_path.stem  # view name without extension
            
            # Normalize tier name to handle mismatches (Tier1 -> Tier 1, Tier3 -> Tier 3)
            tier_normalized = tier
            if tier.startswith("Tier") and not " " in tier:
                # Convert "Tier1" -> "Tier 1", "Tier2" -> "Tier 2", "Tier3" -> "Tier 3"
                tier_number = tier.replace("Tier", "")
                if tier_number.isdigit():
                    tier_normalized = f"Tier {tier_number}"
            
            # Find corresponding parsing map
            parsing_path = parsing_dir / rel_path
            if not parsing_path.exists():
                # Try .pth extension
                parsing_path_pth = parsing_dir / rel_path.with_suffix('.pth')
                if parsing_path_pth.exists():
                    parsing_path = parsing_path_pth
                else:
                    # Try alternative naming
                    alt_parsing = list(parsing_dir.rglob(f"{view_name}*.png")) + \
                                 list(parsing_dir.rglob(f"{view_name}*.pth"))
                    if alt_parsing:
                        parsing_path = alt_parsing[0]
                    else:
                        continue
            
            # Find corresponding pose heatmap - FIX: Check both .pth and .pt extensions
            pose_path = pose_dir / rel_path.with_suffix('.pth')
            if not pose_path.exists():
                # Try .pt extension (PyTorch standard)
                pose_path = pose_dir / rel_path.with_suffix('.pt')
            
            if not pose_path.exists():
                # Try alternative naming with .pth
                alt_pose = list(pose_dir.rglob(f"{view_name}*.pth"))
                if alt_pose:
                    pose_path = alt_pose[0]
                else:
                    # Try alternative naming with .pt
                    alt_pose = list(pose_dir.rglob(f"{view_name}*.pt"))
                    if alt_pose:
                        pose_path = alt_pose[0]
                    else:
                        continue
            
            # Find matching cloth for this sample
            # Try normalized tier first, then original tier
            cloth_dir = clothes_dir / gender / tier_normalized / sample_id
            if not cloth_dir.exists():
                # Try with original tier name
                cloth_dir = clothes_dir / gender / tier / sample_id
                if not cloth_dir.exists():
                    # Try to find any cloth in the normalized tier
                    cloth_dir = clothes_dir / gender / tier_normalized
                    if not cloth_dir.exists():
                        # Try original tier as fallback
                        cloth_dir = clothes_dir / gender / tier
                        if not cloth_dir.exists():
                            continue
            
            cloth_files = list(cloth_dir.rglob("*.png")) + list(cloth_dir.rglob("*.jpg"))
            if not cloth_files:
                continue
            
            # Use first cloth found (or match by sample_id if available)
            cloth_path = None
            for cf in cloth_files:
                # Try to match by sample_id in path
                if sample_id in str(cf):
                    cloth_path = cf
                    break
            if cloth_path is None:
                cloth_path = cloth_files[0]
            
            # Find cloth mask (use normalized tier)
            cloth_mask_dir = cloth_masks_dir / gender / tier_normalized / sample_id
            if not cloth_mask_dir.exists():
                cloth_mask_dir = cloth_masks_dir / gender / tier / sample_id
                if not cloth_mask_dir.exists():
                    cloth_mask_dir = cloth_masks_dir / gender / tier_normalized
                    if not cloth_mask_dir.exists():
                        cloth_mask_dir = cloth_masks_dir / gender / tier
            
            cloth_mask_files = list(cloth_mask_dir.rglob("*.png"))
            cloth_mask_path = None
            if cloth_mask_files:
                # Try to match by sample_id
                for cmf in cloth_mask_files:
                    if sample_id in str(cmf) or cloth_path.stem in str(cmf):
                        cloth_mask_path = cmf
                        break
                if cloth_mask_path is None:
                    cloth_mask_path = cloth_mask_files[0]
            
            # Create sample entry
            sample = {
                'person_img': str(person_img_path),
                'parsing_map': str(parsing_path),
                'pose_heatmap': str(pose_path),
                'cloth_img': str(cloth_path),
                'cloth_mask': str(cloth_mask_path) if cloth_mask_path else None,
                'sample_id': sample_id,
                'gender': gender,
                'tier': tier,
                'view': view_name
            }
            
            # Add target if available (for supervised training)
            if self.load_targets:
                target_dir = self.data_dir / "stage2_outputs"
                target_path = target_dir / rel_path.parent / f"{view_name}_warped.png"
                if target_path.exists():
                    sample['target'] = str(target_path)
            
            samples.append(sample)
        
        return samples
    
    def _load_parsing_map(self, parsing_path: str) -> torch.Tensor:
        """
        Load parsing map as K-channel tensor.
        
        If .pth file: load directly
        If PNG: convert to one-hot encoding with K channels
        """
        parsing_path = Path(parsing_path)
        
        if parsing_path.suffix == '.pth':
            # Load as tensor
            parsing_tensor = torch.load(parsing_path, map_location='cpu')
            
            # Handle different shapes
            if parsing_tensor.dim() == 4:
                parsing_tensor = parsing_tensor.squeeze(0)  # Remove batch dim
            if parsing_tensor.dim() == 2:
                parsing_tensor = parsing_tensor.unsqueeze(0)  # Add channel dim
            
            # If already has K channels, use as-is
            if parsing_tensor.shape[0] == self.num_parsing_classes:
                pass
            elif parsing_tensor.shape[0] == 1:
                # Single channel segmentation map, convert to one-hot
                parsing_tensor = self._segmentation_to_onehot(parsing_tensor.squeeze(0))
            else:
                # Unexpected shape, try to convert
                parsing_tensor = parsing_tensor[:self.num_parsing_classes]
        else:
            # Load PNG image
            parsing_img = Image.open(parsing_path).convert("L")  # Grayscale
            parsing_array = np.array(parsing_img)
            parsing_tensor = torch.from_numpy(parsing_array).long()
            # Convert to one-hot encoding
            parsing_tensor = self._segmentation_to_onehot(parsing_tensor)
        
        # Resize to target size
        if parsing_tensor.shape[1:] != self.image_size:
            parsing_tensor = F.interpolate(
                parsing_tensor.unsqueeze(0).float(),
                size=self.image_size,
                mode='nearest'
            ).squeeze(0)
        
        return parsing_tensor.float()
    
    def _segmentation_to_onehot(self, seg: torch.Tensor) -> torch.Tensor:
        """Convert segmentation map [H, W] to one-hot encoding [K, H, W]"""
        H, W = seg.shape
        onehot = torch.zeros(self.num_parsing_classes, H, W, dtype=torch.float32)
        
        for k in range(self.num_parsing_classes):
            onehot[k] = (seg == k).float()
        
        return onehot
    
    def _load_pose_heatmap(self, pose_path: str) -> torch.Tensor:
        """
        Load pose heatmap as P-channel tensor.
        
        Expected format: .pth or .pt file with shape [P, H, W] or [B, P, H, W]
        """
        pose_tensor = torch.load(pose_path, map_location='cpu')
        
        # Handle different shapes
        if pose_tensor.dim() == 4:
            pose_tensor = pose_tensor.squeeze(0)  # Remove batch dim
        
        # If has P channels, use as-is
        if pose_tensor.shape[0] == self.num_pose_keypoints:
            pass
        elif pose_tensor.shape[0] > self.num_pose_keypoints:
            # Take first P channels
            pose_tensor = pose_tensor[:self.num_pose_keypoints]
        else:
            # Pad with zeros if fewer channels
            P, H, W = pose_tensor.shape
            padded = torch.zeros(self.num_pose_keypoints, H, W, dtype=pose_tensor.dtype)
            padded[:P] = pose_tensor
            pose_tensor = padded
        
        # Resize to target size
        if pose_tensor.shape[1:] != self.image_size:
            pose_tensor = F.interpolate(
                pose_tensor.unsqueeze(0).float(),
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return pose_tensor.float()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        sample = self.samples[idx]
        
        # Load person RGB image
        person_img = Image.open(sample['person_img']).convert("RGB")
        person_rgb = self.transform(person_img)  # [3, H, W]
        
        # Load parsing map
        parsing_map = self._load_parsing_map(sample['parsing_map'])  # [K, H, W]
        
        # Load pose heatmap
        pose_heatmap = self._load_pose_heatmap(sample['pose_heatmap'])  # [P, H, W]
        
        # Load cloth RGB
        cloth_img = Image.open(sample['cloth_img']).convert("RGB")
        cloth_rgb = self.transform(cloth_img)  # [3, H, W]
        
        # Load cloth mask
        if sample['cloth_mask'] and Path(sample['cloth_mask']).exists():
            cloth_mask_img = Image.open(sample['cloth_mask']).convert("L")
            cloth_mask = self.mask_transform(cloth_mask_img)  # [1, H, W]
        else:
            # Create dummy mask if not available
            cloth_mask = torch.ones(1, self.image_size[0], self.image_size[1], dtype=torch.float32)
        
        # Concatenate inputs: person_rgb [3] + parsing_map [K] + pose_heatmap [P] + cloth_rgb [3] + cloth_mask [1]
        input_tensor = torch.cat([
            person_rgb,      # [3, H, W]
            parsing_map,     # [K, H, W]
            pose_heatmap,    # [P, H, W]
            cloth_rgb,       # [3, H, W]
            cloth_mask       # [1, H, W]
        ], dim=0)  # [3+K+P+3+1, H, W]
        
        result = {
            'input_tensor': input_tensor,
            'person_rgb': person_rgb,
            'cloth_rgb': cloth_rgb,
            'metadata': {
                'sample_id': sample['sample_id'],
                'gender': sample['gender'],
                'tier': sample['tier'],
                'view': sample['view']
            }
        }
        
        # Add target if available
        if 'target' in sample and Path(sample['target']).exists():
            target_img = Image.open(sample['target']).convert("RGB")
            target_tensor = self.transform(target_img)
            result['target_tensor'] = target_tensor
        
        return result


class Stage2TensorDataset(Dataset):
    """
    Dataset that loads pre-processed .pth tensor files.
    
    Useful when data has been pre-processed and saved as .pth files
    for faster loading during training.
    """
    
    def __init__(self,
                 data_dir: str,
                 is_train: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_dir: directory containing .pth tensor files
            is_train: whether this is training dataset
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        # Find all .pth files
        pattern = "train_*.pth" if is_train else "val_*.pth"
        self.samples = list(self.data_dir.rglob(pattern))
        
        print(f"Loaded {len(self.samples)} tensor samples from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load pre-processed tensor sample"""
        sample_path = self.samples[idx]
        data = torch.load(sample_path, map_location='cpu')
        return data
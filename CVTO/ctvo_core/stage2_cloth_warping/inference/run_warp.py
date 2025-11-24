"""
Stage 2 Inference Module

Run cloth warping inference using trained UNet model.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
import os
import numpy as np
import torch.nn.functional as F
from PIL import Image

from ..models.unet import UNet
from ..utils.config import Stage2Config
from ..utils.transforms import (
    get_val_transforms,
    load_pose_tensor,
)
from ..utils.visualizer import save_image


class Stage2Inference:
    """Inference wrapper for Stage 2 UNet"""
    
    def __init__(self,
                 checkpoint_path: str,
                 config: Optional[Stage2Config] = None,
                 device: Optional[str] = None):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: path to model checkpoint
            config: optional Stage2Config (will be loaded from checkpoint if not provided)
            device: device to run on (default: cuda if available, else cpu)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        if config is None:
            if 'config' in checkpoint:
                config_dict = checkpoint['config'].copy()
                # Remove 'in_channels' if present (it's a computed property, not a parameter)
                config_dict.pop('in_channels', None)
                # Also remove 'num_keypoints' if present (should be 'num_pose_keypoints')
                if 'num_keypoints' in config_dict and 'num_pose_keypoints' not in config_dict:
                    config_dict['num_pose_keypoints'] = config_dict.pop('num_keypoints')
                self.config = Stage2Config(**config_dict)
            else:
                self.config = Stage2Config()
        else:
            self.config = config
        
        # Create model
        self.model = UNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels
        ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is just state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Setup transforms
        self.transform = get_val_transforms(self.config.image_size)
    
    def _load_parsing_map(self, parsing_path: str) -> torch.Tensor:
        """
        Load parsing map as K-channel tensor.
        
        If .pth file: load directly
        If PNG: convert to one-hot encoding with K channels
        """
        parsing_path_obj = Path(parsing_path)
        
        if parsing_path_obj.suffix == '.pth':
            # Load as tensor
            parsing_tensor = torch.load(parsing_path_obj, map_location='cpu')
            
            # Handle different shapes
            if parsing_tensor.dim() == 4:
                parsing_tensor = parsing_tensor.squeeze(0)  # Remove batch dim
            if parsing_tensor.dim() == 2:
                parsing_tensor = parsing_tensor.unsqueeze(0)  # Add channel dim
            
            # If already has K channels, use as-is
            if parsing_tensor.shape[0] == self.config.num_parsing_channels:
                pass
            elif parsing_tensor.shape[0] == 1:
                # Single channel segmentation map, convert to one-hot
                parsing_tensor = self._segmentation_to_onehot(parsing_tensor.squeeze(0))
            else:
                # Unexpected shape, try to use first K channels
                parsing_tensor = parsing_tensor[:self.config.num_parsing_channels]
        else:
            # Load PNG image as grayscale
            parsing_img = Image.open(parsing_path_obj).convert("L")  # Grayscale
            parsing_array = np.array(parsing_img)
            parsing_tensor = torch.from_numpy(parsing_array).long()
            # Convert to one-hot encoding
            parsing_tensor = self._segmentation_to_onehot(parsing_tensor)
        
        # Resize to target size
        if parsing_tensor.shape[1:] != self.config.image_size:
            parsing_tensor = F.interpolate(
                parsing_tensor.unsqueeze(0).float(),
                size=self.config.image_size,
                mode='nearest'
            ).squeeze(0)
        
        return parsing_tensor.float()
    
    def _segmentation_to_onehot(self, seg: torch.Tensor) -> torch.Tensor:
        """Convert segmentation map [H, W] to one-hot encoding [K, H, W]"""
        K = self.config.num_parsing_channels
        H, W = seg.shape
        
        # Create one-hot encoding
        onehot = torch.zeros(K, H, W, dtype=torch.float32)
        
        # Clamp values to valid range [0, K-1]
        seg_clamped = torch.clamp(seg, 0, K - 1)
        
        # Set corresponding channels
        for k in range(K):
            onehot[k] = (seg_clamped == k).float()
        
        return onehot
    
    def _load_cloth_mask(self, cloth_img_path: str) -> torch.Tensor:
        """
        Generate cloth mask from cloth image.
        Simple approach: create a mask from the cloth image.
        """
        cloth_img = Image.open(cloth_img_path).convert("RGB")
        cloth_tensor = self.transform(cloth_img)
        
        # Create a simple mask: non-black pixels
        # Convert to grayscale and threshold
        gray = cloth_tensor.mean(dim=0)  # [H, W]
        mask = (gray > -0.9).float()  # Threshold for non-black pixels (after normalization)
        mask = mask.unsqueeze(0)  # [1, H, W]
        
        return mask
    
    def warp_cloth(self,
                   person_img_path: str,
                   cloth_img_path: str,
                   parsing_map_path: str,
                   pose_tensor_path: str,
                   output_path: Optional[str] = None,
                   return_tensor: bool = False) -> torch.Tensor:
        """
        Warp cloth image based on person pose and parsing.
        
        Args:
            person_img_path: path to person image
            cloth_img_path: path to cloth image
            parsing_map_path: path to parsing map (PNG or .pth)
            pose_tensor_path: path to pose .pth tensor file
            output_path: optional path to save result
            return_tensor: whether to return tensor instead of saving
            
        Returns:
            warped cloth tensor [3, H, W] in range [0, 1]
        """
        # Load and preprocess images
        cloth_img = Image.open(cloth_img_path).convert("RGB")
        person_img = Image.open(person_img_path).convert("RGB")
        
        cloth = self.transform(cloth_img)
        person = self.transform(person_img)
        
        # Load parsing map as K-channel tensor
        parsing = self._load_parsing_map(parsing_map_path)
        
        # Load pose tensor
        pose = load_pose_tensor(
            pose_tensor_path,
            device=self.device,
            height=self.config.image_size[0],
            width=self.config.image_size[1]
        )
        
        # Generate cloth mask
        cloth_mask = self._load_cloth_mask(cloth_img_path)
        
        # Concatenate inputs: person_rgb [3] + parsing_map [K] + pose_heatmap [P] + cloth_rgb [3] + cloth_mask [1]
        # Note: pose is [3, H, W] but we need [P, H, W], so we'll use the full pose tensor
        # For now, we'll use pose as-is (3 channels) and adjust if needed
        # Actually, let's check what the model expects - it should be [P, H, W] for pose
        
        # Load pose as keypoint heatmaps if available
        try:
            pose_full = torch.load(pose_tensor_path, map_location='cpu')
            if pose_full.dim() == 4:
                pose_full = pose_full.squeeze(0)
            if pose_full.dim() == 3 and pose_full.shape[0] == self.config.num_pose_keypoints:
                # Resize if needed
                if pose_full.shape[1:] != self.config.image_size:
                    pose_full = F.interpolate(
                        pose_full.unsqueeze(0).float(),
                        size=self.config.image_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                pose_keypoints = pose_full.float().to(self.device)
            else:
                # Fallback to 3-channel pose
                pose_keypoints = pose
        except:
            # Fallback to 3-channel pose
            pose_keypoints = pose
        
        # Ensure pose has the right number of channels
        if pose_keypoints.shape[0] != self.config.num_pose_keypoints:
            # If it's 3 channels, we need to expand to P channels
            # For now, duplicate the 3 channels to match P
            if pose_keypoints.shape[0] == 3:
                # Repeat channels to match num_pose_keypoints
                repeats = (self.config.num_pose_keypoints + 2) // 3
                pose_keypoints = pose_keypoints.repeat(repeats, 1, 1)[:self.config.num_pose_keypoints]
            else:
                # Take first P channels or pad
                if pose_keypoints.shape[0] < self.config.num_pose_keypoints:
                    padding = torch.zeros(
                        self.config.num_pose_keypoints - pose_keypoints.shape[0],
                        pose_keypoints.shape[1],
                        pose_keypoints.shape[2],
                        device=pose_keypoints.device
                    )
                    pose_keypoints = torch.cat([pose_keypoints, padding], dim=0)
                else:
                    pose_keypoints = pose_keypoints[:self.config.num_pose_keypoints]
        
        # Move parsing and cloth_mask to device
        parsing = parsing.to(self.device)
        cloth_mask = cloth_mask.to(self.device)
        
        # Concatenate inputs: person_rgb [3] + parsing_map [K] + pose_heatmap [P] + cloth_rgb [3] + cloth_mask [1]
        input_tensor = torch.cat([
            person,      # [3, H, W]
            parsing,     # [K, H, W]
            pose_keypoints,  # [P, H, W]
            cloth,       # [3, H, W]
            cloth_mask   # [1, H, W]
        ], dim=0)
        
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension [1, 3+K+P+3+1, H, W]
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)  # [1, 3, H, W]
            warped_cloth = output[0]  # Remove batch dimension [3, H, W]
        
        # Clamp to [0, 1] (model already outputs in [0, 1] range)
        warped_cloth = torch.clamp(warped_cloth, 0, 1)
        
        # Save if output path provided
        if output_path:
            save_image(warped_cloth, output_path, denormalize=False)
        
        if return_tensor:
            return warped_cloth
        else:
            return warped_cloth.cpu()
    
    def warp_cloth_batch(self,
                        samples: list,
                        output_dir: str) -> Dict[str, torch.Tensor]:
        """
        Warp cloth for multiple samples.
        
        Args:
            samples: list of dicts with keys: person_img, cloth_img, parsing_map, pose_tensor
            output_dir: directory to save outputs
            
        Returns:
            dictionary mapping sample IDs to output tensors
        """
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(samples):
            # Generate output path
            person_name = Path(sample['person_img']).stem
            cloth_name = Path(sample['cloth_img']).stem
            output_path = output_dir / f"{person_name}_{cloth_name}_warped.png"
            
            # Run inference
            warped = self.warp_cloth(
                person_img_path=sample['person_img'],
                cloth_img_path=sample['cloth_img'],
                parsing_map_path=sample['parsing_map'],
                pose_tensor_path=sample['pose_tensor'],
                output_path=str(output_path),
                return_tensor=True
            )
            
            results[f"{person_name}_{cloth_name}"] = warped
        
        return results


def run_stage2_inference(person_img: str,
                         cloth_img: str,
                         parsing_map: str,
                         pose_tensor: str,
                         checkpoint_path: str,
                         output_path: str,
                         device: Optional[str] = None) -> torch.Tensor:
    """
    Convenience function to run Stage 2 inference.
    
    Args:
        person_img: path to person image
        cloth_img: path to cloth image
        parsing_map: path to parsing map
        pose_tensor: path to pose .pth tensor file
        checkpoint_path: path to model checkpoint
        output_path: path to save output
        device: device to run on
        
    Returns:
        warped cloth tensor
    """
    inference = Stage2Inference(checkpoint_path, device=device)
    return inference.warp_cloth(
        person_img_path=person_img,
        cloth_img_path=cloth_img,
        parsing_map_path=parsing_map,
        pose_tensor_path=pose_tensor,
        output_path=output_path
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Stage 2 inference")
    parser.add_argument("--person_img", type=str, required=True)
    parser.add_argument("--cloth_img", type=str, required=True)
    parser.add_argument("--parsing_map", type=str, required=True)
    parser.add_argument("--pose_tensor", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    result = run_stage2_inference(
        person_img=args.person_img,
        cloth_img=args.cloth_img,
        parsing_map=args.parsing_map,
        pose_tensor=args.pose_tensor,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device
    )
    
    print(f"Warped cloth saved to {args.output}")
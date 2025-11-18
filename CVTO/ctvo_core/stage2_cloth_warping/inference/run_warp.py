"""
Stage 2 Inference Module

Run cloth warping inference using trained UNet model.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
import os

from ..models.unet import UNet
from ..utils.config import Stage2Config
from ..utils.transforms import (
    get_val_transforms,
    load_pose_tensor,
    preprocess_inputs
)
from ..utils.visualizer import save_image
from PIL import Image


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
                config_dict = checkpoint['config']
                self.config = Stage2Config(**config_dict)
            else:
                self.config = Stage2Config()
        else:
            self.config = config
        
        # Create model
        self.model = UNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            num_keypoints=self.config.num_keypoints,
            num_parsing_channels=self.config.num_parsing_channels,
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
            parsing_map_path: path to parsing map
            pose_tensor_path: path to pose .pt tensor file
            output_path: optional path to save result
            return_tensor: whether to return tensor instead of saving
            
        Returns:
            warped cloth tensor [3, H, W] in range [0, 1]
        """
        # Load and preprocess images
        cloth_img = Image.open(cloth_img_path).convert("RGB")
        person_img = Image.open(person_img_path).convert("RGB")
        parsing_map = Image.open(parsing_map_path).convert("RGB")
        
        cloth = self.transform(cloth_img)
        person = self.transform(person_img)
        parsing = self.transform(parsing_map)
        
        # Load pose tensor
        pose = load_pose_tensor(
            pose_tensor_path,
            device=self.device,
            height=self.config.image_size[0],
            width=self.config.image_size[1]
        )
        
        # Concatenate inputs
        input_tensor = preprocess_inputs(cloth, person, parsing, pose)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Extract RGB warped cloth (first 3 channels)
            warped_cloth = self.model.extract_warped_cloth(output)
            warped_cloth = warped_cloth[0]  # Remove batch dimension
        
        # Denormalize and clamp
        warped_cloth = warped_cloth * 0.5 + 0.5
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
        pose_tensor: path to pose .pt tensor file
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


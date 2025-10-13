"""
Stage 2: Main Runner for Cloth Warping

This module provides the main interface for Stage 2 cloth warping
"""

import os
import torch
from typing import Optional, Tuple
from .UNet import UNet
from .GMM import GMM, WarpingLayer
from .utils import warp_cloth_once, preprocess_inputs, load_warp_model


class Stage2Processor:
    """Main processor for Stage 2: Cloth Warping"""
    
    def __init__(self, 
                 model_checkpoint: str,
                 model_type: str = "unet",  # "unet" or "gmm"
                 device: str = "cpu"):
        self.device = device
        self.model_type = model_type
        
        if model_type == "unet":
            self.model = load_warp_model(model_checkpoint, UNet)
        elif model_type == "gmm":
            self.model = load_warp_model(model_checkpoint, GMM)
            self.warping_layer = WarpingLayer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def warp_cloth(self, 
                   person_img: str,
                   parsing_map: str, 
                   cloth_img: str,
                   pose_json: str,
                   output_path: Optional[str] = None) -> torch.Tensor:
        """
        Warp cloth image based on person pose and parsing
        
        Args:
            person_img: path to person image
            parsing_map: path to parsing map
            cloth_img: path to cloth image
            pose_json: path to pose JSON
            output_path: optional output path to save result
            
        Returns:
            warped cloth tensor
        """
        if self.model_type == "unet":
            # Use UNet-based warping
            inp = preprocess_inputs(cloth_img, person_img, parsing_map, pose_json).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                warped_cloth = self.model(inp)[0].cpu()
                warped_cloth = warped_cloth.mul_(0.5).add_(0.5).clamp_(0, 1)
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                from torchvision.transforms import ToPILImage
                ToPILImage()(warped_cloth).save(output_path)
                print(f"Warped cloth saved → {output_path}")
            
            return warped_cloth
            
        elif self.model_type == "gmm":
            # Use GMM-based warping
            from torchvision.transforms import ToTensor, Normalize, Resize
            from PIL import Image
            
            transform = torch.nn.Sequential(
                Resize((256, 192)),
                ToTensor(),
                Normalize([0.5]*3, [0.5]*3)
            )
            
            cloth_tensor = transform(Image.open(cloth_img).convert("RGB")).unsqueeze(0).to(self.device)
            person_tensor = transform(Image.open(person_img).convert("RGB")).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                flow = self.model(cloth_tensor, person_tensor)
                warped_cloth = self.warping_layer(cloth_tensor, flow)[0].cpu()
                warped_cloth = warped_cloth.mul_(0.5).add_(0.5).clamp_(0, 1)
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                from torchvision.transforms import ToPILImage
                ToPILImage()(warped_cloth).save(output_path)
                print(f"Warped cloth saved → {output_path}")
            
            return warped_cloth


def run_stage2(person_img: str,
               parsing_map: str,
               cloth_img: str, 
               pose_json: str,
               model_checkpoint: str,
               output_path: str,
               model_type: str = "unet",
               device: str = "cpu") -> torch.Tensor:
    """
    Convenience function to run Stage 2 cloth warping
    
    Args:
        person_img: path to person image
        parsing_map: path to parsing map
        cloth_img: path to cloth image
        pose_json: path to pose JSON
        model_checkpoint: path to model checkpoint
        output_path: output path for warped cloth
        model_type: type of model ("unet" or "gmm")
        device: device to run on
        
    Returns:
        warped cloth tensor
    """
    processor = Stage2Processor(model_checkpoint, model_type, device)
    return processor.warp_cloth(person_img, parsing_map, cloth_img, pose_json, output_path)


if __name__ == "__main__":
    # Example usage
    person_img = "path/to/person.jpg"
    parsing_map = "path/to/parsing.png"
    cloth_img = "path/to/cloth.jpg"
    pose_json = "path/to/pose.json"
    model_checkpoint = "path/to/unet_wrap.pth"
    output_path = "path/to/warped_cloth.jpg"
    
    warped_cloth = run_stage2(
        person_img, parsing_map, cloth_img, pose_json,
        model_checkpoint, output_path
    )
    
    print("Stage 2 processing completed!")
    print(f"Warped cloth shape: {warped_cloth.shape}")

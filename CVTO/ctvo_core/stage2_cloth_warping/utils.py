"""
Stage 2: Utility Functions for Cloth Warping

This module contains utility functions for Stage 2 cloth warping
"""

import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Optional, Tuple


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 256, 192  # (height, width)


def pose_to_heatmap(json_path: str, height: int = H, width: int = W, radius: int = 4) -> torch.Tensor:
    """
    Convert pose JSON to 3-channel heatmap with small Gaussian disks
    
    Args:
        json_path: path to pose JSON file
        height: output height
        width: output width  
        radius: radius for keypoint visualization
        
    Returns:
        heatmap tensor [3, H, W]
    """
    heat = np.zeros((3, height, width), np.float32)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        kps = data["people"][0]["pose_keypoints_2d"]
    except (FileNotFoundError, KeyError, IndexError):
        print(f"Warning: Could not load pose from {json_path}")
        return torch.from_numpy(heat)

    for i in range(0, len(kps), 3):
        x, y, conf = kps[i:i+3]
        if conf < 0.3:  # skip low-confidence keypoints
            continue
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(heat[0], (int(x), int(y)), radius, 1, -1)

    heat[1:] = heat[0]  # duplicate to 3 channels
    return torch.from_numpy(heat)


def load_warp_model(checkpoint_path: str, model_class=None):
    """
    Load warping model with TorchScript first, state-dict fallback
    
    Args:
        checkpoint_path: path to model checkpoint
        model_class: model class to instantiate if TorchScript fails
        
    Returns:
        loaded model in eval mode
    """
    try:
        # Try TorchScript first
        net = torch.jit.load(checkpoint_path, map_location=DEVICE)
        print("Loaded TorchScript warp network ✅")
        return net.eval()
    except Exception:
        pass

    # Fallback to state dict
    if model_class is not None:
        net = model_class().to(DEVICE)
        
        if os.path.exists(checkpoint_path):
            net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
            print("Loaded state-dict into model ✅")
        else:
            print("⚠  No checkpoint found — running model with random weights")
        
        return net.eval()
    
    raise ValueError("No model class provided for state-dict loading")


def preprocess_inputs(cloth_path: str, person_path: str, parsing_path: str, pose_path: str) -> torch.Tensor:
    """
    Preprocess all inputs for cloth warping
    
    Args:
        cloth_path: path to cloth image
        person_path: path to person image  
        parsing_path: path to parsing map
        pose_path: path to pose JSON
        
    Returns:
        concatenated input tensor [12, H, W]
    """
    to_tensor = T.Compose([T.Resize((H, W)), T.ToTensor()])
    normalize = T.Normalize([0.5]*3, [0.5]*3)
    
    # Load and preprocess images
    cloth = normalize(to_tensor(Image.open(cloth_path).convert("RGB")))
    person = normalize(to_tensor(Image.open(person_path).convert("RGB")))
    parsing = normalize(to_tensor(Image.open(parsing_path).convert("RGB")))
    pose = pose_to_heatmap(pose_path)
    
    # Concatenate all inputs: cloth(3) + person(3) + parsing(3) + pose(3) = 12 channels
    input_tensor = torch.cat([cloth, person, parsing, pose], dim=0)
    
    return input_tensor


def warp_cloth_once(person_img: str, 
                   parse_png: str, 
                   cloth_img: str,
                   pose_json: str, 
                   out_path: str,
                   model_checkpoint: str,
                   model_class=None) -> None:
    """
    One-shot cloth warping function
    
    Args:
        person_img: path to person image
        parse_png: path to parsing map
        cloth_img: path to cloth image
        pose_json: path to pose JSON
        out_path: output path for warped cloth
        model_checkpoint: path to model checkpoint
        model_class: model class for loading
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Preprocess inputs
    inp = preprocess_inputs(cloth_img, person_img, parse_png, pose_json).unsqueeze(0).to(DEVICE)
    
    # Load model
    net = load_warp_model(model_checkpoint, model_class)
    
    # Run inference
    with torch.no_grad():
        out = net(inp)[0].cpu().mul_(0.5).add_(0.5).clamp_(0, 1)
        T.ToPILImage()(out).save(out_path)
    
    print(f"Warped cloth saved → {out_path}")


def create_agnostic_parsing(parsing_map: np.ndarray) -> np.ndarray:
    """
    Create agnostic parsing map by removing person-specific details
    
    Args:
        parsing_map: original parsing map
        
    Returns:
        agnostic parsing map
    """
    agnostic = parsing_map.copy()
    
    # Remove face, hair, and other person-specific parts
    # Keep only clothing-related regions
    clothing_regions = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # clothing labels
    
    # Create binary mask for clothing regions
    mask = np.zeros_like(agnostic)
    for region in clothing_regions:
        mask[agnostic == region] = 1
    
    # Apply mask
    agnostic = agnostic * mask
    
    return agnostic

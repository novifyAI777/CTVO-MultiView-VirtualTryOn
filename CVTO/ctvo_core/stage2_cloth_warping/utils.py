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


def pose_tensor_to_json(pose_tensor_path: str, output_json_path: Optional[str] = None, 
                        target_height: int = H, target_width: int = W) -> dict:
    """
    Convert pose tensor (.pt) file to JSON format
    
    Args:
        pose_tensor_path: path to pose tensor file (.pt)
        output_json_path: optional path to save JSON file
        target_height: target height for keypoint scaling (default: 256)
        target_width: target width for keypoint scaling (default: 192)
        
    Returns:
        COCO format pose dictionary
    """
    try:
        pose_tensor = torch.load(pose_tensor_path, map_location='cpu')
        
        # Handle different tensor shapes
        if isinstance(pose_tensor, torch.Tensor):
            if pose_tensor.dim() == 3:  # [num_keypoints, H, W]
                heatmaps = pose_tensor.numpy()
            elif pose_tensor.dim() == 4:  # [1, num_keypoints, H, W] or [B, num_keypoints, H, W]
                heatmaps = pose_tensor.squeeze(0).numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {pose_tensor.shape}")
        else:
            heatmaps = np.array(pose_tensor)
        
        # Extract keypoints from heatmaps
        # heatmaps shape: [num_keypoints, H, W]
        keypoints = []
        for idx in range(heatmaps.shape[0]):  # For each keypoint
            hmap = heatmaps[idx]
            
            # Find peak location
            y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
            conf = float(hmap[y, x])
            
            # Scale coordinates from heatmap size to target size
            # The heatmap is typically smaller than the original image
            # Scale to target dimensions (256x192 for stage 2)
            hmap_h, hmap_w = hmap.shape
            x_scaled = int(x * target_width / hmap_w)
            y_scaled = int(y * target_height / hmap_h)
            
            # Clamp to valid range
            x_scaled = max(0, min(x_scaled, target_width - 1))
            y_scaled = max(0, min(y_scaled, target_height - 1))
            
            keypoints.extend([x_scaled, y_scaled, conf])
        
        coco_dict = {
            "version": 1.0,
            "people": [{
                "person_id": 0,
                "pose_keypoints_2d": keypoints
            }]
        }
        
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(coco_dict, f, indent=2)
        
        return coco_dict
    except Exception as e:
        print(f"Warning: Could not convert pose tensor from {pose_tensor_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty pose dict
        return {
            "version": 1.0,
            "people": [{
                "person_id": 0,
                "pose_keypoints_2d": []
            }]
        }


def load_pose_tensor_from_pt(pose_path: str, device: str = "cpu", height: int = H, width: int = W, radius: int = 4) -> torch.Tensor:
    """
    Load pose heatmap directly from .pt file as a PyTorch tensor.
    Converts keypoint heatmaps to 3-channel visualization by extracting keypoints and drawing circles.
    
    Args:
        pose_path: path to pose tensor file (.pt)
        device: device to move tensor to
        height: target height for resizing (if needed)
        width: target width for resizing (if needed)
        radius: radius for keypoint visualization circles
        
    Returns:
        pose heatmap tensor [3, H, W] with keypoints drawn as circles
    """
    try:
        # Load tensor from .pt file
        pose_tensor = torch.load(pose_path, map_location='cpu')  # Load to CPU first
        
        # Handle different tensor shapes
        if isinstance(pose_tensor, torch.Tensor):
            # Remove batch dimension if present
            if pose_tensor.dim() == 4:  # [B, C, H, W] or [B, num_keypoints, H, W]
                pose_tensor = pose_tensor.squeeze(0)
            elif pose_tensor.dim() == 3:
                # [num_keypoints, H, W] or [C, H, W]
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {pose_tensor.shape}")
            
            # Get original dimensions
            orig_h, orig_w = pose_tensor.shape[1], pose_tensor.shape[2]
            
            # If it's keypoint heatmaps [num_keypoints, H, W], extract keypoints and draw circles
            if pose_tensor.shape[0] > 3:
                # Convert to numpy for keypoint extraction
                heatmaps = pose_tensor.numpy()  # [num_keypoints, H, W]
                
                # Create empty 3-channel heatmap
                heat = np.zeros((3, height, width), dtype=np.float32)
                
                # Extract keypoints from each heatmap
                for idx in range(heatmaps.shape[0]):
                    hmap = heatmaps[idx]
                    
                    # Find peak location (keypoint)
                    y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
                    conf = float(hmap[y, x])
                    
                    # Skip low-confidence keypoints
                    if conf < 0.3:
                        continue
                    
                    # Scale coordinates from heatmap size to target size
                    x_scaled = int(x * width / orig_w)
                    y_scaled = int(y * height / orig_h)
                    
                    # Clamp to valid range
                    x_scaled = max(0, min(x_scaled, width - 1))
                    y_scaled = max(0, min(y_scaled, height - 1))
                    
                    # Draw circle at keypoint location
                    if 0 <= x_scaled < width and 0 <= y_scaled < height:
                        cv2.circle(heat[0], (x_scaled, y_scaled), radius, 1.0, -1)
                
                # Duplicate to 3 channels
                heat[1:] = heat[0]
                
                # Convert back to tensor
                pose_tensor = torch.from_numpy(heat).float()
                
            elif pose_tensor.shape[0] == 1:
                # Single channel, duplicate to 3
                pose_tensor = pose_tensor.repeat(3, 1, 1)  # [3, H, W]
            elif pose_tensor.shape[0] == 3:
                # Already 3 channels, use as-is but resize if needed
                pass
            else:
                raise ValueError(f"Unexpected number of channels: {pose_tensor.shape[0]}")
            
            # Resize if needed
            if pose_tensor.shape[1] != height or pose_tensor.shape[2] != width:
                from torch.nn.functional import interpolate
                pose_tensor = interpolate(
                    pose_tensor.unsqueeze(0), 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Ensure values are in [0, 1] range
            pose_tensor = torch.clamp(pose_tensor, 0.0, 1.0)
            
            # Move to device
            pose_tensor = pose_tensor.to(device)
            
            return pose_tensor
        else:
            raise ValueError(f"Loaded object is not a torch.Tensor: {type(pose_tensor)}")
            
    except Exception as e:
        print(f"Warning: Could not load pose tensor from {pose_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return zero tensor as fallback
        return torch.zeros((3, height, width), dtype=torch.float32, device=device)


def pose_to_heatmap(pose_path: str, height: int = H, width: int = W, radius: int = 4, device: str = "cpu") -> torch.Tensor:
    """
    Convert pose JSON or .pt tensor to 3-channel heatmap with small Gaussian disks
    
    Args:
        pose_path: path to pose JSON file or .pt tensor file
        height: output height
        width: output width  
        radius: radius for keypoint visualization
        device: device to move tensor to (for .pt files)
        
    Returns:
        heatmap tensor [3, H, W]
    """
    # Check if it's a .pt file - load directly as tensor
    if pose_path.endswith('.pt'):
        return load_pose_tensor_from_pt(pose_path, device=device, height=height, width=width)
    
    # Otherwise, handle JSON files (legacy support)
    heat = np.zeros((3, height, width), np.float32)
    
    # Load JSON file
    try:
        with open(pose_path, 'r') as f:
            data = json.load(f)
        kps = data["people"][0]["pose_keypoints_2d"]
    except (FileNotFoundError, KeyError, IndexError):
        print(f"Warning: Could not load pose from {pose_path}")
        return torch.from_numpy(heat).to(device)

    for i in range(0, len(kps), 3):
        x, y, conf = kps[i:i+3]
        if conf < 0.3:  # skip low-confidence keypoints
            continue
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(heat[0], (int(x), int(y)), radius, 1, -1)

    heat[1:] = heat[0]  # duplicate to 3 channels
    return torch.from_numpy(heat).to(device)


def _is_viton_checkpoint(checkpoint_path: str) -> bool:
    """
    Check if checkpoint is a VITON-style GMM (has extractionA, extractionB, regression keys)
    
    Args:
        checkpoint_path: path to checkpoint file
        
    Returns:
        True if checkpoint appears to be VITON GMM format
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            has_extractionA = any('extractionA' in k for k in keys)
            has_extractionB = any('extractionB' in k for k in keys)
            has_regression = any('regression' in k for k in keys)
            return has_extractionA and has_extractionB and has_regression
    except Exception:
        pass
    return False


def load_warp_model(checkpoint_path: str, model_class=None):
    """
    Load warping model with TorchScript first, state-dict fallback.
    Automatically detects VITON-style checkpoints and uses VITONGMM.
    
    Args:
        checkpoint_path: path to model checkpoint
        model_class: model class to instantiate if TorchScript fails
        
    Returns:
        loaded model in eval mode
    """
    try:
        # Try TorchScript first
        net = torch.jit.load(checkpoint_path, map_location=DEVICE)
        print("Loaded TorchScript warp network [OK]")
        return net.eval()
    except Exception:
        pass

    # Check if this is a VITON-style checkpoint
    is_viton = _is_viton_checkpoint(checkpoint_path)
    
    # Fallback to state dict
    if model_class is not None:
        # If checkpoint is VITON-style and model_class is GMM, use VITONGMM
        if is_viton and model_class.__name__ == 'GMM':
            from .GMM import VITONGMM
            net = VITONGMM().to(DEVICE)
            print("Detected VITON-style checkpoint, using VITONGMM [OK]")
        else:
            net = model_class().to(DEVICE)
        
        if os.path.exists(checkpoint_path):
            try:
                net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
                print("Loaded state-dict into model [OK]")
            except Exception as e:
                # Try with strict=False if strict loading fails
                print(f"Warning: Strict loading failed, trying flexible loading: {e}")
                net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
                print("Loaded state-dict into model (flexible mode) [OK]")
        else:
            print("[WARNING] No checkpoint found - running model with random weights")
        
        return net.eval()
    
    raise ValueError("No model class provided for state-dict loading")


def preprocess_inputs(cloth_path: str, person_path: str, parsing_path: str, pose_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Preprocess all inputs for cloth warping
    
    Args:
        cloth_path: path to cloth image
        person_path: path to person image  
        parsing_path: path to parsing map
        pose_path: path to pose .pt tensor file (Stage 1 outputs pose as .pt files)
                  Legacy: also supports JSON files, but .pt is the primary format
        device: device to move tensors to (for .pt pose files)
        
    Returns:
        concatenated input tensor [12, H, W]
    """
    to_tensor = T.Compose([T.Resize((H, W)), T.ToTensor()])
    normalize = T.Normalize([0.5]*3, [0.5]*3)
    
    # Load and preprocess images
    cloth = normalize(to_tensor(Image.open(cloth_path).convert("RGB")))
    person = normalize(to_tensor(Image.open(person_path).convert("RGB")))
    parsing = normalize(to_tensor(Image.open(parsing_path).convert("RGB")))
    
    # Load pose - Stage 1 outputs .pt files (PyTorch tensors)
    # These are loaded directly using torch.load(), not opened as images
    if pose_path.endswith('.pt'):
        # Primary method: Load .pt tensor file directly
        pose = load_pose_tensor_from_pt(pose_path, device=device, height=H, width=W)
        # Move other tensors to device to match
        cloth = cloth.to(device)
        person = person.to(device)
        parsing = parsing.to(device)
    else:
        # Legacy support: JSON files (for backward compatibility)
        pose = pose_to_heatmap(pose_path, device=device)
        # Move other tensors to device to match
        cloth = cloth.to(device)
        person = person.to(device)
        parsing = parsing.to(device)
    
    # Concatenate all inputs: cloth(3) + person(3) + parsing(3) + pose(3) = 12 channels
    input_tensor = torch.cat([cloth, person, parsing, pose], dim=0)
    
    return input_tensor


def warp_cloth_once(person_img: str, 
                   parse_png: str, 
                   cloth_img: str,
                   pose_json: str, 
                   out_path: str,
                   model_checkpoint: str,
                   model_class=None,
                   device: str = None) -> None:
    """
    One-shot cloth warping function
    
    Args:
        person_img: path to person image
        parse_png: path to parsing map
        cloth_img: path to cloth image
        pose_json: path to pose JSON or .pt tensor file
        out_path: output path for warped cloth
        model_checkpoint: path to model checkpoint
        model_class: model class for loading
        device: device to run on (defaults to DEVICE)
    """
    if device is None:
        device = DEVICE
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Preprocess inputs
    inp = preprocess_inputs(cloth_img, person_img, parse_png, pose_json, device=device).unsqueeze(0)
    
    # Load model
    net = load_warp_model(model_checkpoint, model_class)
    
    # Run inference
    with torch.no_grad():
        out = net(inp)[0].cpu().mul_(0.5).add_(0.5).clamp_(0, 1)
        T.ToPILImage()(out).save(out_path)
    
    print(f"Warped cloth saved -> {out_path}")


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

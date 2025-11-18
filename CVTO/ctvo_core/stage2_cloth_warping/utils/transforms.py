"""
Stage 2 Data Transforms
"""

import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional
import numpy as np


def get_train_transforms(image_size: Tuple[int, int] = (256, 192)):
    """Get training transforms"""
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


def get_val_transforms(image_size: Tuple[int, int] = (256, 192)):
    """Get validation transforms"""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


def load_pose_tensor(pose_path: str, 
                     device: str = "cpu",
                     height: int = 256,
                     width: int = 192,
                     radius: int = 4) -> torch.Tensor:
    """
    Load pose heatmap from .pt file as PyTorch tensor.
    
    Args:
        pose_path: path to pose tensor file (.pt)
        device: device to move tensor to
        height: target height for resizing
        width: target width for resizing
        radius: radius for keypoint visualization circles
        
    Returns:
        pose heatmap tensor [3, H, W] with keypoints drawn as circles
    """
    try:
        # Load tensor from .pt file
        pose_tensor = torch.load(pose_path, map_location='cpu')
        
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
                import cv2
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
                pose_tensor = torch.nn.functional.interpolate(
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
        # Return zero tensor as fallback
        return torch.zeros((3, height, width), dtype=torch.float32, device=device)


def preprocess_inputs(cloth: torch.Tensor,
                      person: torch.Tensor,
                      parsing: torch.Tensor,
                      pose: torch.Tensor) -> torch.Tensor:
    """
    Concatenate preprocessed inputs for UNet.
    
    Args:
        cloth: cloth image tensor [3, H, W]
        person: person image tensor [3, H, W]
        parsing: parsing map tensor [3, H, W]
        pose: pose heatmap tensor [3, H, W]
        
    Returns:
        concatenated input tensor [12, H, W]
    """
    return torch.cat([cloth, person, parsing, pose], dim=0)


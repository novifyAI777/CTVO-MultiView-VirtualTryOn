"""
Stage 2 Visualization Utilities
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert tensor to numpy image array.
    
    Args:
        tensor: image tensor [C, H, W] or [B, C, H, W]
        denormalize: whether to denormalize from [-1, 1] to [0, 1]
        
    Returns:
        image array [H, W, C] in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch item
    
    # Detach from computation graph if needed
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if denormalize:
        # Denormalize from [-1, 1] to [0, 1]
        tensor = tensor * 0.5 + 0.5
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    if tensor.shape[0] == 1:
        # Grayscale
        img = tensor.squeeze(0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
    else:
        # RGB
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
    
    return img


def save_image(tensor_or_list, 
               save_path: str,
               denormalize: bool = True,
               titles: Optional[List[str]] = None):
    """
    Save tensor(s) as image file(s).
    
    Args:
        tensor_or_list: image tensor [C, H, W] or [B, C, H, W], or list of tensors
        save_path: path to save image
        denormalize: whether to denormalize
        titles: optional list of titles for each image (if list of tensors provided)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Handle list of tensors
    if isinstance(tensor_or_list, list):
        num_images = len(tensor_or_list)
        if titles is None:
            titles = [f'Image {i+1}' for i in range(num_images)]
        
        # Create grid of images
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
        if num_images == 1:
            axes = [axes]
        
        for i, tensor in enumerate(tensor_or_list):
            img = tensor_to_image(tensor, denormalize)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Single tensor
        tensor = tensor_or_list
        img = tensor_to_image(tensor, denormalize)
        
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        Image.fromarray(img).save(save_path)


def visualize_batch(inputs: torch.Tensor,
                    outputs: torch.Tensor,
                    targets: Optional[torch.Tensor] = None,
                    save_path: Optional[str] = None,
                    num_samples: int = 4):
    """
    Visualize batch of inputs, outputs, and optionally targets.
    
    Args:
        inputs: input tensor [B, C, H, W]
        outputs: output tensor [B, C, H, W]
        targets: optional target tensor [B, C, H, W]
        save_path: optional path to save visualization
        num_samples: number of samples to visualize
    """
    B = min(inputs.shape[0], num_samples)
    
    fig, axes = plt.subplots(B, 3 if targets is not None else 2, figsize=(12, 4 * B))
    if B == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(B):
        # Input (show first 3 channels if more than 3)
        inp = inputs[i, :3] if inputs.shape[1] > 3 else inputs[i]
        axes[i, 0].imshow(tensor_to_image(inp))
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Output (show first 3 channels - RGB warped cloth)
        out = outputs[i, :3] if outputs.shape[1] > 3 else outputs[i]
        axes[i, 1].imshow(tensor_to_image(out))
        axes[i, 1].set_title('Output')
        axes[i, 1].axis('off')
        
        # Target (if provided)
        if targets is not None:
            tgt = targets[i, :3] if targets.shape[1] > 3 else targets[i]
            axes[i, 2].imshow(tensor_to_image(tgt))
            axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_output_channels(output: torch.Tensor,
                             save_path: Optional[str] = None,
                             num_keypoints: int = 18,
                             num_parsing_channels: int = 19):
    """
    Visualize different output channels from UNet.
    
    Args:
        output: model output [C, H, W] or [B, C, H, W]
        save_path: optional path to save visualization
        num_keypoints: number of keypoint channels
        num_parsing_channels: number of parsing channels
    """
    if output.dim() == 4:
        output = output[0]  # Take first batch item
    
    # Extract components
    rgb = output[:3]
    keypoints = output[3:3+num_keypoints]
    parsing = output[3+num_keypoints:3+num_keypoints+num_parsing_channels]
    features = output[3+num_keypoints+num_parsing_channels:-1]
    mask = output[-1:]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RGB warped cloth
    axes[0, 0].imshow(tensor_to_image(rgb))
    axes[0, 0].set_title('RGB Warped Cloth')
    axes[0, 0].axis('off')
    
    # Keypoints (sum all keypoint channels)
    kp_sum = keypoints.sum(dim=0)
    axes[0, 1].imshow(tensor_to_image(kp_sum.unsqueeze(0), denormalize=False))
    axes[0, 1].set_title('Keypoints (sum)')
    axes[0, 1].axis('off')
    
    # Parsing (argmax)
    parsing_argmax = torch.argmax(parsing, dim=0).float() / num_parsing_channels
    axes[0, 2].imshow(tensor_to_image(parsing_argmax.unsqueeze(0), denormalize=False), cmap='viridis')
    axes[0, 2].set_title('Parsing (argmax)')
    axes[0, 2].axis('off')
    
    # Features (show first 3 channels)
    feat_vis = features[:3] if features.shape[0] >= 3 else features
    axes[1, 0].imshow(tensor_to_image(feat_vis))
    axes[1, 0].set_title('Features (first 3)')
    axes[1, 0].axis('off')
    
    # Mask
    axes[1, 1].imshow(tensor_to_image(mask, denormalize=False), cmap='gray')
    axes[1, 1].set_title('Garment Mask')
    axes[1, 1].axis('off')
    
    # All channels overlay (RGB + mask)
    overlay = rgb * 0.7 + mask.repeat(3, 1, 1) * 0.3
    axes[1, 2].imshow(tensor_to_image(overlay))
    axes[1, 2].set_title('RGB + Mask Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


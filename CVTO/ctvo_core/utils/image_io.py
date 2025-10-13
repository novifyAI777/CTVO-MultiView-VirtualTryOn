"""
Shared Image I/O Utilities

This module contains utilities for image loading, saving, and preprocessing
across different stages of the CTVO pipeline.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Union, Tuple, Optional, List
import cv2


class ImageLoader:
    """Image loading utilities"""
    
    @staticmethod
    def load_image(image_path: str, 
                   target_size: Optional[Tuple[int, int]] = None,
                   normalize: bool = True) -> torch.Tensor:
        """
        Load image from path
        
        Args:
            image_path: path to image file
            target_size: target size (height, width)
            normalize: whether to normalize to [-1, 1]
            
        Returns:
            image tensor [3, H, W]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Resize if specified
        if target_size is not None:
            image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        
        # Convert to tensor
        transform = T.ToTensor()
        image_tensor = transform(image)
        
        # Normalize if specified
        if normalize:
            image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        return image_tensor
    
    @staticmethod
    def load_batch_images(image_paths: List[str],
                         target_size: Optional[Tuple[int, int]] = None,
                         normalize: bool = True) -> torch.Tensor:
        """
        Load batch of images
        
        Args:
            image_paths: list of image paths
            target_size: target size (height, width)
            normalize: whether to normalize to [-1, 1]
            
        Returns:
            batch tensor [B, 3, H, W]
        """
        images = []
        for path in image_paths:
            image = ImageLoader.load_image(path, target_size, normalize)
            images.append(image)
        
        return torch.stack(images)


class ImageSaver:
    """Image saving utilities"""
    
    @staticmethod
    def save_image(image_tensor: torch.Tensor,
                   output_path: str,
                   denormalize: bool = True) -> None:
        """
        Save image tensor to file
        
        Args:
            image_tensor: image tensor [3, H, W] or [B, 3, H, W]
            output_path: output file path
            denormalize: whether to denormalize from [-1, 1] to [0, 1]
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image
        
        # Denormalize if specified
        if denormalize:
            image_tensor = (image_tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert to PIL Image
        to_pil = T.ToPILImage()
        image_pil = to_pil(image_tensor)
        
        # Save image
        image_pil.save(output_path)
    
    @staticmethod
    def save_batch_images(image_tensors: torch.Tensor,
                         output_dir: str,
                         prefix: str = "image",
                         denormalize: bool = True) -> List[str]:
        """
        Save batch of images
        
        Args:
            image_tensors: batch tensor [B, 3, H, W]
            output_dir: output directory
            prefix: filename prefix
            denormalize: whether to denormalize from [-1, 1] to [0, 1]
            
        Returns:
            list of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i in range(image_tensors.size(0)):
            output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
            ImageSaver.save_image(image_tensors[i], output_path, denormalize)
            saved_paths.append(output_path)
        
        return saved_paths


class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def resize_image(image: torch.Tensor,
                    target_size: Tuple[int, int],
                    mode: str = "bilinear") -> torch.Tensor:
        """
        Resize image tensor
        
        Args:
            image: image tensor [3, H, W] or [B, 3, H, W]
            target_size: target size (height, width)
            mode: interpolation mode
            
        Returns:
            resized image tensor
        """
        return F.interpolate(
            image.unsqueeze(0) if image.dim() == 3 else image,
            size=target_size,
            mode=mode,
            align_corners=False
        ).squeeze(0) if image.dim() == 3 else F.interpolate(
            image, size=target_size, mode=mode, align_corners=False
        )
    
    @staticmethod
    def normalize_image(image: torch.Tensor,
                       mean: List[float] = [0.5, 0.5, 0.5],
                       std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
        """
        Normalize image tensor
        
        Args:
            image: image tensor [3, H, W] or [B, 3, H, W]
            mean: mean values for normalization
            std: std values for normalization
            
        Returns:
            normalized image tensor
        """
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        
        if image.dim() == 3:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return (image - mean) / std
    
    @staticmethod
    def denormalize_image(image: torch.Tensor,
                         mean: List[float] = [0.5, 0.5, 0.5],
                         std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
        """
        Denormalize image tensor
        
        Args:
            image: image tensor [3, H, W] or [B, 3, H, W]
            mean: mean values for denormalization
            std: std values for denormalization
            
        Returns:
            denormalized image tensor
        """
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        
        if image.dim() == 3:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return image * std + mean


class ImageAugmentation:
    """Image augmentation utilities"""
    
    @staticmethod
    def random_crop(image: torch.Tensor,
                   crop_size: Tuple[int, int]) -> torch.Tensor:
        """
        Random crop image
        
        Args:
            image: image tensor [3, H, W] or [B, 3, H, W]
            crop_size: crop size (height, width)
            
        Returns:
            cropped image tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Random crop
        cropped = T.RandomCrop(crop_size)(image)
        
        if squeeze_output:
            cropped = cropped.squeeze(0)
        
        return cropped
    
    @staticmethod
    def random_flip(image: torch.Tensor,
                   horizontal: bool = True,
                   vertical: bool = False) -> torch.Tensor:
        """
        Random flip image
        
        Args:
            image: image tensor [3, H, W] or [B, 3, H, W]
            horizontal: whether to apply horizontal flip
            vertical: whether to apply vertical flip
            
        Returns:
            flipped image tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Random flip
        if horizontal and torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[3])
        
        if vertical and torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
        
        if squeeze_output:
            image = image.squeeze(0)
        
        return image

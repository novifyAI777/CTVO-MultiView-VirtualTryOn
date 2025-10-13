"""
Stage 4: NeRF Dataset

This module implements dataset classes for NeRF training and evaluation.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import math


class NeRFDataset(Dataset):
    """Dataset for NeRF training"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (256, 256),
                 near: float = 0.0,
                 far: float = 1.0):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.near = near
        self.far = far
        
        # Load metadata
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Load camera parameters
        self.camera_params = self.metadata.get('camera_params', {})
        self.focal_length = self.camera_params.get('focal_length', 500.0)
        
    def __len__(self):
        return len(self.metadata['samples'])
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, sample['image_path'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Load camera pose
        camera_pose = torch.tensor(sample['camera_pose'], dtype=torch.float32)
        
        # Load depth (if available)
        depth = None
        if 'depth_path' in sample:
            depth_path = os.path.join(self.data_dir, sample['depth_path'])
            depth = np.load(depth_path)
            depth = torch.tensor(depth, dtype=torch.float32)
        
        return {
            'image': image,
            'camera_pose': camera_pose,
            'depth': depth,
            'focal_length': self.focal_length
        }


class MultiViewDataset(Dataset):
    """Dataset for multi-view evaluation"""
    
    def __init__(self, 
                 data_dir: str,
                 image_size: Tuple[int, int] = (256, 256)):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "multiview_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self):
        return len(self.metadata['samples'])
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load reference image
        ref_image_path = os.path.join(self.data_dir, sample['reference_image'])
        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_image = self.transform(ref_image)
        
        # Load multi-view images
        multiview_images = []
        for view_path in sample['multiview_images']:
            view_image_path = os.path.join(self.data_dir, view_path)
            view_image = Image.open(view_image_path).convert("RGB")
            view_image = self.transform(view_image)
            multiview_images.append(view_image)
        
        # Load camera poses
        camera_poses = torch.tensor(sample['camera_poses'], dtype=torch.float32)
        
        return {
            'reference_image': ref_image,
            'multiview_images': torch.stack(multiview_images),
            'camera_poses': camera_poses
        }


class SyntheticDataset(Dataset):
    """Dataset for synthetic data generation"""
    
    def __init__(self, 
                 data_dir: str,
                 num_samples: int = 1000,
                 image_size: Tuple[int, int] = (256, 256)):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Create synthetic data
        self.samples = self._create_synthetic_samples()
        
    def _create_synthetic_samples(self):
        """Create synthetic samples for NeRF training"""
        samples = []
        
        for i in range(self.num_samples):
            # Generate random camera pose
            camera_pose = self._generate_random_camera_pose()
            
            # Generate synthetic image (placeholder)
            image_path = f"synthetic_{i:06d}.png"
            
            sample = {
                'image_path': image_path,
                'camera_pose': camera_pose.tolist(),
                'depth_path': f"synthetic_depth_{i:06d}.npy"
            }
            samples.append(sample)
        
        return samples
    
    def _generate_random_camera_pose(self):
        """Generate random camera pose"""
        # Random position on sphere
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        radius = np.random.uniform(1.5, 3.0)
        
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi)
        z = radius * math.sin(phi) * math.sin(theta)
        
        # Look at origin
        forward = torch.tensor([-x, -y, -z], dtype=torch.float32)
        forward = forward / torch.norm(forward)
        
        # Up vector
        up = torch.tensor([0, 1, 0], dtype=torch.float32)
        
        # Right vector
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        # Recompute up
        up = torch.cross(right, forward)
        
        # Create pose matrix
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = torch.tensor([x, y, z], dtype=torch.float32)
        
        return pose
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create synthetic image (placeholder)
        image = torch.randn(3, *self.image_size)
        
        # Load camera pose
        camera_pose = torch.tensor(sample['camera_pose'], dtype=torch.float32)
        
        # Create synthetic depth
        depth = torch.randn(*self.image_size)
        
        return {
            'image': image,
            'camera_pose': camera_pose,
            'depth': depth,
            'focal_length': 500.0
        }


def create_dataloader(dataset: Dataset,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create data loader for NeRF training"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

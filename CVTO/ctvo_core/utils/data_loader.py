"""
Shared Data Loading Utilities

This module contains utilities for data loading and preprocessing
across different stages of the CTVO pipeline.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import random


class CTVODataLoader:
    """Main data loader for CTVO pipeline"""
    
    def __init__(self, 
                 data_dir: str,
                 image_size: Tuple[int, int] = (256, 192),
                 normalize: bool = True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize = normalize
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3) if normalize else T.Lambda(lambda x: x)
        ])
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert("RGB")
        return self.transform(image)
    
    def load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess mask"""
        full_path = os.path.join(self.data_dir, mask_path)
        mask = Image.open(full_path).convert("RGB")
        return self.transform(mask)
    
    def load_pose(self, pose_path: str) -> Dict:
        """Load pose JSON"""
        full_path = os.path.join(self.data_dir, pose_path)
        with open(full_path, 'r') as f:
            return json.load(f)


class CTVODataset(Dataset):
    """Base dataset class for CTVO"""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_path: str,
                 image_size: Tuple[int, int] = (256, 192),
                 normalize: bool = True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize = normalize
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3) if normalize else T.Lambda(lambda x: x)
        ])
    
    def __len__(self):
        return len(self.metadata['samples'])
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert("RGB")
        return self.transform(image)
    
    def load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess mask"""
        full_path = os.path.join(self.data_dir, mask_path)
        mask = Image.open(full_path).convert("RGB")
        return self.transform(mask)
    
    def load_pose(self, pose_path: str) -> Dict:
        """Load pose JSON"""
        full_path = os.path.join(self.data_dir, pose_path)
        with open(full_path, 'r') as f:
            return json.load(f)


class Stage1Dataset(CTVODataset):
    """Dataset for Stage 1: Human Parsing & Pose Estimation"""
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load person image
        person_img = self.load_image(sample['person_image'])
        
        # Load parsing map (if available)
        parsing_map = None
        if 'parsing_map' in sample:
            parsing_map = self.load_mask(sample['parsing_map'])
        
        # Load pose (if available)
        pose_data = None
        if 'pose_json' in sample:
            pose_data = self.load_pose(sample['pose_json'])
        
        return {
            'person_img': person_img,
            'parsing_map': parsing_map,
            'pose_data': pose_data,
            'sample_id': sample.get('id', idx)
        }


class Stage2Dataset(CTVODataset):
    """Dataset for Stage 2: Cloth Warping"""
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load images
        person_img = self.load_image(sample['person_image'])
        cloth_img = self.load_image(sample['cloth_image'])
        parsing_map = self.load_mask(sample['parsing_map'])
        
        # Load pose
        pose_data = self.load_pose(sample['pose_json'])
        
        # Load warped cloth (if available)
        warped_cloth = None
        if 'warped_cloth' in sample:
            warped_cloth = self.load_image(sample['warped_cloth'])
        
        return {
            'person_img': person_img,
            'cloth_img': cloth_img,
            'parsing_map': parsing_map,
            'pose_data': pose_data,
            'warped_cloth': warped_cloth,
            'sample_id': sample.get('id', idx)
        }


class Stage3Dataset(CTVODataset):
    """Dataset for Stage 3: Fusion Generation"""
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load images
        person_img = self.load_image(sample['person_image'])
        warped_cloth = self.load_image(sample['warped_cloth'])
        parsing_map = self.load_mask(sample['parsing_map'])
        target_img = self.load_image(sample['target_image'])
        
        # Load pose
        pose_data = self.load_pose(sample['pose_json'])
        
        return {
            'person_img': person_img,
            'warped_cloth': warped_cloth,
            'parsing_map': parsing_map,
            'target_img': target_img,
            'pose_data': pose_data,
            'sample_id': sample.get('id', idx)
        }


class Stage4Dataset(CTVODataset):
    """Dataset for Stage 4: NeRF Multi-view Generation"""
    
    def __getitem__(self, idx):
        sample = self.metadata['samples'][idx]
        
        # Load reference image
        reference_img = self.load_image(sample['reference_image'])
        
        # Load multi-view images
        multiview_images = []
        for view_path in sample['multiview_images']:
            view_img = self.load_image(view_path)
            multiview_images.append(view_img)
        
        # Load camera poses
        camera_poses = torch.tensor(sample['camera_poses'], dtype=torch.float32)
        
        return {
            'reference_img': reference_img,
            'multiview_images': torch.stack(multiview_images),
            'camera_poses': camera_poses,
            'sample_id': sample.get('id', idx)
        }


class DataAugmentation:
    """Data augmentation utilities"""
    
    @staticmethod
    def random_crop(image: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
        """Random crop image"""
        return T.RandomCrop(crop_size)(image)
    
    @staticmethod
    def random_flip(image: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
        """Random horizontal flip"""
        if horizontal and random.random() > 0.5:
            return torch.flip(image, dims=[2])
        return image
    
    @staticmethod
    def random_rotation(image: torch.Tensor, max_angle: float = 10.0) -> torch.Tensor:
        """Random rotation"""
        angle = random.uniform(-max_angle, max_angle)
        return T.RandomRotation(angle)(image)
    
    @staticmethod
    def color_jitter(image: torch.Tensor, 
                    brightness: float = 0.1,
                    contrast: float = 0.1,
                    saturation: float = 0.1,
                    hue: float = 0.05) -> torch.Tensor:
        """Color jitter augmentation"""
        return T.ColorJitter(brightness, contrast, saturation, hue)(image)


def create_dataloader(dataset: Dataset,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> DataLoader:
    """Create data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def split_dataset(dataset: Dataset, 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset

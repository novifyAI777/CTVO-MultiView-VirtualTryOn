"""
Stage 2: Cloth Warping Module

Modern modular architecture for multi-view cloth warping.
This module has been migrated from the old CP-VTON style to a clean, modular structure.

New Structure:
- models/: UNet architecture
- datasets/: Dataset loaders for .pth tensors
- trainers/: Training loop with L1 + perceptual loss
- inference/: Inference wrapper for Stage 2 outputs
- utils/: Mask generation, transforms, visualization, config
"""

# New modular imports
from .models import UNet, DoubleConv
from .datasets import Stage2Dataset, Stage2TensorDataset
from .trainers import Stage2Trainer, Stage2Loss, PerceptualLoss
from .inference import Stage2Inference, run_stage2_inference
from .utils import (
    Stage2Config,
    get_train_transforms,
    get_val_transforms,
    load_pose_tensor,
    preprocess_inputs,
    generate_garment_mask,
    generate_garment_mask_from_tensor,
    refine_mask_with_morphology,
    apply_mask_to_image,
    tensor_to_image,
    save_image,
    visualize_batch,
    visualize_output_channels
)

__all__ = [
    # Models
    'UNet',
    'DoubleConv',
    
    # Datasets
    'Stage2Dataset',
    'Stage2TensorDataset',
    
    # Trainers
    'Stage2Trainer',
    'Stage2Loss',
    'PerceptualLoss',
    
    # Inference
    'Stage2Inference',
    'run_stage2_inference',
    
    # Utils
    'Stage2Config',
    'get_train_transforms',
    'get_val_transforms',
    'load_pose_tensor',
    'preprocess_inputs',
    'generate_garment_mask',
    'generate_garment_mask_from_tensor',
    'refine_mask_with_morphology',
    'apply_mask_to_image',
    'tensor_to_image',
    'save_image',
    'visualize_batch',
    'visualize_output_channels',
]

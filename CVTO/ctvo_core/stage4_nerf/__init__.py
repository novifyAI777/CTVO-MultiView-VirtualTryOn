"""
Stage 4: NeRF Module

This module handles:
- Neural Radiance Fields (NeRF) for multi-view rendering
- Volume rendering and camera models
- Multi-view virtual try-on generation
- Training and evaluation pipelines
"""

from .model_nerf import NeRFModel, NeRFMLP, PositionalEncoding, VolumeRenderer
from .renderer import NeRFRenderer, Camera, MultiViewGenerator
from .dataset_nerf import NeRFDataset, MultiViewDataset, SyntheticDataset, create_dataloader
from .train_nerf import Stage4NeRFModule, NeRFLoss, train_nerf
from .eval_multiview import Stage4Evaluator, eval_multiview

__all__ = [
    'NeRFModel',
    'NeRFMLP', 
    'PositionalEncoding',
    'VolumeRenderer',
    'NeRFRenderer',
    'Camera',
    'MultiViewGenerator',
    'NeRFDataset',
    'MultiViewDataset',
    'SyntheticDataset',
    'create_dataloader',
    'Stage4NeRFModule',
    'NeRFLoss',
    'train_nerf',
    'Stage4Evaluator',
    'eval_multiview'
]

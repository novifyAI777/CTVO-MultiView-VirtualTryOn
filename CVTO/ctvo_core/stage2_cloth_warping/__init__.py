"""
Stage 2: Cloth Warping Module

This module handles:
- UNet-based cloth warping
- GMM (Geometric Matching Module) for advanced warping
- Integration with Stage 1 outputs
"""

from .UNet import UNet, DoubleConv
from .GMM import GMM, FeatureExtractor, CorrelationLayer, WarpingLayer
from .utils import (
    pose_to_heatmap, 
    load_warp_model, 
    preprocess_inputs, 
    warp_cloth_once,
    create_agnostic_parsing
)
from .run_warp import Stage2Processor, run_stage2

__all__ = [
    'UNet',
    'DoubleConv', 
    'GMM',
    'FeatureExtractor',
    'CorrelationLayer',
    'WarpingLayer',
    'Stage2Processor',
    'run_stage2',
    'pose_to_heatmap',
    'load_warp_model',
    'preprocess_inputs',
    'warp_cloth_once',
    'create_agnostic_parsing'
]

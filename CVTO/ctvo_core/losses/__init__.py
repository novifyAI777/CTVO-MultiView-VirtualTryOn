"""
Shared Loss Functions

This module contains loss functions that can be used across
different stages of the CTVO pipeline.
"""

from .perceptual_loss import VGGPerceptualLoss, LPIPSLoss, StyleLoss, GradientLoss
from .style_loss import GramMatrix, StyleLoss as StyleLossV2, TextureLoss, ColorLoss, HistogramLoss
from .mask_losses import (
    MaskLoss, 
    ClothingRegionLoss, 
    NonClothingRegionLoss, 
    AdaptiveMaskLoss, 
    MaskConsistencyLoss
)

__all__ = [
    'VGGPerceptualLoss',
    'LPIPSLoss', 
    'StyleLoss',
    'GradientLoss',
    'GramMatrix',
    'StyleLossV2',
    'TextureLoss',
    'ColorLoss',
    'HistogramLoss',
    'MaskLoss',
    'ClothingRegionLoss',
    'NonClothingRegionLoss',
    'AdaptiveMaskLoss',
    'MaskConsistencyLoss'
]

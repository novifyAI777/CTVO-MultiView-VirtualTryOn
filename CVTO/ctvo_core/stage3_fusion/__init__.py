"""
Stage 3: Fusion Module

This module handles:
- Try-on generation using various architectures
- Fusion of multiple inputs (person, cloth, mask, pose)
- Training and evaluation pipelines
"""

from .TryOnGenerator import TryOnGenerator
from .FusionNet import FusionNet, AttentionModule, FusionBlock
from .losses import (
    PerceptualLoss, 
    StyleLoss, 
    MaskLoss, 
    AdversarialLoss, 
    FusionLoss
)
from .train_fusion import Stage3FusionModule, train_fusion
from .eval_fusion import Stage3Evaluator, eval_fusion

__all__ = [
    'TryOnGenerator',
    'FusionNet',
    'AttentionModule',
    'FusionBlock',
    'PerceptualLoss',
    'StyleLoss',
    'MaskLoss',
    'AdversarialLoss',
    'FusionLoss',
    'Stage3FusionModule',
    'train_fusion',
    'Stage3Evaluator',
    'eval_fusion'
]

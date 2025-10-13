"""
Stage 1: Human Parsing & Pose Estimation Module

This module handles:
- Human parsing using ONNX models (ATR/LIP)
- Pose estimation using MobileNet-based OpenPose
- Integration of both parsing and pose outputs
"""

from .model_parsing import HumanParsingModel
from .model_pose import PoseEstimationModel
from .run_pose import Stage1Processor, run_stage1

__all__ = [
    'HumanParsingModel',
    'PoseEstimationModel', 
    'Stage1Processor',
    'run_stage1'
]

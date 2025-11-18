"""
CTVO Core Package - Hybrid Architecture for Virtual Try-On

This package contains the modular implementation of the CTVO pipeline:
- Stage 1: Human Parsing & Pose Estimation
- Stage 2: Cloth Warping
- Stage 3: Fusion Generation
- Stage 4: NeRF-based Multi-view Rendering

Author: CTVO Team
Version: 2.0
"""

__version__ = "2.0.0"
__author__ = "CTVO Team"

# Import main modules for easy access
from .stage1_parsing_pose import *
from .stage2_cloth_warping import *

# Stage 3 and 4 imports are optional (require pytorch_lightning)
# Uncomment if you need Stage 3 or 4 functionality
# from .stage3_fusion import *
# from .stage4_nerf import *
try:
    from .stage3_fusion import *
except ImportError:
    pass  # pytorch_lightning not installed, skip Stage 3

try:
    from .stage4_nerf import *
except ImportError:
    pass  # pytorch_lightning not installed, skip Stage 4

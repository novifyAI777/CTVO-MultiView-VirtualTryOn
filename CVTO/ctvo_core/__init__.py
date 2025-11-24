"""
CTVO Core Package - Hybrid Architecture for Virtual Try-On
"""

__version__ = "2.0.0"
__author__ = "CTVO Team"

# Only import Stage 2 for training - Stage 1 outputs already exist
from .stage2_cloth_warping import *

# Make Stage 1 import optional (only needed if running Stage 1)
try:
    from .stage1_parsing_pose import *
except ImportError:
    pass  # Stage 1 not needed for Stage 2 training

# Stage 3 and 4 optional
try:
    from .stage3_fusion import *
except ImportError:
    pass

try:
    from .stage4_nerf import *
except ImportError:
    pass
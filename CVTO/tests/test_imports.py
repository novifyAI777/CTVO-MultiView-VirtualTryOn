"""
Test Import Validation

This module tests that all imports work correctly across the CTVO pipeline.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported correctly"""
    
    def test_stage1_imports(self):
        """Test Stage 1 imports"""
        try:
            from ctvo_core.stage1_parsing_pose import (
                HumanParsingModel, 
                PoseEstimationModel, 
                Stage1Processor, 
                run_stage1
            )
            self.assertTrue(True, "Stage 1 imports successful")
        except ImportError as e:
            self.fail(f"Stage 1 import failed: {e}")
    
    def test_stage2_imports(self):
        """Test Stage 2 imports"""
        try:
            from ctvo_core.stage2_cloth_warping import (
                UNet, 
                GMM, 
                Stage2Processor, 
                run_stage2,
                pose_to_heatmap,
                load_warp_model
            )
            self.assertTrue(True, "Stage 2 imports successful")
        except ImportError as e:
            self.fail(f"Stage 2 import failed: {e}")
    
    def test_stage3_imports(self):
        """Test Stage 3 imports"""
        try:
            from ctvo_core.stage3_fusion import (
                TryOnGenerator,
                FusionNet,
                Stage3FusionModule,
                train_fusion,
                eval_fusion
            )
            self.assertTrue(True, "Stage 3 imports successful")
        except ImportError as e:
            self.fail(f"Stage 3 import failed: {e}")
    
    def test_stage4_imports(self):
        """Test Stage 4 imports"""
        try:
            from ctvo_core.stage4_nerf import (
                NeRFModel,
                NeRFRenderer,
                Stage4NeRFModule,
                train_nerf,
                eval_multiview
            )
            self.assertTrue(True, "Stage 4 imports successful")
        except ImportError as e:
            self.fail(f"Stage 4 import failed: {e}")
    
    def test_losses_imports(self):
        """Test losses imports"""
        try:
            from ctvo_core.losses import (
                VGGPerceptualLoss,
                StyleLoss,
                MaskLoss,
                ClothingRegionLoss
            )
            self.assertTrue(True, "Losses imports successful")
        except ImportError as e:
            self.fail(f"Losses import failed: {e}")
    
    def test_utils_imports(self):
        """Test utils imports"""
        try:
            from ctvo_core.utils import (
                ImageLoader,
                ImageSaver,
                CTVOLogger,
                ResultVisualizer,
                create_dataloader
            )
            self.assertTrue(True, "Utils imports successful")
        except ImportError as e:
            self.fail(f"Utils import failed: {e}")
    
    def test_core_imports(self):
        """Test core package imports"""
        try:
            import ctvo_core
            self.assertTrue(True, "Core package import successful")
        except ImportError as e:
            self.fail(f"Core package import failed: {e}")


if __name__ == "__main__":
    unittest.main()

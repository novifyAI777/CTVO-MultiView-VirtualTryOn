"""
Test Loss Functions

This module tests the loss functions used in the CTVO pipeline.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ctvo_core.losses import (
    VGGPerceptualLoss,
    StyleLoss,
    MaskLoss,
    ClothingRegionLoss,
    AdaptiveMaskLoss
)


class TestLosses(unittest.TestCase):
    """Test loss functions"""
    
    def setUp(self):
        """Set up test data"""
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.channels = 3
        
        # Create dummy images
        self.pred_img = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.target_img = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # Create dummy mask
        self.mask = torch.ones(self.batch_size, 1, self.height, self.width)
        
        # Create dummy parsing map
        self.parsing_map = torch.randint(0, 20, (self.batch_size, 1, self.height, self.width))
    
    def test_vgg_perceptual_loss(self):
        """Test VGG perceptual loss"""
        loss_fn = VGGPerceptualLoss()
        
        loss = loss_fn(self.pred_img, self.target_img)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_style_loss(self):
        """Test style loss"""
        loss_fn = StyleLoss()
        
        loss = loss_fn(self.pred_img, self.target_img)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_mask_loss(self):
        """Test mask loss"""
        loss_fn = MaskLoss()
        
        loss = loss_fn(self.pred_img, self.target_img, self.mask)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_clothing_region_loss(self):
        """Test clothing region loss"""
        loss_fn = ClothingRegionLoss()
        
        loss = loss_fn(self.pred_img, self.target_img, self.parsing_map)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_adaptive_mask_loss(self):
        """Test adaptive mask loss"""
        loss_fn = AdaptiveMaskLoss()
        
        loss = loss_fn(self.pred_img, self.target_img, self.parsing_map)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_loss_gradients(self):
        """Test that losses have gradients"""
        loss_fn = VGGPerceptualLoss()
        
        # Enable gradients
        self.pred_img.requires_grad_(True)
        
        loss = loss_fn(self.pred_img, self.target_img)
        loss.backward()
        
        self.assertIsNotNone(self.pred_img.grad)
        self.assertNotEqual(self.pred_img.grad.sum().item(), 0)
    
    def test_loss_device_consistency(self):
        """Test that losses work on different devices"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
            pred_cuda = self.pred_img.to(device)
            target_cuda = self.target_img.to(device)
            
            loss_fn = VGGPerceptualLoss()
            loss = loss_fn(pred_cuda, target_cuda)
            
            self.assertEqual(loss.device, device)
    
    def test_loss_shapes(self):
        """Test that losses handle different input shapes"""
        # Test single image
        single_pred = self.pred_img[0:1]
        single_target = self.target_img[0:1]
        
        loss_fn = VGGPerceptualLoss()
        loss = loss_fn(single_pred, single_target)
        
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Test different image sizes
        small_pred = torch.randn(1, 3, 32, 32)
        small_target = torch.randn(1, 3, 32, 32)
        
        loss = loss_fn(small_pred, small_target)
        self.assertEqual(loss.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()

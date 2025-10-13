"""
Shared Mask Loss Functions

This module contains mask-aware loss functions for clothing regions
in virtual try-on applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    """Mask-aware loss for clothing regions"""
    
    def __init__(self, loss_type='l1'):
        super(MaskLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, target, mask):
        """
        Compute mask-weighted loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            mask: clothing mask [B, 1, H, W] or [B, 3, H, W]
            
        Returns:
            mask-weighted loss
        """
        # Ensure mask has same number of channels as images
        if mask.size(1) == 1:
            mask = mask.expand_as(pred)
        
        # Compute loss
        loss = self.loss_fn(pred, target)
        
        # Apply mask weighting
        masked_loss = loss * mask
        
        # Compute mean loss
        return masked_loss.mean()


class ClothingRegionLoss(nn.Module):
    """Loss specifically for clothing regions"""
    
    def __init__(self, clothing_labels=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
        super(ClothingRegionLoss, self).__init__()
        self.clothing_labels = clothing_labels
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def create_clothing_mask(self, parsing_map):
        """
        Create binary mask for clothing regions
        
        Args:
            parsing_map: parsing map [B, 1, H, W] with label values
            
        Returns:
            clothing mask [B, 1, H, W]
        """
        mask = torch.zeros_like(parsing_map)
        
        for label in self.clothing_labels:
            mask[parsing_map == label] = 1
        
        return mask
    
    def forward(self, pred, target, parsing_map):
        """
        Compute clothing region loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            parsing_map: parsing map [B, 1, H, W]
            
        Returns:
            clothing region loss
        """
        # Create clothing mask
        clothing_mask = self.create_clothing_mask(parsing_map)
        clothing_mask = clothing_mask.expand_as(pred)
        
        # Compute loss
        loss = self.l1_loss(pred, target)
        
        # Apply clothing mask
        clothing_loss = loss * clothing_mask
        
        # Normalize by mask area
        mask_area = clothing_mask.sum()
        if mask_area > 0:
            clothing_loss = clothing_loss.sum() / mask_area
        else:
            clothing_loss = torch.tensor(0.0, device=pred.device)
        
        return clothing_loss


class NonClothingRegionLoss(nn.Module):
    """Loss for non-clothing regions (background preservation)"""
    
    def __init__(self, clothing_labels=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
        super(NonClothingRegionLoss, self).__init__()
        self.clothing_labels = clothing_labels
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def create_non_clothing_mask(self, parsing_map):
        """
        Create binary mask for non-clothing regions
        
        Args:
            parsing_map: parsing map [B, 1, H, W] with label values
            
        Returns:
            non-clothing mask [B, 1, H, W]
        """
        mask = torch.ones_like(parsing_map)
        
        for label in self.clothing_labels:
            mask[parsing_map == label] = 0
        
        return mask
    
    def forward(self, pred, target, parsing_map):
        """
        Compute non-clothing region loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            parsing_map: parsing map [B, 1, H, W]
            
        Returns:
            non-clothing region loss
        """
        # Create non-clothing mask
        non_clothing_mask = self.create_non_clothing_mask(parsing_map)
        non_clothing_mask = non_clothing_mask.expand_as(pred)
        
        # Compute loss
        loss = self.l1_loss(pred, target)
        
        # Apply non-clothing mask
        non_clothing_loss = loss * non_clothing_mask
        
        # Normalize by mask area
        mask_area = non_clothing_mask.sum()
        if mask_area > 0:
            non_clothing_loss = non_clothing_loss.sum() / mask_area
        else:
            non_clothing_loss = torch.tensor(0.0, device=pred.device)
        
        return non_clothing_loss


class AdaptiveMaskLoss(nn.Module):
    """Adaptive mask loss with different weights for different regions"""
    
    def __init__(self, 
                 clothing_weight=1.0,
                 non_clothing_weight=0.1,
                 clothing_labels=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
        super(AdaptiveMaskLoss, self).__init__()
        self.clothing_weight = clothing_weight
        self.non_clothing_weight = non_clothing_weight
        self.clothing_labels = clothing_labels
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def create_adaptive_mask(self, parsing_map):
        """
        Create adaptive mask with different weights
        
        Args:
            parsing_map: parsing map [B, 1, H, W] with label values
            
        Returns:
            adaptive mask [B, 1, H, W]
        """
        mask = torch.full_like(parsing_map, self.non_clothing_weight)
        
        for label in self.clothing_labels:
            mask[parsing_map == label] = self.clothing_weight
        
        return mask
    
    def forward(self, pred, target, parsing_map):
        """
        Compute adaptive mask loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            parsing_map: parsing map [B, 1, H, W]
            
        Returns:
            adaptive mask loss
        """
        # Create adaptive mask
        adaptive_mask = self.create_adaptive_mask(parsing_map)
        adaptive_mask = adaptive_mask.expand_as(pred)
        
        # Compute loss
        loss = self.l1_loss(pred, target)
        
        # Apply adaptive mask
        adaptive_loss = loss * adaptive_mask
        
        # Compute mean loss
        return adaptive_loss.mean()


class MaskConsistencyLoss(nn.Module):
    """Loss for mask consistency between predicted and target"""
    
    def __init__(self):
        super(MaskConsistencyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_mask, target_mask):
        """
        Compute mask consistency loss
        
        Args:
            pred_mask: predicted mask [B, 1, H, W]
            target_mask: target mask [B, 1, H, W]
            
        Returns:
            mask consistency loss
        """
        return self.bce_loss(pred_mask, target_mask)

"""
Stage 2 Loss Functions

L1 loss and perceptual loss for training UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    """
    
    def __init__(self, 
                 feature_layers: Optional[list] = None,
                 weights: Optional[list] = None):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: list of layer indices to use for loss
            weights: weights for each layer
        """
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        try:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
            self.feature_extractor = vgg.features
        except:
            # Fallback to manual loading
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True)
            self.feature_extractor = vgg.features
        
        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Default layers: conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
        if feature_layers is None:
            feature_layers = [1, 6, 11, 20, 29]
        if weights is None:
            weights = [1.0] * len(feature_layers)
        
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Normalize input to ImageNet stats
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input to ImageNet stats"""
        return (x - self.mean) / self.std
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: predicted image [B, 3, H, W] in range [0, 1]
            target: target image [B, 3, H, W] in range [0, 1]
            
        Returns:
            perceptual loss scalar
        """
        # Normalize to ImageNet stats
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred_norm
        x_target = target_norm
        
        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # Compute L2 loss for each layer
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class Stage2Loss(nn.Module):
    """
    Combined loss for Stage 2 training.
    
    Loss = L1_weight * L1_loss + perceptual_weight * Perceptual_loss
    """
    
    def __init__(self,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 10.0,
                 feature_layers: Optional[list] = None,
                 weights: Optional[list] = None):
        """
        Initialize combined loss.
        
        Args:
            l1_weight: weight for L1 loss
            perceptual_weight: weight for perceptual loss
            feature_layers: VGG layers for perceptual loss
            weights: weights for each VGG layer
        """
        super(Stage2Loss, self).__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(feature_layers, weights)
        else:
            self.perceptual_loss = None
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                pred_rgb_only: bool = True) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred: model output [B, C, H, W]
            target: target warped cloth [B, 3, H, W]
            pred_rgb_only: if True, only use first 3 channels of pred for loss
            
        Returns:
            dictionary with individual and total losses
        """
        # Extract RGB from prediction (first 3 channels)
        if pred_rgb_only and pred.shape[1] > 3:
            pred_rgb = pred[:, :3]
        else:
            pred_rgb = pred
        
        # Ensure target is in [0, 1] range
        target = torch.clamp(target, 0, 1)
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        
        # L1 loss
        l1 = self.l1_loss(pred_rgb, target)
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred_rgb, target)
        else:
            perceptual = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total = self.l1_weight * l1 + self.perceptual_weight * perceptual
        
        return {
            'total': total,
            'l1': l1,
            'perceptual': perceptual,
        }


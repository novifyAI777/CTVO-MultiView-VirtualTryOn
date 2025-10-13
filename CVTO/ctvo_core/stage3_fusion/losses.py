"""
Stage 3: Loss Functions for Fusion Training

This module contains loss functions specific to Stage 3 fusion training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features
        self.feature_layers = feature_layers
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            perceptual loss
        """
        pred_features = []
        target_features = []
        
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_features.append(pred)
                target_features.append(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


class StyleLoss(nn.Module):
    """Style loss for texture preservation"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features[:16]  # Use first 16 layers
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, x):
        """Compute Gram matrix for style representation"""
        B, C, H, W = x.size()
        x = x.view(B, C, H * W)
        gram = torch.bmm(x, x.transpose(1, 2))
        return gram / (C * H * W)
    
    def forward(self, pred, target):
        """
        Compute style loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            style loss
        """
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        pred_gram = self.gram_matrix(pred_features)
        target_gram = self.gram_matrix(target_features)
        
        return F.mse_loss(pred_gram, target_gram)


class MaskLoss(nn.Module):
    """Mask-aware loss for clothing regions"""
    
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, pred, target, mask):
        """
        Compute mask-weighted loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            mask: clothing mask [B, 1, H, W]
            
        Returns:
            mask-weighted loss
        """
        # Expand mask to 3 channels
        mask = mask.expand_as(pred)
        
        # Compute L1 loss weighted by mask
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * mask
        
        return loss.mean()


class AdversarialLoss(nn.Module):
    """Adversarial loss for discriminator"""
    
    def __init__(self, gan_mode='lsgan'):
        super(AdversarialLoss, self).__init__()
        self.gan_mode = gan_mode
        
    def forward(self, pred, is_real):
        """
        Compute adversarial loss
        
        Args:
            pred: discriminator prediction
            is_real: whether input is real (True) or fake (False)
            
        Returns:
            adversarial loss
        """
        if self.gan_mode == 'lsgan':
            if is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            return F.mse_loss(pred, target)
        
        elif self.gan_mode == 'wgangp':
            if is_real:
                return -pred.mean()
            else:
                return pred.mean()
        
        else:
            raise ValueError(f"Unknown GAN mode: {self.gan_mode}")


class FusionLoss(nn.Module):
    """Combined loss for Stage 3 fusion training"""
    
    def __init__(self, 
                 lambda_l1=1.0,
                 lambda_perceptual=10.0,
                 lambda_style=1.0,
                 lambda_mask=5.0,
                 lambda_adv=1.0):
        super(FusionLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_mask = lambda_mask
        self.lambda_adv = lambda_adv
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.mask_loss = MaskLoss()
        self.adv_loss = AdversarialLoss()
    
    def forward(self, pred, target, mask=None, disc_pred=None, is_real=None):
        """
        Compute combined fusion loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            mask: clothing mask [B, 1, H, W] (optional)
            disc_pred: discriminator prediction (optional)
            is_real: whether target is real (optional)
            
        Returns:
            dict of individual losses and total loss
        """
        losses = {}
        
        # L1 loss
        losses['l1'] = self.l1_loss(pred, target)
        
        # Perceptual loss
        losses['perceptual'] = self.perceptual_loss(pred, target)
        
        # Style loss
        losses['style'] = self.style_loss(pred, target)
        
        # Mask loss (if mask provided)
        if mask is not None:
            losses['mask'] = self.mask_loss(pred, target, mask)
        else:
            losses['mask'] = torch.tensor(0.0, device=pred.device)
        
        # Adversarial loss (if discriminator prediction provided)
        if disc_pred is not None and is_real is not None:
            losses['adv'] = self.adv_loss(disc_pred, is_real)
        else:
            losses['adv'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (self.lambda_l1 * losses['l1'] +
                     self.lambda_perceptual * losses['perceptual'] +
                     self.lambda_style * losses['style'] +
                     self.lambda_mask * losses['mask'] +
                     self.lambda_adv * losses['adv'])
        
        losses['total'] = total_loss
        
        return losses

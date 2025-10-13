"""
Shared Perceptual Loss Functions

This module contains perceptual loss functions that can be used
across different stages of the CTVO pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, 
                 feature_layers=[3, 8, 15, 22],
                 weights=[1.0, 1.0, 1.0, 1.0],
                 normalize=True):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.features = vgg
        self.feature_layers = feature_layers
        self.weights = weights
        self.normalize = normalize
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            perceptual loss
        """
        if self.normalize:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        pred_features = []
        target_features = []
        
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            
            if i in self.feature_layers:
                pred_features.append(pred)
                target_features.append(target)
        
        loss = 0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity Loss"""
    
    def __init__(self, net='alex'):
        super(LPIPSLoss, self).__init__()
        # This is a simplified implementation
        # In practice, you would use a pre-trained LPIPS model
        self.net = net
        self.vgg_loss = VGGPerceptualLoss()
    
    def forward(self, pred, target):
        """Compute LPIPS loss"""
        # Simplified implementation using VGG perceptual loss
        return self.vgg_loss(pred, target)


class StyleLoss(nn.Module):
    """Style loss for texture preservation"""
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(StyleLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.features = vgg
        self.feature_layers = feature_layers
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
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
        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
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
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.mse_loss(pred_gram, target_gram)
        
        return loss / len(pred_features)


class GradientLoss(nn.Module):
    """Gradient loss for edge preservation"""
    
    def __init__(self):
        super(GradientLoss, self).__init__()
        
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        """
        Compute gradient loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            gradient loss
        """
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        # Compute gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Compute loss
        loss = F.mse_loss(pred_grad_mag, target_grad_mag)
        
        return loss

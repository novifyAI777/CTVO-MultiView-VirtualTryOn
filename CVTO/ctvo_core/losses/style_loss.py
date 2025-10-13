"""
Shared Style Loss Functions

This module contains style loss functions for texture and style preservation
in virtual try-on applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GramMatrix(nn.Module):
    """Gram matrix computation for style representation"""
    
    def __init__(self):
        super(GramMatrix, self).__init__()
    
    def forward(self, x):
        """
        Compute Gram matrix
        
        Args:
            x: feature map [B, C, H, W]
            
        Returns:
            Gram matrix [B, C, C]
        """
        B, C, H, W = x.size()
        x = x.view(B, C, H * W)
        gram = torch.bmm(x, x.transpose(1, 2))
        return gram / (C * H * W)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""
    
    def __init__(self, 
                 feature_layers=[3, 8, 15, 22],
                 weights=[1.0, 1.0, 1.0, 1.0]):
        super(StyleLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.features = vgg
        self.feature_layers = feature_layers
        self.weights = weights
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Gram matrix computation
        self.gram_matrix = GramMatrix()
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
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
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += weight * F.mse_loss(pred_gram, target_gram)
        
        return loss / len(pred_features)


class TextureLoss(nn.Module):
    """Texture loss for preserving fabric textures"""
    
    def __init__(self, patch_size=3):
        super(TextureLoss, self).__init__()
        self.patch_size = patch_size
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.features = vgg
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def extract_patches(self, x):
        """Extract patches from feature map"""
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=self.patch_size, stride=1, padding=self.patch_size//2)
        patches = patches.view(B, C, self.patch_size, self.patch_size, -1)
        return patches
    
    def forward(self, pred, target):
        """
        Compute texture loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            texture loss
        """
        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features
        pred_feat = self.features[:15](pred)  # Use first 15 layers
        target_feat = self.features[:15](target)
        
        # Extract patches
        pred_patches = self.extract_patches(pred_feat)
        target_patches = self.extract_patches(target_feat)
        
        # Compute patch similarity
        pred_patches = pred_patches.view(pred_patches.size(0), -1, pred_patches.size(-1))
        target_patches = target_patches.view(target_patches.size(0), -1, target_patches.size(-1))
        
        # Compute cosine similarity
        pred_norm = F.normalize(pred_patches, p=2, dim=1)
        target_norm = F.normalize(target_patches, p=2, dim=1)
        
        similarity = torch.bmm(pred_norm.transpose(1, 2), target_norm)
        loss = 1 - similarity.mean()
        
        return loss


class ColorLoss(nn.Module):
    """Color loss for preserving color consistency"""
    
    def __init__(self):
        super(ColorLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Compute color loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            color loss
        """
        # Compute mean color for each image
        pred_mean = torch.mean(pred, dim=(2, 3))  # [B, 3]
        target_mean = torch.mean(target, dim=(2, 3))  # [B, 3]
        
        # Compute color loss
        color_loss = F.mse_loss(pred_mean, target_mean)
        
        return color_loss


class HistogramLoss(nn.Module):
    """Histogram loss for color distribution matching"""
    
    def __init__(self, num_bins=256):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins
    
    def compute_histogram(self, x):
        """Compute histogram for each channel"""
        B, C, H, W = x.size()
        histograms = []
        
        for c in range(C):
            channel = x[:, c].view(B, -1)  # [B, H*W]
            
            # Compute histogram
            hist = torch.zeros(B, self.num_bins, device=x.device)
            for b in range(B):
                hist[b] = torch.histc(channel[b], bins=self.num_bins, min=0, max=1)
            
            histograms.append(hist)
        
        return torch.stack(histograms, dim=1)  # [B, C, num_bins]
    
    def forward(self, pred, target):
        """
        Compute histogram loss
        
        Args:
            pred: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
            
        Returns:
            histogram loss
        """
        # Compute histograms
        pred_hist = self.compute_histogram(pred)
        target_hist = self.compute_histogram(target)
        
        # Normalize histograms
        pred_hist = pred_hist / (pred_hist.sum(dim=2, keepdim=True) + 1e-8)
        target_hist = target_hist / (target_hist.sum(dim=2, keepdim=True) + 1e-8)
        
        # Compute histogram loss
        hist_loss = F.mse_loss(pred_hist, target_hist)
        
        return hist_loss

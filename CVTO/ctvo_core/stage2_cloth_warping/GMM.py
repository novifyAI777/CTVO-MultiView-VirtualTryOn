"""
Stage 2: GMM (Geometric Matching Module) for Cloth Warping

This module implements the Geometric Matching Module for cloth warping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Feature extractor for cloth and person images"""
    
    def __init__(self, in_channels=3, out_channels=256):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class CorrelationLayer(nn.Module):
    """Correlation layer for feature matching"""
    
    def __init__(self, max_displacement=4):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        
    def forward(self, x1, x2):
        """
        Compute correlation between two feature maps
        Args:
            x1: cloth features [B, C, H, W]
            x2: person features [B, C, H, W]
        Returns:
            correlation map [B, (2*max_displacement+1)^2, H, W]
        """
        B, C, H, W = x1.size()
        
        # Pad x2 for correlation
        pad_size = self.max_displacement
        x2_padded = F.pad(x2, (pad_size, pad_size, pad_size, pad_size))
        
        # Compute correlation
        correlation_maps = []
        for i in range(2 * self.max_displacement + 1):
            for j in range(2 * self.max_displacement + 1):
                x2_shifted = x2_padded[:, :, i:i+H, j:j+W]
                corr = torch.sum(x1 * x2_shifted, dim=1, keepdim=True)
                correlation_maps.append(corr)
        
        correlation = torch.cat(correlation_maps, dim=1)
        return correlation


class GMM(nn.Module):
    """Geometric Matching Module for cloth warping"""
    
    def __init__(self, in_channels=3, max_displacement=4):
        super(GMM, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.correlation_layer = CorrelationLayer(max_displacement)
        
        # Flow estimation network
        corr_channels = (2 * max_displacement + 1) ** 2
        self.flow_conv = nn.Sequential(
            nn.Conv2d(corr_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)  # 2 for u, v flow
        )
        
    def forward(self, cloth_img, person_img):
        """
        Compute geometric transformation for cloth warping
        Args:
            cloth_img: cloth image [B, 3, H, W]
            person_img: person image [B, 3, H, W]
        Returns:
            flow: optical flow [B, 2, H, W]
        """
        # Extract features
        cloth_features = self.feature_extractor(cloth_img)
        person_features = self.feature_extractor(person_img)
        
        # Compute correlation
        correlation = self.correlation_layer(cloth_features, person_features)
        
        # Estimate flow
        flow = self.flow_conv(correlation)
        
        return flow


class WarpingLayer(nn.Module):
    """Layer for applying geometric transformation"""
    
    def __init__(self):
        super(WarpingLayer, self).__init__()
        
    def forward(self, cloth_img, flow):
        """
        Apply geometric transformation to cloth image
        Args:
            cloth_img: cloth image [B, 3, H, W]
            flow: optical flow [B, 2, H, W]
        Returns:
            warped_cloth: warped cloth image [B, 3, H, W]
        """
        B, C, H, W = cloth_img.size()
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=cloth_img.device),
            torch.arange(W, device=cloth_img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow to grid
        warped_grid = grid + flow
        
        # Normalize to [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # Permute for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Apply transformation
        warped_cloth = F.grid_sample(
            cloth_img, warped_grid, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_cloth

"""
Stage 3: Fusion Network

This module implements the fusion network that combines multiple inputs
for high-quality virtual try-on generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """Attention module for feature fusion"""
    
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Apply self-attention mechanism
        
        Args:
            x: input feature map [B, C, H, W]
            
        Returns:
            attended feature map [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Project to query, key, value
        proj = self.conv(x)
        query = self.query_conv(proj).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(proj).view(B, -1, H * W)
        value = self.value_conv(proj).view(B, -1, H * W)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection
        return self.gamma * out + x


class FusionBlock(nn.Module):
    """Fusion block for combining multiple feature maps"""
    
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.attention = AttentionModule(out_channels)
        
    def forward(self, x1, x2):
        """
        Fuse two feature maps
        
        Args:
            x1: first feature map [B, C1, H, W]
            x2: second feature map [B, C2, H, W]
            
        Returns:
            fused feature map [B, C_out, H, W]
        """
        # Project to same channel dimension
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        
        # Concatenate and fuse
        fused = torch.cat([f1, f2], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.norm(fused)
        fused = F.relu(fused)
        
        # Apply attention
        fused = self.attention(fused)
        
        return fused


class FusionNet(nn.Module):
    """Main fusion network for Stage 3"""
    
    def __init__(self, 
                 person_channels=3,
                 cloth_channels=3, 
                 mask_channels=3,
                 pose_channels=3,
                 out_channels=3):
        super(FusionNet, self).__init__()
        
        # Feature extractors
        self.person_extractor = nn.Sequential(
            nn.Conv2d(person_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.cloth_extractor = nn.Sequential(
            nn.Conv2d(cloth_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.mask_extractor = nn.Sequential(
            nn.Conv2d(mask_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pose_extractor = nn.Sequential(
            nn.Conv2d(pose_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion blocks
        self.fusion1 = FusionBlock(128, 128, 128)  # person + cloth
        self.fusion2 = FusionBlock(128, 64, 128)    # fused + mask
        self.fusion3 = FusionBlock(128, 64, 128)    # fused + pose
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, person_img, warped_cloth, mask, pose_heatmap):
        """
        Fuse multiple inputs for try-on generation
        
        Args:
            person_img: person image [B, 3, H, W]
            warped_cloth: warped cloth image [B, 3, H, W]
            mask: clothing mask [B, 3, H, W]
            pose_heatmap: pose heatmap [B, 3, H, W]
            
        Returns:
            fused result [B, 3, H, W]
        """
        # Extract features
        person_feat = self.person_extractor(person_img)
        cloth_feat = self.cloth_extractor(warped_cloth)
        mask_feat = self.mask_extractor(mask)
        pose_feat = self.pose_extractor(pose_heatmap)
        
        # Fuse features progressively
        fused1 = self.fusion1(person_feat, cloth_feat)
        fused2 = self.fusion2(fused1, mask_feat)
        fused3 = self.fusion3(fused2, pose_feat)
        
        # Decode to final result
        output = self.decoder(fused3)
        
        return output

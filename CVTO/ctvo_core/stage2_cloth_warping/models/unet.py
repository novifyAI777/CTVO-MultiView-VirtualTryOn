"""
Stage 2: UNet Model for Cloth Warping

Modern UNet architecture for multi-view cloth warping.
Input: person_rgb [3] + parsing_map [K] + pose_heatmap [P] + cloth_rgb [3] + cloth_mask [1]
Output: warped cloth RGB [3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with batch normalization and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet architecture for cloth warping.
    
    Input channels: 3 (person_rgb) + K (parsing_map) + P (pose_heatmap) + 3 (cloth_rgb) + 1 (cloth_mask)
    Output channels: 3 (warped cloth RGB)
    
    Args:
        in_channels: total input channels (3 + K + P + 3 + 1)
        out_channels: output channels (3 for RGB)
        base_channels: number of base channels in first layer
    """
    
    def __init__(self, 
                 in_channels: int = 3 + 19 + 18 + 3 + 1,  # 3+K+P+3+1 = 44 default
                 out_channels: int = 3,
                 base_channels: int = 64):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, in_channels, H, W]
                - x[:, :3]: person_rgb
                - x[:, 3:3+K]: parsing_map
                - x[:, 3+K:3+K+P]: pose_heatmap
                - x[:, 3+K+P:3+K+P+3]: cloth_rgb
                - x[:, -1:]: cloth_mask
            
        Returns:
            output: [B, 3, H, W] - warped cloth RGB
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = self._crop_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._crop_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._crop_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._crop_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        output = self.final_conv(dec1)
        
        # Apply tanh activation to get values in [-1, 1], then normalize to [0, 1]
        output = torch.tanh(output)
        output = (output + 1.0) / 2.0  # Normalize to [0, 1]
        
        return output
    
    def _crop_and_concat(self, x1, x2):
        """Crop x1 to match x2 size and concatenate along channel dimension"""
        # Handle size mismatch
        if x1.size() != x2.size():
            x1 = self._center_crop(x1, x2.size())
        return torch.cat([x2, x1], dim=1)
    
    def _center_crop(self, tensor, target_size):
        """Center crop tensor to match target_size"""
        _, _, h, w = tensor.size()
        _, _, th, tw = target_size
        
        if h > th:
            start_h = (h - th) // 2
            tensor = tensor[:, :, start_h:start_h + th, :]
        elif h < th:
            pad_h = (th - h) // 2
            tensor = F.pad(tensor, (0, 0, pad_h, th - h - pad_h))
            
        if w > tw:
            start_w = (w - tw) // 2
            tensor = tensor[:, :, :, start_w:start_w + tw]
        elif w < tw:
            pad_w = (tw - w) // 2
            tensor = F.pad(tensor, (pad_w, tw - w - pad_w, 0, 0))
            
        return tensor

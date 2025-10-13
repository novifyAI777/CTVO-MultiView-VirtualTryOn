"""
Stage 3: Try-On Generator

This module implements the main try-on generation network that combines
warped cloth with person images to create realistic virtual try-on results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += self.shortcut(residual)
        return F.relu(out)


class TryOnGenerator(nn.Module):
    """Main try-on generation network"""
    
    def __init__(self, input_channels=9, output_channels=3):  # person(3) + warped_cloth(3) + mask(3)
        super(TryOnGenerator, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = ResidualBlock(64, 128, stride=2)
        self.encoder3 = ResidualBlock(128, 256, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # Decoder
        self.decoder1 = ResidualBlock(256, 128, stride=1)
        self.decoder2 = ResidualBlock(128, 64, stride=1)
        
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, person_img, warped_cloth, mask):
        """
        Generate try-on result
        
        Args:
            person_img: person image [B, 3, H, W]
            warped_cloth: warped cloth image [B, 3, H, W]
            mask: clothing mask [B, 3, H, W]
            
        Returns:
            try_on_result: generated try-on image [B, 3, H, W]
        """
        # Concatenate inputs
        x = torch.cat([person_img, warped_cloth, mask], dim=1)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bottleneck
        bottleneck = self.bottleneck(e3)
        
        # Decoder with skip connections
        d1 = self.decoder1(bottleneck + e3)
        d2 = self.decoder2(d1 + e2)
        
        # Final output
        output = self.final_conv(d2 + e1)
        
        return output

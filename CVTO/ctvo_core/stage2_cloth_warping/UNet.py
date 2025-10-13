"""
Stage 2: UNet Model for Cloth Warping

This module contains the UNet architecture used for cloth warping in Stage 2
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels):  
        super(DoubleConv, self).__init__() 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """UNet architecture for cloth warping"""
    def __init__(self, in_channels=12, out_channels=3):  
        super(UNet, self).__init__()  
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        # Handle size mismatch by cropping or padding
        if dec4.size() != enc4.size():
            dec4 = self._center_crop(dec4, enc4.size())
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.size() != enc3.size():
            dec3 = self._center_crop(dec3, enc3.size())
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.size() != enc2.size():
            dec2 = self._center_crop(dec2, enc2.size())
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.size() != enc1.size():
            dec1 = self._center_crop(dec1, enc1.size())
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))  
    
    def _center_crop(self, tensor, target_size):
        """Center crop tensor to match target_size"""
        _, _, h, w = tensor.size()
        _, _, th, tw = target_size
        
        if h > th:
            start_h = (h - th) // 2
            tensor = tensor[:, :, start_h:start_h + th, :]
        elif h < th:
            pad_h = (th - h) // 2
            tensor = nn.functional.pad(tensor, (0, 0, pad_h, th - h - pad_h))
            
        if w > tw:
            start_w = (w - tw) // 2
            tensor = tensor[:, :, :, start_w:start_w + tw]
        elif w < tw:
            pad_w = (tw - w) // 2
            tensor = nn.functional.pad(tensor, (pad_w, tw - w - pad_w, 0, 0))
            
        return tensor

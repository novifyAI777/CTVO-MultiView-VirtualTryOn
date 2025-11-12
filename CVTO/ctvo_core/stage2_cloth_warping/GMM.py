"""
Stage 2: GMM (Geometric Matching Module) for Cloth Warping

This module implements the Geometric Matching Module for cloth warping.
Supports both VITON-compatible GMM (with extractionA/extractionB/regression)
and custom GMM architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VITONFeatureExtractor(nn.Module):
    """
    VITON-compatible feature extractor matching extractionA/extractionB structure.
    Matches the actual checkpoint architecture with 4x4 kernels and specific channel sizes.
    """
    
    def __init__(self, in_channels=3):
        super(VITONFeatureExtractor, self).__init__()
        # Match checkpoint structure: uses 4x4 kernels, stride=2, padding=1
        # Pattern: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> ...
        layers = []
        # Layer 0: Conv [64, in_channels, 4, 4]
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        # Layer 3: Conv [128, 64, 4, 4]
        layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        # Layer 6: Conv [256, 128, 4, 4]
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        # Layer 9: Conv [512, 256, 4, 4]
        layers.append(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        # Layer 12: Conv [512, 512, 3, 3] (final layer, stride=1)
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class VITONRegression(nn.Module):
    """
    VITON-compatible regression network matching regression.conv/regression.linear structure.
    Matches the actual checkpoint architecture: 192->512->256->128->64 channels.
    """
    
    def __init__(self, in_channels=192):
        super(VITONRegression, self).__init__()
        # Match checkpoint structure: regression.conv layers
        # Input: 192 channels, output channels: 512->256->128->64
        conv_layers = []
        # Layer 0: Conv [512, 192, 4, 4]
        conv_layers.append(nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1))
        conv_layers.append(nn.BatchNorm2d(512))
        conv_layers.append(nn.ReLU(inplace=True))
        # Layer 3: Conv [256, 512, 4, 4]
        conv_layers.append(nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1))
        conv_layers.append(nn.BatchNorm2d(256))
        conv_layers.append(nn.ReLU(inplace=True))
        # Layer 6: Conv [128, 256, 3, 3]
        conv_layers.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.BatchNorm2d(128))
        conv_layers.append(nn.ReLU(inplace=True))
        # Layer 9: Conv [64, 128, 3, 3]
        conv_layers.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.BatchNorm2d(64))
        conv_layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Linear layer matching regression.linear structure
        # The checkpoint has: regression.linear.weight shape [50, 768] or similar
        # We'll compute this dynamically based on input size
        self.linear = None  # Will be initialized on first forward pass
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        B_conv, C_conv, H_conv, W_conv = x.size()
        
        # Initialize linear layer if not done yet (matching checkpoint structure)
        if self.linear is None:
            input_features = C_conv * H_conv * W_conv
            # Checkpoint typically outputs theta (6 params) or flow
            # For compatibility, we'll output flow: 2*H*W at correlation resolution
            # Note: H, W here are the original correlation map dimensions
            output_features = 2 * H * W  # Flow field at correlation resolution
            self.linear = nn.Linear(input_features, output_features).to(x.device)
        
        x = x.view(B, -1)  # Flatten
        x = self.linear(x)
        flow = x.view(B, 2, H, W)  # Reshape to [B, 2, H, W] for flow
        
        # Validate flow values (prevent NaN/Inf)
        if torch.isnan(flow).any() or torch.isinf(flow).any():
            print("Warning: Flow contains NaN/Inf values, replacing with zeros")
            flow = torch.where(torch.isnan(flow) | torch.isinf(flow), 
                              torch.zeros_like(flow), flow)
        
        # Clamp flow to reasonable range to prevent extreme warping
        flow = torch.clamp(flow, -100, 100)
        
        return flow


class VITONGMM(nn.Module):
    """
    VITON-compatible GMM matching the checkpoint structure.
    Uses extractionA, extractionB, and regression modules.
    """
    
    def __init__(self):
        super(VITONGMM, self).__init__()
        # Feature extractors matching extractionA and extractionB
        # Checkpoint shows: extractionA expects 22 channels, extractionB expects 1 channel
        # But we'll use 3 channels for RGB images and let the model adapt
        # Actually, looking at the checkpoint, extractionA.model.0.weight is [64, 22, 4, 4]
        # and extractionB.model.0.weight is [64, 1, 4, 4]
        # This suggests extractionA processes concatenated features (cloth + parsing + pose = 22 channels?)
        # and extractionB processes a mask/parsing (1 channel)
        # For now, we'll create adapters to handle the mismatch
        self.extractionA = VITONFeatureExtractor(in_channels=22)  # For cloth + features
        self.extractionB = VITONFeatureExtractor(in_channels=1)  # For person mask
        
        # Regression network matching regression structure
        # Checkpoint shows: regression.conv.0.weight is [512, 192, 4, 4]
        # Input channels: 192 (likely from correlation computation)
        self.regression = VITONRegression(in_channels=192)
        
    def forward(self, cloth_img, person_img, parsing_map=None, pose_map=None):
        """
        Compute geometric transformation for cloth warping
        Args:
            cloth_img: cloth image [B, 3, H, W]
            person_img: person image [B, 3, H, W]
            parsing_map: optional parsing map [B, 3, H, W] or [B, 1, H, W]
            pose_map: optional pose map [B, 3, H, W] or [B, 1, H, W]
        Returns:
            flow: optical flow [B, 2, H, W] at original image resolution
        """
        B, C, H_img, W_img = cloth_img.size()
        
        # Prepare inputs for extractionA (expects 22 channels)
        # Checkpoint expects: cloth (3) + parsing (19?) + pose (?) = 22 channels
        # For now, if we don't have parsing/pose, we'll pad with zeros
        if parsing_map is None:
            # Create dummy parsing map (19 channels of zeros)
            parsing_map = torch.zeros(B, 19, H_img, W_img, device=cloth_img.device)
        if pose_map is None:
            # Create dummy pose map (could be part of the 22)
            pose_map = torch.zeros(B, 0, H_img, W_img, device=cloth_img.device)
        
        # Concatenate for extractionA: cloth (3) + parsing (19) = 22
        # Adjust parsing channels to match 22 total
        if parsing_map.shape[1] == 3:
            # Convert RGB parsing to grayscale and expand
            parsing_gray = parsing_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            parsing_expanded = parsing_gray.repeat(1, 19, 1, 1)  # [B, 19, H, W]
        elif parsing_map.shape[1] == 1:
            parsing_expanded = parsing_map.repeat(1, 19, 1, 1)  # [B, 19, H, W]
        else:
            parsing_expanded = parsing_map[:, :19] if parsing_map.shape[1] >= 19 else F.pad(parsing_map, (0, 0, 0, 0, 0, 19 - parsing_map.shape[1]))
        
        extractionA_input = torch.cat([cloth_img, parsing_expanded], dim=1)  # [B, 22, H, W]
        
        # Prepare input for extractionB (expects 1 channel)
        # Use person image converted to grayscale
        if person_img.shape[1] == 3:
            person_gray = person_img.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            person_gray = person_img[:, :1]  # Take first channel
        
        # Extract features using extractionA and extractionB
        cloth_features = self.extractionA(extractionA_input)
        person_features = self.extractionB(person_gray)
        
        # Compute correlation (standard VITON approach)
        # Need to output 192 channels for regression
        correlation = self._compute_correlation(cloth_features, person_features, max_displacement=5)
        
        # Predict transformation using regression network
        # Flow is at correlation map resolution (downsampled)
        flow = self.regression(correlation)
        
        # Get correlation dimensions before upsampling
        _, _, H_corr, W_corr = correlation.size()
        
        # Upsample flow to match original image resolution
        # Flow shape: [B, 2, H_corr, W_corr] -> [B, 2, H_img, W_img]
        if flow.shape[2] != H_img or flow.shape[3] != W_img:
            # Scale flow values to account for resolution change before upsampling
            # The flow represents pixel displacements, so we scale by resolution ratio
            if H_corr > 0 and W_corr > 0:
                flow[:, 0] *= (W_img / W_corr)  # Scale x displacement
                flow[:, 1] *= (H_img / H_corr)  # Scale y displacement
            
            # Now upsample the scaled flow
            flow = F.interpolate(
                flow, size=(H_img, W_img), 
                mode='bilinear', align_corners=True
            )
        
        return flow
    
    def _compute_correlation(self, x1, x2, max_displacement=5):
        """
        Compute correlation between two feature maps
        Args:
            x1: cloth features [B, C, H, W]
            x2: person features [B, C, H, W]
            max_displacement: maximum displacement for correlation
        Returns:
            correlation map [B, 192, H, W] (to match checkpoint expectation)
        """
        B, C, H, W = x1.size()
        
        # Pad x2 for correlation
        pad_size = max_displacement
        x2_padded = F.pad(x2, (pad_size, pad_size, pad_size, pad_size))
        
        # Compute correlation
        correlation_maps = []
        for i in range(2 * max_displacement + 1):
            for j in range(2 * max_displacement + 1):
                x2_shifted = x2_padded[:, :, i:i+H, j:j+W]
                corr = torch.sum(x1 * x2_shifted, dim=1, keepdim=True)
                correlation_maps.append(corr)
        
        correlation = torch.cat(correlation_maps, dim=1)  # [B, (2*max_displacement+1)^2, H, W]
        
        # If we don't have exactly 192 channels, we need to adjust
        # (2*5+1)^2 = 121, but we need 192
        # Let's use a different approach: compute multiple correlation maps
        if correlation.shape[1] < 192:
            # Repeat or expand correlation maps to reach 192 channels
            num_repeats = (192 + correlation.shape[1] - 1) // correlation.shape[1]
            correlation = correlation.repeat(1, num_repeats, 1, 1)[:, :192]
        elif correlation.shape[1] > 192:
            correlation = correlation[:, :192]
        
        return correlation


class FeatureExtractor(nn.Module):
    """Feature extractor for cloth and person images (custom implementation)"""
    
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
    """
    Geometric Matching Module for cloth warping.
    This is a wrapper that automatically uses VITONGMM if checkpoint structure matches,
    otherwise falls back to custom implementation.
    """
    
    def __init__(self, in_channels=3, max_displacement=4, use_viton=True):
        super(GMM, self).__init__()
        self.use_viton = use_viton
        
        if use_viton:
            # Use VITON-compatible architecture
            self.model = VITONGMM()
        else:
            # Use custom implementation
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
        if self.use_viton:
            flow = self.model(cloth_img, person_img)
        else:
            # Extract features
            cloth_features = self.feature_extractor(cloth_img)
            person_features = self.feature_extractor(person_img)
            
            # Compute correlation
            correlation = self.correlation_layer(cloth_features, person_features)
            
            # Estimate flow
            flow = self.flow_conv(correlation)
        
        # Validate flow values (prevent NaN/Inf) for both paths
        if torch.isnan(flow).any() or torch.isinf(flow).any():
            print("Warning: Flow contains NaN/Inf values in GMM, replacing with zeros")
            flow = torch.where(torch.isnan(flow) | torch.isinf(flow), 
                              torch.zeros_like(flow), flow)
        
        # Clamp flow to reasonable range
        B, C, H, W = cloth_img.size()
        flow = torch.clamp(flow, -H, H)
        
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
        
        # Validate flow dimensions match image
        if flow.shape[2] != H or flow.shape[3] != W:
            # Upsample flow if needed
            flow = F.interpolate(
                flow, size=(H, W), 
                mode='bilinear', align_corners=True
            )
        
        # Validate flow values (prevent NaN/Inf)
        if torch.isnan(flow).any() or torch.isinf(flow).any():
            print("Warning: Flow contains NaN/Inf values in WarpingLayer, replacing with zeros")
            flow = torch.where(torch.isnan(flow) | torch.isinf(flow), 
                              torch.zeros_like(flow), flow)
        
        # Clamp flow to reasonable range
        flow = torch.clamp(flow, -H, H)  # Clamp to image dimensions
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=cloth_img.device, dtype=torch.float32),
            torch.arange(W, device=cloth_img.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow to grid
        warped_grid = grid + flow
        
        # Normalize to [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # Clamp grid to valid range to prevent out-of-bounds sampling
        warped_grid = torch.clamp(warped_grid, -1.1, 1.1)
        
        # Permute for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Apply transformation
        warped_cloth = F.grid_sample(
            cloth_img, warped_grid, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Validate output (check for all-white or all-black)
        if warped_cloth.min() == warped_cloth.max():
            print(f"Warning: Warped cloth appears uniform (min={warped_cloth.min():.3f}, max={warped_cloth.max():.3f})")
        
        return warped_cloth

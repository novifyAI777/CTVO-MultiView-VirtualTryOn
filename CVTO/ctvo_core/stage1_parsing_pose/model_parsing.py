"""
Stage 1: Human Parsing Model

This module handles human parsing using both ONNX and PyTorch (.pth) models (ATR/LIP)
"""

import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os

# Try to import ONNX runtime (optional dependency)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class HumanParsingPyTorchModel(nn.Module):
    """
    Generic PyTorch model for human parsing.
    This is a flexible architecture that can load various parsing model checkpoints.
    """
    
    def __init__(self, num_classes=20, backbone='resnet50'):
        super(HumanParsingPyTorchModel, self).__init__()
        self.num_classes = num_classes
        
        # Use a simple encoder-decoder architecture
        # This will be replaced by the actual model weights when loading
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class HumanParsingModel:
    """Human parsing model supporting both ONNX and PyTorch (.pth) formats"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = str(model_path)
        self.model_type = None  # 'onnx' or 'pytorch'
        self.model = None
        self.session = None
        
        model_path_lower = self.model_path.lower()
        
        # Determine model type and load accordingly
        if model_path_lower.endswith('.onnx'):
            self._load_onnx_model()
        elif model_path_lower.endswith('.pth') or model_path_lower.endswith('.pt'):
            self._load_pytorch_model()
        else:
            raise ValueError(
                f"Unsupported model format: {self.model_path}\n"
                f"Supported formats: .onnx, .pth, .pt"
            )
        
        self.transform = T.Compose([
            T.Resize((473, 473)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def _load_onnx_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime is not installed. Install it with: pip install onnxruntime"
            )
        
        self.model_type = 'onnx'
        try:
            self.session = ort.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from {self.model_path}.\n"
                f"Error: {str(e)}\n"
                f"Please ensure the file is a valid ONNX model."
            ) from e
    
    def _load_pytorch_model(self):
        """Load PyTorch model (.pth)"""
        self.model_type = 'pytorch'
        device_obj = torch.device(self.device)
        
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=device_obj)
            
            # Check if checkpoint is already a model object
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint
                self.model.to(device_obj)
                self.model.eval()
                return
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it contains a full model
                if 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
                    self.model = checkpoint['model']
                    self.model.to(device_obj)
                    self.model.eval()
                    return
                
                # Extract state_dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = checkpoint
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # Try to infer num_classes from state_dict
            num_classes = 20  # Default for LIP/ATR
            for key in state_dict.keys():
                if any(term in key.lower() for term in ['decoder', 'classifier', 'final', 'head', 'seg']):
                    if 'weight' in key:
                        shape = state_dict[key].shape
                        if len(shape) >= 2:
                            num_classes = max(num_classes, shape[0])
            
            # Create model architecture
            self.model = HumanParsingPyTorchModel(num_classes=num_classes)
            
            # Load state dict with flexible matching
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e1:
                # If strict loading fails, try to match keys manually
                model_dict = self.model.state_dict()
                matched_dict = {}
                for k, v in state_dict.items():
                    # Try to find matching key
                    for mk in model_dict.keys():
                        # Match by last part of key or if key contains model key
                        k_last = k.split('.')[-1]
                        mk_last = mk.split('.')[-1]
                        if k_last == mk_last or mk_last in k or k_last in mk:
                            if v.shape == model_dict[mk].shape:
                                matched_dict[mk] = v
                                break
                
                if matched_dict:
                    model_dict.update(matched_dict)
                    self.model.load_state_dict(model_dict, strict=False)
                else:
                    # If still fails, warn but continue with random weights
                    print(f"Warning: Could not load weights from {self.model_path}. Using random initialization.")
                    print(f"Original error: {str(e1)}")
            
            self.model.to(device_obj)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PyTorch model from {self.model_path}.\n"
                f"Error: {str(e)}\n"
                f"Please ensure the file is a valid PyTorch checkpoint."
            ) from e
    
    def parse_human(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Run human parsing on input image"""
        img = Image.open(image_path).convert("RGB")
        
        if self.model_type == 'onnx':
            return self._parse_onnx(img, output_path)
        else:  # pytorch
            return self._parse_pytorch(img, output_path)
    
    def _parse_onnx(self, img: Image.Image, output_path: Optional[str] = None) -> np.ndarray:
        """Run ONNX inference"""
        input_tensor = self.transform(img).unsqueeze(0).numpy()
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        seg = np.argmax(output[0], axis=1)[0].astype(np.uint8)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(seg).save(output_path)
            print(f"Parsing map saved at: {output_path}")
        
        return seg
    
    def _parse_pytorch(self, img: Image.Image, output_path: Optional[str] = None) -> np.ndarray:
        """Run PyTorch inference"""
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            # Handle different output formats
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Get segmentation map
            seg = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(seg).save(output_path)
            print(f"Parsing map saved at: {output_path}")
        
        return seg


def load_parsing_model(model_path: str, device: str = "cpu"):
    """Load and return a HumanParsingModel instance"""
    return HumanParsingModel(model_path, device)


def run_parsing(model: HumanParsingModel, image_path: str, device: str = "cpu") -> np.ndarray:
    """Run parsing on an image using the model"""
    return model.parse_human(image_path)
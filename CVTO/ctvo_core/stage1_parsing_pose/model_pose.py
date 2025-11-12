"""
Stage 1: Pose Estimation Model

This module handles pose estimation using MobileNet-based OpenPose
"""

import torch
import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, Optional

# Import pose model from openpose_pytorch
import sys
import importlib.util
from pathlib import Path

# Dynamically import from openpose_pytorch
openpose_dir = Path(__file__).parent.parent.parent / 'openpose_pytorch'
with_mobilenet_path = openpose_dir / 'models' / 'with_mobilenet.py'

if not with_mobilenet_path.exists():
    raise ImportError(f"Could not find with_mobilenet.py at {with_mobilenet_path}")

# Add openpose_pytorch to path for relative imports within the module (must be before module execution)
if str(openpose_dir) not in sys.path:
    sys.path.insert(0, str(openpose_dir))

spec = importlib.util.spec_from_file_location("models.with_mobilenet", with_mobilenet_path)
with_mobilenet_module = importlib.util.module_from_spec(spec)
sys.modules["models.with_mobilenet"] = with_mobilenet_module
spec.loader.exec_module(with_mobilenet_module)
PoseEstimationWithMobileNet = with_mobilenet_module.PoseEstimationWithMobileNet


class PoseEstimationModel:
    """Pose estimation model using MobileNet-based OpenPose"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
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
        
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()
        
        self.transform = T.Compose([
            T.Resize((473, 473)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_pose(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run pose estimation on input image"""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            stages_output = self.net(tensor)
        
        heatmaps = stages_output[-2]  # final-stage heatmap
        pafs = stages_output[-1]       # part-affinity fields
        
        # Simple argmax peak-finder
        heatmaps = heatmaps.squeeze(0).cpu().numpy()
        keypoints = []
        for idx, hmap in enumerate(heatmaps):  # 0-17 joints
            y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
            conf = hmap[y, x]
            keypoints.extend([int(x), int(y), float(conf)])
        
        coco_dict = {
            "version": 1.0,
            "people": [{
                "person_id": 0,
                "pose_keypoints_2d": keypoints
            }]
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(coco_dict, f, indent=2)
            print(f"Keypoints saved → {output_path}")
        
        return coco_dict


def load_pose_model(checkpoint_path: str, device: str = "cpu"):
    """Load and return a PoseEstimationModel instance"""
    return PoseEstimationModel(checkpoint_path, device)


def run_pose(model: PoseEstimationModel, image_path: str, device: str = "cpu") -> torch.Tensor:
    """Run pose estimation on an image and return heatmaps tensor"""
    img = Image.open(image_path).convert("RGB")
    tensor = model.transform(img).unsqueeze(0).to(model.device)
    
    model.net.to(model.device)
    with torch.no_grad():
        stages_output = model.net(tensor)
    
    heatmaps = stages_output[-2]  # final-stage heatmap
    return heatmaps.squeeze(0).cpu()  # Return as tensor for saving
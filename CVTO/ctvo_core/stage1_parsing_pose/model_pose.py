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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'openpose_pytorch'))
from models.with_mobilenet import PoseEstimationWithMobileNet


class PoseEstimationModel:
    """Pose estimation model using MobileNet-based OpenPose"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.net.load_state_dict(checkpoint)
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

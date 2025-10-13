"""
Stage 1: Main Runner for Human Parsing & Pose Estimation

This module provides the main interface for Stage 1 processing
"""

import os
import torch
from typing import Tuple, Dict, Any, Optional
from .model_parsing import HumanParsingModel
from .model_pose import PoseEstimationModel


class Stage1Processor:
    """Main processor for Stage 1: Human Parsing & Pose Estimation"""
    
    def __init__(self, 
                 parsing_model_path: str,
                 pose_model_path: str,
                 device: str = "cpu"):
        self.device = device
        self.parsing_model = HumanParsingModel(parsing_model_path, device)
        self.pose_model = PoseEstimationModel(pose_model_path, device)
    
    def process_image(self, 
                     image_path: str,
                     parsing_output_path: Optional[str] = None,
                     pose_output_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Process single image for both parsing and pose estimation"""
        
        # Run human parsing
        parsing_result = self.parsing_model.parse_human(image_path, parsing_output_path)
        
        # Run pose estimation
        pose_result = self.pose_model.estimate_pose(image_path, pose_output_path)
        
        return parsing_result, pose_result


def run_stage1(image_path: str, 
               parsing_model_path: str,
               pose_model_path: str,
               output_dir: str,
               device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Convenience function to run Stage 1 processing"""
    
    processor = Stage1Processor(parsing_model_path, pose_model_path, device)
    
    parsing_output_path = os.path.join(output_dir, "parsing_maps", "output.png")
    pose_output_path = os.path.join(output_dir, "keypoints_json", "pose.json")
    
    return processor.process_image(image_path, parsing_output_path, pose_output_path)


if __name__ == "__main__":
    # Example usage
    image_path = "path/to/image.jpg"
    parsing_model_path = "path/to/parsing_lip.onnx"
    pose_model_path = "path/to/body_pose_model.pth"
    output_dir = "outputs"
    
    parsing_result, pose_result = run_stage1(
        image_path, parsing_model_path, pose_model_path, output_dir
    )
    
    print("Stage 1 processing completed!")
    print(f"Parsing shape: {parsing_result.shape}")
    print(f"Pose keypoints: {len(pose_result['people'][0]['pose_keypoints_2d'])}")

"""
Stage 3: Evaluation Script for Fusion Network

This module provides evaluation functionality for Stage 3 fusion network.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, List, Tuple
import json

from .TryOnGenerator import TryOnGenerator
from .FusionNet import FusionNet


class Stage3Evaluator:
    """Evaluator for Stage 3 fusion network"""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "fusion",
                 device: str = "cpu"):
        self.device = device
        self.model_type = model_type
        
        # Load model
        if model_type == "generator":
            self.model = TryOnGenerator()
        elif model_type == "fusion":
            self.model = FusionNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((256, 192)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        self.to_pil = T.ToPILImage()
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)
    
    def preprocess_mask(self, mask_path: str) -> torch.Tensor:
        """Preprocess mask for inference"""
        mask = Image.open(mask_path).convert("RGB")
        return self.transform(mask).unsqueeze(0).to(self.device)
    
    def preprocess_pose(self, pose_json_path: str) -> torch.Tensor:
        """Preprocess pose JSON to heatmap"""
        from ..stage2_cloth_warping.utils import pose_to_heatmap
        return pose_to_heatmap(pose_json_path).unsqueeze(0).to(self.device)
    
    def evaluate_single(self, 
                       person_img_path: str,
                       warped_cloth_path: str,
                       mask_path: str,
                       pose_json_path: str = None,
                       output_path: str = None) -> torch.Tensor:
        """
        Evaluate single sample
        
        Args:
            person_img_path: path to person image
            warped_cloth_path: path to warped cloth image
            mask_path: path to clothing mask
            pose_json_path: path to pose JSON (optional)
            output_path: path to save result (optional)
            
        Returns:
            generated try-on image tensor
        """
        # Preprocess inputs
        person_img = self.preprocess_image(person_img_path)
        warped_cloth = self.preprocess_image(warped_cloth_path)
        mask = self.preprocess_mask(mask_path)
        
        pose_heatmap = None
        if pose_json_path and self.model_type == "fusion":
            pose_heatmap = self.preprocess_pose(pose_json_path)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == "generator":
                result = self.model(person_img, warped_cloth, mask)
            else:  # fusion
                result = self.model(person_img, warped_cloth, mask, pose_heatmap)
        
        # Denormalize
        result = result[0].cpu().mul_(0.5).add_(0.5).clamp_(0, 1)
        
        # Save result
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.to_pil(result).save(output_path)
            print(f"Result saved to: {output_path}")
        
        return result
    
    def evaluate_batch(self, 
                      data_list: List[Dict[str, str]],
                      output_dir: str) -> List[torch.Tensor]:
        """
        Evaluate batch of samples
        
        Args:
            data_list: list of dictionaries containing paths
            output_dir: directory to save results
            
        Returns:
            list of generated images
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, data in enumerate(data_list):
            output_path = os.path.join(output_dir, f"result_{i:04d}.jpg")
            
            result = self.evaluate_single(
                data['person_img'],
                data['warped_cloth'],
                data['mask'],
                data.get('pose_json'),
                output_path
            )
            
            results.append(result)
        
        return results
    
    def compute_metrics(self, 
                       pred_images: List[torch.Tensor],
                       target_images: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            pred_images: list of predicted images
            target_images: list of target images
            
        Returns:
            dictionary of metrics
        """
        metrics = {}
        
        # Convert to tensors
        pred_tensor = torch.stack(pred_images)
        target_tensor = torch.stack(target_images)
        
        # L1 Loss
        metrics['l1_loss'] = torch.nn.functional.l1_loss(pred_tensor, target_tensor).item()
        
        # L2 Loss
        metrics['l2_loss'] = torch.nn.functional.mse_loss(pred_tensor, target_tensor).item()
        
        # PSNR
        mse = metrics['l2_loss']
        metrics['psnr'] = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
        
        # SSIM (simplified version)
        metrics['ssim'] = self._compute_ssim(pred_tensor, target_tensor)
        
        return metrics
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute simplified SSIM"""
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Compute means
        mu_pred = pred_gray.mean()
        mu_target = target_gray.mean()
        
        # Compute variances and covariance
        var_pred = pred_gray.var()
        var_target = target_gray.var()
        cov = ((pred_gray - mu_pred) * (target_gray - mu_target)).mean()
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM formula
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * cov + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (var_pred + var_target + c2))
        
        return ssim.item()


def eval_fusion(model_path: str,
               test_data_path: str,
               output_dir: str,
               model_type: str = "fusion",
               device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate Stage 3 fusion network
    
    Args:
        model_path: path to trained model
        test_data_path: path to test data JSON
        output_dir: directory to save results
        model_type: type of model ("generator" or "fusion")
        device: device to run on
        
    Returns:
        evaluation metrics
    """
    # Initialize evaluator
    evaluator = Stage3Evaluator(model_path, model_type, device)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Evaluate samples
    results = evaluator.evaluate_batch(test_data, output_dir)
    
    # Load target images for metrics
    target_images = []
    for data in test_data:
        target_img = evaluator.preprocess_image(data['target_img'])
        target_images.append(target_img[0].cpu())
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results, target_images)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation completed. Metrics saved to: {metrics_path}")
    print(f"Metrics: {metrics}")
    
    return metrics


if __name__ == "__main__":
    # Example evaluation
    model_path = "checkpoints/stage3_fusion/best_model.pth"
    test_data_path = "data/test_data.json"
    output_dir = "results/stage3_evaluation"
    
    metrics = eval_fusion(model_path, test_data_path, output_dir)

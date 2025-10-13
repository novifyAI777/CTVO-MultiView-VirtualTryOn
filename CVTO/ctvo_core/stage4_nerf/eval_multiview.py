"""
Stage 4: Multi-view Evaluation Script

This module provides evaluation functionality for Stage 4 NeRF multi-view generation.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, List, Tuple
import json
import math

from .model_nerf import NeRFModel, VolumeRenderer
from .renderer import NeRFRenderer, Camera, MultiViewGenerator


class Stage4Evaluator:
    """Evaluator for Stage 4 NeRF multi-view generation"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cpu",
                 image_size: Tuple[int, int] = (256, 256),
                 focal_length: float = 500.0):
        self.device = device
        self.image_size = image_size
        self.focal_length = focal_length
        
        # Initialize NeRF model
        self.nerf_model = NeRFModel().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.nerf_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.nerf_model.load_state_dict(checkpoint)
        
        self.nerf_model.eval()
        
        # Initialize camera and renderer
        self.camera = Camera(focal_length, image_size)
        volume_renderer = VolumeRenderer()
        self.renderer = NeRFRenderer(
            self.nerf_model, volume_renderer, self.camera, device
        )
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        self.to_pil = T.ToPILImage()
    
    def generate_multiview(self, 
                          num_views: int = 8,
                          radius: float = 2.0,
                          height: float = 0.0,
                          output_dir: str = None) -> List[torch.Tensor]:
        """
        Generate multi-view virtual try-on results
        
        Args:
            num_views: number of views to generate
            radius: radius of camera path
            height: height of camera path
            output_dir: directory to save results (optional)
            
        Returns:
            list of rendered images
        """
        # Create multi-view generator
        generator = MultiViewGenerator(
            self.nerf_model, self.camera, self.device
        )
        
        # Generate views
        rendered_images = generator.generate_multiview(
            num_views, radius, height
        )
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, image in enumerate(rendered_images):
                # Denormalize image
                image_np = image.cpu().mul_(0.5).add_(0.5).clamp_(0, 1)
                image_pil = self.to_pil(image_np.permute(2, 0, 1))
                
                # Save image
                output_path = os.path.join(output_dir, f"view_{i:03d}.jpg")
                image_pil.save(output_path)
                print(f"View {i} saved to: {output_path}")
        
        return rendered_images
    
    def generate_custom_views(self, 
                            camera_poses: List[torch.Tensor],
                            output_dir: str = None) -> List[torch.Tensor]:
        """
        Generate custom views from given camera poses
        
        Args:
            camera_poses: list of camera pose matrices
            output_dir: directory to save results (optional)
            
        Returns:
            list of rendered images
        """
        # Create multi-view generator
        generator = MultiViewGenerator(
            self.nerf_model, self.camera, self.device
        )
        
        # Generate views
        rendered_images = generator.generate_custom_views(camera_poses)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, image in enumerate(rendered_images):
                # Denormalize image
                image_np = image.cpu().mul_(0.5).add_(0.5).clamp_(0, 1)
                image_pil = self.to_pil(image_np.permute(2, 0, 1))
                
                # Save image
                output_path = os.path.join(output_dir, f"custom_view_{i:03d}.jpg")
                image_pil.save(output_path)
                print(f"Custom view {i} saved to: {output_path}")
        
        return rendered_images
    
    def evaluate_multiview_quality(self, 
                                  generated_images: List[torch.Tensor],
                                  reference_images: List[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate quality of multi-view generation
        
        Args:
            generated_images: list of generated images
            reference_images: list of reference images (optional)
            
        Returns:
            dictionary of quality metrics
        """
        metrics = {}
        
        # Convert to tensors
        gen_tensor = torch.stack(generated_images)
        
        # Compute consistency metrics across views
        metrics['view_consistency'] = self._compute_view_consistency(gen_tensor)
        
        # Compute sharpness
        metrics['sharpness'] = self._compute_sharpness(gen_tensor)
        
        # Compute color consistency
        metrics['color_consistency'] = self._compute_color_consistency(gen_tensor)
        
        # If reference images provided, compute similarity metrics
        if reference_images is not None:
            ref_tensor = torch.stack(reference_images)
            metrics['similarity'] = self._compute_similarity(gen_tensor, ref_tensor)
        
        return metrics
    
    def _compute_view_consistency(self, images: torch.Tensor) -> float:
        """Compute consistency across views"""
        # Compute variance across views
        variance = torch.var(images, dim=0)
        consistency = 1.0 / (1.0 + variance.mean().item())
        return consistency
    
    def _compute_sharpness(self, images: torch.Tensor) -> float:
        """Compute average sharpness across views"""
        # Convert to grayscale
        gray_images = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        
        # Compute Laplacian for sharpness
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        
        sharpness_scores = []
        for i in range(images.shape[0]):
            gray = gray_images[i].unsqueeze(0).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
            sharpness = torch.var(laplacian).item()
            sharpness_scores.append(sharpness)
        
        return np.mean(sharpness_scores)
    
    def _compute_color_consistency(self, images: torch.Tensor) -> float:
        """Compute color consistency across views"""
        # Compute mean color for each view
        mean_colors = torch.mean(images, dim=(1, 2))  # [num_views, 3]
        
        # Compute variance of mean colors
        color_variance = torch.var(mean_colors, dim=0)
        consistency = 1.0 / (1.0 + color_variance.mean().item())
        
        return consistency
    
    def _compute_similarity(self, gen_images: torch.Tensor, ref_images: torch.Tensor) -> float:
        """Compute similarity between generated and reference images"""
        # Compute MSE
        mse = torch.nn.functional.mse_loss(gen_images, ref_images)
        
        # Convert to similarity score
        similarity = 1.0 / (1.0 + mse.item())
        
        return similarity


def eval_multiview(model_path: str,
                  output_dir: str,
                  num_views: int = 8,
                  radius: float = 2.0,
                  height: float = 0.0,
                  device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate Stage 4 NeRF multi-view generation
    
    Args:
        model_path: path to trained NeRF model
        output_dir: directory to save results
        num_views: number of views to generate
        radius: radius of camera path
        height: height of camera path
        device: device to run on
        
    Returns:
        evaluation metrics
    """
    # Initialize evaluator
    evaluator = Stage4Evaluator(model_path, device)
    
    # Generate multi-view results
    rendered_images = evaluator.generate_multiview(
        num_views, radius, height, output_dir
    )
    
    # Compute quality metrics
    metrics = evaluator.evaluate_multiview_quality(rendered_images)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "multiview_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Multi-view evaluation completed. Metrics saved to: {metrics_path}")
    print(f"Metrics: {metrics}")
    
    return metrics


if __name__ == "__main__":
    # Example evaluation
    model_path = "checkpoints/stage4_nerf/best_model.ckpt"
    output_dir = "results/stage4_multiview"
    
    metrics = eval_multiview(model_path, output_dir)

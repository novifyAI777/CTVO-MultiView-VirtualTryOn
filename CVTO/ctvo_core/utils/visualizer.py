"""
Shared Visualization Utilities

This module contains utilities for visualizing results and debugging
across different stages of the CTVO pipeline.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict, Any
import json


class ResultVisualizer:
    """Visualizer for CTVO results"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (256, 192),
                 font_size: int = 12):
        self.image_size = image_size
        self.font_size = font_size
        
        # Image transforms
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()
    
    def denormalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image from [-1, 1] to [0, 1]"""
        return (image_tensor + 1.0) / 2.0
    
    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        
        # Denormalize
        image_tensor = self.denormalize_image(image_tensor)
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        return self.to_pil(image_tensor)
    
    def create_comparison_grid(self, 
                              images: List[torch.Tensor],
                              titles: List[str],
                              grid_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Create comparison grid of images
        
        Args:
            images: list of image tensors
            titles: list of titles for each image
            grid_size: grid size (rows, cols)
            
        Returns:
            PIL Image with comparison grid
        """
        if grid_size is None:
            grid_size = (1, len(images))
        
        rows, cols = grid_size
        img_h, img_w = self.image_size
        
        # Create canvas
        canvas_w = cols * img_w
        canvas_h = rows * img_h + rows * 30  # Extra space for titles
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        
        # Draw images
        for i, (image, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            # Convert to PIL
            img_pil = self.tensor_to_pil(image)
            
            # Paste image
            x = col * img_w
            y = row * img_h + row * 30
            canvas.paste(img_pil, (x, y))
            
            # Draw title
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", self.font_size)
            except:
                font = ImageFont.load_default()
            
            text_w, text_h = draw.textsize(title, font=font)
            text_x = x + (img_w - text_w) // 2
            text_y = y + img_h + 5
            
            draw.text((text_x, text_y), title, fill='black', font=font)
        
        return canvas
    
    def visualize_stage1_results(self, 
                               person_img: torch.Tensor,
                               parsing_map: torch.Tensor,
                               pose_data: Dict,
                               output_path: str) -> None:
        """Visualize Stage 1 results"""
        # Create pose visualization
        pose_img = self.visualize_pose(pose_data)
        
        # Create comparison grid
        images = [person_img, parsing_map, pose_img]
        titles = ['Person Image', 'Parsing Map', 'Pose Keypoints']
        
        grid = self.create_comparison_grid(images, titles)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)
    
    def visualize_stage2_results(self,
                               person_img: torch.Tensor,
                               cloth_img: torch.Tensor,
                               warped_cloth: torch.Tensor,
                               output_path: str) -> None:
        """Visualize Stage 2 results"""
        images = [person_img, cloth_img, warped_cloth]
        titles = ['Person Image', 'Cloth Image', 'Warped Cloth']
        
        grid = self.create_comparison_grid(images, titles)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)
    
    def visualize_stage3_results(self,
                               person_img: torch.Tensor,
                               warped_cloth: torch.Tensor,
                               target_img: torch.Tensor,
                               pred_img: torch.Tensor,
                               output_path: str) -> None:
        """Visualize Stage 3 results"""
        images = [person_img, warped_cloth, target_img, pred_img]
        titles = ['Person Image', 'Warped Cloth', 'Target Image', 'Predicted Image']
        
        grid = self.create_comparison_grid(images, titles, grid_size=(2, 2))
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)
    
    def visualize_stage4_results(self,
                               reference_img: torch.Tensor,
                               multiview_images: List[torch.Tensor],
                               output_path: str) -> None:
        """Visualize Stage 4 multi-view results"""
        # Create grid with reference and multi-view images
        images = [reference_img] + multiview_images
        titles = ['Reference'] + [f'View {i}' for i in range(len(multiview_images))]
        
        grid = self.create_comparison_grid(images, titles)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)
    
    def visualize_pose(self, pose_data: Dict) -> torch.Tensor:
        """Visualize pose keypoints"""
        # Create blank image
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Get keypoints
        if 'people' in pose_data and len(pose_data['people']) > 0:
            keypoints = pose_data['people'][0]['pose_keypoints_2d']
            
            # Draw keypoints
            for i in range(0, len(keypoints), 3):
                x, y, conf = keypoints[i:i+3]
                if conf > 0.3:  # Only draw high-confidence keypoints
                    # Scale coordinates to image size
                    x_scaled = int(x * self.image_size[1] / 473)  # Assuming original size 473
                    y_scaled = int(y * self.image_size[0] / 473)
                    
                    # Draw circle
                    radius = 3
                    draw.ellipse([x_scaled-radius, y_scaled-radius, 
                                x_scaled+radius, y_scaled+radius], 
                               fill='red')
        
        # Convert to tensor
        return self.to_tensor(img)
    
    def create_loss_plot(self, 
                        losses: Dict[str, List[float]],
                        output_path: str) -> None:
        """Create loss plot"""
        plt.figure(figsize=(12, 8))
        
        for loss_name, loss_values in losses.items():
            plt.plot(loss_values, label=loss_name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_plot(self, 
                          metrics: Dict[str, List[float]],
                          output_path: str) -> None:
        """Create metrics plot"""
        plt.figure(figsize=(12, 8))
        
        for metric_name, metric_values in metrics.items():
            plt.plot(metric_values, label=metric_name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class DebugVisualizer:
    """Debug visualizer for intermediate results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Save tensor as image"""
        # Denormalize
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        to_pil = T.ToPILImage()
        img = to_pil(tensor)
        
        # Save
        output_path = os.path.join(self.output_dir, f"{name}.png")
        img.save(output_path)
    
    def save_attention_map(self, attention_map: torch.Tensor, name: str) -> None:
        """Save attention map"""
        # Normalize attention map
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / attention_map.max()
        
        # Convert to PIL
        if attention_map.dim() == 4:
            attention_map = attention_map[0]
        
        to_pil = T.ToPILImage()
        img = to_pil(attention_map)
        
        # Save
        output_path = os.path.join(self.output_dir, f"{name}_attention.png")
        img.save(output_path)
    
    def save_feature_maps(self, feature_maps: List[torch.Tensor], name: str) -> None:
        """Save feature maps"""
        for i, feat_map in enumerate(feature_maps):
            # Take first channel or average across channels
            if feat_map.dim() == 4:
                feat_map = feat_map[0]
            
            if feat_map.size(0) > 1:
                feat_map = feat_map.mean(dim=0, keepdim=True)
            
            # Normalize
            feat_map = feat_map - feat_map.min()
            feat_map = feat_map / feat_map.max()
            
            # Convert to PIL
            to_pil = T.ToPILImage()
            img = to_pil(feat_map)
            
            # Save
            output_path = os.path.join(self.output_dir, f"{name}_feat_{i}.png")
            img.save(output_path)

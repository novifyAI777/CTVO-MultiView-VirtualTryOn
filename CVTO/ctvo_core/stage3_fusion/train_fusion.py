"""
Stage 3: Training Script for Fusion Network

This module provides PyTorch Lightning training for Stage 3 fusion network.
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .TryOnGenerator import TryOnGenerator
from .FusionNet import FusionNet
from .losses import FusionLoss


class Stage3FusionModule(pl.LightningModule):
    """PyTorch Lightning module for Stage 3 fusion training"""
    
    def __init__(self, 
                 model_type: str = "fusion",  # "generator" or "fusion"
                 learning_rate: float = 2e-4,
                 lambda_l1: float = 1.0,
                 lambda_perceptual: float = 10.0,
                 lambda_style: float = 1.0,
                 lambda_mask: float = 5.0,
                 lambda_adv: float = 1.0):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize model
        if model_type == "generator":
            self.model = TryOnGenerator()
        elif model_type == "fusion":
            self.model = FusionNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize loss function
        self.criterion = FusionLoss(
            lambda_l1=lambda_l1,
            lambda_perceptual=lambda_perceptual,
            lambda_style=lambda_style,
            lambda_mask=lambda_mask,
            lambda_adv=lambda_adv
        )
    
    def forward(self, person_img, warped_cloth, mask, pose_heatmap=None):
        """Forward pass"""
        if isinstance(self.model, TryOnGenerator):
            return self.model(person_img, warped_cloth, mask)
        else:  # FusionNet
            return self.model(person_img, warped_cloth, mask, pose_heatmap)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        person_img = batch['person_img']
        warped_cloth = batch['warped_cloth']
        mask = batch['mask']
        target_img = batch['target_img']
        pose_heatmap = batch.get('pose_heatmap', None)
        
        # Forward pass
        pred_img = self.forward(person_img, warped_cloth, mask, pose_heatmap)
        
        # Compute loss
        losses = self.criterion(pred_img, target_img, mask)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train/{key}_loss', value, on_step=True, on_epoch=True)
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        person_img = batch['person_img']
        warped_cloth = batch['warped_cloth']
        mask = batch['mask']
        target_img = batch['target_img']
        pose_heatmap = batch.get('pose_heatmap', None)
        
        # Forward pass
        pred_img = self.forward(person_img, warped_cloth, mask, pose_heatmap)
        
        # Compute loss
        losses = self.criterion(pred_img, target_img, mask)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'val/{key}_loss', value, on_step=False, on_epoch=True)
        
        # Log sample images
        if batch_idx == 0:
            self.logger.log_image("val_samples", [
                person_img[0], warped_cloth[0], mask[0], 
                pred_img[0], target_img[0]
            ])
        
        return losses['total']
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=100, 
            gamma=0.5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


def train_fusion(train_loader: DataLoader,
                val_loader: DataLoader,
                config: Dict[str, Any],
                checkpoint_dir: str = "checkpoints/stage3_fusion"):
    """
    Train Stage 3 fusion network
    
    Args:
        train_loader: training data loader
        val_loader: validation data loader
        config: training configuration
        checkpoint_dir: directory to save checkpoints
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = Stage3FusionModule(**config)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 200),
        gpus=config.get('gpus', 1),
        precision=config.get('precision', 16),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        default_root_dir=checkpoint_dir,
        log_every_n_steps=config.get('log_every_n_steps', 50),
        val_check_interval=config.get('val_check_interval', 0.5)
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model


if __name__ == "__main__":
    # Example training configuration
    config = {
        'model_type': 'fusion',
        'learning_rate': 2e-4,
        'lambda_l1': 1.0,
        'lambda_perceptual': 10.0,
        'lambda_style': 1.0,
        'lambda_mask': 5.0,
        'lambda_adv': 1.0,
        'max_epochs': 200,
        'gpus': 1,
        'precision': 16
    }
    
    # This would be called with actual data loaders
    # train_fusion(train_loader, val_loader, config)

"""
Stage 2 Training Script for UNet

Training loop with L1 + perceptual loss.
Trains UNet from scratch using multiview dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict
import os
from datetime import datetime

from ..models.unet import UNet
from ..datasets.stage2_dataset import Stage2Dataset
from ..trainers.losses import Stage2Loss
from ..utils.config import Stage2Config
from ..utils.visualizer import save_image, visualize_batch


class Stage2Trainer:
    """Trainer for Stage 2 UNet model"""
    
    def __init__(self, config: Stage2Config):
        """
        Initialize trainer.
        
        Args:
            config: Stage2Config object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Calculate input channels: 3 (person_rgb) + K (parsing) + P (pose) + 3 (cloth_rgb) + 1 (cloth_mask)
        in_channels = 3 + config.num_parsing_channels + config.num_pose_keypoints + 3 + 1
        
        # Create model
        self.model = UNet(
            in_channels=in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels
        ).to(self.device)
        
        # Create loss function
        self.criterion = Stage2Loss(
            l1_weight=config.l1_weight,
            perceptual_weight=config.perceptual_weight
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size if hasattr(config, 'scheduler_step_size') else 30,
            gamma=0.5
        )
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        # Create output directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.reconstruction_dir = Path(config.output_dir)
        self.reconstruction_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self):
        """Setup data loaders"""
        if self.config.train_data_dir:
            self.train_dataset = Stage2Dataset(
                data_dir=self.config.train_data_dir,
                image_size=self.config.image_size,
                is_train=True,
                load_targets=self.config.load_targets if hasattr(self.config, 'load_targets') else False,
                num_parsing_classes=self.config.num_parsing_channels,
                num_pose_keypoints=self.config.num_pose_keypoints
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=True
            )
        
        if self.config.val_data_dir:
            self.val_dataset = Stage2Dataset(
                data_dir=self.config.val_data_dir,
                image_size=self.config.image_size,
                is_train=False,
                load_targets=True,
                num_parsing_classes=self.config.num_parsing_channels,
                num_pose_keypoints=self.config.num_pose_keypoints
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_l1 = 0.0
        total_perceptual = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input_tensor'].to(self.device)
            
            # Get target if available (for supervised training)
            if 'target_tensor' in batch:
                targets = batch['target_tensor'].to(self.device)
            else:
                # For unsupervised training, use person image as target (self-supervised)
                targets = batch['person_rgb'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets, pred_rgb_only=True)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_l1 += loss_dict['l1'].item()
            total_perceptual += loss_dict['perceptual'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['l1'].item():.4f}",
                'perceptual': f"{loss_dict['perceptual'].item():.4f}"
            })
            
            # Save periodic reconstructions
            if batch_idx % 100 == 0 and batch_idx > 0:
                self._save_reconstructions(outputs, targets, batch, batch_idx)
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_l1 = total_l1 / num_batches if num_batches > 0 else 0.0
        avg_perceptual = total_perceptual / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'l1': avg_l1,
            'perceptual': avg_perceptual
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_l1 = 0.0
        total_perceptual = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                inputs = batch['input_tensor'].to(self.device)
                
                # Get target
                if 'target_tensor' in batch:
                    targets = batch['target_tensor'].to(self.device)
                else:
                    targets = batch['person_rgb'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss_dict = self.criterion(outputs, targets, pred_rgb_only=True)
                loss = loss_dict['total']
                
                # Accumulate losses
                total_loss += loss.item()
                total_l1 += loss_dict['l1'].item()
                total_perceptual += loss_dict['perceptual'].item()
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_l1 = total_l1 / num_batches if num_batches > 0 else 0.0
        avg_perceptual = total_perceptual / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'l1': avg_l1,
            'perceptual': avg_perceptual
        }
    
    def _save_reconstructions(self, outputs, targets, batch, batch_idx):
        """Save reconstruction images"""
        # Take first batch item and detach from computation graph
        output_img = outputs[0].detach().cpu()
        target_img = targets[0].detach().cpu()
        person_img = batch['person_rgb'][0].detach().cpu()
        cloth_img = batch['cloth_rgb'][0].detach().cpu()
        
        # Denormalize images
        def denormalize(tensor):
            return (tensor + 1.0) / 2.0
        
        output_img = denormalize(output_img)
        target_img = denormalize(target_img)
        person_img = denormalize(person_img)
        cloth_img = denormalize(cloth_img)
        
        # Save reconstruction
        save_path = self.reconstruction_dir / f"epoch_{self.current_epoch:03d}_batch_{batch_idx:04d}.png"
        save_image(
            [person_img, cloth_img, output_img, target_img],
            str(save_path),
            titles=['Person', 'Cloth', 'Output', 'Target']
        )
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
    
    def train(self):
        """Main training loop"""
        # Setup data
        self.setup_data()
        
        if self.train_loader is None:
            raise ValueError("No training data provided")
        
        print(f"\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Input channels: {3 + self.config.num_parsing_channels + self.config.num_pose_keypoints + 3 + 1}")
        print(f"Output channels: {self.config.out_channels}")
        print("=" * 60)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            history_entry = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.train_history.append(history_entry)
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            self.save_checkpoint(str(checkpoint_path))
            
            # Save best model
            if val_metrics and val_metrics.get('loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(str(self.checkpoint_dir / 'best_model.pth'), is_best=True)
            
            # Save final model
            if epoch == self.config.num_epochs - 1:
                final_path = self.checkpoint_dir / 'unet_trained.pth'
                self.save_checkpoint(str(final_path))
                print(f"Saved final model to {final_path}")
            
            # Print metrics
            print(f"\nEpoch {epoch}/{self.config.num_epochs - 1}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, Perceptual: {train_metrics['perceptual']:.4f})")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f} (L1: {val_metrics['l1']:.4f}, Perceptual: {val_metrics['perceptual']:.4f})")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save training history
            history_path = self.log_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Reconstructions saved to: {self.reconstruction_dir}")
        print("=" * 60)


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 2 UNet")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default="data/multiview_dataset", 
                       help="Path to multiview dataset directory")
    parser.add_argument("--val_data_dir", type=str, default=None,
                       help="Path to validation dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Stage2Config.from_yaml(args.config)
    else:
        config = Stage2Config()
    
    # Override with command line arguments
    if args.data_dir:
        config.train_data_dir = args.data_dir
    if args.val_data_dir:
        config.val_data_dir = args.val_data_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    # Create trainer
    trainer = Stage2Trainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from epoch {trainer.current_epoch}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

"""
Stage 2 Configuration Management
"""

from dataclasses import dataclass
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class Stage2Config:
    """Configuration for Stage 2 training and inference"""
    
    # Model architecture
    # Input channels: 3 (person_rgb) + K (parsing) + P (pose) + 3 (cloth_rgb) + 1 (cloth_mask)
    num_parsing_channels: int = 19  # K
    num_pose_keypoints: int = 18    # P
    out_channels: int = 3            # RGB warped cloth
    base_channels: int = 64
    
    # Training
    batch_size: int = 4
    learning_rate: float = 0.0002  # 2e-4
    num_epochs: int = 100
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0001
    scheduler_step_size: int = 30
    
    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 10.0
    
    # Data
    image_size: tuple = (256, 192)  # (H, W)
    num_workers: int = 4
    pin_memory: bool = True
    load_targets: bool = False  # Whether to load target warped cloth for supervised training
    
    # Paths
    train_data_dir: Optional[str] = None
    val_data_dir: Optional[str] = None
    checkpoint_dir: str = "pretrained_weights"
    log_dir: str = "experiments/logs"
    output_dir: str = "experiments/reconstructions"
    
    # Device
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # Inference
    checkpoint_path: Optional[str] = None
    
    @property
    def in_channels(self) -> int:
        """Calculate total input channels"""
        return 3 + self.num_parsing_channels + self.num_pose_keypoints + 3 + 1
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Stage2Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract stage2-specific config if nested
        if 'stage2' in config_dict:
            config_dict = config_dict['stage2']
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'num_parsing_channels': self.num_parsing_channels,
            'num_pose_keypoints': self.num_pose_keypoints,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'base_channels': self.base_channels,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'scheduler_step_size': self.scheduler_step_size,
            'l1_weight': self.l1_weight,
            'perceptual_weight': self.perceptual_weight,
            'image_size': self.image_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'load_targets': self.load_targets,
            'train_data_dir': self.train_data_dir,
            'val_data_dir': self.val_data_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'output_dir': self.output_dir,
            'device': self.device,
            'checkpoint_path': self.checkpoint_path,
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

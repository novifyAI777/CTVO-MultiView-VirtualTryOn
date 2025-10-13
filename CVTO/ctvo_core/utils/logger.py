"""
Shared Logging Utilities

This module contains utilities for logging and monitoring
across different stages of the CTVO pipeline.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import numpy as np


class CTVOLogger:
    """Main logger for CTVO pipeline"""
    
    def __init__(self, 
                 log_dir: str,
                 stage: str,
                 level: int = logging.INFO):
        self.log_dir = log_dir
        self.stage = stage
        self.level = level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"ctvo_{stage}")
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(log_dir, f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics = {}
        self.start_time = time.time()
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def log_metrics(self, metrics: Dict[str, Any], epoch: int = None) -> None:
        """Log metrics"""
        if epoch is not None:
            self.logger.info(f"Epoch {epoch}: {metrics}")
        else:
            self.logger.info(f"Metrics: {metrics}")
        
        # Store metrics
        if epoch is not None:
            if epoch not in self.metrics:
                self.metrics[epoch] = {}
            self.metrics[epoch].update(metrics)
        else:
            if 'final' not in self.metrics:
                self.metrics['final'] = {}
            self.metrics['final'].update(metrics)
    
    def log_losses(self, losses: Dict[str, float], epoch: int = None) -> None:
        """Log losses"""
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        
        if epoch is not None:
            self.logger.info(f"Epoch {epoch} Losses: {loss_str}")
        else:
            self.logger.info(f"Losses: {loss_str}")
        
        # Store losses
        if epoch is not None:
            if epoch not in self.metrics:
                self.metrics[epoch] = {}
            self.metrics[epoch]['losses'] = losses
        else:
            if 'final' not in self.metrics:
                self.metrics['final'] = {}
            self.metrics['final']['losses'] = losses
    
    def log_timing(self, operation: str, duration: float) -> None:
        """Log timing information"""
        self.logger.info(f"{operation} took {duration:.2f} seconds")
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """Log model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def save_metrics(self, output_path: str = None) -> None:
        """Save metrics to JSON file"""
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"{self.stage}_metrics.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {output_path}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger initialization"""
        return time.time() - self.start_time


class TrainingLogger:
    """Logger specifically for training"""
    
    def __init__(self, log_dir: str, stage: str):
        self.logger = CTVOLogger(log_dir, stage)
        self.epoch_start_time = None
        self.batch_times = []
    
    def start_epoch(self, epoch: int) -> None:
        """Start logging for epoch"""
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch}")
    
    def end_epoch(self, epoch: int, losses: Dict[str, float], metrics: Dict[str, float] = None) -> None:
        """End logging for epoch"""
        epoch_duration = time.time() - self.epoch_start_time
        self.logger.log_timing(f"Epoch {epoch}", epoch_duration)
        
        # Log losses
        self.logger.log_losses(losses, epoch)
        
        # Log metrics if provided
        if metrics:
            self.logger.log_metrics(metrics, epoch)
        
        # Log average batch time
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            self.logger.log_timing(f"Average batch time in epoch {epoch}", avg_batch_time)
            self.batch_times = []
    
    def log_batch(self, batch_idx: int, losses: Dict[str, float]) -> None:
        """Log batch information"""
        batch_start_time = time.time()
        
        # Log losses every 10 batches
        if batch_idx % 10 == 0:
            self.logger.log_losses(losses, batch_idx)
        
        # Record batch time
        batch_time = time.time() - batch_start_time
        self.batch_times.append(batch_time)
    
    def log_validation(self, epoch: int, losses: Dict[str, float], metrics: Dict[str, float] = None) -> None:
        """Log validation results"""
        self.logger.info(f"Validation results for epoch {epoch}:")
        self.logger.log_losses(losses, epoch)
        
        if metrics:
            self.logger.log_metrics(metrics, epoch)


class EvaluationLogger:
    """Logger specifically for evaluation"""
    
    def __init__(self, log_dir: str, stage: str):
        self.logger = CTVOLogger(log_dir, stage)
    
    def log_evaluation_start(self, dataset_size: int) -> None:
        """Log start of evaluation"""
        self.logger.info(f"Starting evaluation on {dataset_size} samples")
    
    def log_evaluation_progress(self, current: int, total: int) -> None:
        """Log evaluation progress"""
        if current % 10 == 0 or current == total:
            progress = (current / total) * 100
            self.logger.info(f"Evaluation progress: {current}/{total} ({progress:.1f}%)")
    
    def log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results"""
        self.logger.info("Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Store metrics
        self.logger.log_metrics(metrics)
    
    def log_sample_results(self, sample_id: str, results: Dict[str, Any]) -> None:
        """Log results for individual sample"""
        self.logger.info(f"Sample {sample_id} results: {results}")


class SystemLogger:
    """Logger for system information"""
    
    def __init__(self, log_dir: str):
        self.logger = CTVOLogger(log_dir, "system")
    
    def log_system_info(self) -> None:
        """Log system information"""
        import platform
        import psutil
        
        self.logger.info("System Information:")
        self.logger.info(f"  Platform: {platform.platform()}")
        self.logger.info(f"  Python: {platform.python_version()}")
        self.logger.info(f"  PyTorch: {torch.__version__}")
        self.logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"  CUDA Version: {torch.version.cuda}")
            self.logger.info(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.logger.info(f"  CPU Count: {psutil.cpu_count()}")
        self.logger.info(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    def log_gpu_memory(self) -> None:
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.info(f"GPU {i} Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

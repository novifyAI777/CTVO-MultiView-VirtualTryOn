#!/usr/bin/env python3
"""
Stage 3 Runner: Fusion Generation

This script runs Stage 3 of the CTVO pipeline to perform fusion generation
for virtual try-on results.
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ctvo_core.stage3_fusion import Stage3FusionModule, train_fusion, eval_fusion
from ctvo_core.utils import CTVOLogger, create_dataloader, split_dataset
from ctvo_core.utils.data_loader import Stage3Dataset


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3: Fusion Generation")
    parser.add_argument("--config", type=str, default="configs/stage3_fusion.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "eval", "inference"],
                       help="Mode to run: train, eval, or inference")
    parser.add_argument("--data_dir", type=str, default="data/custom_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/stage3_previews",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                       help="Learning rate")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization of results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'device': args.device
    })
    
    # Setup logging
    logger = CTVOLogger("logs/stage3", "stage3")
    logger.info(f"Starting Stage 3: Fusion Generation - {args.mode} mode")
    logger.log_config(vars(args))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode == "train":
            # Training mode
            logger.info("Starting training...")
            
            # Load dataset
            train_metadata_path = os.path.join(args.data_dir, "train_metadata.json")
            val_metadata_path = os.path.join(args.data_dir, "val_metadata.json")
            
            if not os.path.exists(train_metadata_path):
                logger.error(f"Training metadata not found: {train_metadata_path}")
                return
            
            train_dataset = Stage3Dataset(args.data_dir, train_metadata_path)
            
            if os.path.exists(val_metadata_path):
                val_dataset = Stage3Dataset(args.data_dir, val_metadata_path)
            else:
                # Split training dataset for validation
                train_dataset, val_dataset, _ = split_dataset(train_dataset, 0.8, 0.2, 0.0)
            
            # Create data loaders
            train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Train model
            model = train_fusion(
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir="checkpoints/stage3_fusion"
            )
            
            logger.info("Training completed successfully")
            
        elif args.mode == "eval":
            # Evaluation mode
            logger.info("Starting evaluation...")
            
            if args.checkpoint is None:
                logger.error("Checkpoint path required for evaluation mode")
                return
            
            if not os.path.exists(args.checkpoint):
                logger.error(f"Checkpoint not found: {args.checkpoint}")
                return
            
            # Load test dataset
            test_metadata_path = os.path.join(args.data_dir, "test_metadata.json")
            if not os.path.exists(test_metadata_path):
                logger.error(f"Test metadata not found: {test_metadata_path}")
                return
            
            # Run evaluation
            metrics = eval_fusion(
                model_path=args.checkpoint,
                test_data_path=test_metadata_path,
                output_dir=args.output_dir,
                device=args.device
            )
            
            logger.info("Evaluation completed successfully")
            logger.log_metrics(metrics)
            
        elif args.mode == "inference":
            # Inference mode
            logger.info("Starting inference...")
            
            if args.checkpoint is None:
                logger.error("Checkpoint path required for inference mode")
                return
            
            if not os.path.exists(args.checkpoint):
                logger.error(f"Checkpoint not found: {args.checkpoint}")
                return
            
            # Load model
            model = Stage3FusionModule.load_from_checkpoint(args.checkpoint)
            model.eval()
            model.to(args.device)
            
            # TODO: Implement inference on single samples
            logger.info("Inference mode not yet implemented")
            
        # Generate visualization if requested
        if args.visualize and args.mode in ["eval", "inference"]:
            logger.info("Generating visualization...")
            # TODO: Implement visualization for Stage 3 results
            logger.info("Visualization not yet implemented for Stage 3")
        
        # Save results summary
        results_summary = {
            "mode": args.mode,
            "config": args.config,
            "data_dir": args.data_dir,
            "checkpoint": args.checkpoint,
            "output_dir": args.output_dir,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate
        }
        
        import json
        summary_path = os.path.join(args.output_dir, "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during Stage 3 processing: {str(e)}")
        raise
    
    finally:
        elapsed_time = logger.get_elapsed_time()
        logger.info(f"Stage 3 completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

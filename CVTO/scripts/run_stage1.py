#!/usr/bin/env python3
"""
Stage 1 Runner: Human Parsing & Pose Estimation

This script runs Stage 1 of the CTVO pipeline to perform human parsing
and pose estimation on input images.
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

from ctvo_core.stage1_parsing_pose import Stage1Processor, run_stage1
from ctvo_core.utils import CTVOLogger, ResultVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1: Human Parsing & Pose Estimation")
    parser.add_argument("--config", type=str, default="configs/base.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--input_image", type=str, required=True,
                       help="Path to input person image")
    parser.add_argument("--output_dir", type=str, default="results/stage1",
                       help="Output directory for results")
    parser.add_argument("--parsing_model", type=str, 
                       default="ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx",
                       help="Path to parsing model")
    parser.add_argument("--pose_model", type=str,
                       default="ctvo_core/stage1_parsing_pose/pretrained_models/body_pose_model.pth",
                       help="Path to pose model")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization of results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = CTVOLogger("logs/stage1", "stage1")
    logger.info("Starting Stage 1: Human Parsing & Pose Estimation")
    logger.log_config(vars(args))
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        logger.error(f"Input image not found: {args.input_image}")
        return
    
    # Check if models exist
    if not os.path.exists(args.parsing_model):
        logger.error(f"Parsing model not found: {args.parsing_model}")
        return
    
    if not os.path.exists(args.pose_model):
        logger.error(f"Pose model not found: {args.pose_model}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run Stage 1 processing
        logger.info(f"Processing image: {args.input_image}")
        
        parsing_result, pose_result = run_stage1(
            image_path=args.input_image,
            parsing_model_path=args.parsing_model,
            pose_model_path=args.pose_model,
            output_dir=args.output_dir,
            device=args.device
        )
        
        logger.info("Stage 1 processing completed successfully")
        logger.info(f"Parsing result shape: {parsing_result.shape}")
        logger.info(f"Pose keypoints: {len(pose_result['people'][0]['pose_keypoints_2d'])}")
        
        # Generate visualization if requested
        if args.visualize:
            logger.info("Generating visualization...")
            
            # Load input image for visualization
            from ctvo_core.utils import ImageLoader
            image_loader = ImageLoader()
            person_img = image_loader.load_image(args.input_image, normalize=False)
            
            # Create visualizer
            visualizer = ResultVisualizer()
            
            # Generate visualization
            vis_output_path = os.path.join(args.output_dir, "stage1_visualization.png")
            visualizer.visualize_stage1_results(
                person_img=person_img,
                parsing_map=torch.from_numpy(parsing_result).unsqueeze(0),
                pose_data=pose_result,
                output_path=vis_output_path
            )
            
            logger.info(f"Visualization saved to: {vis_output_path}")
        
        # Save results summary
        results_summary = {
            "input_image": args.input_image,
            "parsing_result_shape": list(parsing_result.shape),
            "pose_keypoints_count": len(pose_result['people'][0]['pose_keypoints_2d']),
            "output_dir": args.output_dir,
            "device": args.device
        }
        
        import json
        summary_path = os.path.join(args.output_dir, "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during Stage 1 processing: {str(e)}")
        raise
    
    finally:
        elapsed_time = logger.get_elapsed_time()
        logger.info(f"Stage 1 completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

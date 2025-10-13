#!/usr/bin/env python3
"""
Stage 2 Runner: Cloth Warping

This script runs Stage 2 of the CTVO pipeline to perform cloth warping
based on person pose and parsing information.
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

from ctvo_core.stage2_cloth_warping import Stage2Processor, run_stage2
from ctvo_core.utils import CTVOLogger, ResultVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2: Cloth Warping")
    parser.add_argument("--config", type=str, default="configs/base.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--person_img", type=str, required=True,
                       help="Path to person image")
    parser.add_argument("--cloth_img", type=str, required=True,
                       help="Path to cloth image")
    parser.add_argument("--parsing_map", type=str, required=True,
                       help="Path to parsing map")
    parser.add_argument("--pose_json", type=str, required=True,
                       help="Path to pose JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for warped cloth")
    parser.add_argument("--model_checkpoint", type=str,
                       default="ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="unet",
                       choices=["unet", "gmm"],
                       help="Type of model to use")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization of results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = CTVOLogger("logs/stage2", "stage2")
    logger.info("Starting Stage 2: Cloth Warping")
    logger.log_config(vars(args))
    
    # Check if input files exist
    input_files = {
        "person_img": args.person_img,
        "cloth_img": args.cloth_img,
        "parsing_map": args.parsing_map,
        "pose_json": args.pose_json
    }
    
    for file_type, file_path in input_files.items():
        if not os.path.exists(file_path):
            logger.error(f"{file_type} not found: {file_path}")
            return
    
    # Check if model exists
    if not os.path.exists(args.model_checkpoint):
        logger.error(f"Model checkpoint not found: {args.model_checkpoint}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # Run Stage 2 processing
        logger.info("Starting cloth warping...")
        
        warped_cloth = run_stage2(
            person_img=args.person_img,
            parsing_map=args.parsing_map,
            cloth_img=args.cloth_img,
            pose_json=args.pose_json,
            model_checkpoint=args.model_checkpoint,
            output_path=args.output_path,
            model_type=args.model_type,
            device=args.device
        )
        
        logger.info("Stage 2 processing completed successfully")
        logger.info(f"Warped cloth shape: {warped_cloth.shape}")
        logger.info(f"Warped cloth saved to: {args.output_path}")
        
        # Generate visualization if requested
        if args.visualize:
            logger.info("Generating visualization...")
            
            # Load input images for visualization
            from ctvo_core.utils import ImageLoader
            image_loader = ImageLoader()
            person_img = image_loader.load_image(args.person_img, normalize=False)
            cloth_img = image_loader.load_image(args.cloth_img, normalize=False)
            
            # Create visualizer
            visualizer = ResultVisualizer()
            
            # Generate visualization
            vis_output_path = os.path.join(os.path.dirname(args.output_path), "stage2_visualization.png")
            visualizer.visualize_stage2_results(
                person_img=person_img,
                cloth_img=cloth_img,
                warped_cloth=warped_cloth,
                output_path=vis_output_path
            )
            
            logger.info(f"Visualization saved to: {vis_output_path}")
        
        # Save results summary
        results_summary = {
            "person_img": args.person_img,
            "cloth_img": args.cloth_img,
            "parsing_map": args.parsing_map,
            "pose_json": args.pose_json,
            "warped_cloth_shape": list(warped_cloth.shape),
            "output_path": args.output_path,
            "model_type": args.model_type,
            "device": args.device
        }
        
        import json
        summary_path = os.path.join(os.path.dirname(args.output_path), "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during Stage 2 processing: {str(e)}")
        raise
    
    finally:
        elapsed_time = logger.get_elapsed_time()
        logger.info(f"Stage 2 completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

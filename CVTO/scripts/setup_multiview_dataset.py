#!/usr/bin/env python3
"""
Simple Multi-View Dataset Setup

This script helps you quickly organize your 80-image dataset 
and provides step-by-step instructions for running the CTVO pipeline.
"""

import os
import json
from pathlib import Path


def create_dataset_structure():
    """Create the proper directory structure for your dataset"""
    
    print("🏗️  Creating dataset structure...")
    
    # Base directories
    base_dir = Path("data/multiview_dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Person directories (10 persons, 8 views each)
    for i in range(1, 11):
        person_dir = base_dir / f"person_{i:03d}"
        person_dir.mkdir(exist_ok=True)
        
        # Create view directories
        for j in range(1, 9):
            view_dir = person_dir / f"view_{j:02d}"
            view_dir.mkdir(exist_ok=True)
    
    # Cloth directory
    cloth_dir = base_dir / "clothes"
    cloth_dir.mkdir(exist_ok=True)
    
    # Output directories
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    stage1_dir = outputs_dir / "stage1"
    stage1_dir.mkdir(exist_ok=True)
    
    stage2_dir = outputs_dir / "stage2"
    stage2_dir.mkdir(exist_ok=True)
    
    print("✅ Dataset structure created!")
    return base_dir


def create_dataset_template():
    """Create a template for organizing your images"""
    
    template = {
        "dataset_info": {
            "description": "Multi-view virtual try-on dataset",
            "total_persons": 10,
            "images_per_person": 8,
            "total_images": 80,
            "total_clothes": 5
        },
        "organization_instructions": {
            "step1": "Copy your 80 person images to data/multiview_dataset/person_XXX/view_XX/ directories",
            "step2": "Copy your cloth images to data/multiview_dataset/clothes/ directory",
            "step3": "Update the file paths in this template",
            "step4": "Run the pipeline using the provided scripts"
        },
        "person_images": {
            "person_001": {
                "view_01": "path/to/person1_view1.jpg",
                "view_02": "path/to/person1_view2.jpg",
                "view_03": "path/to/person1_view3.jpg",
                "view_04": "path/to/person1_view4.jpg",
                "view_05": "path/to/person1_view5.jpg",
                "view_06": "path/to/person1_view6.jpg",
                "view_07": "path/to/person1_view7.jpg",
                "view_08": "path/to/person1_view8.jpg"
            },
            "person_002": {
                "view_01": "path/to/person2_view1.jpg",
                "view_02": "path/to/person2_view2.jpg",
                "view_03": "path/to/person2_view3.jpg",
                "view_04": "path/to/person2_view4.jpg",
                "view_05": "path/to/person2_view5.jpg",
                "view_06": "path/to/person2_view6.jpg",
                "view_07": "path/to/person2_view7.jpg",
                "view_08": "path/to/person2_view8.jpg"
            }
            # ... continue for all 10 persons
        },
        "cloth_images": {
            "cloth_01": "path/to/cloth1.jpg",
            "cloth_02": "path/to/cloth2.jpg",
            "cloth_03": "path/to/cloth3.jpg",
            "cloth_04": "path/to/cloth4.jpg",
            "cloth_05": "path/to/cloth5.jpg"
        }
    }
    
    # Save template
    template_path = Path("data/multiview_dataset/dataset_template.json")
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📝 Dataset template created: {template_path}")
    return template_path


def print_instructions():
    """Print step-by-step instructions"""
    
    print("\n" + "="*60)
    print("📋 STEP-BY-STEP INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣  ORGANIZE YOUR IMAGES:")
    print("   • Copy your 80 person images to:")
    print("     data/multiview_dataset/person_XXX/view_XX/")
    print("   • Copy your cloth images to:")
    print("     data/multiview_dataset/clothes/")
    
    print("\n2️⃣  UPDATE THE TEMPLATE:")
    print("   • Edit data/multiview_dataset/dataset_template.json")
    print("   • Replace 'path/to/...' with your actual file paths")
    
    print("\n3️⃣  RUN STAGE 1 (Human Parsing & Pose):")
    print("   python scripts/run_stage1.py \\")
    print("       --input_image data/multiview_dataset/person_001/view_01/person1_view1.jpg \\")
    print("       --output_dir results/stage1/person_001/view_01 \\")
    print("       --visualize")
    
    print("\n4️⃣  RUN STAGE 2 (Cloth Warping):")
    print("   python scripts/run_stage2.py \\")
    print("       --person_img data/multiview_dataset/person_001/view_01/person1_view1.jpg \\")
    print("       --cloth_img data/multiview_dataset/clothes/cloth1.jpg \\")
    print("       --parsing_map results/stage1/person_001/view_01/parsing_maps/output.png \\")
    print("       --pose_json results/stage1/person_001/view_01/keypoints_json/pose.json \\")
    print("       --output_path results/stage2/person_001_view01_cloth1_warped.jpg \\")
    print("       --visualize")
    
    print("\n5️⃣  RUN STAGE 3 (Fusion Generation):")
    print("   python scripts/run_stage3.py \\")
    print("       --mode train \\")
    print("       --data_dir data/multiview_dataset \\")
    print("       --config configs/stage3_fusion.yaml")
    
    print("\n6️⃣  RUN STAGE 4 (NeRF Multi-view):")
    print("   python scripts/run_stage4.py \\")
    print("       --mode train \\")
    print("       --data_dir data/multiview_dataset \\")
    print("       --config configs/stage4_nerf.yaml")
    
    print("\n" + "="*60)
    print("🎯 QUICK START COMMANDS")
    print("="*60)
    
    print("\n# Test with one image first:")
    print("python scripts/run_stage1.py --input_image YOUR_IMAGE.jpg --output_dir test_output --visualize")
    
    print("\n# Run all stages (after organizing data):")
    print("bash scripts/train_all.sh")
    
    print("\n# Check if everything works:")
    print("python tests/test_imports.py")


def main():
    print("🎯 CTVO Multi-View Dataset Setup")
    print("="*50)
    
    # Create structure
    base_dir = create_dataset_structure()
    
    # Create template
    template_path = create_dataset_template()
    
    # Print instructions
    print_instructions()
    
    print(f"\n✅ Setup complete!")
    print(f"📁 Dataset directory: {base_dir}")
    print(f"📝 Template file: {template_path}")
    print(f"\n🚀 Ready to organize your 80 images and run the pipeline!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-View Dataset Organizer and Pipeline Runner

This script helps organize your 80-image dataset (8 images per person) 
and runs the complete CTVO pipeline from Stage 1 to Stage 4.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import sys


class MultiViewDatasetOrganizer:
    """Organizes multi-view dataset for CTVO pipeline"""
    
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.persons_dir = self.output_dir / "persons"
        self.clothes_dir = self.output_dir / "clothes"
        
        # Create directories
        self.persons_dir.mkdir(parents=True, exist_ok=True)
        self.clothes_dir.mkdir(parents=True, exist_ok=True)
    
    def organize_person_images(self, person_id: str, image_paths: List[str]) -> str:
        """
        Organize 8 images for a single person
        
        Args:
            person_id: Person identifier (e.g., "person_001")
            image_paths: List of 8 image paths for this person
            
        Returns:
            Path to person directory
        """
        person_dir = self.persons_dir / person_id
        person_dir.mkdir(exist_ok=True)
        
        # Copy images with view naming
        for i, img_path in enumerate(image_paths, 1):
            src = Path(img_path)
            dst = person_dir / f"view_{i:02d}.jpg"
            
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied {src.name} -> {dst}")
            else:
                print(f"Warning: {src} not found")
        
        return str(person_dir)
    
    def organize_cloth_images(self, cloth_paths: List[str]) -> None:
        """
        Organize cloth images
        
        Args:
            cloth_paths: List of cloth image paths
        """
        for i, cloth_path in enumerate(cloth_paths, 1):
            src = Path(cloth_path)
            dst = self.clothes_dir / f"cloth_{i:02d}.jpg"
            
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied cloth {src.name} -> {dst}")
            else:
                print(f"Warning: {src} not found")
    
    def create_dataset_metadata(self, person_data: Dict[str, List[str]], 
                              cloth_data: List[str]) -> str:
        """
        Create metadata file for the dataset
        
        Args:
            person_data: Dictionary mapping person_id to list of image paths
            cloth_data: List of cloth image paths
            
        Returns:
            Path to metadata file
        """
        metadata = {
            "dataset_info": {
                "total_persons": len(person_data),
                "images_per_person": 8,
                "total_images": sum(len(images) for images in person_data.values()),
                "total_clothes": len(cloth_data)
            },
            "persons": {},
            "clothes": []
        }
        
        # Add person data
        for person_id, image_paths in person_data.items():
            metadata["persons"][person_id] = {
                "views": [f"view_{i:02d}.jpg" for i in range(1, 9)],
                "original_paths": image_paths
            }
        
        # Add cloth data
        for i, cloth_path in enumerate(cloth_data, 1):
            metadata["clothes"].append({
                "cloth_id": f"cloth_{i:02d}",
                "filename": f"cloth_{i:02d}.jpg",
                "original_path": cloth_path
            })
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created metadata file: {metadata_path}")
        return str(metadata_path)


class CTVOPipelineRunner:
    """Runs the complete CTVO pipeline"""
    
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).parent.parent
    
    def run_stage1_batch(self, person_dirs: List[str]) -> Dict[str, str]:
        """
        Run Stage 1 (Human Parsing & Pose) for all persons
        
        Args:
            person_dirs: List of person directory paths
            
        Returns:
            Dictionary mapping person_id to output directory
        """
        print("\n" + "="*50)
        print("RUNNING STAGE 1: Human Parsing & Pose Estimation")
        print("="*50)
        
        stage1_outputs = {}
        
        for person_dir in person_dirs:
            person_id = Path(person_dir).name
            print(f"\nProcessing {person_id}...")
            
            # Create output directory for this person
            person_output_dir = self.output_dir / "stage1_outputs" / person_id
            person_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each view
            for view_file in sorted(Path(person_dir).glob("view_*.jpg")):
                view_id = view_file.stem
                print(f"  Processing {view_id}...")
                
                # Run Stage 1 for this view
                cmd = [
                    sys.executable, "scripts/run_stage1.py",
                    "--input_image", str(view_file),
                    "--output_dir", str(person_output_dir / view_id),
                    "--parsing_model", "ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx",
                    "--pose_model", "ctvo_core/stage1_parsing_pose/pretrained_models/body_pose_model.pth",
                    "--device", "cpu"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                    if result.returncode == 0:
                        print(f"    ✅ {view_id} completed successfully")
                    else:
                        print(f"    ❌ {view_id} failed: {result.stderr}")
                except Exception as e:
                    print(f"    ❌ {view_id} error: {e}")
            
            stage1_outputs[person_id] = str(person_output_dir)
        
        return stage1_outputs
    
    def run_stage2_batch(self, person_dirs: List[str], cloth_dir: str, 
                        stage1_outputs: Dict[str, str]) -> Dict[str, str]:
        """
        Run Stage 2 (Cloth Warping) for all person-cloth combinations
        
        Args:
            person_dirs: List of person directory paths
            cloth_dir: Path to cloth directory
            stage1_outputs: Stage 1 output directories
            
        Returns:
            Dictionary mapping person_id to Stage 2 output directory
        """
        print("\n" + "="*50)
        print("RUNNING STAGE 2: Cloth Warping")
        print("="*50)
        
        stage2_outputs = {}
        cloth_files = list(Path(cloth_dir).glob("cloth_*.jpg"))
        
        for person_dir in person_dirs:
            person_id = Path(person_dir).name
            print(f"\nProcessing {person_id}...")
            
            # Create output directory for this person
            person_output_dir = self.output_dir / "stage2_outputs" / person_id
            person_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each view with each cloth
            for view_file in sorted(Path(person_dir).glob("view_*.jpg")):
                view_id = view_file.stem
                stage1_view_dir = Path(stage1_outputs[person_id]) / view_id
                
                for cloth_file in cloth_files:
                    cloth_id = cloth_file.stem
                    print(f"  Processing {view_id} with {cloth_id}...")
                    
                    # Run Stage 2
                    cmd = [
                        sys.executable, "scripts/run_stage2.py",
                        "--person_img", str(view_file),
                        "--cloth_img", str(cloth_file),
                        "--parsing_map", str(stage1_view_dir / "parsing_maps" / "output.png"),
                        "--pose_json", str(stage1_view_dir / "keypoints_json" / "pose.json"),
                        "--output_path", str(person_output_dir / f"{view_id}_{cloth_id}_warped.jpg"),
                        "--model_checkpoint", "ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth",
                        "--device", "cpu"
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                        if result.returncode == 0:
                            print(f"    ✅ {view_id}_{cloth_id} completed successfully")
                        else:
                            print(f"    ❌ {view_id}_{cloth_id} failed: {result.stderr}")
                    except Exception as e:
                        print(f"    ❌ {view_id}_{cloth_id} error: {e}")
            
            stage2_outputs[person_id] = str(person_output_dir)
        
        return stage2_outputs
    
    def create_training_data(self, stage2_outputs: Dict[str, str]) -> str:
        """
        Create training data structure for Stage 3 and Stage 4
        
        Args:
            stage2_outputs: Stage 2 output directories
            
        Returns:
            Path to training data directory
        """
        print("\n" + "="*50)
        print("CREATING TRAINING DATA STRUCTURE")
        print("="*50)
        
        training_dir = self.output_dir / "training_data"
        training_dir.mkdir(exist_ok=True)
        
        # Create Stage 3 training data
        stage3_dir = training_dir / "stage3"
        stage3_dir.mkdir(exist_ok=True)
        
        # Create Stage 4 training data (NeRF format)
        stage4_dir = training_dir / "stage4"
        stage4_dir.mkdir(exist_ok=True)
        
        # TODO: Implement data preparation for Stage 3 and Stage 4
        print("Training data structure created. Manual preparation needed for Stage 3 & 4.")
        
        return str(training_dir)


def main():
    parser = argparse.ArgumentParser(description="Organize multi-view dataset and run CTVO pipeline")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing your 80 images")
    parser.add_argument("--output_dir", type=str, default="data/multiview_dataset",
                       help="Output directory for organized dataset")
    parser.add_argument("--persons", type=int, default=10,
                       help="Number of persons in dataset")
    parser.add_argument("--views_per_person", type=int, default=8,
                       help="Number of views per person")
    parser.add_argument("--clothes", type=int, default=5,
                       help="Number of cloth images")
    parser.add_argument("--run_pipeline", action="store_true",
                       help="Run the complete pipeline after organization")
    
    args = parser.parse_args()
    
    print("🎯 CTVO Multi-View Dataset Organizer")
    print("="*50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Persons: {args.persons}")
    print(f"Views per person: {args.views_per_person}")
    print(f"Clothes: {args.clothes}")
    
    # Initialize organizer
    organizer = MultiViewDatasetOrganizer(args.input_dir, args.output_dir)
    
    # TODO: You need to provide the actual image paths
    print("\n📋 MANUAL SETUP REQUIRED:")
    print("Please provide the following information:")
    print("1. List of person directories with 8 images each")
    print("2. List of cloth image paths")
    print("3. Update the script with your actual file paths")
    
    # Example structure (you need to update with your actual paths):
    """
    person_data = {
        "person_001": [
            "path/to/person1_view1.jpg",
            "path/to/person1_view2.jpg",
            # ... 8 images total
        ],
        "person_002": [
            "path/to/person2_view1.jpg",
            # ... 8 images total
        ],
        # ... 10 persons total
    }
    
    cloth_data = [
        "path/to/cloth1.jpg",
        "path/to/cloth2.jpg",
        # ... 5 clothes total
    ]
    
    # Organize dataset
    person_dirs = []
    for person_id, image_paths in person_data.items():
        person_dir = organizer.organize_person_images(person_id, image_paths)
        person_dirs.append(person_dir)
    
    organizer.organize_cloth_images(cloth_data)
    metadata_path = organizer.create_dataset_metadata(person_data, cloth_data)
    
    if args.run_pipeline:
        # Run pipeline
        runner = CTVOPipelineRunner(args.input_dir, args.output_dir)
        
        # Stage 1
        stage1_outputs = runner.run_stage1_batch(person_dirs)
        
        # Stage 2
        stage2_outputs = runner.run_stage2_batch(person_dirs, organizer.clothes_dir, stage1_outputs)
        
        # Create training data
        training_dir = runner.create_training_data(stage2_outputs)
        
        print("\n🎉 Pipeline completed!")
        print(f"Results saved to: {args.output_dir}")
    """
    
    print("\n📝 NEXT STEPS:")
    print("1. Update this script with your actual image paths")
    print("2. Run: python scripts/organize_multiview_dataset.py --input_dir YOUR_PATH --run_pipeline")
    print("3. Check results in the output directory")


if __name__ == "__main__":
    main()

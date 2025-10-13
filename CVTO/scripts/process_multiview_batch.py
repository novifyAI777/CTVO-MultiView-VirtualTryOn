#!/usr/bin/env python3
"""
Multi-View Batch Processing Script

This script processes your 80-image dataset (8 images per person) 
through the complete CTVO pipeline.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict
import argparse


class MultiViewBatchProcessor:
    """Processes multi-view dataset through CTVO pipeline"""
    
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.project_root = Path(__file__).parent.parent
        
        # Create output directories
        self.stage1_output = self.output_dir / "stage1"
        self.stage2_output = self.output_dir / "stage2"
        self.stage3_output = self.output_dir / "stage3"
        self.stage4_output = self.output_dir / "stage4"
        
        for output_dir in [self.stage1_output, self.stage2_output, self.stage3_output, self.stage4_output]:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_person_images(self) -> Dict[str, List[Path]]:
        """Find all person images organized by person"""
        person_images = {}
        
        persons_dir = self.dataset_dir / "persons"
        if not persons_dir.exists():
            print(f"❌ Persons directory not found: {persons_dir}")
            return person_images
        
        for person_dir in sorted(persons_dir.iterdir()):
            if person_dir.is_dir():
                person_id = person_dir.name
                view_images = []
                
                # Find all view images
                for view_file in sorted(person_dir.glob("view_*.jpg")):
                    view_images.append(view_file)
                
                if view_images:
                    person_images[person_id] = view_images
                    print(f"✅ Found {len(view_images)} images for {person_id}")
                else:
                    print(f"⚠️  No images found for {person_id}")
        
        return person_images
    
    def find_cloth_images(self) -> List[Path]:
        """Find all cloth images"""
        cloth_images = []
        
        clothes_dir = self.dataset_dir / "clothes"
        if not clothes_dir.exists():
            print(f"❌ Clothes directory not found: {clothes_dir}")
            return cloth_images
        
        for cloth_file in sorted(clothes_dir.glob("cloth_*.jpg")):
            cloth_images.append(cloth_file)
        
        print(f"✅ Found {len(cloth_images)} cloth images")
        return cloth_images
    
    def run_stage1_for_person(self, person_id: str, person_images: List[Path]) -> Dict[str, str]:
        """Run Stage 1 for all views of a person"""
        print(f"\n🎯 Processing {person_id} - Stage 1 (Human Parsing & Pose)")
        print("-" * 50)
        
        person_output_dir = self.stage1_output / person_id
        person_output_dir.mkdir(exist_ok=True)
        
        view_outputs = {}
        
        for i, image_path in enumerate(person_images, 1):
            view_id = f"view_{i:02d}"
            view_output_dir = person_output_dir / view_id
            
            print(f"  Processing {view_id}...")
            
            # Run Stage 1
            cmd = [
                sys.executable, "scripts/run_stage1.py",
                "--input_image", str(image_path),
                "--output_dir", str(view_output_dir),
                "--parsing_model", "ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx",
                "--pose_model", "ctvo_core/stage1_parsing_pose/pretrained_models/body_pose_model.pth",
                "--device", "cpu"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                if result.returncode == 0:
                    print(f"    ✅ {view_id} completed")
                    view_outputs[view_id] = str(view_output_dir)
                else:
                    print(f"    ❌ {view_id} failed: {result.stderr}")
            except Exception as e:
                print(f"    ❌ {view_id} error: {e}")
        
        return view_outputs
    
    def run_stage2_for_person(self, person_id: str, person_images: List[Path], 
                             cloth_images: List[Path], stage1_outputs: Dict[str, str]) -> Dict[str, str]:
        """Run Stage 2 for all person-cloth combinations"""
        print(f"\n🎯 Processing {person_id} - Stage 2 (Cloth Warping)")
        print("-" * 50)
        
        person_output_dir = self.stage2_output / person_id
        person_output_dir.mkdir(exist_ok=True)
        
        combination_outputs = {}
        
        for i, person_image in enumerate(person_images, 1):
            view_id = f"view_{i:02d}"
            stage1_view_dir = Path(stage1_outputs[view_id])
            
            for j, cloth_image in enumerate(cloth_images, 1):
                cloth_id = f"cloth_{j:02d}"
                combination_id = f"{view_id}_{cloth_id}"
                
                print(f"  Processing {combination_id}...")
                
                # Check if Stage 1 outputs exist
                parsing_map = stage1_view_dir / "parsing_maps" / "output.png"
                pose_json = stage1_view_dir / "keypoints_json" / "pose.json"
                
                if not parsing_map.exists() or not pose_json.exists():
                    print(f"    ⚠️  Skipping {combination_id} - Stage 1 outputs missing")
                    continue
                
                # Run Stage 2
                output_path = person_output_dir / f"{combination_id}_warped.jpg"
                
                cmd = [
                    sys.executable, "scripts/run_stage2.py",
                    "--person_img", str(person_image),
                    "--cloth_img", str(cloth_image),
                    "--parsing_map", str(parsing_map),
                    "--pose_json", str(pose_json),
                    "--output_path", str(output_path),
                    "--model_checkpoint", "ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth",
                    "--device", "cpu"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                    if result.returncode == 0:
                        print(f"    ✅ {combination_id} completed")
                        combination_outputs[combination_id] = str(output_path)
                    else:
                        print(f"    ❌ {combination_id} failed: {result.stderr}")
                except Exception as e:
                    print(f"    ❌ {combination_id} error: {e}")
        
        return combination_outputs
    
    def process_all_persons(self, person_images: Dict[str, List[Path]], 
                           cloth_images: List[Path]) -> Dict[str, Dict]:
        """Process all persons through Stage 1 and Stage 2"""
        print("\n" + "="*60)
        print("🚀 STARTING MULTI-VIEW BATCH PROCESSING")
        print("="*60)
        
        all_results = {}
        
        for person_id, images in person_images.items():
            print(f"\n👤 Processing Person: {person_id}")
            print(f"   Images: {len(images)}")
            print(f"   Cloth combinations: {len(images)} × {len(cloth_images)} = {len(images) * len(cloth_images)}")
            
            # Stage 1
            stage1_outputs = self.run_stage1_for_person(person_id, images)
            
            # Stage 2
            stage2_outputs = self.run_stage2_for_person(person_id, images, cloth_images, stage1_outputs)
            
            all_results[person_id] = {
                "stage1": stage1_outputs,
                "stage2": stage2_outputs,
                "total_combinations": len(stage2_outputs)
            }
        
        return all_results
    
    def create_training_metadata(self, results: Dict[str, Dict]) -> str:
        """Create metadata for Stage 3 and Stage 4 training"""
        print("\n📝 Creating training metadata...")
        
        metadata = {
            "dataset_info": {
                "total_persons": len(results),
                "total_combinations": sum(person_data["total_combinations"] for person_data in results.values()),
                "stages_completed": ["stage1", "stage2"]
            },
            "persons": results,
            "training_ready": True
        }
        
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Training metadata saved: {metadata_path}")
        return str(metadata_path)
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print processing summary"""
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        
        total_persons = len(results)
        total_combinations = sum(person_data["total_combinations"] for person_data in results.values())
        
        print(f"👥 Total Persons Processed: {total_persons}")
        print(f"🔄 Total Combinations Created: {total_combinations}")
        print(f"📁 Output Directory: {self.output_dir}")
        
        print(f"\n📋 Per-Person Results:")
        for person_id, person_data in results.items():
            stage1_count = len(person_data["stage1"])
            stage2_count = person_data["total_combinations"]
            print(f"  {person_id}: {stage1_count} views → {stage2_count} combinations")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Review results in: {self.output_dir}")
        print(f"  2. Run Stage 3 training: python scripts/run_stage3.py --mode train")
        print(f"  3. Run Stage 4 training: python scripts/run_stage4.py --mode train")


def main():
    parser = argparse.ArgumentParser(description="Process multi-view dataset through CTVO pipeline")
    parser.add_argument("--dataset_dir", type=str, default="data/multiview_dataset",
                       help="Directory containing organized dataset")
    parser.add_argument("--output_dir", type=str, default="results/multiview_batch",
                       help="Output directory for results")
    parser.add_argument("--stages", type=str, nargs="+", default=["stage1", "stage2"],
                       choices=["stage1", "stage2", "stage3", "stage4"],
                       help="Stages to run")
    
    args = parser.parse_args()
    
    print("🎯 CTVO Multi-View Batch Processor")
    print("="*50)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stages to run: {', '.join(args.stages)}")
    
    # Initialize processor
    processor = MultiViewBatchProcessor(args.dataset_dir, args.output_dir)
    
    # Find images
    person_images = processor.find_person_images()
    cloth_images = processor.find_cloth_images()
    
    if not person_images:
        print("❌ No person images found. Please organize your dataset first.")
        print("   Run: python scripts/setup_multiview_dataset.py")
        return
    
    if not cloth_images:
        print("❌ No cloth images found. Please add cloth images to the clothes directory.")
        return
    
    # Process dataset
    if "stage1" in args.stages or "stage2" in args.stages:
        results = processor.process_all_persons(person_images, cloth_images)
        
        # Create training metadata
        metadata_path = processor.create_training_metadata(results)
        
        # Print summary
        processor.print_summary(results)
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📁 Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()

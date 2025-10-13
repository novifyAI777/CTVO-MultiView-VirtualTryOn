"""
Test Dataset Integrity

This module tests the integrity of datasets and data loading functionality.
"""

import unittest
import os
import json
import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ctvo_core.utils.data_loader import (
    Stage1Dataset,
    Stage2Dataset, 
    Stage3Dataset,
    Stage4Dataset,
    create_dataloader,
    split_dataset
)


class TestDatasetIntegrity(unittest.TestCase):
    """Test dataset integrity and data loading"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data_dir = "tests/test_data"
        self.create_test_data()
    
    def create_test_data(self):
        """Create test data for testing"""
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create dummy images
        dummy_img = Image.new('RGB', (256, 192), color='red')
        dummy_mask = Image.new('RGB', (256, 192), color='blue')
        
        # Save dummy images
        dummy_img.save(os.path.join(self.test_data_dir, "person_001.jpg"))
        dummy_img.save(os.path.join(self.test_data_dir, "cloth_001.jpg"))
        dummy_mask.save(os.path.join(self.test_data_dir, "parsing_001.png"))
        
        # Create dummy pose JSON
        pose_data = {
            "version": 1.0,
            "people": [{
                "person_id": 0,
                "pose_keypoints_2d": [100, 100, 0.9] * 18  # 18 keypoints
            }]
        }
        
        with open(os.path.join(self.test_data_dir, "pose_001.json"), 'w') as f:
            json.dump(pose_data, f)
        
        # Create metadata files
        self.create_metadata_files()
    
    def create_metadata_files(self):
        """Create metadata files for different stages"""
        
        # Stage 1 metadata
        stage1_metadata = {
            "samples": [{
                "id": "001",
                "person_image": "person_001.jpg",
                "parsing_map": "parsing_001.png",
                "pose_json": "pose_001.json"
            }]
        }
        
        with open(os.path.join(self.test_data_dir, "stage1_metadata.json"), 'w') as f:
            json.dump(stage1_metadata, f)
        
        # Stage 2 metadata
        stage2_metadata = {
            "samples": [{
                "id": "001",
                "person_image": "person_001.jpg",
                "cloth_image": "cloth_001.jpg",
                "parsing_map": "parsing_001.png",
                "pose_json": "pose_001.json",
                "warped_cloth": "cloth_001.jpg"  # Using same for test
            }]
        }
        
        with open(os.path.join(self.test_data_dir, "stage2_metadata.json"), 'w') as f:
            json.dump(stage2_metadata, f)
        
        # Stage 3 metadata
        stage3_metadata = {
            "samples": [{
                "id": "001",
                "person_image": "person_001.jpg",
                "warped_cloth": "cloth_001.jpg",
                "parsing_map": "parsing_001.png",
                "target_image": "person_001.jpg",  # Using same for test
                "pose_json": "pose_001.json"
            }]
        }
        
        with open(os.path.join(self.test_data_dir, "stage3_metadata.json"), 'w') as f:
            json.dump(stage3_metadata, f)
        
        # Stage 4 metadata
        stage4_metadata = {
            "samples": [{
                "id": "001",
                "reference_image": "person_001.jpg",
                "multiview_images": ["person_001.jpg", "person_001.jpg"],
                "camera_poses": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
            }]
        }
        
        with open(os.path.join(self.test_data_dir, "stage4_metadata.json"), 'w') as f:
            json.dump(stage4_metadata, f)
    
    def test_stage1_dataset(self):
        """Test Stage 1 dataset"""
        dataset = Stage1Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage1_metadata.json")
        )
        
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertIn('person_img', sample)
        self.assertIn('parsing_map', sample)
        self.assertIn('pose_data', sample)
        self.assertIn('sample_id', sample)
        
        # Check tensor shapes
        self.assertEqual(sample['person_img'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['parsing_map'].shape, torch.Size([3, 192, 256]))
        self.assertIsInstance(sample['pose_data'], dict)
    
    def test_stage2_dataset(self):
        """Test Stage 2 dataset"""
        dataset = Stage2Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage2_metadata.json")
        )
        
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertIn('person_img', sample)
        self.assertIn('cloth_img', sample)
        self.assertIn('parsing_map', sample)
        self.assertIn('pose_data', sample)
        self.assertIn('warped_cloth', sample)
        
        # Check tensor shapes
        self.assertEqual(sample['person_img'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['cloth_img'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['parsing_map'].shape, torch.Size([3, 192, 256]))
    
    def test_stage3_dataset(self):
        """Test Stage 3 dataset"""
        dataset = Stage3Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage3_metadata.json")
        )
        
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertIn('person_img', sample)
        self.assertIn('warped_cloth', sample)
        self.assertIn('parsing_map', sample)
        self.assertIn('target_img', sample)
        self.assertIn('pose_data', sample)
        
        # Check tensor shapes
        self.assertEqual(sample['person_img'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['warped_cloth'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['target_img'].shape, torch.Size([3, 192, 256]))
    
    def test_stage4_dataset(self):
        """Test Stage 4 dataset"""
        dataset = Stage4Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage4_metadata.json")
        )
        
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertIn('reference_img', sample)
        self.assertIn('multiview_images', sample)
        self.assertIn('camera_poses', sample)
        
        # Check tensor shapes
        self.assertEqual(sample['reference_img'].shape, torch.Size([3, 192, 256]))
        self.assertEqual(sample['multiview_images'].shape, torch.Size([2, 3, 192, 256]))
        self.assertEqual(sample['camera_poses'].shape, torch.Size([2, 4, 4]))
    
    def test_dataloader_creation(self):
        """Test data loader creation"""
        dataset = Stage1Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage1_metadata.json")
        )
        
        dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertEqual(len(dataloader), 1)
        
        # Test iteration
        for batch in dataloader:
            self.assertIn('person_img', batch)
            self.assertEqual(batch['person_img'].shape, torch.Size([1, 3, 192, 256]))
            break
    
    def test_dataset_splitting(self):
        """Test dataset splitting"""
        dataset = Stage1Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "stage1_metadata.json")
        )
        
        # Create multiple samples for testing
        extended_metadata = {
            "samples": [
                {"id": "001", "person_image": "person_001.jpg", "parsing_map": "parsing_001.png", "pose_json": "pose_001.json"},
                {"id": "002", "person_image": "person_001.jpg", "parsing_map": "parsing_001.png", "pose_json": "pose_001.json"},
                {"id": "003", "person_image": "person_001.jpg", "parsing_map": "parsing_001.png", "pose_json": "pose_001.json"},
                {"id": "004", "person_image": "person_001.jpg", "parsing_map": "parsing_001.png", "pose_json": "pose_001.json"},
                {"id": "005", "person_image": "person_001.jpg", "parsing_map": "parsing_001.png", "pose_json": "pose_001.json"}
            ]
        }
        
        with open(os.path.join(self.test_data_dir, "extended_metadata.json"), 'w') as f:
            json.dump(extended_metadata, f)
        
        extended_dataset = Stage1Dataset(
            self.test_data_dir,
            os.path.join(self.test_data_dir, "extended_metadata.json")
        )
        
        train_dataset, val_dataset, test_dataset = split_dataset(
            extended_dataset, 0.6, 0.2, 0.2
        )
        
        self.assertEqual(len(train_dataset), 3)
        self.assertEqual(len(val_dataset), 1)
        self.assertEqual(len(test_dataset), 1)
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Lightning AI Training Script for CTVO
This script is optimized to run directly on Lightning AI Studio

Usage:
    python lightning_train.py
    python lightning_train.py --epochs 50 --batch_size 16
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """Check if environment is set up correctly"""
    print("=" * 60)
    print("CTVO Training Environment Check")
    print("=" * 60)
    
    # Check CUDA
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Fix: Use getattr to avoid type checker error with torch.version.cuda
        cuda_version = getattr(torch.version, 'cuda', 'N/A')
        print(f"CUDA Version: {cuda_version}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: CUDA not available! Training will be slow on CPU.")
    
    # Check project structure
    print(f"\nProject Root: {project_root}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check config file
    config_path = project_root / "configs" / "stage3_fusion.yaml"
    if config_path.exists():
        print(f"✓ Config file found: {config_path}")
    else:
        print(f"✗ Config file not found: {config_path}")
    
    # Check data directory
    data_dir = project_root / "data" / "custom_dataset"
    if data_dir.exists():
        print(f"✓ Data directory exists: {data_dir}")
    else:
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   You may need to upload your data to the Studio")
    
    print("=" * 60)
    print()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CTVO Training on Lightning AI")
    parser.add_argument("--config", type=str, default="configs/stage3_fusion.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "inference"],
                       help="Mode to run")
    parser.add_argument("--data_dir", type=str, default="data/custom_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/stage3_previews",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                       help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for resume/eval")
    parser.add_argument("--test_run", action="store_true",
                       help="Run a quick test (1 epoch, small dataset)")
    
    args = parser.parse_args()
    
    # Check environment
    check_environment()
    
    # Adjust for test run
    if args.test_run:
        print("🧪 Running in TEST MODE (1 epoch, limited batches)")
        args.num_epochs = 1
        args.batch_size = min(args.batch_size, 4)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Prepare arguments for run_stage3
    sys.argv = [
        'lightning_train.py',
        '--config', args.config,
        '--mode', args.mode,
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
    ]
    
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    
    # Import and run
    try:
        # Import the run_stage3 module
        import importlib.util
        script_path = project_root / "scripts" / "run_stage3.py"
        spec = importlib.util.spec_from_file_location("run_stage3", script_path)
        
        # Fix: Check if spec is None before using it
        if spec is None:
            raise ImportError(f"Could not load spec from {script_path}")
        
        # Fix: Check if spec.loader is None before using it
        if spec.loader is None:
            raise ImportError(f"Could not get loader from spec for {script_path}")
        
        run_stage3_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_stage3_module)
        run_stage3 = run_stage3_module.main
        
        print("\n🚀 Starting CTVO Training...")
        print(f"   Mode: {args.mode}")
        print(f"   Device: {args.device}")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Epochs: {args.num_epochs}")
        print(f"   Learning Rate: {args.learning_rate}")
        print()
        
        run_stage3()
        
        print("\n✅ Training completed successfully!")
        print(f"   Checkpoints saved to: checkpoints/stage3_fusion/")
        print(f"   Results saved to: {args.output_dir}")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Make sure you're in the CVTO directory and dependencies are installed")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
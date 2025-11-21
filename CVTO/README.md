# CTVO - Hybrid Virtual Try-On Pipeline

A modular, hybrid architecture for virtual try-on applications combining traditional computer vision with modern deep learning techniques.

## 🏗️ Architecture Overview

CTVO implements a 4-stage pipeline for high-quality virtual try-on:

- **Stage 1**: Human Parsing & Pose Estimation
- **Stage 2**: Cloth Warping  
- **Stage 3**: Fusion Generation
- **Stage 4**: NeRF Multi-view Rendering

## 📁 Project Structure

```
ctvo-project/
│
├── ctvo_core/                               # Main source package
│   ├── stage1_parsing_pose/                 # Human parsing & pose estimation
│   │   ├── model_parsing.py
│   │   ├── model_pose.py
│   │   ├── run_pose.py
│   │   └── __init__.py
│   │
│   ├── stage2_cloth_warping/                # Cloth warping
│   │   ├── UNet.py
│   │   ├── GMM.py
│   │   ├── utils.py
│   │   ├── run_warp.py
│   │   └── pretrained_weights/
│   │       └── unet_wrap.pth
│   │
│   ├── stage3_fusion/                       # Fusion generation
│   │   ├── TryOnGenerator.py
│   │   ├── FusionNet.py
│   │   ├── losses.py
│   │   ├── train_fusion.py
│   │   ├── eval_fusion.py
│   │   └── __init__.py
│   │
│   ├── stage4_nerf/                        # NeRF multi-view
│   │   ├── model_nerf.py
│   │   ├── renderer.py
│   │   ├── dataset_nerf.py
│   │   ├── train_nerf.py
│   │   ├── eval_multiview.py
│   │   └── __init__.py
│   │
│   ├── losses/                              # Shared loss functions
│   │   ├── perceptual_loss.py
│   │   ├── style_loss.py
│   │   ├── mask_losses.py
│   │   └── __init__.py
│   │
│   ├── utils/                               # Shared utilities
│   │   ├── image_io.py
│   │   ├── data_loader.py
│   │   ├── visualizer.py
│   │   ├── logger.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── data/                                    # Data directory
│   ├── custom_dataset/
│   │   ├── train/
│   │   │   ├── image/
│   │   │   ├── cloth/
│   │   │   ├── image-parse/
│   │   │   ├── agnostic-parse/
│   │   │   ├── openpose/
│   │   │   ├── warped_cloth/
│   │   │   ├── pairs.txt
│   │   │   └── meta.json
│   │   └── test/
│   │       ├── image/
│   │       ├── cloth/
│   │       ├── image-parse/
│   │       ├── openpose/
│   │       └── warped_cloth/
│   │
│   ├── synthetic_augmented/
│   │   ├── zero123/
│   │   ├── triposr/
│   │   ├── depth/
│   │   └── readme.txt
│   │
│   └── marquis_viton_hd/
│       ├── train/
│       ├── test/
│       └── metadata.txt
│
├── results/                                 # Results directory
│   ├── stage2_samples/
│   ├── stage3_previews/
│   └── stage4_multiview/
│
├── configs/                                 # Configuration files
│   ├── base.yaml
│   ├── stage3_fusion.yaml
│   ├── stage4_nerf.yaml
│   └── lightning_trainer.yaml
│
├── scripts/                                 # Run scripts
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_stage3.py
│   ├── run_stage4.py
│   └── train_all.sh
│
├── tests/                                   # Test files
│   ├── test_imports.py
│   ├── test_losses.py
│   └── test_dataset_integrity.py
│
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ctvo-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
```bash
# Download Stage 1 models
wget <parsing-model-url> -O ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx
wget <pose-model-url> -O ctvo_core/stage1_parsing_pose/pretrained_models/body_pose_model.pth

# Download Stage 2 models
wget <warp-model-url> -O ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth
```

### Running Individual Stages

#### Stage 1: Human Parsing & Pose Estimation
```bash
python scripts/run_stage1.py \
    --input_image data/person.jpg \
    --output_dir results/stage1 \
    --visualize
```

#### Stage 2: Cloth Warping
```bash
python scripts/run_stage2.py \
    --person_img data/person.jpg \
    --cloth_img data/cloth.jpg \
    --parsing_map results/stage1/parsing_maps/output.png \
    --pose_json results/stage1/keypoints_json/pose.json \
    --output_path results/stage2/warped_cloth.jpg \
    --visualize
```

#### Stage 3: Fusion Generation
```bash
# Training
python scripts/run_stage3.py \
    --mode train \
    --data_dir data/custom_dataset \
    --config configs/stage3_fusion.yaml

# Evaluation
python scripts/run_stage3.py \
    --mode eval \
    --checkpoint checkpoints/stage3_fusion/best_model.pth \
    --data_dir data/custom_dataset \
    --output_dir results/stage3_previews
```

#### Stage 4: NeRF Multi-view Generation
```bash
# Training
python scripts/run_stage4.py \
    --mode train \
    --data_dir data/synthetic_augmented \
    --config configs/stage4_nerf.yaml

# Evaluation
python scripts/run_stage4.py \
    --mode eval \
    --checkpoint checkpoints/stage4_nerf/best_model.pth \
    --output_dir results/stage4_multiview \
    --num_views 8
```

### Training All Stages
```bash
bash scripts/train_all.sh
```

## 🧪 Testing

Run the test suite to validate the installation:

```bash
# Test imports
python tests/test_imports.py

# Test loss functions
python tests/test_losses.py

# Test dataset integrity
python tests/test_dataset_integrity.py
```

## 📊 Configuration

The pipeline uses YAML configuration files for easy customization:

- `configs/base.yaml`: Base configuration shared across stages
- `configs/stage3_fusion.yaml`: Stage 3 specific settings
- `configs/stage4_nerf.yaml`: Stage 4 specific settings
- `configs/lightning_trainer.yaml`: PyTorch Lightning trainer settings

## 🔧 Key Features

### Stage 1: Human Parsing & Pose Estimation
- ONNX-based human parsing using LIP/ATR models
- MobileNet-based pose estimation
- Integrated processing pipeline

### Stage 2: Cloth Warping
- UNet-based cloth warping
- GMM (Geometric Matching Module) support
- Pose-aware warping

### Stage 3: Fusion Generation
- Multiple fusion architectures (TryOnGenerator, FusionNet)
- Advanced loss functions (perceptual, style, mask-aware)
- PyTorch Lightning integration

### Stage 4: NeRF Multi-view Generation
- Neural Radiance Fields for multi-view rendering
- Volume rendering pipeline
- Camera path generation

### Shared Components
- Comprehensive loss functions
- Data loading utilities
- Visualization tools
- Logging system

## 📈 Performance

The pipeline is designed for:
- **Efficiency**: Modular design allows for stage-wise optimization
- **Scalability**: PyTorch Lightning for distributed training
- **Quality**: Advanced loss functions and multi-view consistency
- **Flexibility**: Configurable architecture and training parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenPose for pose estimation models
- VITON for cloth warping techniques
- NeRF for neural rendering
- PyTorch Lightning for training infrastructure

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.
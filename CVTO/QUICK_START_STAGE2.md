# Quick Start Guide: Stage 2 - Cloth Warping

This guide will help you run Stage 2 (Cloth Warping) on your multiview dataset.

## Prerequisites

1. **Stage 1 outputs**: Make sure you have completed Stage 1 processing. You need:
   - Parsing maps in `data/multiview_dataset/stage1_outputs/parsing_maps/`
   - Pose tensors in `data/multiview_dataset/stage1_outputs/pose_heatmaps/`

2. **Cloth images**: Flat cloth images should be in `data/multiview_dataset/clothes/`

3. **Pretrained model**: The GMM (Geometric Matching Module) warping model should be at:
   - `ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth`
   - **Note**: Despite the filename, this is actually a VITON-compatible GMM model, not a UNet

## Quick Start

### Option 1: Using the Batch File (Windows)

Simply double-click or run:
```batch
run_stage2_batch.bat
```

This will process all person-cloth combinations in your dataset.

### Option 2: Using Python Script

```bash
python scripts/run_stage2_batch.py
```

### Advanced Usage

#### Process specific gender/tier:
```bash
python scripts/run_stage2_batch.py --gender Men --tier "Tier 1"
```

#### Use GPU (if available):
```bash
python scripts/run_stage2_batch.py --device cuda
```

#### Process single combination:
```bash
python scripts/run_stage2.py \
    --person_img "path/to/person.png" \
    --cloth_img "path/to/cloth.png" \
    --parsing_map "path/to/parsing.png" \
    --pose_json "path/to/pose.pt" \
    --output_path "path/to/output.png"
```

## Output Structure

Results will be saved to `data/multiview_dataset/stage2_outputs/` with the same directory structure as input images.

Each warped cloth image will be named: `{person_name}_{cloth_name}_warped.png`

## Configuration Options

- `--person_images_dir`: Directory containing person images (default: `data/multiview_dataset/images/train`)
- `--cloth_images_dir`: Directory containing cloth images (default: `data/multiview_dataset/clothes`)
- `--stage1_parsing_dir`: Directory with Stage 1 parsing maps (default: `data/multiview_dataset/stage1_outputs/parsing_maps`)
- `--stage1_pose_dir`: Directory with Stage 1 pose tensors (default: `data/multiview_dataset/stage1_outputs/pose_heatmaps`)
- `--output_dir`: Output directory (default: `data/multiview_dataset/stage2_outputs`)
- `--model_checkpoint`: Path to model checkpoint (default: `ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth`)
- `--model_type`: Model type - "unet" or "gmm" (default: "gmm" - the unet_wrap.pth checkpoint is actually a VITON GMM model)
- `--device`: Device to use - "cpu" or "cuda" (default: "cpu")
- `--gender`: Filter by gender - "all", "Men", or "Women" (default: "all")
- `--tier`: Filter by tier - "all", "Tier 1", "Tier 2", "Tier 3" (default: "all")
- `--skip_existing`: Skip combinations that already have outputs
- `--max_combinations`: Limit number of combinations (for testing)

## Troubleshooting

### Error: "Parsing map not found"
- Make sure Stage 1 has been run successfully
- Check that parsing maps exist in the expected directory structure

### Error: "Pose tensor not found"
- Make sure Stage 1 has been run successfully
- Check that pose tensors (.pt files) exist in the expected directory structure

### Error: "Model checkpoint not found"
- Verify that `unet_wrap.pth` exists in `ctvo_core/stage2_cloth_warping/pretrained_weights/`
- If missing, you may need to download it or train the model first

### Low quality warping results
- Ensure Stage 1 outputs are of good quality
- Check that cloth images are flat/unwarped
- Try using GPU for better performance: `--device cuda`

## Next Steps

After Stage 2 is complete, you can proceed to:
- **Stage 3**: Fusion Generation (combining warped cloth with person image)
- **Stage 4**: NeRF-based Multi-view Rendering

## Notes

- Stage 2 automatically handles pose tensor (.pt) files from Stage 1 - no conversion needed
- The script will match cloth images to person images based on gender and tier when possible
- Processing time depends on number of combinations and device (CPU vs GPU)


# Quick Start: Stage 1 Implementation

This guide will help you quickly set up and run Stage 1 of the CTVO pipeline.

## Step 1: Download Pretrained Models

You need **two pretrained models** for Stage 1:

### 1. Pose Estimation Model (Automatic Download)

**Windows:**
```batch
scripts\download_stage1_models.bat
```

**Linux/Mac:**
```bash
python scripts/download_stage1_models.py
```

This will automatically download the pose model. The parsing model needs to be downloaded manually (see below).

### 2. Human Parsing Model (Manual Download)

You need to download the human parsing model manually. Here are the best options:

**Option A: Self-Correction-Human-Parsing (Recommended)**
1. Visit: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
2. Download the pretrained model
3. Convert to ONNX format (see STAGE1_SETUP.md for conversion script)
4. Save as: `ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx`

**Option B: Direct ONNX Download**
- Search for "human parsing ONNX" on HuggingFace or GitHub
- Download a pre-converted ONNX model
- Save as: `ctvo_core/stage1_parsing_pose/pretrained_models/parsing_lip.onnx`

**Quick Download Links (if available):**
- Check HuggingFace: https://huggingface.co/models?search=human+parsing
- Check ONNX Model Zoo: https://github.com/onnx/models

## Step 2: Verify Models

Check that both models exist:

**Windows:**
```batch
dir ctvo_core\stage1_parsing_pose\pretrained_models\
```

**Linux/Mac:**
```bash
ls -lh ctvo_core/stage1_parsing_pose/pretrained_models/
```

You should see:
- `body_pose_model.pth` (~40-50 MB)
- `parsing_lip.onnx` (~100-200 MB)

## Step 3: Test with Single Image

Test Stage 1 with one image first:

**Windows:**
```batch
python scripts\run_stage1.py --input_image "data\multiview_dataset\images\train\Men\Tier 1\<your_image>.png" --output_dir "results\stage1_test" --visualize
```

**Linux/Mac:**
```bash
python scripts/run_stage1.py --input_image "data/multiview_dataset/images/train/Men/Tier 1/your_image.png" --output_dir "results/stage1_test" --visualize
```

If this works, you'll see:
- Parsing map saved
- Pose keypoints JSON saved
- Visualization image (if `--visualize` flag used)

## Step 4: Process All Images

Once testing works, process all your images:

**Windows:**
```batch
run_stage1_batch.bat
```

**Or manually:**
```batch
python scripts\run_stage1_batch.py --input_dir "data\multiview_dataset\images\train" --output_dir "data\multiview_dataset\stage1_outputs"
```

**Linux/Mac:**
```bash
python scripts/run_stage1_batch.py --input_dir "data/multiview_dataset/images/train" --output_dir "data/multiview_dataset/stage1_outputs"
```

### Processing Options

Process specific genders or tiers:
```bash
# Process only Men
python scripts/run_stage1_batch.py --gender Men

# Process only Tier 1
python scripts/run_stage1_batch.py --tier "Tier 1"

# Process only Women, Tier 2
python scripts/run_stage1_batch.py --gender Women --tier "Tier 2"

# Skip already processed images
python scripts/run_stage1_batch.py --skip_existing

# Use GPU (if available)
python scripts/run_stage1_batch.py --device cuda
```

## Step 5: Verify Results

Check the output directory structure:
```
data/multiview_dataset/stage1_outputs/
├── Men/
│   ├── Tier 1/
│   │   └── <outfit_name>/
│   │       ├── <image_name>_parsing.png
│   │       └── <image_name>_pose.json
│   ├── Tier 2/
│   └── Tier 3/
└── Women/
    ├── Tier 1/
    ├── Tier 2/
    └── Tier 3/
```

Also check `stage1_batch_summary.json` for processing statistics.

## Troubleshooting

### "Model not found" Error
- Ensure both model files are in `ctvo_core/stage1_parsing_pose/pretrained_models/`
- Check file names match exactly

### "ONNX Runtime" Error
- Install: `pip install onnxruntime` (or `onnxruntime-gpu` for GPU)

### "Out of Memory" Error
- Use CPU: `--device cpu`
- Process in smaller batches (use `--tier` or `--gender` filters)

### Processing is Slow
- Use GPU if available: `--device cuda`
- Process in parallel batches (split by tier/gender)

## Next Steps

After Stage 1 completes:
1. ✅ Verify parsing maps and pose keypoints look correct
2. ✅ Check `stage1_batch_summary.json` for any errors
3. ➡️ Proceed to Stage 2 (Cloth Warping)

## Need Help?

- See `STAGE1_SETUP.md` for detailed model download instructions
- Check `scripts/run_stage1.py` for single image processing
- Review error messages in `stage1_batch_summary.json`


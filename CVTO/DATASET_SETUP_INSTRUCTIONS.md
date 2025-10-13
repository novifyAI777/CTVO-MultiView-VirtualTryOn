# 🎯 CTVO Multi-View Dataset Setup Instructions

## 📁 **Your Dataset Structure (80 Images = 8 Images per Person)**

Your dataset should be organized as follows:

```
data/multiview_dataset/
├── persons/
│   ├── person_001/
│   │   ├── view_01/     ← Put person 1, view 1 image here
│   │   ├── view_02/     ← Put person 1, view 2 image here
│   │   ├── view_03/     ← Put person 1, view 3 image here
│   │   ├── view_04/     ← Put person 1, view 4 image here
│   │   ├── view_05/     ← Put person 1, view 5 image here
│   │   ├── view_06/     ← Put person 1, view 6 image here
│   │   ├── view_07/     ← Put person 1, view 7 image here
│   │   └── view_08/     ← Put person 1, view 8 image here
│   ├── person_002/
│   │   ├── view_01/     ← Put person 2, view 1 image here
│   │   ├── view_02/     ← Put person 2, view 2 image here
│   │   └── ... (8 views total)
│   └── ... (10 persons total)
└── clothes/
    ├── cloth_01.jpg     ← Put your cloth images here
    ├── cloth_02.jpg
    ├── cloth_03.jpg
    ├── cloth_04.jpg
    └── cloth_05.jpg
```

## 🚀 **Step-by-Step Instructions**

### 1️⃣ **Organize Your Images**

**Copy your 80 person images to the correct directories:**

- **Person 1, View 1** → `data/multiview_dataset/persons/person_001/view_01/`
- **Person 1, View 2** → `data/multiview_dataset/persons/person_001/view_02/`
- **Person 1, View 3** → `data/multiview_dataset/persons/person_001/view_03/`
- **Person 1, View 4** → `data/multiview_dataset/persons/person_001/view_04/`
- **Person 1, View 5** → `data/multiview_dataset/persons/person_001/view_05/`
- **Person 1, View 6** → `data/multiview_dataset/persons/person_001/view_06/`
- **Person 1, View 7** → `data/multiview_dataset/persons/person_001/view_07/`
- **Person 1, View 8** → `data/multiview_dataset/persons/person_001/view_08/`

**Repeat for all 10 persons (person_002 through person_010)**

**Copy your cloth images to:**
- `data/multiview_dataset/clothes/cloth_01.jpg`
- `data/multiview_dataset/clothes/cloth_02.jpg`
- `data/multiview_dataset/clothes/cloth_03.jpg`
- `data/multiview_dataset/clothes/cloth_04.jpg`
- `data/multiview_dataset/clothes/cloth_05.jpg`

### 2️⃣ **Fix Python PATH Issue**

**Option A: Use Full Python Path**
```bash
# Find your Python installation
where python
# or
where python3

# Use the full path, for example:
C:\Python39\python.exe scripts/setup_multiview_dataset.py
```

**Option B: Add Python to PATH**
1. Open System Properties → Environment Variables
2. Add Python installation directory to PATH
3. Restart command prompt

**Option C: Use Anaconda/Miniconda**
```bash
conda activate your_env_name
python scripts/setup_multiview_dataset.py
```

### 3️⃣ **Run Stage 1 (Human Parsing & Pose Estimation)**

**Test with one image first:**
```bash
python scripts/run_stage1.py --input_image "data/multiview_dataset/persons/person_001/view_01/YOUR_IMAGE.jpg" --output_dir "results/stage1_test" --visualize
```

**Run for all images (after organizing):**
```bash
python scripts/process_multiview_batch.py --dataset_dir "data/multiview_dataset" --output_dir "results/multiview_batch"
```

### 4️⃣ **Run Stage 2 (Cloth Warping)**

**Test with one combination:**
```bash
python scripts/run_stage2.py --person_img "data/multiview_dataset/persons/person_001/view_01/YOUR_IMAGE.jpg" --cloth_img "data/multiview_dataset/clothes/cloth_01.jpg" --parsing_map "results/stage1_test/parsing_maps/output.png" --pose_json "results/stage1_test/keypoints_json/pose.json" --output_path "results/stage2_test/warped_cloth.jpg" --visualize
```

**Run for all combinations (after Stage 1):**
```bash
# This will be handled by the batch processor
python scripts/process_multiview_batch.py --dataset_dir "data/multiview_dataset" --output_dir "results/multiview_batch"
```

### 5️⃣ **Run Stage 3 (Fusion Generation)**

**Training:**
```bash
python scripts/run_stage3.py --mode train --data_dir "data/multiview_dataset" --config "configs/stage3_fusion.yaml"
```

**Evaluation:**
```bash
python scripts/run_stage3.py --mode eval --checkpoint "checkpoints/stage3_fusion/best_model.ckpt" --data_dir "data/multiview_dataset" --output_dir "results/stage3_previews"
```

### 6️⃣ **Run Stage 4 (NeRF Multi-view Generation)**

**Training:**
```bash
python scripts/run_stage4.py --mode train --data_dir "data/multiview_dataset" --config "configs/stage4_nerf.yaml"
```

**Evaluation:**
```bash
python scripts/run_stage4.py --mode eval --checkpoint "checkpoints/stage4_nerf/best_model.ckpt" --output_dir "results/stage4_multiview" --num_views 8
```

## 🔧 **Quick Commands (After Organizing Data)**

**Run all stages:**
```bash
bash scripts/train_all.sh
```

**Test everything works:**
```bash
python tests/test_imports.py
```

## 📊 **Expected Results**

After running the pipeline, you should have:

- **Stage 1**: Human parsing maps and pose keypoints for all 80 images
- **Stage 2**: Warped cloth images for all person-cloth combinations (80 × 5 = 400 combinations)
- **Stage 3**: Fused virtual try-on results
- **Stage 4**: Multi-view rendered results

## ⚠️ **Important Notes**

1. **Image Format**: Use JPG format for all images
2. **Image Size**: Recommended 256×192 or 512×384 pixels
3. **File Names**: Keep original names, the script will handle organization
4. **Python Path**: Make sure Python is accessible from command line
5. **Dependencies**: Install requirements.txt first: `pip install -r requirements.txt`

## 🆘 **Troubleshooting**

**If Python not found:**
- Use full path: `C:\Python39\python.exe` instead of `python`
- Install Python from python.org
- Use Anaconda/Miniconda

**If scripts fail:**
- Check file paths are correct
- Ensure all dependencies are installed
- Run tests first: `python tests/test_imports.py`

**If images not found:**
- Double-check directory structure
- Verify file names and extensions
- Check file permissions

## 🎉 **Success Indicators**

✅ All 80 images organized in correct directories  
✅ Python commands work without "not found" errors  
✅ Stage 1 produces parsing maps and pose keypoints  
✅ Stage 2 produces warped cloth images  
✅ Stage 3 trains fusion model successfully  
✅ Stage 4 generates multi-view results  

---

**Ready to start? Begin with organizing your 80 images into the directory structure above!**

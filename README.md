# Pancreas Segmentation and Classification with multitask nnUNet

## Overview
This repository extends **nnUNetv2** to perform *multi-task learning* on pancreas CT scans:
- **Semantic segmentation** (pancreas, lesion)
- **Subtype classification** (3-class problem)

The project modifies nnU-Net’s architecture by adding a **classification head** on top of the shared 3D ResEnc-M encoder, while keeping the standard decoder for segmentation.

## Repository Contents

| File | Description |
|------|-------------|
| `dataset_conversion.py` | Converts original dataset into nnUNetv2 format and generates metadata files. |
| `nnUNetTrainer_multitask.py` | Custom multitask trainer extending nnUNetTrainer (segmentation + classification). |
| `multitask_predict.py` | Standalone inference script for multi-task prediction. |
| `comparing_speed.py` | Script to benchmark baseline vs optimized inference speed. |
  
## Environments and Requirements
**Run Environment**
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4090 24GB
- CUDA version: 12.9
- python version: 3.12.12

To install environments:
```setup
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Clone nnUNet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

pip install -r requirements.txt
```

Set nnUNet paths (Windows Command Prompt)
```
set nnUNet_raw=nnUNet_data/nnUNet_raw
set nnUNet_preprocessed=nnUNet_data/nnUNet_preprocessed
set nnUNet_results=nnUNet_data/nnUNet_results
```

## Data Preparation
`dataset_conversion.py`
1. Place original data into imagesTr, imagesTs
2. Place segmentation mask into labelsTr
3. Generate:
  * dataset.json
  * splits_final.json 
  * classification_labels.csv
  
```python
python dataset_conversion.py \
    --src_dir "/path/to/original/data" \
    --nnUNet_dir "/nnUNet/data/folder/" \
    --dataset_id 101
```

## Preprocessing
```
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity -pl nnUNetPlannerResEncM
```

## Train
`nnUNetTrainer_multitask.py`

**Move nnUNetTrainer_multitask.py into "/nnunetv2/training/nnUNetTrainer"**

Then start training with:
```
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_multitask -p nnUNetResEncUNetMPlans
```

## Evaluation
Validation metrics (following Metrics Reloaded) are computed automatically after training process and stored at: ""nnUNet_results/<dataset_folder>/<model_folder>/fold_0/final_metrics.json""

- Segmentation metrics:
  * Dice score (dsc)
  * Normalized surface dice (nsd)
  * Absolute volume different (avd)
- Classification metrics:
  * Balanced accuracy (balanced_accuracy_score)
  * Macro average F1 (f1_macro)
  * AUROC (auroc)
  * Likelihood ratio+ (lr_plus)
  * Expected calibration error (ece)
  * Brier score (brier_score)

## Results

| Segmentation               |  Pancreas (label > 0)  |   Lesion (label = 2)   |
| :------------------------- | :--------: | :--------: |
| Dice                       |  0.89      |       0.58     | 

| Classification             |            |
| :------------------------- | :--------: |
| F1 macro                   |    0.19        | 

## Inference
`multitask_predict.py`
This repo uses a standalone multi-task inference script (multitask_predict.py) instead of nnUNetPredictor (due to multi-head architecture).
multitask_predict.py 
```
python multitask_predict.py \
    --model_folder nnUNet_results/<dataset_folder>/<model_folder> \
    --input_folder nnUNet_data/nnUNet_raw/<dataset_folder>/imagesTs
    --output_folder predictions --step_size 0.5 --enable_tta
```
Faster inference
* Increase --step_size (e.g., 0.85)
* Remove --enable_tta

Comparing speed between BASELINE and OPTIMIZED inferences: `comparing_speed.py`
```
python comparing_speed.py \
    --input_folder nnUNet_data/nnUNet_raw/<dataset_folder>/imagesTs \
    --output_folder predictions
    --model_folder nnUNet_data/nnUNet_results/<dataset_folder>/<model_folder>
```
Outputs:
* Segmentation masks (*.nii.gz)
* Classification results &arr subtype_results.csv
* Speed comparison (optional: uncomment in script to save) &arr speed_comparison.json

| Baseline (seconds per case) | Optimized (seconds per case) | Speed up (%) |
|:--------:|:---------:|:------------:|
| 1.0        |    0.51      | 48.8%           |

## References
* Pancreatic cancer detection via non-contrast CT and deep learning: Cao et al., Nature Medicine, 2023
* nnU-Net: Isensee et al., Nature Methods, 2021
* Metrics Reloaded: Maier-Hein et al., Medical Image Analysis, 2022


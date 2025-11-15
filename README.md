# Pancreas Segmentation and Classification with multitask nnUNet

## Overview
Extend the original nnUNetv2 to perform:
- Semantic Segmentation (Pancreas, Lesion)
- Classification (Subtype prediction)
Added files:
- dataset_conversion.py
- nnUNetTrainer_multitask.py
- MultiTaskPredictor.py (run inference)
  
## Environments and Requirements
Run Environment:
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4090
- RAM: 
- CUDA version: 12.9
- python version: 3.12.12

To install environments:
```setup
# Make sure install pytorch properly
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

pip install -r requirements.txt
```

Set up nnUNet paths
```
set nnUNet_raw=nnUNet_data/nnUNet_raw
set nnUNet_preprocessed=nnUNet_data/nnUNet_preprocessed
set nnUNet_results=nnUNet_data/nnUNet_results
```

## Data Preparation
1. Copy original data into imagesTr, imagesTs
2. Copy segmentation mask into labelsTr
3. Create:
  * dataset.json
  * final_splits.json 
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
>Describe how to train the models, with example commands, including the full training procedure and appropriate hyper-parameters.
**Move nnUNetTrainer_multitask.py into "/nnunetv2/training/nnUNetTrainer"**
Then start training with:
```
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_multitask -p nnUNetResEncUNetMPlans
```

## Evaluation
Validation metrics (from Metrics Reloaded paper) was combined into the trainer script (after training process):
- For segmentation:
  * Dice score (dsc)
  * Normalized surface dice (nsd)
  * Absolute volume different (avd)
- For classification:
  * Balanced accuracy (balanced_accuracy_score)
  * Macro average F1 (f1_macro)
  * AUROC (auroc)
  * Likelihood ratio+ (lr_plus)
  * Expected calibration error (ece)
  * Brier score (brier_score)
These metrics are reported per class and stored in:
nnUNet_results/.../final_metrics.json

## Results

| Segmentation               |  Pancreas (label > 0)  |   Lesion (label = 2)   |
| :------------------------- | :--------: | :--------: |
| Dice                       |            |            | 
| Normalized Surface Dice    |            |            |   
| Absolute volume different  |            |            |   

| Classification             |  Subtype0  |  Subtype1  |  Subtype2  |
| :------------------------- | :--------: | :--------: | :--------: |
| F1 macro                   |            |            |            |

## Inference
Custom multitask inference script
| Mode          | Description                                                     |
|:--------------|:----------------------------------------------------------------|
| Default       | Run baseline inference                                          |
| --compare     | Runs baseline + optimized inference and prints time comparison  |
Outputs:
- Segmentation predictions
- Classification results &arr classifications.json
- Speed comparison (optional: uncomment in script to save) &arr *speed_comparison.json*

```
python MultiTaskPredictor.py -i imagesTs -o predictions -d 001 --compare
```
  
| Baseline | Optimized | Speed up (%) |
|:--------:|:---------:|:------------:|
| x        | y         | z%           |


## References
* nnU-Net: Isensee et al., Nature Methods, 2021
* Metrics Reloaded: Maier-Hein et al., Medical Image Analysis, 2022


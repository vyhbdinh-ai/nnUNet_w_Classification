# nnUNet_w_Classification

## Environments and Requirements

- Windows/Ubuntu version
- CPU, RAM, GPU information
- CUDA version
- python version

To install environments:

```setup
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

pip install -r requirements.txt
```

Set up nnUNet environment variables

```
set nnUNet_raw=nnUNet_data/nnUNet_raw
set nnUNet_preprocessed=nnUNet_data/nnUNet_preprocessed
set nnUNet_results=nnUNet_data/nnUNet_results
```

## Data Preparation
- Copy original data into imagesTr, imagesTs, labels in "Dataset101_Pancreas/nnUNet_raw"
- Create dataset.json
- Create final_splits.json to keep the original train, val data
- Create labels for segmentation (csv file)
  
```python
python dataset_conversion.py \
    --src_dir "/path/to/original/data" \
    --nnUNet_dir "/nnUNet/data/folder/" \
    --dataset_id 101
```

**Move nnUNetTrainer_multitask.py into "/nnunetv2/training/nnUNetTrainer"**

## Preprocessing

```
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity -pl nnUNetPlannerResEncM
```

## Train
>Describe how to train the models, with example commands, including the full training procedure and appropriate hyper-parameters.
```
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_multitask -p nnUNetResEncUNetMPlans
```

## Evaluation

Validation metrics (from Metrics Reloaded paper) was combined into the trainer script at the end of training process and saved into final_metrics.json:
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
Metrics are available per class.

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
- For default speed: nnUNetv2_predict ...
- For faster speed: nnUNetv2_predict ...
  
| Default | Speed up |
|:-------:|:--------:|
| x       | y        |

## Notes
- Evaluation can be seperated into an individually file

## References
- nnUNet paper
- Metrics Reloaded


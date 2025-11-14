# nnUNet_w_Classification

## Environments and Requirements

- Windows/Ubuntu version
- CPU, RAM, GPU information
- CUDA version
- python version

To install environments:

```setup
pip install -r requirements.txt
```

## Preprocess

```
python dataset_conversion.py \
    --raw_dir "/path/to/original/data" \
    --out_dir "/nnUNet/data/" \
    --dataset_id 101
```
export nnUNet_raw=nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=nnUNet_data/nnUNet_preprocessed
export nnUNet_results=nnUNet_data/nnUNet_results

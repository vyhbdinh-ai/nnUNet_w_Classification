import os
import shutil
import json
from pathlib import Path
from glob import glob
import csv
from typing import List, Tuple
import SimpleITK as sitk
import numpy as np

def fix_labels(input_file: Path, output_file: Path = None):
    """Convert floating point labels to integers"""
    if output_file is None:
        output_file = input_file
    
    # Read the image
    img = sitk.ReadImage(str(input_file))
    array = sitk.GetArrayFromImage(img)
    
    # Check if labels need fixing (if they're floats)
    if array.dtype in [np.float32, np.float64, np.float16]:
        print(f"Fixing float labels in: {input_file.name}")
        # Round to nearest integer and convert to int
        array_fixed = np.round(array).astype(np.int64)
        
        # Create new image
        img_fixed = sitk.GetImageFromArray(array_fixed)
        img_fixed.CopyInformation(img)
        
        # Save
        sitk.WriteImage(img_fixed, str(output_file))
        return True
    else:
        # Just copy if already integers
        if input_file != output_file:
            shutil.copy2(input_file, output_file)
        return False


def create_classification_labels(nnunet_raw_dir: Path, cases: List[Tuple[str, int]]):
    """
    cases: list of (case_id, subtype_int)
    """
    csv_path = nnunet_raw_dir / f"classification_labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case_id", "Subtype"])
        for case_id, subtype in cases:
            w.writerow([case_id, int(subtype)])

    print(f"Wrote {csv_path.name} with {len(cases)} rows")


def extract_cases(source_dir: Path, target_nnunet_dataset_dir: Path):
    """
    Extract cases from path, create labels for classification and copy files to nnUNet_raw folder
    """
    # Create directories
    (target_nnunet_dataset_dir / "imagesTr").mkdir(exist_ok=True)
    (target_nnunet_dataset_dir / "labelsTr").mkdir(exist_ok=True)
    (target_nnunet_dataset_dir / "imagesTs").mkdir(exist_ok=True)

    # Process training data
    train_cases: List[Tuple[str, int]] = []
    source_train = source_dir / "train"

    for subtype in os.listdir(source_train):
        subtype_path = source_train / subtype

        all_files = glob(os.path.join(subtype_path, "*.nii.gz"))
        lbl_files = [f for f in all_files if not f.endswith("_0000.nii.gz")]

        for lbl_file_path  in lbl_files:
            lbl_file = Path(lbl_file_path)
            case_id = lbl_file.stem.replace(".nii","")

            img_file = subtype_path / f"{case_id}_0000.nii.gz"

            shutil.copy2(img_file, target_nnunet_dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz")
            label_output = target_nnunet_dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
            fix_labels(lbl_file, label_output)

            subtype_num = int(subtype.replace("subtype", ""))
            train_cases.append([case_id, subtype_num])

    # Process validation data 
    val_cases : List[Tuple[str, int]] = []
    source_val = source_dir / "validation"

    for subtype in os.listdir(source_val):
        subtype_path = source_val / subtype

        all_files = glob(os.path.join(subtype_path, "*.nii.gz"))
        lbl_files = [f for f in all_files if not f.endswith("_0000.nii.gz")]

        for lbl_file_path in lbl_files:
            lbl_file = Path(lbl_file_path)
            case_id = lbl_file.stem.replace(".nii","")

            img_file = subtype_path / f"{case_id}_0000.nii.gz"

            shutil.copy2(img_file, target_nnunet_dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz")
            label_output = target_nnunet_dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
            fix_labels(lbl_file, label_output)
            
            subtype_num = int(subtype.replace("subtype", ""))
            val_cases.append([case_id, subtype_num])

    create_classification_labels(target_nnunet_dataset_dir, sorted(train_cases + val_cases))
    
    # Process test data
    source_test = source_dir / "test"
    test_files = list(source_test.glob("*.nii.gz"))

    test_cases: List[str] = []
    for test_file in test_files:
        case_id = test_file.stem.replace(".nii", "")
        shutil.copy2(test_file, target_nnunet_dataset_dir / "imagesTs" / f"{case_id}.nii.gz")
        test_cases.append(case_id)

    return train_cases, val_cases, test_cases


def create_dataset_json(out_dir: Path, dataset_name: str, training_num: int):
    """
    Create dataset.json
    """
    dataset_json = {
        "name": dataset_name,
        "description": "Pancreas segmentation with subtype classification",
        "channel_names": {
            "0": "CT"  
        },
        "labels": {
            "background": "0",
            "pancreas": "1", 
            "lesion": "2"
        },
        "numTraining": training_num,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open(out_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    return dataset_json


def create_splits_json(out_dir: Path, train_cases, val_cases):
    """
    Create splits_final.json to keep the original validation split
    """
    splits_data =[{
        "train": [c[0] for c in train_cases],
        "val": [c[0] for c in val_cases]
    }]

    out_json = out_dir / "splits_final.json"
    with open(out_json, "w") as f:
        json.dump(splits_data, f, indent=2)
    return out_json

# Run the conversion
if __name__ == "__main__":
    import argparse
    parser =  argparse.ArgumentParser(description="Convert pancreas dataset to nnUNet format")
    parser.add_argument("--src_dir", required=True, type=Path,
                    help="Directory that contains train/ validation/ test folders")
    parser.add_argument("--nnUNet_dir", required=True, type=Path,
                    help="Destination: will contain nnUNet_raw, nnUNet_preprocessed, nnUNet_results")
    parser.add_argument("--dataset_id", type=int, default=101,
                    help="Dataset ID for nnUNet (default: 101)")
    args = parser.parse_args()

    src_dir = args.src_dir
    nnUNet_dir = args.nnUNet_dir
    (nnUNet_dir / "nnUNet_raw").mkdir(exist_ok=True)
    (nnUNet_dir / "nnUNet_preprocessed").mkdir(exist_ok=True)
    (nnUNet_dir / "nnUNet_results").mkdir(exist_ok=True)

    dataset_id = args.dataset_id
    dataset_name = f"Dataset{dataset_id:03d}_PancreasCT"
    dataset_raw_dir = nnUNet_dir / "nnUNet_raw" / dataset_name
    os.makedirs(dataset_raw_dir, exist_ok=True)
    dataset_preprocessed_dir = nnUNet_dir / "nnUNet_preprocessed" / dataset_name
    os.makedirs(dataset_preprocessed_dir, exist_ok=True)

    # Extract cases
    train_cases, val_cases, test_cases = extract_cases(src_dir, dataset_raw_dir)

    # Create dataset.json
    create_dataset_json(dataset_raw_dir, dataset_name, len(train_cases) + len(val_cases))

    # Create splits file
    create_splits_json(dataset_raw_dir, train_cases, val_cases)




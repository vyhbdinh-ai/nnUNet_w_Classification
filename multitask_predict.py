import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import SimpleITK as sitk
from typing import List, Tuple
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from tqdm import tqdm
import sys
import os


def load_model(model_folder: str, fold: int, checkpoint_name: str, device: torch.device):
    from nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer_multitask import nnUNetTrainer_multitask

    plans_file = join(model_folder, 'nnUNetResEncUNetMPlans.json')
    if not os.path.exists(plans_file):
        plans_file = join(model_folder, 'plans.json')
    
    plans = load_json(plans_file)
    dataset_json = load_json(join(model_folder, 'dataset.json'))
    configuration_name = '3d_fullres'
    trainer = nnUNetTrainer_multitask(
        plans=plans,
        configuration=configuration_name,
        fold=fold,
        dataset_json=dataset_json,
        device=device)
    
    trainer.initialize()

    checkpoint_path = join(model_folder, f'fold_{fold}', checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    with torch.no_grad():
        patch_size = plans['configurations'][configuration_name]['patch_size']
        dummy_input = torch.randn(1, 1, *patch_size).to(device)
        trainer.network.eval()
        _ = trainer.network(dummy_input)
        enc_features = trainer.network.encoder(dummy_input)
        _ = trainer.network.ClassificationHead(enc_features)
    
    trainer.network.load_state_dict(checkpoint['network_weights'])
    trainer.network.eval()
    
    return trainer.network, plans


def normalize_image(image: np.ndarray) -> np.ndarray:
    """CT-specific normalization"""
    image = np.clip(image, -1000, 1000)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image


def sliding_window_inference(
    network,
    image: torch.Tensor,
    patch_size: List[int],
    step_size: float,
    device: torch.device,
    use_tta: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sliding window inference with Gaussian weighting
    Returns: segmentation logits, classification logits
    """
    img_shape = image.shape[2:]  # [D, H, W]
    
    # Pad image if smaller than patch size
    pad_needed = [max(0, p - s) for p, s in zip(patch_size, img_shape)]
    if any(pad_needed):
        padding = []
        for pad in reversed(pad_needed):
            padding.extend([pad // 2, pad - pad // 2])
        image = torch.nn.functional.pad(image, padding, mode='constant', value=image.min().item())
        img_shape = image.shape[2:]
    
    steps = [max(1, int(p * step_size)) for p in patch_size]
    
    num_classes = 3
    seg_output = torch.zeros((1, num_classes) + img_shape, dtype=torch.float32, device=device)
    seg_counts = torch.zeros((1, num_classes) + img_shape, dtype=torch.float32, device=device)
    cls_logits_list = []
    
    gaussian = torch.zeros((1, 1) + tuple(patch_size), device=device)
    center = [p // 2 for p in patch_size]
    for d in range(patch_size[0]):
        for h in range(patch_size[1]):
            for w in range(patch_size[2]):
                dist = ((d - center[0]) / (patch_size[0] / 2)) ** 2 + \
                       ((h - center[1]) / (patch_size[1] / 2)) ** 2 + \
                       ((w - center[2]) / (patch_size[2] / 2)) ** 2
                gaussian[0, 0, d, h, w] = np.exp(-dist / 2)
    
    positions = []
    for d in range(0, img_shape[0], steps[0]):
        for h in range(0, img_shape[1], steps[1]):
            for w in range(0, img_shape[2], steps[2]):
                # Adjust position to ensure patch fits within image
                d_start = min(d, img_shape[0] - patch_size[0])
                h_start = min(h, img_shape[1] - patch_size[1])
                w_start = min(w, img_shape[2] - patch_size[2])
                positions.append((d_start, h_start, w_start))
    
    positions = list(set(positions))
    
    if not positions:
        positions = [(0, 0, 0)]
    
    with torch.no_grad():
        for d, h, w in positions:
            d_end = d + patch_size[0]
            h_end = h + patch_size[1]
            w_end = w + patch_size[2]
            
            patch = image[:, :, d:d_end, h:h_end, w:w_end]
            
            seg_pred = network(patch)
            if isinstance(seg_pred, list):
                seg_pred = seg_pred[0]
            
            enc_features = network.encoder(patch)
            cls_pred = network.ClassificationHead(enc_features)
            cls_logits_list.append(cls_pred)
            
            weighted = seg_pred * gaussian
            seg_output[:, :, d:d_end, h:h_end, w:w_end] += weighted
            seg_counts[:, :, d:d_end, h:h_end, w:w_end] += gaussian
            
            if use_tta:
                patch_flip = torch.flip(patch, [2])
                seg_pred_flip = network(patch_flip)
                if isinstance(seg_pred_flip, list):
                    seg_pred_flip = seg_pred_flip[0]
                seg_pred_flip = torch.flip(seg_pred_flip, [2])
                
                enc_features_flip = network.encoder(patch_flip)
                cls_pred_flip = network.ClassificationHead(enc_features_flip)
                cls_logits_list.append(cls_pred_flip)
                
                weighted_flip = seg_pred_flip * gaussian
                seg_output[:, :, d:d_end, h:h_end, w:w_end] += weighted_flip
                seg_counts[:, :, d:d_end, h:h_end, w:w_end] += gaussian
    # Normalize
    seg_output = seg_output / (seg_counts + 1e-8)
    
    # Remove padding if pad
    if any(pad_needed):
        crop_slices = [slice(None), slice(None)]  # Keep batch and channel dims
        for pad in pad_needed:
            if pad > 0:
                crop_slices.append(slice(pad // 2, -(pad - pad // 2)))
            else:
                crop_slices.append(slice(None))
        seg_output = seg_output[tuple(crop_slices)]
    
    # Average classification logits
    cls_logits = torch.stack(cls_logits_list).mean(dim=0)
    
    return seg_output, cls_logits


def predict_case(
    network,
    image_path: str,
    patch_size: List[int],
    step_size: float,
    device: torch.device,
    use_tta: bool = False
) -> Tuple[np.ndarray, int, dict]:
    """
    Predict segmentation and classification for one case
    Returns: segmentation mask, predicted class, metadata
    """
    # Load image
    sitk_img = sitk.ReadImage(image_path)
    image_np = sitk.GetArrayFromImage(sitk_img)  # [D, H, W]
    
    metadata = {
        'spacing': sitk_img.GetSpacing(),
        'origin': sitk_img.GetOrigin(),
        'direction': sitk_img.GetDirection(),
        'size': sitk_img.GetSize()}
    
    image_np = normalize_image(image_np)
    image_np = image_np[np.newaxis, ...]  # [1, D, H, W]
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)  # [1, 1, D, H, W]
    seg_logits, cls_logits = sliding_window_inference(
        network, image_tensor, patch_size, step_size, device, use_tta)
    
    seg_pred = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()  # [D, H, W]
    cls_pred = torch.argmax(cls_logits, dim=1).item()
    
    return seg_pred, cls_pred, metadata


def save_segmentation(seg_mask: np.ndarray, output_path: str, metadata: dict):
    seg_sitk = sitk.GetImageFromArray(seg_mask.astype(np.uint8))
    seg_sitk.SetSpacing(metadata['spacing'])
    seg_sitk.SetOrigin(metadata['origin'])
    seg_sitk.SetDirection(metadata['direction'])
    sitk.WriteImage(seg_sitk, output_path)


def main():
    parser = argparse.ArgumentParser(description='Standalone Multi-task nnUNet Inference')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                       help='Input folder with test images (*_0000.nii.gz)')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                       help='Output folder for predictions')
    parser.add_argument('-m', '--model_folder', type=str, required=True,
                       help='Model training output folder')
    parser.add_argument('-f', '--fold', type=int, default=0,
                       help='Fold to use (default: 0)')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoint_best.pth',
                       help='Checkpoint name (default: checkpoint_best.pth)')
    parser.add_argument('--step_size', type=float, default=0.5,
                       help='Sliding window step size (default: 0.5, higher=faster)')
    parser.add_argument('--enable_tta', action='store_true',
                       help='Enable test-time augmentation (slower but may improve accuracy)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    network, plans = load_model(args.model_folder, args.fold, args.checkpoint, device)
    network = network.to(device)
    
    patch_size = plans['configurations']['3d_fullres']['patch_size']
    print(f"Patch size: {patch_size}")
    print(f"Step size: {args.step_size}")
    print(f"TTA: {'Enabled' if args.enable_tta else 'Disabled'}")
    print("Model loaded\n")
    
    input_folder = Path(args.input_folder)
    test_files = sorted(list(input_folder.glob('*_0000.nii.gz')))
    print(f"Found {len(test_files)} test cases\n")
    
    classification_results = []
    
    for test_file in tqdm(test_files, desc="Processing"):
        case_id = test_file.stem.replace('_0000.nii', '')
        
        seg_pred, cls_pred, metadata = predict_case(
            network,
            str(test_file),
            patch_size,
            args.step_size,
            device,
            args.enable_tta)
        
        output_file = output_folder / f"{case_id}.nii.gz"
        save_segmentation(seg_pred, str(output_file), metadata)
        
        classification_results.append({
            'Names': f"{case_id}.nii.gz",
            'Subtype': cls_pred})
        
        print(f"  {case_id}: Class {cls_pred}")

    csv_path = output_folder / 'subtype_results.csv'
    df = pd.DataFrame(classification_results)
    df.to_csv(csv_path, index=False)
    
    print(f"- Segmentations: {output_folder}")
    print(f"- Classifications: {csv_path}")
    print(f"\nSpeed optimizations applied:")
    print(f"- TTA: {'Enabled (slower)' if args.enable_tta else 'Disabled (faster)'}")
    print(f"- Step size: {args.step_size} (default 0.5, higher=faster)")

if __name__ == '__main__':
    main()
import torch
from torch import nn
import torch.nn.functional as F
import wandb
import os
import json
import numpy as np
import pandas as pd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.paths import nnUNet_raw
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma 
    
    def forward(self, inputs, targets):
        """
        Args: logits, class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


class Classification_Head(nn.Module):
    def __init__(self, in_channels: int, num_classes: int=3, hidden_dim: int=128, 
                 pooling_type: str="avg", 
                 feature_level: str="bottleneck", num_scale_features=None,
                 dropout_rate: float=0.2):
        super().__init__()

        self.num_classes = num_classes
        self.feature_level = feature_level
        self.pooling_type = pooling_type

        if isinstance(in_channels, list):
            bottleneck_channels = in_channels[-1]
        else:
            bottleneck_channels = in_channels

        self.pooling_op = nn.AdaptiveAvgPool3d(1) if pooling_type != 'max' else nn.AdaptiveMaxPool3d(1)

        # Feature selection strategy: single tensor from bottleneck / multi-scale features / all features 
        if self.feature_level == "bottleneck": # single feature
            self.num_scale_features = 1
            classifier_input_dim = bottleneck_channels 
            print("Using SINGLE bottleneck feature")
        elif  self.feature_level == "multi_scale": # last N start from bottleneck, default num_scale_features = 3
            self.num_scale_features = num_scale_features if num_scale_features else 3
            self.feature_adapters = None
            self.feature_fusion = None
            classifier_input_dim = 128  #fusion output
            print(f"Using MULTI-SCALE with {self.num_scale_features} features")
        elif self.feature_level == "all_features": # all features from encoders
            self.num_scale_features = "all"  
            self.feature_adapters = None
            self.feature_fusion = None
            classifier_input_dim = 128  #fusion output
            print("Using ALL available features")
        else:
            raise ValueError(f"Unknown feature_level: {feature_level}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes))
    
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _setup_multi_scale_layers(self, features):
        if self.feature_level == "bottleneck":
            return  
            
        # Used features
        if self.num_scale_features == "all":
            num_to_use = len(features)
        else:
            #num_to_use = min(self.num_scale_features, len(features)) #features availability
            num_to_use = self.num_scale_features
        
        print(f"Setting up multi-scale layers for {num_to_use} features")

        selected_indices = -num_to_use  # Last N features
        channel_dims = [f.shape[1] for f in features[selected_indices:]]
        
        device = features[0].device
        # Feature adapters
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((5, 5, 5)) 
            ) for ch in channel_dims
        ]).to(device)
        
        # Fusion layer
        fusion_input_channels = 64 * num_to_use
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(fusion_input_channels, 128, kernel_size=3, padding=1), 
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        ).to(device)

    def forward(self, features):
        # Bottleneck 
        if self.feature_level == "bottleneck":
            if isinstance(features, list):
                x = features[-1]  
            else:
                x = features

            pooled = self.pooling_op(x) 
                
            pooled_flat = pooled.view(pooled.size(0), -1) 

        # Multi-scale + All features
        else:
            if self.feature_adapters is None:
                self._setup_multi_scale_layers(features)
            
            # Select features to use
            if self.num_scale_features == "all":
                selected_features = features  # All features
            else:
                selected_features = features[-self.num_scale_features:]  # Last N
            
            # Adapt and fuse features
            adapted_features = []
            for feature, adapter in zip(selected_features, self.feature_adapters):
                adapted = adapter(feature)
                adapted_features.append(adapted)
        
            fused = torch.cat(adapted_features, dim=1)
            x = self.feature_fusion(fused)
            pooled_flat = x.view(x.size(0), -1) # [B, 128]
            

        # Classify
        logits = self.classifier(pooled_flat)  
        
        return logits

class nnUNetTrainer_multitask(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        #self.enable_compile = False 

        wandb.init(
            project="pancreas-multi-task",
            name=f"MultiTask_fold{fold}",
            config={
                "plans": self.plans_manager.plans_name,
                "configuration": configuration,
                "fold": fold,
                "architecture": "3D_ResNet_M_MultiTask"
            })

        self.subtype_dict = {}
        self.cls_class_weights = torch.ones(3, dtype=torch.float32, device=self.device)
        self._load_classification_labels() #call calculate_class_weight
        self.cls_criterion = FocalLoss(
            alpha=self.cls_class_weights,
            gamma=2.0)
        
        # Early stopping setup 
        self.best_combined_score = 0
        self.early_stop_patience = 15
        self.early_stop_counter = 0
        #self.min_epochs = 100

        self._enc_features = None
        self.classification_head = None 

    def _load_classification_labels(self):
        dataset_name = self.plans_manager.dataset_name
        dataset_path = os.path.join(nnUNet_raw, dataset_name)

        csv_path = os.path.join(dataset_path, "classification_labels.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.subtype_dict = dict(zip(df['Case_id'], df['Subtype']))
        else:
            self.subtype_dict = {}
            #print(f"Loaded classification labels from CSV: {len(self.classification_labels)} cases")
        
        self._calculate_class_weights()

    def _calculate_class_weights(self):
        labels = list(self.subtype_dict.values())
        
        # Get class distribution
        label_counts = np.bincount(labels, minlength=3)
        total_samples = len(labels)
        
        print(f"Class distribution: {label_counts}")
        
        # MANUAL weight calculation - inverse frequency
        class_weights = total_samples / (3 * label_counts.astype(float))
        class_weights = class_weights / class_weights.sum() * 3
        
        self.cls_class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        print(f"Class weights: {class_weights}")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        """Override to add classification head to the network"""
        network = super().build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        encoder_output_channels = network.encoder.output_channels
        print(f" Encoder channels: {encoder_output_channels}")
        
        ###################### Add classification head ######################
        network.ClassificationHead = Classification_Head(
            encoder_output_channels,
            num_classes=3,
            feature_level="multi_scale",
            pooling_type="avg", 
            hidden_dim=128,
            dropout_rate=0.4
        ).to(self.device)

        return network

    def train_step(self, batch: dict):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        # Get classification labels
        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long,
            device=self.device)
        
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        seg_output = self.network(data)
        enc_features = self.network.encoder(data)

        class_logits = self.network.ClassificationHead(enc_features)

        seg_loss = self.loss(seg_output, target)
        cls_loss = self.cls_criterion(class_logits, subtype)
        total_loss = seg_loss + 0.5*cls_loss #Should implement dynamic

        total_loss.backward()
        #nn.utils.clip_grad_norm_(self.network.ClassificationHead.parameters(), max_norm=6.0)
        nn.utils.clip_grad_norm_(self.network.parameters(), 12) 
        self.optimizer.step()

        return {
        'loss': total_loss.detach().cpu().numpy(),
        'seg_loss': seg_loss.detach().cpu().numpy(),
        'class_loss': cls_loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long, 
            device=self.device
        )
        
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        seg_output = self.network(data)
        enc_features = self.network.encoder(data) 
        class_logits = self.network.ClassificationHead(enc_features)

        seg_loss = self.loss(seg_output, target)
        cls_loss = self.cls_criterion(class_logits, subtype)
        total_loss = seg_loss + 0.5 * cls_loss 

        class_probs = torch.softmax(class_logits, dim=1)
        class_preds = torch.argmax(class_logits, dim=1)

        if isinstance(seg_output, list):
        # Store only the highest resolution output for metrics
            seg_output_for_metrics = seg_output[0].detach().cpu().numpy()
        else:
            seg_output_for_metrics = seg_output.detach().cpu().numpy()

        if isinstance(target, list):
            target_for_metrics = target[0].detach().cpu().numpy()
        else:
            target_for_metrics = target.detach().cpu().numpy()

        # Store detailed metrics for epoch-end calculation 
        batch_metrics = { 
            'case_ids': case_ids,
            'subtype_true': subtype.cpu().numpy(),
            'class_probs': class_probs.detach().cpu().numpy(),
            'class_preds': class_preds.detach().cpu().numpy(),
            'seg_output': seg_output_for_metrics,
            'target': target_for_metrics
        }

        result = self._calculate_basic_segmentation_metrics(seg_output, target)
        result.update({
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'class_loss': cls_loss.detach().cpu().numpy(),
            'batch_metrics': batch_metrics
        })

        return result
 
    def _calculate_basic_segmentation_metrics(self, seg_output, target):
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        axes = [0] + list(range(2, seg_output.ndim))

        # Convert to one-hot for metric calculation
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, 
                                                    device=seg_output.device, 
                                                    dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        # Handle ignore labels
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:] if target.dtype != torch.bool else ~target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, 
                                    axes=axes, mask=mask)

        return {
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(), 
            'fn_hard': fn.detach().cpu().numpy()
        }

    def on_train_epoch_end(self, train_outputs):
        train_outputs_collated = collate_outputs(train_outputs)
        self.current_train_loss = np.mean(train_outputs_collated['loss'])
        wandb.log({
            'epoch': self.current_epoch,
            'train_loss': np.mean(train_outputs_collated['loss']),
            'train_seg_loss': np.mean(train_outputs_collated['seg_loss']),
            'train_class_loss': np.mean(train_outputs_collated['class_loss']),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })

    def on_validation_epoch_end(self, val_outputs):
        """log to W&B"""
        batch_metrics_list = [output.pop('batch_metrics') for output in val_outputs]
        val_outputs_collated = collate_outputs(val_outputs)
        self.current_val_loss = float(np.mean(val_outputs_collated['loss']))

        # Segmentation metrics
        tp = np.sum(val_outputs_collated['tp_hard'], 0) 
        fp = np.sum(val_outputs_collated['fp_hard'], 0)
        fn = np.sum(val_outputs_collated['fn_hard'], 0)
        
        global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 for i, j, k in zip(tp, fp, fn)] 
        mean_fg_dice = np.nanmean(global_dc_per_class) 

        # Classification metrics
        all_true = []
        all_preds = []
        for batch in batch_metrics_list:
            all_true.extend(batch['subtype_true'])
            all_preds.extend(batch['class_preds'])
        cls_metrics = balanced_accuracy_score(all_true, all_preds)
        cls_f1_macro = f1_score(all_true, all_preds, average='macro') 

        # Store metrics
        wandb.log({
            'epoch': self.current_epoch,
            'val_loss': float(np.mean(val_outputs_collated['loss'])),
            'val_seg_loss': float(np.mean(val_outputs_collated['seg_loss'])),
            'val_class_loss': float(np.mean(val_outputs_collated['class_loss'])),
            'val_mean_dice': float(mean_fg_dice),
            'val_balanced_accuracy': float(cls_metrics),
            'val_f1_macro': float(cls_f1_macro), 
            **{f'val_dice_class_{i}': float(dice_score) for i, dice_score in enumerate(global_dc_per_class)}
            })

        # Store for early stopping
        self.current_val_metrics = {
            'mean_dice': mean_fg_dice,
            'balanced_accuracy': cls_metrics,
            'combined_score': (0.7 * mean_fg_dice + 0.3 * cls_metrics)
        }

        self.print_to_log_file(f"=== Epoch {self.current_epoch} ===")
        self.print_to_log_file(f"Segmentation: Dice = {mean_fg_dice:.4f}")
        self.print_to_log_file(f"Classification: F1-macro = {cls_f1_macro:.4f}, Balanced Acc = {cls_metrics:.4f}")
        self.print_to_log_file(f"Combined Score: {self.current_val_metrics['combined_score']:.4f}")

    def run_training(self):
        """Override training loop for multi-task learning"""
        self.on_train_start()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            #Training    
            self.on_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs) 

            #Validation
            self.network.eval()
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            # Early stopping check
            should_stop = self.on_epoch_end()
            if should_stop:
                self.print_to_log_file(f"Early stopping at epoch {epoch}")
                break    
        
        self.on_train_end()
        self.final_validation()
        wandb.finish()

    def on_epoch_end(self): 
        """Check on early stopping conditions"""    
        self.current_epoch += 1  
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        if hasattr(self, 'current_val_metrics'):
            current_score = self.current_val_metrics['combined_score']
            if current_score > self.best_combined_score:
                self.best_combined_score = current_score
                self.early_stop_counter = 0
                self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_best.pth'))
                self.print_to_log_file(f"New best model! Combined score: {current_score:.4f}")
            else:
                self.early_stop_counter += 1
                self.print_to_log_file(f"No improvement. Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")

            #if self.current_epoch >= self.min_epochs and self.early_stop_counter >= self.early_stop_patience:
            if self.early_stop_counter >= self.early_stop_patience:    
                self.print_to_log_file(f"Early stopping triggered at epoch {self.current_epoch}")
                return True 
        
        return False

        #-----------------------------Metrics Reloaded: Helper Function-----------------------------
    ## Classification
    def _expected_calibration_error(self, y_true, probs, n_bins=10): #arbitrary n_bins
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correctness = (predictions == y_true).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1) 
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if not np.any(mask):
                continue
            acc_bin = correctness[mask].mean()
            conf_bin = confidences[mask].mean()
            ece += mask.mean() * abs(acc_bin - conf_bin)
        return float(ece)

    def _brier_score_multiclass(self, y_true, probs):
        n_classes = probs.shape[1]
        onehot = np.eye(n_classes)[y_true]
        sq_error = (probs - onehot) ** 2
        brier = sq_error.sum(axis=1).mean()
        return float(brier)

    def _positive_likelihood_ratio_per_class(self, y_true, y_pred, n_classes):
        """
        LR+ = TPR / FPR = sensitivity/(1-specificity)
        Returns the macro-average LR+.
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        lrp = []
        for k in range(n_classes):
            tp = cm[k, k]
            fn = cm[k, :].sum() - tp
            fp = cm[:, k].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            if fpr == 0:
                lrp.append(np.inf)
            else:
                lrp.append(float(tpr / fpr))
        
        # Handle infinite values by taking finite mean
        finite_lrp = [x for x in lrp if np.isfinite(x)] #Ignore perfect classes when averaging
        return np.mean(finite_lrp) if finite_lrp else 0.0
    
    ## Segmentation
    def _get_surface(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        struct = generate_binary_structure(mask.ndim, 1) 
        eroded = binary_erosion(mask, structure=struct, border_value=0) 
        return mask & ~eroded 

    def _normalized_surface_dice(self, gt: np.ndarray, pred: np.ndarray, spacing, tol_mm=1.0) -> float:
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        if not gt.any() and not pred.any(): 
            return 1.0
        if not gt.any() or not pred.any(): 
            return 0.0

        gt_surf = self._get_surface(gt)
        pred_surf = self._get_surface(pred)

        dt_gt = distance_transform_edt(~gt_surf, sampling=spacing) 
        dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)

        dist_pred_to_gt = dt_gt[pred_surf]
        dist_gt_to_pred = dt_pred[gt_surf]

        tp_pred = (dist_pred_to_gt <= tol_mm).sum() 
        tp_gt = (dist_gt_to_pred <= tol_mm).sum()

        nsd_pred = tp_pred / (pred_surf.sum() + 1e-8)
        nsd_gt = tp_gt / (gt_surf.sum() + 1e-8)

        return float(0.5 * (nsd_pred + nsd_gt))

    def _dice_coefficient(self, gt: np.ndarray, pred: np.ndarray) -> float:
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        if not gt.any() and not pred.any():
            return 1.0
        inter = np.logical_and(gt, pred).sum()
        return float(2.0 * inter / (gt.sum() + pred.sum() + 1e-8))

    def _abs_volume_diff_ml(self, gt: np.ndarray, pred: np.ndarray, spacing) -> float:
        vx_vol_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        gt_vol_ml = gt.sum() * vx_vol_mm3 / 1000.0
        pred_vol_ml = pred.sum() * vx_vol_mm3 / 1000.0
        return float(abs(gt_vol_ml - pred_vol_ml))
    
#-----------------------------Metrics Reloaded: Finished Training-----------------------------
    def final_validation(self):
        """Comprehensive metrics after training finishes"""
        self.print_to_log_file("Running comprehensive final validation...")
        self.network.eval()

        batch_metrics_list = []
        
        with torch.no_grad():
            for batch_id in range(self.num_val_iterations_per_epoch):
                batch = next(self.dataloader_val)
                val_result = self.validation_step(batch)
                batch_metrics_list.append(val_result['batch_metrics'])

        
        cls_metrics = self._calculate_classification_metrics(batch_metrics_list)
        seg_metrics = self._calculate_segmentation_metrics(batch_metrics_list)
        
        comprehensive_metrics = {**cls_metrics, **seg_metrics}
        wandb.log(comprehensive_metrics)
        
        # Save to file
        self._save_final_metrics(comprehensive_metrics)
        
        self.print_to_log_file("Final validation completed!")
        return comprehensive_metrics

    def _calculate_segmentation_metrics(self, batch_metrics): 
        all_dsc_class1 = []  
        all_dsc_class2 = []
        all_dsc_whole = []  
        all_nsd = []
        all_nsd_class1 = []  
        all_nsd_class2 = []         
        all_avd = []
        all_avd_class1 = []  
        all_avd_class2 = []  
        
        for batch in batch_metrics:
            seg_outputs = batch['seg_output']  
            targets = batch['target']          
            
            preds = np.argmax(seg_outputs, axis=1)

            if targets.shape[1] == 1:
                targets = targets[:, 0]  # [B, D, H, W]
            
            for i in range(len(preds)):
                pred_mask = preds[i]
                gt_mask = targets[i]
                spacing = (1.0, 1.0, 1.0)
                # Class 1 (Pancreas)
                pred_class1 = (pred_mask == 1)
                gt_class1 = (gt_mask == 1)
                if np.any(gt_class1) or np.any(pred_class1):
                    dsc1 = self._dice_coefficient(gt_class1, pred_class1)
                    all_dsc_class1.append(dsc1)
                    nsd1 = self._normalized_surface_dice(gt_class1, pred_class1, spacing)
                    all_nsd_class1.append(nsd1)
                    all_nsd.append(nsd1)
                    avd1 = self._abs_volume_diff_ml(gt_class1, pred_class1, spacing)
                    all_avd_class1.append(avd1)
                    all_avd.append(avd1)
                # Class 2 (Lesion)
                pred_class2 = (pred_mask == 2)
                gt_class2 = (gt_mask == 2)
                if np.any(gt_class2) or np.any(pred_class2):
                    dsc2 = self._dice_coefficient(gt_class2, pred_class2)
                    all_dsc_class2.append(dsc2)
                    nsd2 = self._normalized_surface_dice(gt_class2, pred_class2, spacing)
                    all_nsd_class2.append(nsd2)
                    all_nsd.append(nsd2)
                    avd2 = self._abs_volume_diff_ml(gt_class2, pred_class2, spacing)
                    all_avd_class2.append(avd2)
                    all_avd.append(avd2)
                # Whole pancreas (label>0)
                pred_whole = (pred_mask > 0)  
                gt_whole = (gt_mask > 0)      
                if np.any(gt_whole) or np.any(pred_whole):
                    dsc_whole = self._dice_coefficient(gt_whole, pred_whole)
                    all_dsc_whole.append(dsc_whole)
        
        # Aggregate metrics
        metrics = {
        'val_dsc_whole_pancreas_mean': np.mean(all_dsc_whole) if all_dsc_whole else 0.0,
        'val_dsc_class1_mean': np.mean(all_dsc_class1) if all_dsc_class1 else 0.0,
        'val_dsc_class2_mean': np.mean(all_dsc_class2) if all_dsc_class2 else 0.0,
        'val_nsd_class1_mean': np.mean(all_nsd_class1) if all_nsd_class1 else 0.0,
        'val_nsd_class2_mean': np.mean(all_nsd_class2) if all_nsd_class2 else 0.0,
        'val_avd_class1_mean': np.mean(all_avd_class1) if all_avd_class1 else 0.0,
        'val_avd_class2_mean': np.mean(all_avd_class2) if all_avd_class2 else 0.0,
    }
        
        return metrics
    
    def _calculate_classification_metrics(self, batch_metrics_list):
        all_true = []
        all_preds = []
        all_probs = []
        
        for batch in batch_metrics_list:
            all_true.extend(batch['subtype_true'])
            all_preds.extend(batch['class_preds'])
            all_probs.extend(batch['class_probs'])

        all_true = np.array(all_true)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        n_classes = all_probs.shape[1]

        per_class_metrics = {}
        for class_id in range(n_classes):
            # Binary indicators for this class
            y_true_class = (all_true == class_id)
            y_pred_class = (all_preds == class_id)
            y_probs_class = all_probs[:, class_id]
            if not np.any(y_true_class):
                continue
            accuracy = (y_pred_class == y_true_class).mean()
            tp = np.sum(y_true_class & y_pred_class)
            fp = np.sum(~y_true_class & y_pred_class)
            fn = np.sum(y_true_class & ~y_pred_class)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            try:
                auroc = roc_auc_score(y_true_class, y_probs_class)
            except:
                auroc = 0.0

            per_class_metrics[f'class_{class_id}_accuracy'] = float(accuracy)
            per_class_metrics[f'class_{class_id}_precision'] = float(precision)
            per_class_metrics[f'class_{class_id}_recall'] = float(recall)
            per_class_metrics[f'class_{class_id}_f1'] = float(f1)
            per_class_metrics[f'class_{class_id}_auroc'] = float(auroc)
            per_class_metrics[f'class_{class_id}_support'] = int(np.sum(y_true_class))
        
        # Overall metrics (keep your existing ones)
        overall_metrics = {
            'val_balanced_accuracy': balanced_accuracy_score(all_true, all_preds),
            'val_f1_macro': f1_score(all_true, all_preds, average='macro'),
            'val_f1_weighted': f1_score(all_true, all_preds, average='weighted'),
            'val_auroc': roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro'),
            'val_lr_plus': self._positive_likelihood_ratio_per_class(all_true, all_preds, n_classes),
            'val_ece': self._expected_calibration_error(all_true, all_probs),
            'val_brier_score': self._brier_score_multiclass(all_true, all_probs),
        }
        
        # Add per-class confusion matrix for detailed analysis
        cm = confusion_matrix(all_true, all_preds, labels=list(range(n_classes)))
        per_class_metrics['confusion_matrix'] = cm.tolist()
        
        return {**overall_metrics, **per_class_metrics}
    
    def _save_final_metrics(self, metrics):
        metrics_file = os.path.join(self.output_folder, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.print_to_log_file(f"Final metrics saved to: {metrics_file}")

if __name__ == "__main__":
    pass


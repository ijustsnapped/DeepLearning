#!/usr/bin/env python
# your_project_name/train_cv_with_meta.py
from __future__ import annotations

import os

import argparse
from pathlib import Path
import time
import copy
import logging
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam # Added Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from torchmetrics import AUROC, F1Score, AveragePrecision
from sklearn.metrics import precision_recall_curve
from torch.amp import autocast, GradScaler

# MODIFIED IMPORTS
from data_handling import (
    FlatDatasetWithMeta, build_transform, build_gpu_transform_pipeline, # MODIFIED
    ClassBalancedSampler
)
from models import get_model as get_base_cnn_model, CNNWithMetadata # MODIFIED
from losses import focal_ce_loss, LDAMLoss
from utils import (
    set_seed, load_config, cast_config_values,
    update_ema,
    get_device, CudaTimer, reset_cuda_peak_memory_stats, empty_cuda_cache,
    TensorBoardLogger
)

try:
    from torch.profiler import ProfilerActivity
except ImportError:
    pass # ProfilerActivity will be None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def _get_path_from_config(cfg: dict, key: str, default: str | None = None, base_path: Path | None = None) -> Path:
    paths_cfg = cfg.get("paths", {})
    path_str = paths_cfg.get(key)
    if path_str is None:
        if default is not None:
            path_str = default
            logger.warning(f"Path for '{key}' not in config's 'paths' section. Using default: '{default}'")
        else:
            logger.error(f"Required path for '{key}' not found in config's 'paths' section. Configured paths: {paths_cfg}")
            raise ValueError(f"Missing path configuration for '{key}'")
    path = Path(path_str)
    if base_path and not path.is_absolute():
        path = base_path / path
    return path.resolve()

def get_class_counts(df: pd.DataFrame, label2idx: dict) -> np.ndarray:
    # (Same as original)
    num_classes = len(label2idx)
    counts = np.zeros(num_classes, dtype=int)
    mapped_indices = df['label'].map(label2idx)
    if mapped_indices.isnull().any():
        unmapped_originals = df['label'][mapped_indices.isnull()].unique()
        logger.warning(
            f"Some labels in DataFrame could not be mapped using label2idx and resulted in NaN. "
            f"These will be ignored in class counts. Unique unmapped original labels: {unmapped_originals}. "
            f"label2idx keys (first few): {list(label2idx.keys())[:5]}"
        )
        class_series = mapped_indices.dropna().astype(int).value_counts()
    else:
        class_series = mapped_indices.astype(int).value_counts()
    for class_idx, count_val in class_series.items():
        if 0 <= class_idx < num_classes:
            counts[class_idx] = count_val
        else:
            logger.warning(f"Out-of-bounds mapped class index {class_idx} found after label2idx mapping. "
                           f"Num_classes: {num_classes}. This count ({count_val}) will be ignored.")
    return counts


# --- CORE TRAINING AND VALIDATION FUNCTION (MODIFIED FOR METADATA) ---
def run_training_phase(
    phase_name: str,
    model: nn.Module, # This will be CNNWithMetadata
    train_ld: DataLoader,
    val_ld: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict, # Full config
    phase_cfg: dict, # Config for this specific phase (e.g., training_cfg or meta_tuning_cfg)
    fold: int,
    label2idx: dict, # For metrics
    tb_logger: TensorBoardLogger,
    run_ckpt_dir: Path,
    exp_name_for_files: str,
    start_epoch: int = 0, # For continuing training phases
    ema_model: nn.Module | None = None, # Pass EMA model if used
    initial_best_metric_val: float = 0.0 # Carry over best metric if needed
) -> tuple[float, int, dict | None]: # Returns best_metric_val, best_epoch, best_thresholds

    num_epochs_phase = phase_cfg.get("num_epochs", phase_cfg.get("epochs")) # Allow 'epochs' or 'num_epochs'
    accum_steps = cfg.get("training", {}).get("accum_steps", 1) # Accum steps from main config
    amp_enabled = cfg.get("training", {}).get("amp_enabled", True) # AMP from main config
    ema_decay = cfg.get("training", {}).get("ema_decay", 0.0)
    use_ema_for_val = cfg.get("training", {}).get("use_ema_for_val", True)

    model_selection_metric_name = cfg.get("training",{}).get("model_selection_metric", "macro_auc").lower()
    best_metric_val = initial_best_metric_val
    best_epoch_for_metric = -1
    optimal_thresholds_best_ckpt = None
    patience_counter = 0
    early_stopping_patience = phase_cfg.get("early_stopping_patience", 10)

    num_classes = cfg["numClasses"] # Should be set in main cfg
    loss_func_type = cfg.get("training", {}).get("loss", {}).get("type", "cross_entropy").lower() # For focal loss args
    loss_cfg_main = cfg.get("training", {}).get("loss", {})


    logger.info(f"Starting {phase_name} for {num_epochs_phase} epochs (Epochs {start_epoch}-{start_epoch + num_epochs_phase -1}).")
    for epoch_offset in range(num_epochs_phase):
        epoch = start_epoch + epoch_offset
        # DRW logic might need to be adapted if used across phases, or phase-specific
        # For now, assuming DRW applies based on global epoch number if configured in main 'training.loss'
        # (DRW logic from original train_one_fold can be inserted here if needed, using `epoch`)

        if hasattr(train_ld.sampler, 'set_epoch'): train_ld.sampler.set_epoch(epoch) # For distributed or custom samplers
        
        # Profiler setup (can be adapted from original if needed per phase)
        # current_epoch_profiler = tb_logger.setup_profiler(epoch, fold_specific_tb_log_dir) # Needs log dir

        # Freeze/Unfreeze logic (e.g. backbone) - for metadata, CNN backbone is frozen in phase 2
        # This is handled before calling this function by setting requires_grad on model parts

        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train()
        # if gpu_augmentation_pipeline_train: gpu_augmentation_pipeline_train.train() # Add if GPU augs used

        cumulative_train_loss_for_pbar = 0.0; cumulative_train_corrects_for_pbar = 0; cumulative_train_samples_for_pbar = 0
        epoch_gpu_time_ms = 0.0; optimizer.zero_grad(); epoch_start_time = time.time()
        
        train_pbar_desc = f"F{fold} E{epoch} {phase_name} Train"
        train_pbar = tqdm(train_ld, desc=train_pbar_desc, ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",100))

        for batch_idx, (inputs, labels_cpu) in enumerate(train_pbar):
            img_cpu, meta_cpu = inputs # Unpack inputs
            img_device = img_cpu.to(device, non_blocking=True)
            meta_device = meta_cpu.to(device, non_blocking=True)
            labels_device = labels_cpu.to(device, non_blocking=True)

            # if gpu_augmentation_pipeline_train:
            #     with torch.set_grad_enabled(True): img_device = gpu_augmentation_pipeline_train(img_device)
            
            batch_gpu_time_ms = 0.0
            with CudaTimer(device) as timer:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(img_device, meta_device) # Pass both image and meta
                    if loss_func_type == "focal_ce_loss":
                        current_focal_alpha = loss_cfg_main.get("focal_alpha", 1.0)
                        current_focal_gamma = loss_cfg_main.get("focal_gamma", 2.0)
                        loss = criterion(logits.float(), labels_device, alpha=current_focal_alpha, gamma=current_focal_gamma)
                    else:
                        loss = criterion(logits.float(), labels_device)
                    if accum_steps > 1: loss = loss / accum_steps
                
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_ld):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if ema_model is not None and ema_decay > 0: update_ema(ema_model, model, ema_decay)
            
            batch_gpu_time_ms = timer.get_elapsed_time_ms(); epoch_gpu_time_ms += batch_gpu_time_ms
            batch_loss_val = loss.item() * (accum_steps if accum_steps > 1 else 1)
            preds = logits.argmax(dim=1)
            batch_corrects = (preds == labels_device).float().sum().item()
            batch_samples = img_device.size(0)
            
            cumulative_train_loss_for_pbar += batch_loss_val * batch_samples
            cumulative_train_corrects_for_pbar += batch_corrects
            cumulative_train_samples_for_pbar += batch_samples
            avg_epoch_loss_pbar = cumulative_train_loss_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            avg_epoch_acc_pbar = cumulative_train_corrects_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            train_pbar.set_postfix(loss=f"{avg_epoch_loss_pbar:.4f}", acc=f"{avg_epoch_acc_pbar:.4f}")
            
            current_batch_acc_for_tb = batch_corrects / batch_samples if batch_samples > 0 else 0.0
            # tb_logger.log_train_batch_metrics (use global_train_step if defined across phases)
            # For now, log with simple epoch and batch_idx for phase-specific view
            tb_logger.writer.add_scalar(f"Loss/train_batch_{phase_name}", batch_loss_val, epoch * len(train_ld) + batch_idx)
            tb_logger.writer.add_scalar(f"Acc/train_batch_{phase_name}", current_batch_acc_for_tb, epoch * len(train_ld) + batch_idx)


        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = avg_epoch_loss_pbar; avg_train_acc = avg_epoch_acc_pbar
        train_pbar.close(); scheduler.step()
        
        epoch_metrics_tb = {
            f"Loss/train_epoch_{phase_name}": avg_train_loss,
            f"Accuracy/train_epoch_{phase_name}": avg_train_acc,
            f"LearningRate/epoch_{phase_name}": optimizer.param_groups[0]['lr'],
            f"Time/Train_epoch_duration_sec_{phase_name}": epoch_duration,
        }
        if device.type == 'cuda': epoch_metrics_tb[f"Time/GPU_ms_per_train_epoch_{phase_name}"] = epoch_gpu_time_ms
        
        # Validation
        val_interval = phase_cfg.get("val_interval", cfg.get("training",{}).get("val_interval",1))
        if epoch % val_interval == 0 or epoch == (start_epoch + num_epochs_phase -1):
            eval_model_to_use = ema_model if ema_model is not None and use_ema_for_val else model
            eval_model_to_use.eval()
            
            logger.info(f"Validation E{epoch} ({phase_name}) using {'EMA' if eval_model_to_use is ema_model else 'primary'} model.")
            val_loss_sum, val_acc_sum, val_seen_samples = 0.0, 0.0, 0
            all_probs_val, all_labels_val = [], []
            
            val_pbar_desc = f"F{fold} E{epoch} {phase_name} Val"
            val_pbar = tqdm(val_ld, desc=val_pbar_desc, ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",100))
            with torch.no_grad():
                for batch_idx_val, (inputs_val, labels_val_cpu) in enumerate(val_pbar):
                    img_val_cpu, meta_val_cpu = inputs_val
                    img_val_dev = img_val_cpu.to(device, non_blocking=True)
                    meta_val_dev = meta_val_cpu.to(device, non_blocking=True)
                    labels_val_dev = labels_val_cpu.to(device, non_blocking=True)

                    with autocast(device_type=device.type, enabled=amp_enabled):
                        logits_val = eval_model_to_use(img_val_dev, meta_val_dev)
                        if loss_func_type == "focal_ce_loss":
                            loss_v = criterion(logits_val.float(), labels_val_dev, alpha=loss_cfg_main.get("focal_alpha",1.0), gamma=loss_cfg_main.get("focal_gamma",2.0))
                        else:
                            loss_v = criterion(logits_val.float(), labels_val_dev)
                    
                    current_val_loss = loss_v.item()
                    current_val_correct = (logits_val.argmax(dim=1) == labels_val_dev).float().sum().item()
                    val_loss_sum += current_val_loss * img_val_dev.size(0)
                    val_acc_sum += current_val_correct
                    val_seen_samples += img_val_dev.size(0)
                    all_probs_val.append(logits_val.softmax(dim=1).cpu())
                    all_labels_val.append(labels_val_dev.cpu())
                    val_pbar.set_postfix(avg_loss=f"{val_loss_sum/val_seen_samples:.4f}", avg_acc=f"{val_acc_sum/val_seen_samples:.4f}")

            all_probs_val_cat = torch.cat(all_probs_val); all_labels_val_cat = torch.cat(all_labels_val)
            avg_val_loss = val_loss_sum / val_seen_samples if val_seen_samples > 0 else 0
            avg_val_acc = val_acc_sum / val_seen_samples if val_seen_samples > 0 else 0
            val_pbar.close()
            
            epoch_metrics_tb[f"Loss/val_epoch_{phase_name}"] = avg_val_loss
            epoch_metrics_tb[f"Accuracy/val_epoch_{phase_name}"] = avg_val_acc

            # --- Calculate Metrics (AUROC, F1, etc.) ---
            auc_metric_fn = AUROC(task="multiclass", num_classes=num_classes, average="macro")
            pauc_max_fpr = cfg.get("training", {}).get("pauc_max_fpr", 0.2)
            pauc_metric_fn = AUROC(task="multiclass", num_classes=num_classes, average="macro", max_fpr=pauc_max_fpr)
            f1_metric_fn_default = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            
            current_macro_auc = auc_metric_fn(all_probs_val_cat, all_labels_val_cat).item()
            current_pauc = pauc_metric_fn(all_probs_val_cat, all_labels_val_cat).item()
            current_f1_macro_default = f1_metric_fn_default(all_probs_val_cat, all_labels_val_cat).item()
            epoch_metrics_tb[f"AUROC/val_macro_{phase_name}"] = current_macro_auc
            epoch_metrics_tb[f"pAUROC{int(pauc_max_fpr*100)}/val_macro_{phase_name}"] = current_pauc
            epoch_metrics_tb[f"F1Score/val_macro_default_thresh_{phase_name}"] = current_f1_macro_default
            
            optimal_thresholds_for_ckpt_epoch = {}
            current_mean_optimal_f1 = 0.0
            current_mean_optimal_sensitivity = 0.0

            if num_classes > 1:
                labels_oh_val = torch.nn.functional.one_hot(all_labels_val_cat, num_classes).numpy()
                probs_np_val = all_probs_val_cat.numpy()
                # ... (AP, per-class optimal F1/Sensitivity calculation - same as original)
                # Make sure to add phase_name to metric keys for TensorBoard
                # Example for one metric:
                # epoch_metrics_tb[f"F1Score_optimal/val_class_{i}_{phase_name}"] = optimal_f1_class

                # Simplified version for brevity (full calculation from original can be pasted here)
                _f1_scores, _sens_scores = [], []
                for i in range(num_classes): # Placeholder for detailed PR curve analysis
                    # This is where you'd calculate optimal_f1_class, optimal_threshold_for_f1_class, optimal_sensitivity_class
                    # For now, let's assume some placeholder values or skip for brevity of this example
                    # For a real run, copy the logic from the original `train_one_fold`
                    prec, rec, thr = precision_recall_curve(labels_oh_val[:, i], probs_np_val[:, i])
                    f1s = (2 * prec * rec) / (prec + rec + 1e-8)
                    valid_f1_indices = np.where(np.isfinite(f1s[1:]) & (prec[1:] + rec[1:] > 0))[0]
                    if len(valid_f1_indices) > 0:
                        best_idx = valid_f1_indices[np.argmax(f1s[1:][valid_f1_indices])]
                        opt_f1 = float(f1s[1:][best_idx])
                        opt_thr = float(thr[best_idx])
                        opt_sens = float(rec[1:][best_idx])
                    else:
                        opt_f1, opt_thr, opt_sens = 0.0, 0.5, 0.0
                    
                    _f1_scores.append(opt_f1)
                    _sens_scores.append(opt_sens)
                    optimal_thresholds_for_ckpt_epoch[i] = opt_thr
                    epoch_metrics_tb[f"F1Score_optimal/val_class_{i}_{phase_name}"] = opt_f1
                    epoch_metrics_tb[f"Sensitivity_optimal/val_class_{i}_{phase_name}"] = opt_sens


                if _f1_scores: current_mean_optimal_f1 = np.mean(_f1_scores)
                if _sens_scores: current_mean_optimal_sensitivity = np.mean(_sens_scores)
                epoch_metrics_tb[f"F1Score/val_mean_optimal_per_class_{phase_name}"] = current_mean_optimal_f1
                epoch_metrics_tb[f"Sensitivity/val_mean_optimal_per_class_{phase_name}"] = current_mean_optimal_sensitivity


            current_primary_metric_val = 0.0
            if model_selection_metric_name == "macro_auc": current_primary_metric_val = current_macro_auc
            elif model_selection_metric_name == "mean_optimal_f1": current_primary_metric_val = current_mean_optimal_f1
            elif model_selection_metric_name == "mean_optimal_sensitivity": current_primary_metric_val = current_mean_optimal_sensitivity
            
            logger.info(f"F{fold} E{epoch} {phase_name} Val -> Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f} "
                        f"SelectedMetric ({model_selection_metric_name})={current_primary_metric_val:.4f}")

            if current_primary_metric_val > best_metric_val:
                best_metric_val = current_primary_metric_val
                best_epoch_for_metric = epoch
                patience_counter = 0
                ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_fold{fold}_{phase_name}_best.pt"
                
                model_sd_to_save = getattr(model, '_orig_mod', model).state_dict()
                ema_model_sd_to_save = getattr(ema_model, '_orig_mod', ema_model).state_dict() if ema_model else None
                
                checkpoint_data = {
                    'epoch': epoch, 'model_state_dict': model_sd_to_save, 
                    'ema_model_state_dict': ema_model_sd_to_save,
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    f'best_{model_selection_metric_name}': best_metric_val,
                    'config_runtime': cfg, 'label2idx': label2idx,
                    'phase_name': phase_name
                }
                if cfg.get("training",{}).get("save_optimal_thresholds", False) and \
                   model_selection_metric_name in ["mean_optimal_f1", "mean_optimal_sensitivity"] and optimal_thresholds_for_ckpt_epoch:
                    checkpoint_data['optimal_thresholds_val'] = optimal_thresholds_for_ckpt_epoch
                    optimal_thresholds_best_ckpt = optimal_thresholds_for_ckpt_epoch # Save for return
                    logger.info(f"Saving optimal thresholds with ckpt for {phase_name}.")
                
                torch.save(checkpoint_data, str(ckpt_path))
                logger.info(f"Saved best {phase_name} model to {ckpt_path} based on {model_selection_metric_name}: {best_metric_val:.4f} at E{epoch}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping for {phase_name} at E{epoch} (Patience: {early_stopping_patience}).")
                    # tb_logger.log_epoch_summary(epoch_metrics_tb, epoch) # Log before breaking
                    # tb_logger.flush()
                    # return best_metric_val, best_epoch_for_metric, optimal_thresholds_best_ckpt # Return current best
                    break # Break from epoch loop for this phase

        tb_logger.log_epoch_summary(epoch_metrics_tb, epoch) # Log all collected metrics for the epoch
        tb_logger.flush()
        if patience_counter >= early_stopping_patience: break


    # Save last model for this phase
    last_ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_fold{fold}_{phase_name}_last_E{epoch}.pt"
    torch.save({
        'epoch': epoch, 'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
        'ema_model_state_dict': getattr(ema_model, '_orig_mod', ema_model).state_dict() if ema_model else None,
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'current_primary_metric': current_primary_metric_val, # Last validated metric
        'config_runtime': cfg, 'label2idx': label2idx, 'phase_name': phase_name
    }, str(last_ckpt_path))
    logger.info(f"Saved last {phase_name} model checkpoint to {last_ckpt_path} at E{epoch}")

    logger.info(f"Finished {phase_name}. Best {model_selection_metric_name}: {best_metric_val:.4f} at E{best_epoch_for_metric}")
    return best_metric_val, best_epoch_for_metric, optimal_thresholds_best_ckpt


def train_one_fold(
    fold: int,
    train_df_fold: pd.DataFrame, # Already filtered for this fold's training
    val_df_fold: pd.DataFrame,   # Already filtered for this fold's validation
    meta_df_full: pd.DataFrame, # Full metadata df, FlatDatasetWithMeta will handle filtering/merging
    cfg: dict,
    label2idx: dict[str,int],
    train_root_path: Path, # Image root
    run_log_dir: Path,
    run_ckpt_dir: Path,
    exp_name_for_files: str,
    device: torch.device,
) -> float | None:
    
    fold_specific_tb_log_dir = run_log_dir / f"fold_{fold}"; fold_specific_tb_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Config sections ---
    main_training_cfg = cfg.get("training", {})
    meta_tuning_cfg = cfg.get("meta_tuning", {}) # Config for metadata fine-tuning phase
    data_handling_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    # ... other configs (loss, optimizer from main_training_cfg initially)

    # --- TensorBoard Logger ---
    # Calculate train_loader_len based on the largest batch size used if it varies by phase
    # For simplicity, use main training batch size for initial TB setup.
    _batch_size_tb = main_training_cfg.get("batch_size", 32)
    tb_logger_train_loader_len = (len(train_df_fold) + _batch_size_tb - 1) // _batch_size_tb
    try:
        tb_logger = TensorBoardLogger(log_dir=fold_specific_tb_log_dir, experiment_config=cfg,
                                      train_loader_len=tb_logger_train_loader_len)
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoardLogger for fold {fold}: {e}", exc_info=True)
        return None

    # --- Data Transforms (CPU & GPU) ---
    tf_train_cpu = build_transform(data_handling_cfg.get("cpu_augmentations",{}), train=True)
    tf_val_cpu = build_transform(data_handling_cfg.get("cpu_augmentations",{}), train=False)
    # gpu_augmentation_pipeline_train = build_gpu_transform_pipeline(...) # If used

    # --- Datasets & DataLoaders (Phase 1 - Joint Training) ---
    logger.info("Setting up DataLoaders for Phase 1 (Joint Training)...")
    train_ds_phase1 = FlatDatasetWithMeta(
        df=train_df_fold, meta_df=meta_df_full, root=train_root_path, label2idx=label2idx, tf=tf_train_cpu,
        image_loader=data_handling_cfg.get("image_loader", "pil"),
        enable_ram_cache=data_handling_cfg.get("enable_ram_cache", False),
        meta_features_names=data_handling_cfg.get("meta_features_names"), # from config
        meta_augmentation_p=data_handling_cfg.get("meta_augmentation_p", 0.0),
        meta_nan_fill_value=data_handling_cfg.get("meta_nan_fill_value", 0.0)
    )
    val_ds_phase1 = FlatDatasetWithMeta(
        df=val_df_fold, meta_df=meta_df_full, root=train_root_path, label2idx=label2idx, tf=tf_val_cpu,
        image_loader=data_handling_cfg.get("image_loader", "pil"),
        enable_ram_cache=data_handling_cfg.get("enable_ram_cache", False),
        meta_features_names=data_handling_cfg.get("meta_features_names"),
        meta_augmentation_p=0.0 # No meta aug for val
    )
    
    metadata_dim = train_ds_phase1.metadata_dim # Get actual metadata dimension
    logger.info(f"Actual metadata dimension from dataset: {metadata_dim}")

    sampler_cfg = data_handling_cfg.get("sampler", {}); train_sampler_p1 = None
    if sampler_cfg.get("type") == "class_balanced_sqrt":
        train_sampler_p1 = ClassBalancedSampler(train_ds_phase1, num_samples=len(train_ds_phase1))
        logger.info("Using ClassBalancedSampler for Phase 1.")
    
    train_ld_p1 = DataLoader(train_ds_phase1, batch_size=main_training_cfg.get("batch_size"), 
                             sampler=train_sampler_p1, shuffle=(train_sampler_p1 is None),
                             num_workers=data_handling_cfg.get("num_workers"), pin_memory=True,
                             persistent_workers=data_handling_cfg.get("num_workers",0)>0)
    val_ld_p1 = DataLoader(val_ds_phase1, batch_size=main_training_cfg.get("batch_size"), shuffle=False,
                           num_workers=data_handling_cfg.get("num_workers"), pin_memory=True,
                           persistent_workers=data_handling_cfg.get("num_workers",0)>0)

    # --- Model Initialization (CNNWithMetadata) ---
    base_cnn_config = {"MODEL_TYPE": model_cfg.get("base_cnn_type"), 
                       "numClasses": cfg["numClasses"], # Temp numClasses for base model factory
                       "pretrained": model_cfg.get("pretrained_cnn", True)}
    base_cnn = get_base_cnn_model(base_cnn_config) # This loads the CNN (e.g., EfficientNet)
    
    model = CNNWithMetadata(
        base_cnn_model=base_cnn,
        num_classes=cfg["numClasses"], # Final number of classes
        metadata_input_dim=metadata_dim,
        meta_mlp_hidden_dim=model_cfg.get("meta_mlp_hidden_dim", 256),
        meta_mlp_output_dim=model_cfg.get("meta_mlp_output_dim", 256),
        meta_dropout_p=model_cfg.get("meta_dropout_p", 0.4),
        post_concat_dim=model_cfg.get("post_concat_dim", 1024),
        post_concat_dropout_p=model_cfg.get("post_concat_dropout_p", 0.4)
    ).to(device)
    
    model_name_for_log = f"{model_cfg.get('base_cnn_type')}_WithMeta"
    logger.info(f"Model '{model_name_for_log}' loaded on device {device}.")

    # torch.compile (if enabled)
    # ... (original torch.compile logic can be applied to `model`)

    ema_model = None
    if main_training_cfg.get("ema_decay", 0.0) > 0:
        ema_model = copy.deepcopy(model).to(device); [p.requires_grad_(False) for p in ema_model.parameters()]
        logger.info(f"EMA enabled with decay {main_training_cfg.get('ema_decay')}.")

    # --- Loss Function (shared or phase-specific) ---
    # Using one criterion instance, configured from main training block
    class_counts_train_fold = get_class_counts(train_df_fold, label2idx) # For LDAM/WCE
    criterion = None; loss_type = main_training_cfg.get("loss", {}).get("type", "cross_entropy").lower()
    loss_cfg = main_training_cfg.get("loss", {})
    
    if loss_type == "focal_ce_loss":
        criterion = focal_ce_loss # Function, not module
        logger.info(f"Using Focal CE Loss.")
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CE Loss.")
    elif loss_type == "weighted_cross_entropy":
        # ... WCE setup from original, using class_counts_train_fold ...
        k_factor = loss_cfg.get("wce_k_factor", 1.0)
        weights_ni = np.ones(cfg["numClasses"], dtype=float) # Default
        if k_factor > 0 and np.sum(class_counts_train_fold) > 0 :
            total_samples_N = np.sum(class_counts_train_fold)
            safe_counts = np.maximum(class_counts_train_fold, 1e-8) # Avoid div by zero
            weights_ni = (total_samples_N / safe_counts) ** k_factor
            if np.sum(weights_ni) > 1e-12: weights_ni = weights_ni / np.sum(weights_ni) * cfg["numClasses"]
            else: weights_ni = np.ones(cfg["numClasses"], dtype=float)
        initial_loss_weights_wce = torch.tensor(weights_ni, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=initial_loss_weights_wce)
        logger.info(f"Using Weighted CE Loss with k={k_factor}.")

    elif loss_type == "ldam_loss":
        criterion = LDAMLoss(class_counts=class_counts_train_fold,
                             max_margin=loss_cfg.get("ldam_max_margin", 0.5),
                             # ... other LDAM params from config ...
                             scale=loss_cfg.get("ldam_scale", 30.0)).to(device)
        logger.info(f"Using LDAM Loss.")
    else: raise ValueError(f"Unsupported loss type: {loss_type}")

    scaler = GradScaler(enabled=(device.type == 'cuda' and main_training_cfg.get("amp_enabled", True)))
    
    # ========================= PHASE 1: Joint Training =========================
    logger.info("===== Starting Phase 1: Joint Training =====")
    model.set_base_cnn_trainable(True) # Ensure CNN backbone is trainable
    # Optimizer and Scheduler for Phase 1
    optimizer_cfg_p1 = main_training_cfg.get("optimizer", {})
    # Parameter groups (if differentiated LR for backbone needed)
    # For simplicity, all params of CNNWithMetadata are trained with same LR initially
    # Or, adapt original backbone/head splitting logic for model.base_cnn_model vs other parts
    
    all_params_p1 = model.parameters() # Train all parameters
    if optimizer_cfg_p1.get("type", "AdamW").lower() == "adamw":
        optimizer_p1 = AdamW(all_params_p1, lr=optimizer_cfg_p1.get("lr"), weight_decay=optimizer_cfg_p1.get("weight_decay"))
    else: # Add Adam or other optimizers if needed
        optimizer_p1 = Adam(all_params_p1, lr=optimizer_cfg_p1.get("lr"), weight_decay=optimizer_cfg_p1.get("weight_decay",0))


    scheduler_cfg_p1 = main_training_cfg.get("scheduler", {})
    if scheduler_cfg_p1.get("type", "StepLR").lower() == "steplr":
        scheduler_p1 = StepLR(optimizer_p1, step_size=scheduler_cfg_p1.get("step_size"), gamma=scheduler_cfg_p1.get("gamma"))
    elif scheduler_cfg_p1.get("type", "").lower() == "cosineannealinglr":
        scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=scheduler_cfg_p1.get("t_max", main_training_cfg.get("num_epochs")), 
                                         eta_min=scheduler_cfg_p1.get("min_lr", 0.0))
    else: raise ValueError(f"Unsupported scheduler type for Phase 1: {scheduler_cfg_p1.get('type')}")

    best_metric_p1, best_epoch_p1, best_thresholds_p1 = run_training_phase(
        phase_name="Phase1_Joint", model=model, train_ld=train_ld_p1, val_ld=val_ld_p1,
        criterion=criterion, optimizer=optimizer_p1, scheduler=scheduler_p1, scaler=scaler,
        device=device, cfg=cfg, phase_cfg=main_training_cfg, fold=fold, label2idx=label2idx,
        tb_logger=tb_logger, run_ckpt_dir=run_ckpt_dir, exp_name_for_files=exp_name_for_files,
        start_epoch=0, ema_model=ema_model, initial_best_metric_val=0.0
    )
    
    current_overall_best_metric = best_metric_p1
    current_best_epoch = best_epoch_p1
    # Load the best model from phase 1 if continuing to phase 2
    if meta_tuning_cfg.get("enable", False) and best_epoch_p1 != -1:
        best_p1_ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_fold{fold}_Phase1_Joint_best.pt"
        if best_p1_ckpt_path.exists():
            logger.info(f"Loading best model from Phase 1 (E{best_epoch_p1}) for Meta Tuning: {best_p1_ckpt_path}")
            checkpoint = torch.load(best_p1_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if ema_model and checkpoint.get('ema_model_state_dict'):
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            # Optimizer/scheduler state not reloaded as we use new ones for phase 2
        else:
            logger.warning(f"Best checkpoint from Phase 1 not found at {best_p1_ckpt_path}. Proceeding with current model state for Phase 2.")


    # ========================= PHASE 2: Metadata Head Fine-tuning =========================
    if meta_tuning_cfg.get("enable", False):
        logger.info("===== Starting Phase 2: Metadata Head Fine-tuning =====")
        model.set_base_cnn_trainable(False) # Freeze CNN backbone

        # DataLoaders for Phase 2 (potentially different batch size)
        logger.info("Setting up DataLoaders for Phase 2 (Meta Tuning)...")
        # Re-use datasets, but new DataLoaders for different batch_size
        train_ds_phase2 = train_ds_phase1 # Can reuse dataset instance
        val_ds_phase2 = val_ds_phase1
        
        # Potentially new sampler if class balancing is desired for phase 2 specifically
        # For simplicity, reusing sampler config from main data config if any
        train_sampler_p2 = None
        if sampler_cfg.get("type") == "class_balanced_sqrt": # Example
             train_sampler_p2 = ClassBalancedSampler(train_ds_phase2, num_samples=len(train_ds_phase2))
             logger.info("Using ClassBalancedSampler for Phase 2.")

        train_ld_p2 = DataLoader(train_ds_phase2, batch_size=meta_tuning_cfg.get("batch_size"),
                                 sampler=train_sampler_p2, shuffle=(train_sampler_p2 is None),
                                 num_workers=data_handling_cfg.get("num_workers"), pin_memory=True,
                                 persistent_workers=data_handling_cfg.get("num_workers",0)>0)
        val_ld_p2 = DataLoader(val_ds_phase2, batch_size=meta_tuning_cfg.get("batch_size"), shuffle=False,
                               num_workers=data_handling_cfg.get("num_workers"), pin_memory=True,
                               persistent_workers=data_handling_cfg.get("num_workers",0)>0)

        # Optimizer and Scheduler for Phase 2
        # Only optimize unfrozen parameters (metadata_mlp, post_concat_fc, final_classifier)
        params_to_tune = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Number of parameters to tune in Phase 2: {sum(p.numel() for p in params_to_tune)}")

        opt_type_p2 = meta_tuning_cfg.get("optimizer_type", "AdamW").lower()
        if opt_type_p2 == "adamw":
            optimizer_p2 = AdamW(params_to_tune, lr=meta_tuning_cfg.get("lr"), weight_decay=meta_tuning_cfg.get("weight_decay", 0.0))
        elif opt_type_p2 == "adam":
            optimizer_p2 = Adam(params_to_tune, lr=meta_tuning_cfg.get("lr"), weight_decay=meta_tuning_cfg.get("weight_decay", 0.0))
        else: raise ValueError(f"Unsupported optimizer type for Phase 2: {opt_type_p2}")
        
        # Scheduler for Phase 2 - simple Cosine or StepLR, or even fixed LR
        # For simplicity, let's use a fixed LR by not stepping a scheduler, or a simple Cosine
        num_epochs_p2 = meta_tuning_cfg.get("epochs",50)
        scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=num_epochs_p2, eta_min=meta_tuning_cfg.get("lr")/10.0 if num_epochs_p2 > 1 else meta_tuning_cfg.get("lr")) # Example

        # Start epochs for phase 2 from where phase 1 left off for logging purposes
        phase1_epochs = main_training_cfg.get("num_epochs",0)

        best_metric_p2, best_epoch_p2, _ = run_training_phase(
            phase_name="Phase2_MetaTune", model=model, train_ld=train_ld_p2, val_ld=val_ld_p2,
            criterion=criterion, optimizer=optimizer_p2, scheduler=scheduler_p2, scaler=scaler,
            device=device, cfg=cfg, phase_cfg=meta_tuning_cfg, fold=fold, label2idx=label2idx,
            tb_logger=tb_logger, run_ckpt_dir=run_ckpt_dir, exp_name_for_files=exp_name_for_files,
            start_epoch=phase1_epochs, # Continue epoch count for logging
            ema_model=ema_model, # Pass EMA model
            initial_best_metric_val=0.0 # Reset best metric for this phase or carry over? For now, reset.
                                        # If carrying over: initial_best_metric_val = current_overall_best_metric
        )
        if best_metric_p2 > current_overall_best_metric : # Update if phase 2 was better
             current_overall_best_metric = best_metric_p2
             current_best_epoch = best_epoch_p2 # This epoch is relative to start of phase 2, needs global context

    tb_logger.close()
    logger.info(f"Finished training fold {fold}. Overall best {cfg.get('training',{}).get('model_selection_metric')}: {current_overall_best_metric:.4f}")
    return float(current_overall_best_metric) if current_overall_best_metric is not None else None


def main():
    ap = argparse.ArgumentParser(description="Train models with CV using YAML config, with metadata.")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logger.info(f"Starting experiment with metadata: {args.exp_name} with CLI args: {args}")

    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
    if not cfg_path.exists():
        fallback_cfg_path = Path(args.config_dir) / "config_metadata_example.yaml" # Default to new example
        if not args.config_file and fallback_cfg_path.exists():
            logger.warning(f"Config {cfg_path} not found. Using fallback {fallback_cfg_path}"); cfg_path = fallback_cfg_path
        else: raise FileNotFoundError(f"Config file {cfg_path} (and fallback) not found.")
    cfg = load_config(cfg_path); logger.info(f"Loaded config from {cfg_path}"); # cfg = cast_config_values(cfg) # Cast later or be careful

    exp_setup_cfg = cfg.get("experiment_setup", {})
    current_seed = args.seed if args.seed is not None else exp_setup_cfg.get("seed", 42)
    set_seed(current_seed); logger.info(f"Seed set to {current_seed}")
    exp_setup_cfg["seed"] = current_seed; cfg["experiment_setup"] = exp_setup_cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg = cfg.get("paths", {})
    config_proj_root = paths_cfg.get("project_root")
    base_path_for_config_paths = Path(config_proj_root).resolve() if config_proj_root else cfg_path.parent
    
    base_log_dir = _get_path_from_config(cfg, "log_dir", default="../outputs/tensorboard", base_path=base_path_for_config_paths)
    base_ckpt_dir = _get_path_from_config(cfg, "ckpt_dir", default="../outputs/checkpoints", base_path=base_path_for_config_paths)

    run_log_dir = base_log_dir / args.exp_name / timestamp
    run_ckpt_dir = base_ckpt_dir / args.exp_name / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True); run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run logs: {run_log_dir}"); logger.info(f"Run checkpoints: {run_ckpt_dir}")

    labels_csv_p = _get_path_from_config(cfg, "labels_csv", base_path=base_path_for_config_paths)
    meta_csv_p = _get_path_from_config(cfg, "meta_csv", base_path=base_path_for_config_paths) # Get meta_csv path
    train_root_p = _get_path_from_config(cfg, "train_root", base_path=base_path_for_config_paths)

    df_labels = pd.read_csv(labels_csv_p)
    meta_df = pd.read_csv(meta_csv_p) # Load metadata
    logger.info(f"Loaded labels from {labels_csv_p} ({len(df_labels)} rows)")
    logger.info(f"Loaded metadata from {meta_csv_p} ({len(meta_df)} rows)")
    
    required_cols = ['fold', 'label', 'dataset', 'filename']
    if any(c not in df_labels.columns for c in required_cols):
        raise ValueError(f"Missing required columns in {labels_csv_p}: {required_cols}")

    labels_unique = sorted(df_labels['label'].unique()); label2idx = {name: i for i, name in enumerate(labels_unique)}
    cfg["numClasses"] = len(labels_unique); cfg['label2idx'] = label2idx
    logger.info(f"Classes: {cfg['numClasses']}, Map: {label2idx}")

    device_str = exp_setup_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(device_str); logger.info(f"Using device: {device}")
    exp_setup_cfg["device"] = str(device); cfg["experiment_setup"] = exp_setup_cfg
    
    fold_primary_metric_results = []
    folds = sorted(df_labels['fold'].unique())
    if not list(folds): raise ValueError("No folds found in labels_csv.")
    logger.info(f"Found folds: {folds}")

    for fold_num in folds:
        logger.info(f"\n===== Processing Fold {fold_num} / {folds[-1]} =====")
        train_df_fold_split = df_labels[df_labels['fold'] != fold_num].reset_index(drop=True)
        val_df_fold_split = df_labels[df_labels['fold'] == fold_num].reset_index(drop=True)

        if train_df_fold_split.empty or val_df_fold_split.empty:
            logger.warning(f"Fold {fold_num} has empty train/val set. Skipping.")
            fold_primary_metric_results.append(None); continue
        
        logger.info(f"Fold {fold_num}: Train samples={len(train_df_fold_split)}, Val samples={len(val_df_fold_split)}")

        best_fold_metric_val = train_one_fold(
            fold=fold_num, train_df_fold=train_df_fold_split, val_df_fold=val_df_fold_split,
            meta_df_full=meta_df, # Pass full metadata
            cfg=cfg, label2idx=label2idx, train_root_path=train_root_p,
            run_log_dir=run_log_dir, run_ckpt_dir=run_ckpt_dir,
            exp_name_for_files=args.exp_name, device=device
        )
        fold_primary_metric_results.append(best_fold_metric_val)

    # --- Summarize CV Results ---
    # (Same as original, using fold_primary_metric_results)
    chosen_metric_name_summary = cfg.get("training",{}).get("model_selection_metric", "macro_auc").lower()
    successful_metrics = [m for m in fold_primary_metric_results if m is not None and not np.isnan(m)]
    
    if successful_metrics:
        mean_metric = float(np.mean(successful_metrics))
        std_metric = float(np.std(successful_metrics, ddof=1)) if len(successful_metrics) > 1 else 0.0
        logger.info(f"\n===== {len(folds)}-Fold CV Results (Primary Metric: {chosen_metric_name_summary}) =====")
        for i, metric_val in enumerate(fold_primary_metric_results):
            fold_id_display = folds[i]
            logger.info(f"Fold {fold_id_display} Best {chosen_metric_name_summary}: {metric_val:.4f}" if metric_val is not None else f"Fold {fold_id_display}: SKIPPED/N/A")
        logger.info(f"Average Best {chosen_metric_name_summary} (over {len(successful_metrics)} successful folds) = {mean_metric:.4f} ± {std_metric:.4f}")
    else:
        logger.warning("No folds yielded valid primary metric values for averaging.")
        
    logger.info(f"Experiment {args.exp_name} (timestamp: {timestamp}) finished.")


if __name__ == "__main__":
    main()
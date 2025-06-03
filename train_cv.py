#!/usr/bin/env python
# your_project_name/train_cv_final_optimized.py
from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from pathlib import Path
import time
import copy
import logging
import sys
from datetime import datetime
import contextlib # For conditional CudaTimer

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from torchmetrics import AUROC, F1Score, AveragePrecision
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from data_handling import (
    FlatDataset, build_transform, build_gpu_transform_pipeline,
    ClassBalancedSampler
)
from models import get_model
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
    pass

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
    num_classes = len(label2idx)
    counts = np.zeros(num_classes, dtype=int)
    mapped_indices = df['label'].map(label2idx)
    if mapped_indices.isnull().any():
        unmapped_originals = df['label'][mapped_indices.isnull()].unique()
        logger.warning(
            f"Some labels in DataFrame could not be mapped. Unique unmapped: {unmapped_originals}. label2idx keys: {list(label2idx.keys())[:5]}"
        )
        class_series = mapped_indices.dropna().astype(int).value_counts()
    else:
        class_series = mapped_indices.astype(int).value_counts()
    for class_idx, count_val in class_series.items():
        if 0 <= class_idx < num_classes: counts[class_idx] = count_val
        else: logger.warning(f"Out-of-bounds mapped class index {class_idx}. Num_classes: {num_classes}. Ignored.")
    return counts

def generate_confusion_matrix_figure(true_labels: np.ndarray, pred_labels: np.ndarray, display_labels: list[str], title: str):
    """Generates a matplotlib figure for the confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(max(8, len(display_labels)*0.8), max(8, len(display_labels)*0.8))) # Dynamic sizing
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    label2idx: dict[str,int],
    train_root_path: Path,
    run_log_dir: Path,
    run_ckpt_dir: Path,
    exp_name_for_files: str,
    device: torch.device,
) -> float | None:
    fold_specific_tb_log_dir = run_log_dir / f"fold_{fold}"; fold_specific_tb_log_dir.mkdir(parents=True, exist_ok=True)

    experiment_setup_cfg = cfg.get("experiment_setup", {})
    model_yaml_cfg = cfg.get("model", {})
    training_loop_cfg = cfg.get("training", {})
    data_handling_cfg = cfg.get("data", {})
    cpu_augmentations_cfg = data_handling_cfg.get("cpu_augmentations", {})
    gpu_augmentations_cfg = data_handling_cfg.get("gpu_augmentations", {})
    torch_compile_cfg = cfg.get("torch_compile", {})
    loss_cfg = training_loop_cfg.get("loss", {})
    optimizer_yaml_cfg = training_loop_cfg.get("optimizer", {})
    scheduler_yaml_cfg = training_loop_cfg.get("scheduler", {})
    tb_logging_cfg = cfg.get("tensorboard_logging", {})

    logger.info(f"Starting training for fold {fold} of experiment '{exp_name_for_files}'. TB logs: {fold_specific_tb_log_dir}")
    try:
        tb_logger = TensorBoardLogger(log_dir=fold_specific_tb_log_dir, experiment_config=cfg,
                                      train_loader_len=len(train_df) // training_loop_cfg.get("batch_size", 32) +1)
    except Exception as e:
        logger.error(f"Failed to initialize TensorBoardLogger for fold {fold}: {e}", exc_info=True)
        return None

    tf_train_cpu = build_transform(cpu_augmentations_cfg, train=True)
    tf_val_cpu = build_transform(cpu_augmentations_cfg, train=False)

    train_ds = FlatDataset(train_df, train_root_path, label2idx, tf_train_cpu,
                           image_loader=data_handling_cfg.get("image_loader", "pil"),
                           enable_ram_cache=data_handling_cfg.get("enable_ram_cache", False))
    val_ds = FlatDataset(val_df, train_root_path, label2idx, tf_val_cpu,
                         image_loader=data_handling_cfg.get("image_loader", "pil"),
                         enable_ram_cache=data_handling_cfg.get("enable_ram_cache", False))

    train_sampler = None
    if data_handling_cfg.get("sampler", {}).get("type") == "class_balanced_sqrt":
        train_sampler = ClassBalancedSampler(train_ds, num_samples=len(train_ds)); logger.info("Using ClassBalancedSampler.")
    shuffle_train = train_sampler is None

    train_ld = DataLoader(train_ds, batch_size=training_loop_cfg.get("batch_size"), sampler=train_sampler, shuffle=shuffle_train,
        num_workers=data_handling_cfg.get("num_workers", 0),
        pin_memory=(device.type == 'cuda' and data_handling_cfg.get("num_workers", 0) > 0),
        persistent_workers=data_handling_cfg.get("persistent_workers", False) and data_handling_cfg.get("num_workers", 0) > 0,
        prefetch_factor=data_handling_cfg.get("prefetch_factor", 2) if data_handling_cfg.get("num_workers", 0) > 0 else None)
    val_ld = DataLoader(val_ds, batch_size=training_loop_cfg.get("batch_size"), shuffle=False,
        num_workers=data_handling_cfg.get("num_workers", 0),
        pin_memory=(device.type == 'cuda' and data_handling_cfg.get("num_workers", 0) > 0),
        persistent_workers=data_handling_cfg.get("persistent_workers", False) and data_handling_cfg.get("num_workers", 0) > 0,
        prefetch_factor=data_handling_cfg.get("prefetch_factor", 2) if data_handling_cfg.get("num_workers", 0) > 0 else None)

    model_factory_cfg = model_yaml_cfg.copy(); model_factory_cfg["numClasses"] = len(label2idx)
    if "type" in model_factory_cfg: model_factory_cfg["MODEL_TYPE"] = model_factory_cfg.pop("type")
    model = get_model(model_factory_cfg).to(device)
    # ... (Compile logic) ...

    ema_model = None
    ema_decay = training_loop_cfg.get("ema_decay", 0.0)
    if ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device); [p.requires_grad_(False) for p in ema_model.parameters()]
        logger.info(f"EMA enabled with decay {ema_decay}.")

    optimizer_params = [{'params': model.parameters(), 'lr': optimizer_yaml_cfg.get("lr")}]
    optimizer = AdamW(optimizer_params, weight_decay=optimizer_yaml_cfg.get("weight_decay"))
    scheduler = CosineAnnealingLR(optimizer, T_max=training_loop_cfg.get("num_epochs"), eta_min=scheduler_yaml_cfg.get("min_lr", 0.0))
    scaler = GradScaler(enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True)))
    accum_steps = training_loop_cfg.get("accum_steps", 1)

    post_hoc_unk_threshold = training_loop_cfg.get("post_hoc_unk_threshold", None)
    unk_label_str_config = training_loop_cfg.get("unk_label_string", "UNK")
    known_class_indices_model = []
    unk_idx_model = -1
    if post_hoc_unk_threshold is not None:
        logger.info(f"Fold {fold}: Post-hoc '{unk_label_str_config}' classification with threshold: {post_hoc_unk_threshold}")
        if unk_label_str_config in label2idx: unk_idx_model = label2idx[unk_label_str_config]
        else: logger.warning(f"'{unk_label_str_config}' not in label2idx. Cannot assign post-hoc UNK index.")
        for label_name, idx_val in label2idx.items():
            if label_name != unk_label_str_config: known_class_indices_model.append(idx_val)
        if not known_class_indices_model or unk_idx_model == -1 : # If no known classes OR UNK index is invalid
            logger.warning(f"Fold {fold}: Invalid setup for post-hoc UNK. Thresholding skipped.")
            post_hoc_unk_threshold = None

    model_selection_metric_name = training_loop_cfg.get("model_selection_metric", "macro_auc").lower()
    best_metric_val = -float('inf') # Initialize to handle metrics that can be negative
    patience_counter = 0
    current_primary_metric_val_for_last_ckpt = -float('inf')

    num_classes = cfg["numClasses"]
    class_counts_train_fold = get_class_counts(train_df, label2idx)
    criterion = None; loss_type = loss_cfg.get("type", "cross_entropy").lower()
    # ... (Loss function setup as in previous train_cv_optimized.py) ...
    if loss_type == "focal_ce_loss":
        criterion = focal_ce_loss; focal_alpha = loss_cfg.get("focal_alpha",1.0); focal_gamma = loss_cfg.get("focal_gamma",2.0)
        logger.info(f"Using Focal CE Loss (alpha={focal_alpha}, gamma={focal_gamma}).")
    elif loss_type == "cross_entropy": criterion = nn.CrossEntropyLoss() # DRW handled separately
    # ... (other loss types)

    val_metrics_heavy_interval = training_loop_cfg.get("val_metrics_heavy_interval", 9999) # Default to very high
    profiler_cfg_tb = tb_logging_cfg.get("profiler", {})
    enable_batch_timing_always = profiler_cfg_tb.get("enable_batch_timing_always", False)
    drw_schedule_epochs_list = training_loop_cfg.get("drw_schedule_epochs", []); current_drw_stage = 0


    logger.info(f"Starting training loop for {training_loop_cfg.get('num_epochs')} epochs.")
    for epoch in range(training_loop_cfg.get("num_epochs")):
        # ... (DRW stage updates) ...
        if drw_schedule_epochs_list: # Simplified DRW logic for brevity
            # ... (DRW weight update logic) ...
            pass

        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(epoch)
        current_epoch_profiler = tb_logger.setup_profiler(epoch, fold_specific_tb_log_dir)
        model.train()
        epoch_gpu_time_ms = 0.0; optimizer.zero_grad(); epoch_start_time = time.time()
        cumulative_train_loss_for_pbar, cumulative_train_corrects_for_pbar, cumulative_train_samples_for_pbar = 0.0,0,0
        train_pbar = tqdm(train_ld, desc=f"Fold {fold} E{epoch} Train", ncols=experiment_setup_cfg.get("TQDM_NCOLS", 100))

        for batch_idx, (imgs_cpu, labels_cpu) in enumerate(train_pbar):
            imgs_device = imgs_cpu.to(device, non_blocking=True); labels_device = labels_cpu.to(device, non_blocking=True)
            
            batch_gpu_time_ms_this_batch = 0.0
            timer_active_this_batch = current_epoch_profiler is not None or \
                                   (enable_batch_timing_always and tb_logger._should_log_batch(tb_logger.log_interval_train_batch, batch_idx))
            
            timer_context = CudaTimer(device) if timer_active_this_batch and device.type == 'cuda' else contextlib.nullcontext()

            with timer_context as timer:
                with autocast(device_type=device.type, enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))):
                    logits = model(imgs_device)
                    if loss_type == "focal_ce_loss":
                        loss = criterion(logits.float(), labels_device, alpha=focal_alpha, gamma=focal_gamma)
                    else:
                        loss = criterion(logits.float(), labels_device)
                    if accum_steps > 1: loss = loss / accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_ld):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if ema_model is not None and ema_decay > 0: update_ema(ema_model, model, ema_decay)
            
            if timer_active_this_batch and device.type == 'cuda': batch_gpu_time_ms_this_batch = timer.get_elapsed_time_ms()
            epoch_gpu_time_ms += batch_gpu_time_ms_this_batch
            
            # Update pbar
            batch_loss_val = loss.item() * (accum_steps if accum_steps > 1 else 1)
            preds_train = logits.argmax(dim=1)
            batch_corrects = (preds_train == labels_device).float().sum().item()
            batch_samples = imgs_device.size(0)
            cumulative_train_loss_for_pbar += batch_loss_val * batch_samples
            cumulative_train_corrects_for_pbar += batch_corrects
            cumulative_train_samples_for_pbar += batch_samples
            avg_epoch_loss_pbar = cumulative_train_loss_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            avg_epoch_acc_pbar = cumulative_train_corrects_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            train_pbar.set_postfix(loss=f"{avg_epoch_loss_pbar:.4f}", acc=f"{avg_epoch_acc_pbar:.4f}")
            
            # Log to TensorBoard (respecting intervals)
            tb_logger.log_train_batch_metrics(loss=batch_loss_val, acc=(batch_corrects/batch_samples if batch_samples>0 else 0),
                                              lr=optimizer.param_groups[0]['lr'], epoch=epoch, batch_idx=batch_idx,
                                              batch_gpu_time_ms=batch_gpu_time_ms_this_batch if timer_active_this_batch else None)
            if current_epoch_profiler: tb_logger.step_profiler()
        train_pbar.close()

        epoch_duration = time.time() - epoch_start_time; scheduler.step()
        epoch_metrics = {"Loss/train_epoch": avg_epoch_loss_pbar, "Accuracy/train_epoch": avg_epoch_acc_pbar, 
                         "LearningRate/epoch": optimizer.param_groups[0]['lr'],
                         "Time/Train_epoch_duration_sec": epoch_duration, 
                         "Throughput/train_samples_per_sec": cumulative_train_samples_for_pbar / epoch_duration if epoch_duration > 0 else 0}
        if device.type == 'cuda': epoch_metrics["Time/GPU_ms_per_train_epoch"] = epoch_gpu_time_ms
        if current_epoch_profiler: tb_logger.stop_and_process_profiler(); current_epoch_profiler = None

        # --- Validation ---
        if epoch % training_loop_cfg.get("val_interval", 1) == 0 or epoch == training_loop_cfg.get("num_epochs") - 1:
            eval_model = ema_model if ema_model is not None and training_loop_cfg.get("use_ema_for_val", True) else model
            eval_model.eval()
            all_logits_val, all_labels_val_cpu_true = [], []
            val_loss_sum, val_corrects_original_argmax, val_seen_samples = 0.0,0,0
            val_pbar = tqdm(val_ld, desc=f"Fold {fold} E{epoch} Val", ncols=experiment_setup_cfg.get("TQDM_NCOLS",100))
            with torch.no_grad():
                for batch_idx_val, (imgs_val_cpu, labels_val_cpu_batch) in enumerate(val_pbar):
                    imgs_val_dev = imgs_val_cpu.to(device,non_blocking=True); labels_val_dev_batch = labels_val_cpu_batch.to(device,non_blocking=True)
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))):
                        logits_val_batch = eval_model(imgs_val_dev)
                        if loss_type == "focal_ce_loss": loss_v = criterion(logits_val_batch.float(), labels_val_dev_batch, alpha=focal_alpha, gamma=focal_gamma)
                        else: loss_v = criterion(logits_val_batch.float(), labels_val_dev_batch)
                    val_loss_sum += loss_v.item() * imgs_val_dev.size(0)
                    val_corrects_original_argmax += (logits_val_batch.argmax(dim=1) == labels_val_dev_batch).float().sum().item()
                    val_seen_samples += imgs_val_dev.size(0)
                    all_logits_val.append(logits_val_batch.cpu()); all_labels_val_cpu_true.append(labels_val_cpu_batch.cpu())
                    val_pbar.set_postfix(avg_loss=f"{val_loss_sum/val_seen_samples:.4f}", avg_acc=f"{val_corrects_original_argmax/val_seen_samples:.4f}")
                    tb_logger.log_val_batch_metrics(loss=loss_v.item(), acc=((logits_val_batch.argmax(dim=1) == labels_val_dev_batch).float().sum().item()/imgs_val_dev.size(0) if imgs_val_dev.size(0)>0 else 0),
                                                    epoch=epoch, batch_idx=batch_idx_val)
            val_pbar.close()
            all_logits_val_cat = torch.cat(all_logits_val); all_labels_val_cat_cpu = torch.cat(all_labels_val_cpu_true)
            all_probs_val_cat = F.softmax(all_logits_val_cat, dim=1)
            
            epoch_metrics["Loss/val_epoch"] = val_loss_sum / val_seen_samples if val_seen_samples > 0 else 0
            epoch_metrics["Accuracy/val_epoch_original_argmax"] = val_corrects_original_argmax / val_seen_samples if val_seen_samples > 0 else 0

            model_original_preds_indices = torch.argmax(all_probs_val_cat, dim=1)
            all_final_preds_for_fold_cpu = model_original_preds_indices.clone()
            if post_hoc_unk_threshold is not None and known_class_indices_model : # unk_idx_model !=-1 is implicitly handled by post_hoc_unk_threshold not being None
                for sample_idx in range(all_probs_val_cat.size(0)):
                    probs_for_known = all_probs_val_cat[sample_idx][known_class_indices_model]
                    if probs_for_known.numel() > 0 and torch.max(probs_for_known).item() < post_hoc_unk_threshold:
                        if unk_idx_model != -1: # Ensure UNK is a valid index to assign to
                           all_final_preds_for_fold_cpu[sample_idx] = unk_idx_model
            
            epoch_metrics["Accuracy/val_epoch_post_hoc_unk"] = (all_final_preds_for_fold_cpu == all_labels_val_cat_cpu).sum().item() / val_seen_samples if val_seen_samples > 0 else 0
            epoch_metrics["AUROC/val_macro"] = AUROC(task="multiclass", num_classes=num_classes, average="macro")(all_probs_val_cat, all_labels_val_cat_cpu).item()
            epoch_metrics["F1Score/val_macro_post_hoc_unk"] = F1Score(task="multiclass", num_classes=num_classes, average="macro")(all_final_preds_for_fold_cpu, all_labels_val_cat_cpu).item()

            optimal_thresholds_for_ckpt_pr = {}
            if epoch % val_metrics_heavy_interval == 0 or epoch == training_loop_cfg.get("num_epochs") - 1:
                logger.info(f"Fold {fold} E{epoch}: Calculating HEAVY validation metrics.")
                if num_classes > 1:
                    labels_oh_val = F.one_hot(all_labels_val_cat_cpu, num_classes).numpy(); probs_np_val = all_probs_val_cat.numpy()
                    current_optimal_f1_scores_pr, current_optimal_sensitivities_pr = [], []
                    for i in range(num_classes):
                        opt_f1_pr, opt_thr_pr, opt_sens_pr = 0.0, 0.5, 0.0
                        try:
                            prec, rec, thr = precision_recall_curve(labels_oh_val[:, i], probs_np_val[:, i])
                            if len(prec) > 1 and len(rec) > 1 and len(thr) >0:
                                f1s_curve=(2*prec*rec)/(prec+rec+1e-8);rel_f1s=f1s_curve[1:];rel_recs=rec[1:]
                                valid_f1_idx=np.where(np.isfinite(rel_f1s)&(prec[1:]+rec[1:]>0))[0]
                                if len(valid_f1_idx)>0:
                                    best_idx=valid_f1_idx[np.argmax(rel_f1s[valid_f1_idx])]
                                    opt_f1_pr=float(rel_f1s[best_idx]);opt_thr_pr=float(thr[best_idx]);opt_sens_pr=float(rel_recs[best_idx])
                        except Exception as e_pr: logger.warning(f"PR curve fail C{i} E{epoch}: {e_pr}")
                        current_optimal_f1_scores_pr.append(opt_f1_pr); current_optimal_sensitivities_pr.append(opt_sens_pr)
                        optimal_thresholds_for_ckpt_pr[i] = opt_thr_pr
                    if current_optimal_f1_scores_pr: epoch_metrics["F1Score/val_mean_optimal_per_class_from_PR"] = np.mean(current_optimal_f1_scores_pr)
                    if current_optimal_sensitivities_pr: epoch_metrics["Sensitivity/val_mean_optimal_per_class_from_PR"] = np.mean(current_optimal_sensitivities_pr)
            
            current_primary_metric_val = epoch_metrics.get(model_selection_metric_name.replace("mean_optimal_f1", "F1Score/val_mean_optimal_per_class_from_PR")
                                                                            .replace("mean_optimal_sensitivity", "Sensitivity/val_mean_optimal_per_class_from_PR")
                                                                            .replace("f1_macro_post_hoc_unk", "F1Score/val_macro_post_hoc_unk")
                                                                            .replace("accuracy_post_hoc_unk", "Accuracy/val_epoch_post_hoc_unk")
                                                                            .replace("macro_auc", "AUROC/val_macro"), # Fallback for simple macro_auc
                                                            0.0) # Default if key missing after replacement
            current_primary_metric_val_for_last_ckpt = current_primary_metric_val

            logger.info(f"Fold{fold} E{epoch} Val -> Loss={epoch_metrics['Loss/val_epoch']:.4f} AccPostHoc={epoch_metrics['Accuracy/val_epoch_post_hoc_unk']:.4f} SelectedMetric ({model_selection_metric_name})={current_primary_metric_val:.4f}")

            if current_primary_metric_val > best_metric_val:
                best_metric_val = current_primary_metric_val; patience_counter=0
                ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_fold{fold}_best.pt"
                checkpoint_data = {'epoch':epoch, 'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
                                   'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                                   'scaler_state_dict':scaler.state_dict(), f'best_{model_selection_metric_name}': best_metric_val,
                                   'config_runtime':cfg,'label2idx':label2idx}
                if ema_model: checkpoint_data['ema_model_state_dict'] = getattr(ema_model, '_orig_mod', ema_model).state_dict()
                if (training_loop_cfg.get("save_optimal_thresholds_from_pr", False) or \
                    model_selection_metric_name in ["mean_optimal_f1", "mean_optimal_sensitivity"]) and optimal_thresholds_for_ckpt_pr:
                    checkpoint_data['optimal_thresholds_val_from_pr'] = optimal_thresholds_for_ckpt_pr
                if post_hoc_unk_threshold is not None: checkpoint_data['post_hoc_unk_threshold_used'] = post_hoc_unk_threshold
                torch.save(checkpoint_data, str(ckpt_path))
                logger.info(f"Saved best model to {ckpt_path} based on {model_selection_metric_name}: {best_metric_val:.4f}")
            else:
                patience_counter+=1
                if patience_counter >= training_loop_cfg.get("early_stopping_patience",10):
                    logger.info(f"Early stopping E{epoch} F{fold}."); break
        
        tb_logger.log_epoch_summary(epoch_metrics, epoch) # This calls tb_logger.flush() internally

    # --- End of Fold: Log Confusion Matrix for Best Model ---
    if best_metric_val > -float('inf'): # Check if any best model was saved
        best_ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_fold{fold}_best.pt"
        if best_ckpt_path.exists():
            logger.info(f"Fold {fold}: Loading best model from {best_ckpt_path} for final Confusion Matrix.")
            checkpoint_best = torch.load(best_ckpt_path, map_location=device) # Load to current device
            
            # Re-create model instance and load state dict
            # Ensure numClasses is correct for the loaded model config
            model_cfg_best = checkpoint_best['config_runtime']['model']
            if "numClasses" not in model_cfg_best: model_cfg_best["numClasses"] = len(checkpoint_best['label2idx'])
            if "type" in model_cfg_best and "MODEL_TYPE" not in model_cfg_best: model_cfg_best["MODEL_TYPE"] = model_cfg_best.pop("type")

            model_for_cm = get_model(model_cfg_best).to(device)
            
            # Decide whether to load EMA or standard model weights
            final_weights_to_load_key = 'model_state_dict'
            if training_loop_cfg.get("use_ema_for_val", True) and 'ema_model_state_dict' in checkpoint_best and checkpoint_best['ema_model_state_dict'] is not None:
                final_weights_to_load_key = 'ema_model_state_dict'
                logger.info(f"Fold {fold}: Using EMA weights for final CM.")
            
            model_for_cm.load_state_dict(checkpoint_best[final_weights_to_load_key])
            model_for_cm.eval()

            all_logits_cm, all_labels_cm = [], []
            with torch.no_grad():
                for imgs_val_cpu, labels_val_cpu_batch in val_ld: # Iterate over val_ld again
                    imgs_val_dev = imgs_val_cpu.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))):
                        logits_val_batch = model_for_cm(imgs_val_dev)
                    all_logits_cm.append(logits_val_batch.cpu())
                    all_labels_cm.append(labels_val_cpu_batch.cpu())
            
            all_logits_cm_cat = torch.cat(all_logits_cm)
            all_labels_cm_cat = torch.cat(all_labels_cm)
            all_probs_cm_cat = F.softmax(all_logits_cm_cat, dim=1)

            final_preds_cm = torch.argmax(all_probs_cm_cat, dim=1) # Start with argmax
            # Apply post-hoc UNK if it was configured for the run
            cm_post_hoc_unk_threshold = checkpoint_best.get('post_hoc_unk_threshold_used', training_loop_cfg.get("post_hoc_unk_threshold", None)) # Get from ckpt or current cfg
            cm_unk_label_str = checkpoint_best['config_runtime'].get("training",{}).get("unk_label_string", "UNK")
            cm_label2idx = checkpoint_best['label2idx']
            cm_unk_idx = cm_label2idx.get(cm_unk_label_str, -1)
            cm_known_indices = [idx for label, idx in cm_label2idx.items() if label != cm_unk_label_str]

            if cm_post_hoc_unk_threshold is not None and cm_known_indices and cm_unk_idx != -1:
                logger.info(f"Fold {fold}: Applying post-hoc '{cm_unk_label_str}' (thresh {cm_post_hoc_unk_threshold}) for CM.")
                for i in range(all_probs_cm_cat.size(0)):
                    probs_known = all_probs_cm_cat[i][cm_known_indices]
                    if probs_known.numel() > 0 and torch.max(probs_known).item() < cm_post_hoc_unk_threshold:
                        final_preds_cm[i] = cm_unk_idx
            
            cm_display_labels = [label for label, idx in sorted(cm_label2idx.items(), key=lambda item: item[1])]
            cm_title = f"Fold {fold} Val CM (Best Model E{checkpoint_best['epoch']})"
            if cm_post_hoc_unk_threshold is not None: cm_title += f" UNK Thresh={cm_post_hoc_unk_threshold:.2f}"
            
            cm_fig = generate_confusion_matrix_figure(all_labels_cm_cat.numpy(), final_preds_cm.numpy(), cm_display_labels, cm_title)
            tb_logger.writer.add_figure(f"Fold_{fold}/ConfusionMatrix_BestModel_ValSet", cm_fig, global_step=training_loop_cfg.get("num_epochs")-1) # Log at last epoch step
            plt.close(cm_fig) # Close figure to free memory
        else:
            logger.warning(f"Fold {fold}: Best checkpoint file not found at {best_ckpt_path}. Cannot log confusion matrix.")


    # Save last model checkpoint
    if epoch == training_loop_cfg.get("num_epochs") -1 : # Check if loop completed fully
        last_ckpt_path=run_ckpt_dir/f"{exp_name_for_files}_fold{fold}_last_E{epoch}.pt"
        # ... (construct and save last checkpoint data as before) ...
        model_sd_to_save = getattr(model, '_orig_mod', model).state_dict()
        ema_sd_to_save = getattr(ema_model, '_orig_mod', ema_model).state_dict() if ema_model else None
        last_ckpt_data = {'epoch':epoch,'model_state_dict':model_sd_to_save,'ema_model_state_dict':ema_sd_to_save,
                    'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),
                    'current_primary_metric': current_primary_metric_val_for_last_ckpt,
                    'config_runtime':cfg,'label2idx':label2idx}
        if post_hoc_unk_threshold is not None: last_ckpt_data['post_hoc_unk_threshold_used'] = post_hoc_unk_threshold
        torch.save(last_ckpt_data, str(last_ckpt_path))
        logger.info(f"Saved last model checkpoint to {last_ckpt_path}")

    tb_logger.close() # This calls flush() internally
    logger.info(f"Finished training fold {fold}. Best {model_selection_metric_name}: {best_metric_val:.4f}")
    return float(best_metric_val) if best_metric_val > -float('inf') else None


def main():
    ap = argparse.ArgumentParser(description="Train models with CV using YAML config (Optimized Version).")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config if set.")
    args = ap.parse_args()
    # ... (logging setup, config loading using new default name, seed setting, path setup as before) ...
    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml" # Standard behavior
    if not cfg_path.exists(): # Fallback to new default config name
        fallback_cfg_path = Path(args.config_dir) / "config_cv_final_optimized.yaml"
        if not args.config_file and fallback_cfg_path.exists(): # Only use fallback if no specific file was given
            logger.warning(f"Config {cfg_path} not found. Using fallback {fallback_cfg_path}"); cfg_path = fallback_cfg_path
        else: raise FileNotFoundError(f"Config file {cfg_path} not found (and fallback '{fallback_cfg_path}' if applicable).")
    cfg = load_config(cfg_path); logger.info(f"Loaded config from {cfg_path}"); cfg = cast_config_values(cfg)

    exp_setup_cfg = cfg.get("experiment_setup", {})
    current_seed = args.seed if args.seed is not None else exp_setup_cfg.get("seed", 42)
    set_seed(current_seed); logger.info(f"Seed set to {current_seed}")
    exp_setup_cfg["seed"] = current_seed; cfg["experiment_setup"] = exp_setup_cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg = cfg.get("paths", {})
    config_proj_root = paths_cfg.get("project_root") # Optional project root from config
    base_path_for_config_paths = Path(config_proj_root).resolve() if config_proj_root else cfg_path.parent
    
    base_log_dir = _get_path_from_config(cfg, "log_dir", default="../outputs/tensorboard_cv_final", base_path=base_path_for_config_paths)
    base_ckpt_dir = _get_path_from_config(cfg, "ckpt_dir", default="../outputs/checkpoints_cv_final", base_path=base_path_for_config_paths)
    run_specific_log_dir = base_log_dir / args.exp_name / timestamp
    run_specific_ckpt_dir = base_ckpt_dir / args.exp_name / timestamp
    run_specific_log_dir.mkdir(parents=True, exist_ok=True); run_specific_ckpt_dir.mkdir(parents=True, exist_ok=True)

    labels_csv_p = _get_path_from_config(cfg, "labels_csv", base_path=base_path_for_config_paths)
    train_root_p = _get_path_from_config(cfg, "train_root", base_path=base_path_for_config_paths)
    
    df = pd.read_csv(labels_csv_p)
    labels_unique = sorted(df['label'].unique()); label2idx = {name: i for i, name in enumerate(labels_unique)}
    cfg["numClasses"] = len(labels_unique); cfg['label2idx'] = label2idx
    logger.info(f"Classes: {cfg['numClasses']}, Map: {label2idx}")

    device_str = exp_setup_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(device_str)
    exp_setup_cfg["device"] = str(device); cfg["experiment_setup"] = exp_setup_cfg
    
    fold_primary_metric_results = []
    folds_str_series = df['fold'].astype(str).unique() # Convert to string first for consistent unique
    folds = sorted([int(f) if f.isdigit() else f for f in folds_str_series]) # Attempt conversion for sorting
    if not list(folds): raise ValueError("No folds found.")
    logger.info(f"Found folds: {folds}")

    for fold_num_orig in folds:
        current_fold_for_df_query = fold_num_orig
        if df['fold'].dtype != type(fold_num_orig): # Try to match type for robust querying
            try:
                if pd.api.types.is_numeric_dtype(df['fold'].dtype) and isinstance(fold_num_orig, str) and fold_num_orig.isdigit():
                    current_fold_for_df_query = int(fold_num_orig)
                elif pd.api.types.is_string_dtype(df['fold'].dtype) and isinstance(fold_num_orig, int):
                    current_fold_for_df_query = str(fold_num_orig)
            except ValueError: pass

        logger.info(f"\n===== Processing Fold {fold_num_orig} =====")
        train_df_fold = df[df['fold'] != current_fold_for_df_query].reset_index(drop=True)
        val_df_fold = df[df['fold'] == current_fold_for_df_query].reset_index(drop=True)

        if train_df_fold.empty or val_df_fold.empty:
            logger.warning(f"Fold {fold_num_orig} has empty train/val. Skipping."); fold_primary_metric_results.append(None); continue
        
        best_fold_metric_val = train_one_fold(
            fold=fold_num_orig, train_df=train_df_fold, val_df=val_df_fold, cfg=cfg, label2idx=label2idx,
            train_root_path=train_root_p, run_log_dir=run_specific_log_dir, run_ckpt_dir=run_specific_ckpt_dir,
            exp_name_for_files=args.exp_name, device=device
        )
        fold_primary_metric_results.append(best_fold_metric_val)
        if device.type == 'cuda': torch.cuda.empty_cache()
    
    # ... (Summarize CV results as before) ...
    chosen_metric_name_summary = cfg.get("training",{}).get("model_selection_metric", "macro_auc").lower()
    successful_metrics = [m_val for m_val in fold_primary_metric_results if m_val is not None and not np.isnan(m_val)]
    if successful_metrics:
        mean_metric = float(np.mean(successful_metrics)); std_metric = float(np.std(successful_metrics, ddof=1)) if len(successful_metrics) > 1 else 0.0
        logger.info(f"\n===== {len(folds)}-Fold CV Results (Primary Metric: {chosen_metric_name_summary}) =====")
        for i, metric_val in enumerate(fold_primary_metric_results):
            fold_id_display = folds[i]
            logger.info(f"Fold {fold_id_display} Best {chosen_metric_name_summary}: {metric_val:.4f}" if metric_val is not None else f"Fold {fold_id_display}: SKIPPED/N/A")
        logger.info(f"Average Best {chosen_metric_name_summary} (over {len(successful_metrics)} successful folds) = {mean_metric:.4f} ± {std_metric:.4f}")
    else: logger.warning("No folds yielded valid primary metric values for averaging.")
    logger.info(f"Experiment {args.exp_name} (timestamp: {timestamp}) finished.")

if __name__ == "__main__":
    main()
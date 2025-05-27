#!/usr/bin/env python
# your_project_name/train_cv.py
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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from torchmetrics import AUROC, F1Score, AveragePrecision
from sklearn.metrics import precision_recall_curve
from torch.amp import autocast, GradScaler

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
    if df['label'].dtype == 'object':
        class_series = df['label'].map(label2idx).value_counts()
    else:
        class_series = df['label'].value_counts()
    for class_idx, count in class_series.items():
        if 0 <= class_idx < num_classes:
            counts[class_idx] = count
        else:
            logger.warning(f"Out-of-bounds class index {class_idx} while getting class counts. Num_classes: {num_classes}. This class will be ignored in counts.")
    return counts

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
) -> float:
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

    logger.info(f"Starting training for fold {fold} of experiment '{exp_name_for_files}'. TB logs: {fold_specific_tb_log_dir}")
    tb_logger = TensorBoardLogger(log_dir=fold_specific_tb_log_dir, experiment_config=cfg,
                                  train_loader_len=len(train_df) // training_loop_cfg.get("batch_size", 32) +1)

    if not cpu_augmentations_cfg: raise ValueError("Missing 'data.cpu_augmentations' section in YAML.")
    tf_train_cpu = build_transform(cpu_augmentations_cfg, train=True)
    tf_val_cpu = build_transform(cpu_augmentations_cfg, train=False)
    train_ds = FlatDataset(train_df, train_root_path, label2idx, tf_train_cpu)
    val_ds = FlatDataset(val_df, train_root_path, label2idx, tf_val_cpu)
    
    sampler_cfg = data_handling_cfg.get("sampler", {}); train_sampler = None
    if sampler_cfg.get("type") == "class_balanced_sqrt":
        if hasattr(train_ds, 'samples') or hasattr(train_ds, 'targets') or hasattr(train_ds, 'get_labels'):
            train_sampler = ClassBalancedSampler(train_ds, num_samples=len(train_ds)); logger.info("Using ClassBalancedSampler.")
        else: logger.warning("ClassBalancedSampler selected, but dataset lacks attributes. Using default sampler.")
    shuffle_train = train_sampler is None

    train_ld = DataLoader(train_ds, batch_size=training_loop_cfg.get("batch_size"), sampler=train_sampler, shuffle=shuffle_train,
        num_workers=data_handling_cfg.get("num_workers"), pin_memory=True, persistent_workers=data_handling_cfg.get("num_workers", 0) > 0,
        prefetch_factor=cfg.get("PREFETCH_FACTOR", 4))
    val_ld = DataLoader(val_ds, batch_size=training_loop_cfg.get("batch_size"), shuffle=False, num_workers=data_handling_cfg.get("num_workers"),
        pin_memory=True, persistent_workers=data_handling_cfg.get("num_workers", 0) > 0, prefetch_factor=cfg.get("PREFETCH_FACTOR", 4))

    gpu_augmentation_pipeline_train = None
    if gpu_augmentations_cfg.get("enable", False):
        gpu_augmentation_pipeline_train = build_gpu_transform_pipeline(gpu_augmentations_cfg, device)
        if gpu_augmentation_pipeline_train: logger.info("GPU augmentation (from gpu_augmentations.pipeline) enabled.")

    if data_handling_cfg.get("num_workers", 0) > 0 and len(val_ld) > 0:
        try: _ = next(iter(val_ld)); logger.debug("Val DataLoader pre-warmed.")
        except StopIteration: logger.debug("Val DataLoader pre-warming StopIteration.")

    model_factory_cfg = model_yaml_cfg.copy(); model_factory_cfg["numClasses"] = len(label2idx)
    if "type" in model_factory_cfg: model_factory_cfg["MODEL_TYPE"] = model_factory_cfg.pop("type")
    model = get_model(model_factory_cfg).to(device)
    model_name_for_log = model_yaml_cfg.get("type", 'UnknownModel')
    logger.info(f"Model '{model_name_for_log}' loaded on device {device}.")

    if torch_compile_cfg.get("enable", False):
        if hasattr(torch, "compile") and sys.version_info >= (3, 8):
            logger.info(f"Attempting torch.compile() for model '{model_name_for_log}'...")
            compile_options = {"mode": torch_compile_cfg.get("mode", "default"), 
                               "fullgraph": torch_compile_cfg.get("fullgraph", False), 
                               "dynamic": torch_compile_cfg.get("dynamic", False)}
            if "backend" in torch_compile_cfg and torch_compile_cfg["backend"]: compile_options["backend"] = torch_compile_cfg["backend"]
            compile_options = {k: v for k, v in compile_options.items() if v is not None}
            try:
                cs_time = time.time(); model = torch.compile(model, **compile_options); cd_time = time.time() - cs_time
                logger.info(f"torch.compile() applied to '{model_name_for_log}' in {cd_time:.2f}s. Options: {compile_options}")
            except Exception as e: logger.error(f"torch.compile() failed for '{model_name_for_log}': {e}. Proceeding without.", exc_info=True)
        else: logger.warning(f"torch.compile() enabled, but unavailable. PyTorch: {torch.__version__}, Python: {sys.version.split()[0]}")

    ema_model = None
    if training_loop_cfg.get("ema_decay", 0) > 0:
        ema_model = copy.deepcopy(model).to(device); [p.requires_grad_(False) for p in ema_model.parameters()]; ema_decay = training_loop_cfg.get("ema_decay")
        logger.info(f"EMA enabled with decay {ema_decay}.")

    backbone_params, head_params = [], []
    if hasattr(model, 'named_parameters'):
        for name, param in model.named_parameters():
            is_head = False; model_type_str_from_cfg = model_yaml_cfg.get("type", "").lower()
            if model_type_str_from_cfg.startswith("dino"):
                if "classifier" in name: is_head = True
            elif model_type_str_from_cfg.startswith("deeplab"):
                if "net.classifier" in name: is_head = True
            else:
                classifier_name_str = ""; cfm_obj = None; orig_model_obj = getattr(model, '_orig_mod', model)
                if hasattr(orig_model_obj, 'get_classifier') and callable(orig_model_obj.get_classifier): cfm_obj = orig_model_obj.get_classifier()
                if cfm_obj and hasattr(cfm_obj, '_get_name'): classifier_name_str = cfm_obj._get_name().lower()
                head_kw_list = ["head", "fc", "classifier", "top"]; current_search_kw_list = list(head_kw_list)
                if classifier_name_str: current_search_kw_list.append(classifier_name_str)
                current_search_kw_list = list(set(s for s in current_search_kw_list if s))
                if any(kw in name.lower() for kw in current_search_kw_list): is_head = True
            if is_head: head_params.append(param)
            else: backbone_params.append(param)
    else: logger.warning("Model has no named_parameters. Treating all parameters as head parameters."); head_params = list(model.parameters())
    
    freeze_epochs_val = training_loop_cfg.get("freeze_epochs", 0)
    if freeze_epochs_val > 0 and backbone_params: logger.info(f"Freezing {len(backbone_params)} backbone params for {freeze_epochs_val} epochs."); [p.requires_grad_(False) for p in backbone_params]
    elif not backbone_params and freeze_epochs_val > 0: logger.warning("FREEZE_EPOCHS > 0 but no backbone params identified for freezing.")

    optimizer_params = [{'params': head_params, 'lr': optimizer_yaml_cfg.get("lr")}]
    if backbone_params: optimizer_params.append({'params': backbone_params, 'lr': optimizer_yaml_cfg.get("lr") * training_loop_cfg.get("backbone_lr_mult", 0.1)})
    if optimizer_yaml_cfg.get("type", "AdamW").lower() == "adamw": optimizer = AdamW(optimizer_params, weight_decay=optimizer_yaml_cfg.get("weight_decay"))
    else: raise ValueError(f"Unsupported optimizer type: {optimizer_yaml_cfg.get('type')}")
    scheduler_type_str = scheduler_yaml_cfg.get("type", "StepLR").lower()
    if scheduler_type_str == "steplr": scheduler = StepLR(optimizer, step_size=scheduler_yaml_cfg.get("step_size"), gamma=scheduler_yaml_cfg.get("gamma"))
    elif scheduler_type_str == "cosineannealinglr": scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_yaml_cfg.get("t_max", training_loop_cfg.get("num_epochs")), eta_min=scheduler_yaml_cfg.get("min_lr", 0.0))
    else: raise ValueError(f"Unsupported scheduler type: {scheduler_type_str}")
    
    scaler = GradScaler(enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True)))
    accum_steps = training_loop_cfg.get("accum_steps", 1)
    
    model_selection_metric_name = training_loop_cfg.get("model_selection_metric", "macro_auc").lower()
    best_metric_val = 0.0 
    if model_selection_metric_name not in ["macro_auc", "mean_optimal_f1", "mean_optimal_sensitivity"]:
        logger.warning(f"Unsupported model_selection_metric: '{model_selection_metric_name}'. Defaulting to 'macro_auc'.")
        model_selection_metric_name = "macro_auc"
    logger.info(f"Primary metric for model selection and early stopping: {model_selection_metric_name}")
    patience_counter = 0
    
    num_classes = cfg.get("numClasses")
    class_counts_train_fold = get_class_counts(train_df, label2idx)
    criterion = None; loss_type = loss_cfg.get("type", "cross_entropy").lower()
    initial_loss_weights = None

    if loss_type == "focal_ce_loss": criterion = focal_ce_loss; logger.info("Using Focal CE Loss.")
    elif loss_type == "cross_entropy": criterion = nn.CrossEntropyLoss(weight=initial_loss_weights); logger.info(f"Using standard CE Loss (DRW active: {bool(training_loop_cfg.get('drw_schedule_epochs'))}).")
    elif loss_type == "weighted_cross_entropy":
        k_factor = loss_cfg.get("wce_k_factor", 1.0)
        if k_factor > 0:
            total_samples_N = np.sum(class_counts_train_fold)
            safe_counts = np.maximum(class_counts_train_fold, 1e-8)
            weights_ni = (total_samples_N / safe_counts) ** k_factor
            initial_loss_weights = torch.tensor(weights_ni, dtype=torch.float, device=device)
            logger.info(f"Using Weighted CE Loss with k={k_factor}. Initial weights (first 5): {initial_loss_weights[:5]}")
        else: logger.info("WCE Loss selected, but k_factor <= 0. Using unweighted CE.")
        criterion = nn.CrossEntropyLoss(weight=initial_loss_weights)
    elif loss_type == "ldam_loss":
        if class_counts_train_fold is None or len(class_counts_train_fold) != num_classes: raise ValueError("Class counts required for LDAM loss and seem incorrect.")
        criterion = LDAMLoss(class_counts=class_counts_train_fold, max_margin=loss_cfg.get("ldam_max_margin",0.5),
                             use_effective_number_margin=loss_cfg.get("ldam_use_effective_number_margin",True),
                             effective_number_beta=loss_cfg.get("ldam_effective_number_beta",0.999), scale=30.0, weight=initial_loss_weights).to(device)
        logger.info(f"Using LDAM Loss (DRW active: {bool(training_loop_cfg.get('drw_schedule_epochs'))}).")
    else: raise ValueError(f"Unsupported loss type: {loss_type}")
    
    drw_schedule_epochs_list = training_loop_cfg.get("drw_schedule_epochs", []); current_drw_stage = 0

    logger.info(f"Starting training loop for {training_loop_cfg.get('num_epochs')} epochs.")
    for epoch in range(training_loop_cfg.get("num_epochs")):
        if drw_schedule_epochs_list:
            new_stage_candidate = -1
            for stage_idx, start_epoch_drw in enumerate(drw_schedule_epochs_list):
                if epoch >= start_epoch_drw: new_stage_candidate = stage_idx + 1
            if new_stage_candidate != -1 and new_stage_candidate != current_drw_stage:
                current_drw_stage = new_stage_candidate; logger.info(f"Epoch {epoch}: Entering DRW Stage {current_drw_stage}.")
                eff_num_beta_drw = loss_cfg.get("ldam_effective_number_beta",0.999)
                safe_class_counts = np.maximum(class_counts_train_fold, 1)
                eff_num = 1.0 - np.power(eff_num_beta_drw, safe_class_counts)
                per_cls_weights_drw = (1.0 - eff_num_beta_drw) / eff_num
                per_cls_weights_drw = per_cls_weights_drw / np.sum(per_cls_weights_drw) * num_classes
                per_cls_weights_drw_tensor = torch.tensor(per_cls_weights_drw, dtype=torch.float, device=device)
                if isinstance(criterion, LDAMLoss): criterion.update_weights(per_cls_weights_drw_tensor)
                elif isinstance(criterion, nn.CrossEntropyLoss): criterion = nn.CrossEntropyLoss(weight=per_cls_weights_drw_tensor).to(device)
                logger.info(f"DRW Stage {current_drw_stage}: Updated loss with class-balanced weights.")

        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(epoch)
        current_epoch_profiler = tb_logger.setup_profiler(epoch, fold_specific_tb_log_dir)
        if epoch == freeze_epochs_val and freeze_epochs_val > 0 and backbone_params: logger.info(f"E{epoch}: Unfreezing {len(backbone_params)} params."); [p.requires_grad_(True) for p in backbone_params]
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train();
        if gpu_augmentation_pipeline_train: gpu_augmentation_pipeline_train.train()
        
        # Initialize cumulative metrics for the epoch
        cumulative_train_loss_for_pbar = 0.0
        cumulative_train_corrects_for_pbar = 0
        cumulative_train_samples_for_pbar = 0
        
        epoch_gpu_time_ms = 0.0; optimizer.zero_grad(); epoch_start_time = time.time()
        train_pbar = tqdm(train_ld, desc=f"Fold {fold} E{epoch} Train", ncols=experiment_setup_cfg.get("TQDM_NCOLS", 100))

        for batch_idx, (imgs_cpu, labels_cpu) in enumerate(train_pbar):
            imgs_device = imgs_cpu.to(device, non_blocking=True); labels_device = labels_cpu.to(device, non_blocking=True)
            if gpu_augmentation_pipeline_train:
                with torch.set_grad_enabled(True): imgs_device = gpu_augmentation_pipeline_train(imgs_device)
            
            batch_gpu_time_ms = 0.0
            with CudaTimer(device) as timer:
                with autocast(device_type=device.type, enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))):
                    logits = model(imgs_device)
                    if loss_type == "focal_ce_loss": loss = criterion(logits.float(), labels_device, alpha=loss_cfg.get("focal_alpha",1.0), gamma=loss_cfg.get("focal_gamma",2.0))
                    else: loss = criterion(logits.float(), labels_device)
                    if accum_steps > 1: loss = loss / accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_ld):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if ema_model is not None: update_ema(ema_model, model, ema_decay)
            
            batch_gpu_time_ms = timer.get_elapsed_time_ms(); epoch_gpu_time_ms += batch_gpu_time_ms
            
            batch_loss_val = loss.item() * (accum_steps if accum_steps > 1 else 1)
            preds = logits.argmax(dim=1)
            batch_corrects = (preds == labels_device).float().sum().item()
            batch_samples = imgs_device.size(0)

            cumulative_train_loss_for_pbar += batch_loss_val * batch_samples # Weighted sum for epoch avg
            cumulative_train_corrects_for_pbar += batch_corrects
            cumulative_train_samples_for_pbar += batch_samples
            
            # Update progress bar with cumulative epoch averages
            avg_epoch_loss_pbar = cumulative_train_loss_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            avg_epoch_acc_pbar = cumulative_train_corrects_for_pbar / cumulative_train_samples_for_pbar if cumulative_train_samples_for_pbar > 0 else 0.0
            train_pbar.set_postfix(loss=f"{avg_epoch_loss_pbar:.4f}", acc=f"{avg_epoch_acc_pbar:.4f}")
            
            # Log batch metrics (can still be current batch or cumulative, your choice)
            # For tb_logger, let's log current batch loss/acc for finer grain if interval is small
            current_batch_acc_for_tb = batch_corrects / batch_samples if batch_samples > 0 else 0.0
            tb_logger.log_train_batch_metrics(loss=batch_loss_val, acc=current_batch_acc_for_tb, 
                                              lr=optimizer.param_groups[0]['lr'], epoch=epoch, batch_idx=batch_idx,
                                              imgs=(imgs_device.detach() if batch_idx==0 and gpu_augmentation_pipeline_train else imgs_cpu if batch_idx==0 else None))
            if current_epoch_profiler: tb_logger.step_profiler()

        epoch_duration = time.time() - epoch_start_time
        # Final epoch averages (already calculated for pbar)
        avg_train_loss = avg_epoch_loss_pbar
        avg_train_acc = avg_epoch_acc_pbar
        train_pbar.set_postfix(avg_loss=f"{avg_train_loss:.4f}", avg_acc=f"{avg_train_acc:.4f}"); train_pbar.close(); scheduler.step()
        
        epoch_metrics = {"Loss/train_epoch": avg_train_loss, "Accuracy/train_epoch": avg_train_acc, "LearningRate/epoch": optimizer.param_groups[0]['lr'],
                         "Time/Train_epoch_duration_sec": epoch_duration, "Throughput/train_samples_per_sec": cumulative_train_samples_for_pbar / epoch_duration if epoch_duration > 0 else 0}
        if device.type == 'cuda': epoch_metrics["Time/GPU_ms_per_train_epoch"] = epoch_gpu_time_ms
        if current_epoch_profiler: tb_logger.stop_and_process_profiler(); current_epoch_profiler = None

        if epoch % training_loop_cfg.get("val_interval", 1) == 0 or epoch == training_loop_cfg.get("num_epochs") - 1:
            eval_model = ema_model if ema_model is not None and training_loop_cfg.get("use_ema_for_val", True) else model
            eval_model.eval();
            if gpu_augmentation_pipeline_train: gpu_augmentation_pipeline_train.eval()
            logger.info(f"Validation E{epoch} using {'EMA' if eval_model is ema_model else 'primary'} model.")
            val_loss_sum, val_acc_sum, val_seen_samples = 0.0, 0.0, 0; all_probs_val, all_labels_val = [], []
            val_pbar = tqdm(val_ld, desc=f"Fold {fold} E{epoch} Val  ", ncols=experiment_setup_cfg.get("TQDM_NCOLS", 100))
            with torch.no_grad():
                for batch_idx_val, (imgs_val_cpu, labels_val_cpu) in enumerate(val_pbar):
                    imgs_val_dev = imgs_val_cpu.to(device, non_blocking=True); labels_val_dev = labels_val_cpu.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))):
                        logits_val = eval_model(imgs_val_dev)
                        if loss_type == "focal_ce_loss": loss_v = criterion(logits_val.float(), labels_val_dev,alpha=loss_cfg.get("focal_alpha",1.0),gamma=loss_cfg.get("focal_gamma",2.0))
                        else: loss_v = criterion(logits_val.float(), labels_val_dev)
                    current_val_loss = loss_v.item(); current_val_correct = (logits_val.argmax(dim=1) == labels_val_dev).float().sum().item()
                    current_val_acc = current_val_correct / imgs_val_dev.size(0) if imgs_val_dev.size(0) > 0 else 0.0
                    val_loss_sum += current_val_loss * imgs_val_dev.size(0); val_acc_sum += current_val_correct; val_seen_samples += imgs_val_dev.size(0)
                    all_probs_val.append(logits_val.softmax(dim=1).cpu()); all_labels_val.append(labels_val_dev.cpu())
                    val_pbar.set_postfix(avg_loss=f"{val_loss_sum / val_seen_samples:.4f}", avg_acc=f"{val_acc_sum / val_seen_samples:.4f}") # Cumulative for val pbar
                    tb_logger.log_val_batch_metrics(loss=current_val_loss, acc=current_val_acc, epoch=epoch, batch_idx=batch_idx_val,
                                                    imgs=(imgs_val_cpu if batch_idx_val == 0 else None) )
            all_probs_val_cat = torch.cat(all_probs_val); all_labels_val_cat = torch.cat(all_labels_val)
            avg_val_loss = val_loss_sum / val_seen_samples if val_seen_samples > 0 else 0; avg_val_acc = val_acc_sum / val_seen_samples if val_seen_samples > 0 else 0
            val_pbar.set_postfix(avg_loss=f"{avg_val_loss:.4f}", avg_acc=f"{avg_val_acc:.4f}"); val_pbar.close()
            epoch_metrics["Loss/val_epoch"] = avg_val_loss; epoch_metrics["Accuracy/val_epoch"] = avg_val_acc
            
            auc_metric_fn = AUROC(task="multiclass", num_classes=num_classes, average="macro"); pauc_max_fpr = training_loop_cfg.get("pauc_max_fpr", 0.2)
            pauc_metric_fn = AUROC(task="multiclass", num_classes=num_classes, average="macro", max_fpr=pauc_max_fpr)
            f1_metric_fn_default = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            current_macro_auc = auc_metric_fn(all_probs_val_cat, all_labels_val_cat).item()
            current_pauc = pauc_metric_fn(all_probs_val_cat, all_labels_val_cat).item()
            current_f1_macro_default = f1_metric_fn_default(all_probs_val_cat, all_labels_val_cat).item()
            epoch_metrics["AUROC/val_macro"] = current_macro_auc; epoch_metrics[f"pAUROC{int(pauc_max_fpr*100)}/val_macro"] = current_pauc
            epoch_metrics["F1Score/val_macro_default_thresh"] = current_f1_macro_default
            
            optimal_thresholds_for_ckpt = {}
            if num_classes > 1:
                labels_oh_val = torch.nn.functional.one_hot(all_labels_val_cat, num_classes).numpy(); probs_np_val = all_probs_val_cat.numpy()
                try:
                    ap_metric_fn = AveragePrecision(task="multiclass", num_classes=num_classes, average=None)
                    ap_vals = ap_metric_fn(all_probs_val_cat, all_labels_val_cat)
                    for i, ap in enumerate(ap_vals): epoch_metrics[f"PRAUC/val_class_{i}"] = ap.item()
                except Exception as e_ap: logger.warning(f"AP calc failed E{epoch}: {e_ap}")
                
                current_optimal_f1_scores = []; current_optimal_sensitivities = []
                for i in range(num_classes):
                    optimal_f1_class, optimal_threshold_for_f1_class, optimal_sensitivity_class = 0.0, 0.5, 0.0
                    try:
                        prec, rec, thr = precision_recall_curve(labels_oh_val[:, i], probs_np_val[:, i])
                        if len(prec) > 1 and len(rec) > 1 and len(thr) > 0:
                            f1_scores_curve = (2 * prec * rec) / (prec + rec + 1e-8)
                            relevant_f1s = f1_scores_curve[1:]; relevant_recalls = rec[1:]
                            valid_indices_f1 = np.where(np.isfinite(relevant_f1s))[0]
                            if len(valid_indices_f1) > 0:
                                best_idx_in_relevant_f1 = np.argmax(relevant_f1s[valid_indices_f1])
                                best_f1_val = relevant_f1s[valid_indices_f1[best_idx_in_relevant_f1]]
                                best_thr_idx_f1 = valid_indices_f1[best_idx_in_relevant_f1]
                                optimal_f1_class = float(best_f1_val)
                                optimal_threshold_for_f1_class = float(thr[best_thr_idx_f1])
                            
                            valid_indices_recall = np.where(np.isfinite(relevant_recalls))[0]
                            if len(valid_indices_recall) > 0:
                                best_idx_in_relevant_recall = np.argmax(relevant_recalls[valid_indices_recall])
                                best_recall_val = relevant_recalls[valid_indices_recall[best_idx_in_relevant_recall]]
                                optimal_sensitivity_class = float(best_recall_val)
                    except Exception as e_pr: logger.warning(f"PR curve fail C{i} E{epoch}: {e_pr}")
                    epoch_metrics[f"F1Score_optimal/val_class_{i}"] = optimal_f1_class
                    epoch_metrics[f"Threshold_optimal_F1/val_class_{i}"] = optimal_threshold_for_f1_class
                    epoch_metrics[f"Sensitivity_optimal/val_class_{i}"] = optimal_sensitivity_class
                    current_optimal_f1_scores.append(optimal_f1_class)
                    current_optimal_sensitivities.append(optimal_sensitivity_class)
                    optimal_thresholds_for_ckpt[i] = optimal_threshold_for_f1_class
                
                if current_optimal_f1_scores: epoch_metrics["F1Score/val_mean_optimal_per_class"] = np.mean(current_optimal_f1_scores)
                if current_optimal_sensitivities: epoch_metrics["Sensitivity/val_mean_optimal_per_class"] = np.mean(current_optimal_sensitivities)
            
            current_primary_metric_val = 0.0
            if model_selection_metric_name == "macro_auc": current_primary_metric_val = current_macro_auc
            elif model_selection_metric_name == "mean_optimal_f1": current_primary_metric_val = epoch_metrics.get("F1Score/val_mean_optimal_per_class", 0.0)
            elif model_selection_metric_name == "mean_optimal_sensitivity": current_primary_metric_val = epoch_metrics.get("Sensitivity/val_mean_optimal_per_class", 0.0)

            logger.info(f"Fold{fold} E{epoch} Val -> Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f} SelectedMetric ({model_selection_metric_name})={current_primary_metric_val:.4f}")
            
            if current_primary_metric_val > best_metric_val:
                best_metric_val = current_primary_metric_val; patience_counter=0; ckpt_path=run_ckpt_dir/f"{exp_name_for_files}_fold{fold}_best.pt"
                model_sd_to_save = getattr(model, '_orig_mod', model).state_dict(); ema_model_sd_to_save = getattr(ema_model, '_orig_mod', ema_model).state_dict() if ema_model else None
                checkpoint_data = {'epoch':epoch,'model_state_dict': model_sd_to_save, 'ema_model_state_dict': ema_model_sd_to_save,
                                   'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                                   'scaler_state_dict':scaler.state_dict(), f'best_{model_selection_metric_name}': best_metric_val,
                                   'config_runtime':cfg,'label2idx':label2idx}
                if training_loop_cfg.get("save_optimal_thresholds", False) and model_selection_metric_name in ["mean_optimal_f1", "mean_optimal_sensitivity"] and optimal_thresholds_for_ckpt:
                    checkpoint_data['optimal_thresholds_val'] = optimal_thresholds_for_ckpt; logger.info(f"Saving optimal thresholds with ckpt.")
                torch.save(checkpoint_data, str(ckpt_path))
                logger.info(f"Saved best model to {ckpt_path} based on {model_selection_metric_name}: {best_metric_val:.4f}")
            else:
                patience_counter+=1
                if patience_counter >= training_loop_cfg.get("early_stopping_patience",10): logger.info(f"Early stopping E{epoch} F{fold}."); break
        
        tb_logger.log_epoch_summary(epoch_metrics, epoch)
        tb_logger.flush()
        if epoch == training_loop_cfg.get("num_epochs") -1 :
            last_ckpt_path=run_ckpt_dir/f"{exp_name_for_files}_fold{fold}_last.pt"
            model_sd_to_save = getattr(model, '_orig_mod', model).state_dict(); ema_model_sd_to_save = getattr(ema_model, '_orig_mod', ema_model).state_dict() if ema_model else None
            torch.save({'epoch':epoch,'model_state_dict':model_sd_to_save,'ema_model_state_dict':ema_model_sd_to_save,
                        'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),
                        'current_primary_metric':current_primary_metric_val if 'current_primary_metric_val' in locals() else -1.,
                        'config_runtime':cfg,'label2idx':label2idx}, str(last_ckpt_path))
            logger.info(f"Saved last model checkpoint to {last_ckpt_path}")

    tb_logger.close()
    logger.info(f"Finished training fold {fold}. Best {model_selection_metric_name}: {best_metric_val:.4f}")
    return float(best_metric_val)

def main():
    ap = argparse.ArgumentParser(description="Train models with CV using YAML config.")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config if set.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-[%(name)s]-%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.info(f"Starting experiment: {args.exp_name} with CLI args: {args}")
    
    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
    if not cfg_path.exists():
        fallback_cfg_path = Path(args.config_dir) / "base_experiment.yaml"
        if not args.config_file and fallback_cfg_path.exists():
            logger.warning(f"Config {cfg_path} not found. Using fallback {fallback_cfg_path}"); cfg_path = fallback_cfg_path
        else: raise FileNotFoundError(f"Config file {cfg_path} (and fallback if applicable) not found.")
    cfg = load_config(cfg_path); logger.info(f"Loaded config from {cfg_path}"); cfg = cast_config_values(cfg)

    exp_setup_cfg = cfg.get("experiment_setup", {})
    current_seed = args.seed if args.seed is not None else exp_setup_cfg.get("seed", 42)
    set_seed(current_seed); logger.info(f"Seed set to {current_seed}")
    exp_setup_cfg["seed"] = current_seed; cfg["experiment_setup"] = exp_setup_cfg
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg = cfg.get("paths", {})
    config_proj_root = paths_cfg.get("project_root")
    base_path_for_config_paths = Path(config_proj_root).resolve() if config_proj_root else cfg_path.parent
    logger.info(f"Base path for resolving relative paths in config: {base_path_for_config_paths}")
    base_log_dir_from_cfg = _get_path_from_config(cfg, "log_dir", base_path=base_path_for_config_paths)
    base_ckpt_dir_from_cfg = _get_path_from_config(cfg, "ckpt_dir", base_path=base_path_for_config_paths)
    run_specific_log_dir = base_log_dir_from_cfg / args.exp_name / timestamp
    run_specific_ckpt_dir = base_ckpt_dir_from_cfg / args.exp_name / timestamp
    run_specific_log_dir.mkdir(parents=True, exist_ok=True); run_specific_ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run-specific log directory: {run_specific_log_dir}"); logger.info(f"Run-specific checkpoint directory: {run_specific_ckpt_dir}")

    labels_csv_p = _get_path_from_config(cfg, "labels_csv", base_path=base_path_for_config_paths)
    train_root_p = _get_path_from_config(cfg, "train_root", base_path=base_path_for_config_paths)
    logger.info(f"Paths: Labels={labels_csv_p}, TrainRoot={train_root_p}")

    df = pd.read_csv(labels_csv_p)
    required_cols = ['fold', 'label', 'dataset', 'filename']; missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols: raise ValueError(f"Missing required columns in {labels_csv_p}: {missing_cols}")
    
    labels_unique = sorted(df['label'].unique()); label2idx = {name: i for i, name in enumerate(labels_unique)}
    num_classes_val = len(labels_unique)
    model_cfg_section = cfg.get("model", {}); model_cfg_section["numClasses"] = num_classes_val; cfg["model"] = model_cfg_section
    cfg["numClasses"] = num_classes_val; cfg['label2idx'] = label2idx
    logger.info(f"Classes: {cfg['numClasses']}, Map: {label2idx}")
    
    device_str = exp_setup_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(device_str); logger.info(f"Using device: {device}")
    exp_setup_cfg["device"] = str(device); cfg["experiment_setup"] = exp_setup_cfg

    fold_primary_metric_results = []
    folds = sorted(df['fold'].unique())
    if not folds: raise ValueError("No folds found in 'fold' column of labels_csv.")
    logger.info(f"Found folds: {folds}")

    for fold_num in folds:
        logger.info(f"\n===== Processing Fold {fold_num} / {folds[-1]} =====")
        train_df_fold = df[df['fold'] != fold_num].reset_index(drop=True)
        val_df_fold = df[df['fold'] == fold_num].reset_index(drop=True)
        if train_df_fold.empty or val_df_fold.empty:
            logger.warning(f"Fold {fold_num} has empty train/val set. Skipping."); fold_primary_metric_results.append(0.0); continue
        logger.info(f"Fold {fold_num}: Train samples={len(train_df_fold)}, Val samples={len(val_df_fold)}")
        
        best_fold_metric_val = train_one_fold(
            fold_num, train_df_fold, val_df_fold, cfg, label2idx, train_root_path=train_root_p,
            run_log_dir=run_specific_log_dir, run_ckpt_dir=run_specific_ckpt_dir,
            exp_name_for_files=args.exp_name, device=device
        )
        fold_primary_metric_results.append(best_fold_metric_val)
        if device.type == 'cuda': torch.cuda.empty_cache()

    chosen_metric_name_summary = cfg.get("training",{}).get("model_selection_metric", "macro_auc").lower()
    if any(m_val != 0 or not np.isnan(m_val) for m_val in fold_primary_metric_results):
        successful_metrics = [m_val for m_val in fold_primary_metric_results if not np.isnan(m_val)]
        if successful_metrics:
            mean_metric = float(np.mean(successful_metrics))
            std_metric = float(np.std(successful_metrics, ddof=1)) if len(successful_metrics) > 1 else 0.0
            logger.info(f"\n===== {len(folds)}-Fold CV Results (Primary Metric: {chosen_metric_name_summary}) =====")
            for i, metric_val in enumerate(fold_primary_metric_results): logger.info(f"Fold {folds[i]} Best {chosen_metric_name_summary}: {metric_val:.4f}")
            logger.info(f"Average Best {chosen_metric_name_summary} (over {len(successful_metrics)} folds with valid primary metric) = {mean_metric:.4f} ± {std_metric:.4f}")
        else: logger.warning("No folds yielded valid primary metric values for averaging.")
    else: logger.warning("No folds were successfully trained or processed to yield primary metric values.")
    logger.info(f"Experiment {args.exp_name} (timestamp: {timestamp}) finished.")

if __name__ == "__main__":
    main()
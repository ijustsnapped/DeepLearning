#!/usr/bin/env python
# your_project_name/train_single_fold_with_meta_fully_updated.py
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
import contextlib

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler

try:
    from tqdm.notebook import tqdm as tqdm_notebook
    if 'IPython' in sys.modules and hasattr(sys.modules['IPython'], 'core') and hasattr(sys.modules['IPython'].core, 'getipython') and sys.modules['IPython.core.getipython'].get_ipython() is not None: # type: ignore
            tqdm_iterator = tqdm_notebook
    else:
        from tqdm import tqdm as tqdm_cli
        tqdm_iterator = tqdm_cli
except ImportError:
    from tqdm import tqdm as tqdm_cli
    tqdm_iterator = tqdm_cli

from torchmetrics import AUROC, F1Score, AveragePrecision
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_handling import (
    FlatDatasetWithMeta, build_transform, build_gpu_transform_pipeline,
    ClassBalancedSampler
)
from models import get_model as get_base_cnn_model, CNNWithMetadata
from losses import focal_ce_loss, LDAMLoss
from utils import (
    set_seed, load_config, cast_config_values,
    update_ema,
    get_device, CudaTimer, reset_cuda_peak_memory_stats, empty_cuda_cache,
    TensorBoardLogger
)

try:
    from torch.profiler import ProfilerActivity # type: ignore
except ImportError:
    ProfilerActivity = None # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__)

if 'LDAMLoss' not in globals(): # Placeholder
    class LDAMLoss(nn.Module): # type: ignore
        def __init__(self, *args, **kwargs): raise NotImplementedError("LDAMLoss not implemented/imported")
        def update_weights(self, *args, **kwargs): pass


def _get_path_from_config(cfg: dict, key: str, default: str | None = None, base_path: Path | None = None) -> Path:
    paths_cfg = cfg.get("paths", {})
    path_str = paths_cfg.get(key)
    if path_str is None:
        if default is not None: path_str = default; logger.warning(f"Path for '{key}' not in config. Using default: '{default}'")
        else: logger.error(f"Required path for '{key}' not found. Config paths: {paths_cfg}"); raise ValueError(f"Missing path for '{key}'")
    path = Path(path_str)
    if base_path and not path.is_absolute(): path = base_path / path
    return path.resolve()

def get_class_counts(df: pd.DataFrame, label2idx: dict[str, int]) -> np.ndarray:
    num_classes = len(label2idx)
    counts = np.zeros(num_classes, dtype=int)
    valid_labels_mask = df['label'].isin(label2idx.keys())
    if not valid_labels_mask.all():
        unmapped_in_df = df.loc[~valid_labels_mask, 'label'].unique()
        logger.warning(f"get_class_counts: DataFrame has labels not in label2idx: {unmapped_in_df}. Ignored.")
    mapped_indices = df['label'].map(lambda x: label2idx.get(x, -1))
    valid_mapped_indices = mapped_indices[mapped_indices != -1]
    if valid_mapped_indices.empty and not df.empty: logger.warning("get_class_counts: No labels mapped. Zero counts."); return counts
    elif valid_mapped_indices.empty and df.empty: return counts
    class_series = valid_mapped_indices.astype(int).value_counts()
    for class_idx, count_val in class_series.items():
        if 0 <= class_idx < num_classes: counts[class_idx] = count_val
        else: logger.warning(f"Out-of-bounds mapped class index {class_idx}. Count ({count_val}) ignored.")
    return counts

def generate_confusion_matrix_figure(true_labels: np.ndarray, pred_labels: np.ndarray, display_labels: list[str], title: str):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig_s_base, fig_s_factor = 8, 0.6
    fig_w = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig_h = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d'); ax.set_title(title); plt.tight_layout()
    return fig


# --- CORE TRAINING AND VALIDATION FUNCTION FOR A SINGLE PHASE ---
def run_training_phase(
    phase_name: str, model: CNNWithMetadata,
    train_ld: DataLoader, val_ld: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler,
    device: torch.device, cfg: dict, phase_cfg: dict, fold_run_id_str: str,
    label2idx_model_map_phase: dict[str, int], label2idx_eval_map_phase: dict[str, int],
    tb_logger: TensorBoardLogger, run_ckpt_dir_phase: Path, exp_name_for_files_phase: str,
    start_epoch: int = 0, ema_model_phase: CNNWithMetadata | None = None,
    initial_best_metric_val_phase: float = -float('inf')
) -> tuple[float, int, str, dict | None]:

    main_training_cfg = cfg.get("training", {})
    amp_enabled_phase = main_training_cfg.get("amp_enabled", True)
    ema_decay_phase = main_training_cfg.get("ema_decay", 0.0)
    use_ema_for_val_phase = main_training_cfg.get("use_ema_for_val", True)
    model_selection_metric_name_phase = main_training_cfg.get("model_selection_metric", "macro_auc").lower()
    unk_label_string_cfg_phase = main_training_cfg.get("unk_label_string", "UNK")
    post_hoc_unk_thresh_cfg_phase = main_training_cfg.get("post_hoc_unk_threshold")
    val_metrics_heavy_interval_phase = main_training_cfg.get("val_metrics_heavy_interval", 99999)

    num_epochs_this_phase = phase_cfg.get("num_epochs", phase_cfg.get("epochs", 10))
    accum_steps_phase = phase_cfg.get("accum_steps", main_training_cfg.get("accum_steps", 1))
    early_stopping_patience_phase = phase_cfg.get("early_stopping_patience", main_training_cfg.get("early_stopping_patience", 10))


    known_model_indices_phase = [idx for lbl, idx in label2idx_model_map_phase.items() if lbl != unk_label_string_cfg_phase and idx != label2idx_model_map_phase.get(unk_label_string_cfg_phase, -1)]
    unk_eval_assign_idx_phase = label2idx_eval_map_phase.get(unk_label_string_cfg_phase, -1)
    idx2label_model_phase = {v: k for k, v in label2idx_model_map_phase.items()}
    idx2label_eval_phase = {v: k for k, v in label2idx_eval_map_phase.items()}

    post_hoc_active_this_phase = (post_hoc_unk_thresh_cfg_phase is not None and
                                  bool(known_model_indices_phase) and
                                  unk_eval_assign_idx_phase != -1)
    if post_hoc_unk_thresh_cfg_phase is not None:
        logger.info(f"{phase_name} F{fold_run_id_str}: Post-hoc '{unk_label_string_cfg_phase}' (Th={post_hoc_unk_thresh_cfg_phase}). Active: {post_hoc_active_this_phase}")

    best_metric_val_current_phase = initial_best_metric_val_phase
    best_epoch_for_metric_current_phase = -1
    best_ckpt_path_current_phase = ""
    optimal_thresholds_best_ckpt_current_phase = None
    patience_counter_current_phase = 0

    num_model_classes_this_phase = len(label2idx_model_map_phase)
    num_eval_classes_this_phase = len(label2idx_eval_map_phase)
    
    # Loss specific args (e.g. for FocalCE)
    # Use loss config from the main training block, phase_cfg might not have it if it's shared
    loss_params_from_main_cfg = main_training_cfg.get("loss",{})
    loss_type_for_criterion = loss_params_from_main_cfg.get("type","cross_entropy").lower()
    focal_alpha_cfg = loss_params_from_main_cfg.get("focal_alpha",1.0)
    focal_gamma_cfg = loss_params_from_main_cfg.get("focal_gamma",2.0)

    epoch_global_phase = start_epoch # Initialize for potential early exit if num_epochs_this_phase is 0

    logger.info(f"Starting {phase_name} F{fold_run_id_str}: {num_epochs_this_phase} epochs (Global Epochs {start_epoch}-{start_epoch + num_epochs_this_phase -1}).")
    for epoch_offset_phase in range(num_epochs_this_phase):
        epoch_global_phase = start_epoch + epoch_offset_phase

        if hasattr(train_ld.sampler, 'set_epoch') and train_ld.sampler is not None: # type: ignore
            train_ld.sampler.set_epoch(epoch_global_phase) # type: ignore

        active_profiler_phase = tb_logger.setup_profiler(epoch_global_phase, Path(tb_logger.writer.log_dir if tb_logger.writer else run_ckpt_dir_phase)) # type: ignore

        model.train()
        # if gpu_aug_train: gpu_aug_train.train()

        epoch_gpu_time_ms_phase, epoch_start_time_phase = 0.0, time.time()
        train_loss_sum_pbar, train_corrects_sum_pbar, train_samples_sum_pbar = 0.0,0,0
        optimizer.zero_grad()

        train_pbar_desc_phase = f"F{fold_run_id_str} E{epoch_global_phase} {phase_name} Train"
        train_pbar_phase = tqdm_iterator(train_ld, desc=train_pbar_desc_phase, ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",120), leave=False)

        for batch_idx_train_ph, (inputs_train_ph, labels_train_cpu_ph) in enumerate(train_pbar_phase):
            img_train_cpu_ph, meta_train_cpu_ph = inputs_train_ph
            img_train_dev_ph = img_train_cpu_ph.to(device, non_blocking=True)
            meta_train_dev_ph = meta_train_cpu_ph.to(device, non_blocking=True)
            labels_train_dev_ph = labels_train_cpu_ph.to(device, non_blocking=True) # In label2idx_model_map_phase space

            batch_gpu_time_ph, timer_active_ph = 0.0, active_profiler_phase is not None or \
                (cfg.get("tensorboard_logging",{}).get("profiler",{}).get("enable_batch_timing_always",False) and \
                 tb_logger._should_log_batch(tb_logger.log_interval_train_batch, batch_idx_train_ph)) # type: ignore

            timer_ctx_ph = CudaTimer(device) if timer_active_ph and device.type=='cuda' else contextlib.nullcontext() # type: ignore
            with timer_ctx_ph as batch_timer_ph: # type: ignore
                with autocast(device_type=device.type, enabled=amp_enabled_phase):
                    logits_train_ph = model(img_train_dev_ph, meta_train_dev_ph)
                    if loss_type_for_criterion == "focal_ce_loss":
                         loss_train_ph = criterion(logits_train_ph.float(), labels_train_dev_ph, alpha=focal_alpha_cfg, gamma=focal_gamma_cfg)
                    else:
                         loss_train_ph = criterion(logits_train_ph.float(), labels_train_dev_ph)

                    if accum_steps_phase > 1: loss_train_ph = loss_train_ph / accum_steps_phase # type: ignore
                scaler.scale(loss_train_ph).backward()
                if (batch_idx_train_ph + 1) % accum_steps_phase == 0 or (batch_idx_train_ph + 1) == len(train_ld): # type: ignore
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if ema_model_phase is not None and ema_decay_phase > 0: update_ema(ema_model_phase, model, ema_decay_phase)

            if timer_active_ph and device.type=='cuda' and batch_timer_ph: batch_gpu_time_ph = batch_timer_ph.get_elapsed_time_ms() # type: ignore
            epoch_gpu_time_ms_phase += batch_gpu_time_ph

            batch_loss_val_ph = loss_train_ph.item() * (accum_steps_phase if accum_steps_phase > 1 else 1) # type: ignore
            preds_train_ph = logits_train_ph.argmax(dim=1)
            batch_corrects_ph = (preds_train_ph == labels_train_dev_ph).float().sum().item()
            batch_samples_ph = img_train_dev_ph.size(0)

            train_loss_sum_pbar += batch_loss_val_ph * batch_samples_ph
            train_corrects_sum_pbar += batch_corrects_ph
            train_samples_sum_pbar += batch_samples_ph
            avg_loss_pbar_ph = train_loss_sum_pbar / train_samples_sum_pbar if train_samples_sum_pbar > 0 else 0.0
            avg_acc_pbar_ph = train_corrects_sum_pbar / train_samples_sum_pbar if train_samples_sum_pbar > 0 else 0.0
            train_pbar_phase.set_postfix(loss=f"{avg_loss_pbar_ph:.4f}", acc=f"{avg_acc_pbar_ph:.4f}")

            tb_logger.log_train_batch_metrics(loss=batch_loss_val_ph,
                                              acc=(batch_corrects_ph / batch_samples_ph if batch_samples_ph > 0 else 0.0),
                                              lr=optimizer.param_groups[0]['lr'], epoch=epoch_global_phase, batch_idx=batch_idx_train_ph,
                                              batch_gpu_time_ms=batch_gpu_time_ph if timer_active_ph else None)
            if active_profiler_phase: active_profiler_phase.step()
        train_pbar_phase.close()

        epoch_duration_ph = time.time() - epoch_start_time_phase
        scheduler.step()

        current_epoch_metrics_combined_phase = {
            f"Loss/train_epoch_{phase_name}": avg_loss_pbar_ph,
            f"Accuracy/train_epoch_{phase_name}": avg_acc_pbar_ph,
            f"LearningRate/epoch_{phase_name}": optimizer.param_groups[0]['lr'],
            f"Time/Train_epoch_duration_sec_{phase_name}": epoch_duration_ph,
        }
        if device.type == 'cuda': current_epoch_metrics_combined_phase[f"Time/GPU_ms_per_train_epoch_{phase_name}"] = epoch_gpu_time_ms_phase
        if active_profiler_phase: tb_logger.stop_and_process_profiler(); active_profiler_phase = None # type: ignore

        val_interval_this_phase = phase_cfg.get("val_interval", main_training_cfg.get("val_interval",1))
        if epoch_global_phase % val_interval_this_phase == 0 or epoch_offset_phase == (num_epochs_this_phase -1) :
            eval_model_to_use_phase = ema_model_phase if ema_model_phase is not None and use_ema_for_val_phase else model
            eval_model_to_use_phase.eval()

            logger.info(f"Validation E{epoch_global_phase} ({phase_name}) using {'EMA' if eval_model_to_use_phase is ema_model_phase else 'primary'} model.")

            all_val_logits_list_ph, all_val_true_labels_eval_list_ph = [], []
            epoch_val_loss_sum_ph, epoch_val_seen_samples_ph, epoch_val_loss_samples_ph = 0.0, 0, 0
            epoch_val_corrects_orig_ph, epoch_val_orig_samples_ph, epoch_val_corrects_posthoc_ph = 0, 0, 0

            val_pbar_desc_phase = f"F{fold_run_id_str} E{epoch_global_phase} {phase_name} Val"
            val_pbar_phase = tqdm_iterator(val_ld, desc=val_pbar_desc_phase, ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",120), leave=False)
            with torch.no_grad():
                for batch_val_idx_ph, (inputs_val_ph, labels_val_cpu_ph) in enumerate(val_pbar_phase):
                    img_val_cpu_ph, meta_val_cpu_ph = inputs_val_ph
                    img_val_dev_ph = img_val_cpu_ph.to(device, non_blocking=True)
                    meta_val_dev_ph = meta_val_cpu_ph.to(device, non_blocking=True)
                    true_labels_eval_idx_batch_cpu_ph = labels_val_cpu_ph.cpu()

                    batch_val_labels_model_list_ph, batch_loss_mask_list_ph = [],[]
                    for true_eval_idx_s_ph in true_labels_eval_idx_batch_cpu_ph.tolist():
                        true_label_str_s_ph = idx2label_eval_phase.get(true_eval_idx_s_ph, "___ERR___")
                        model_idx_s_ph = label2idx_model_map_phase.get(true_label_str_s_ph,-1) if true_label_str_s_ph!="___ERR___" else -1
                        batch_val_labels_model_list_ph.append(model_idx_s_ph); batch_loss_mask_list_ph.append(model_idx_s_ph!=-1)

                    batch_val_labels_model_dev_ph = torch.tensor(batch_val_labels_model_list_ph,dtype=torch.long,device=device)
                    batch_loss_active_mask_ph = torch.tensor(batch_loss_mask_list_ph,dtype=torch.bool,device=device)

                    with autocast(device_type=device.type, enabled=amp_enabled_phase):
                        logits_val_batch_ph = eval_model_to_use_phase(img_val_dev_ph, meta_val_dev_ph)
                        if batch_loss_active_mask_ph.any():
                            num_loss_samples_b_ph = img_val_dev_ph[batch_loss_active_mask_ph].size(0)
                            logits_for_loss = logits_val_batch_ph[batch_loss_active_mask_ph].float()
                            labels_for_loss = batch_val_labels_model_dev_ph[batch_loss_active_mask_ph]
                            if loss_type_for_criterion == "focal_ce_loss":
                                 loss_v_b_ph = criterion(logits_for_loss, labels_for_loss, alpha=focal_alpha_cfg, gamma=focal_gamma_cfg)
                            else:
                                 loss_v_b_ph = criterion(logits_for_loss, labels_for_loss)
                            epoch_val_loss_sum_ph += loss_v_b_ph.item()*num_loss_samples_b_ph; epoch_val_loss_samples_ph += num_loss_samples_b_ph
                        else: loss_v_b_ph=torch.tensor(0.0,device=device)

                    preds_orig_model_b_ph = logits_val_batch_ph.argmax(dim=1)
                    if batch_loss_active_mask_ph.any():
                        ok_orig_b_ph = (preds_orig_model_b_ph[batch_loss_active_mask_ph]==batch_val_labels_model_dev_ph[batch_loss_active_mask_ph]).sum().item()
                        epoch_val_corrects_orig_ph+=ok_orig_b_ph; epoch_val_orig_samples_ph+=batch_loss_active_mask_ph.sum().item()

                    epoch_val_seen_samples_ph += img_val_dev_ph.size(0)
                    all_val_logits_list_ph.append(logits_val_batch_ph.cpu()); all_val_true_labels_eval_list_ph.append(true_labels_eval_idx_batch_cpu_ph)

                    tqdm_pf_ph = {'avg_loss':f"{epoch_val_loss_sum_ph/epoch_val_loss_samples_ph:.4f}" if epoch_val_loss_samples_ph>0 else "N/A",
                                  'avg_acc_orig':f"{epoch_val_corrects_orig_ph/epoch_val_orig_samples_ph:.4f}" if epoch_val_orig_samples_ph>0 else "N/A"}
                    if post_hoc_active_this_phase:
                        probs_v_b_cpu_ph = F.softmax(logits_val_batch_ph.cpu(),dim=1)
                        preds_model_b_cpu_ph = probs_v_b_cpu_ph.argmax(dim=1)
                        final_preds_eval_b_tqdm_ph = torch.full_like(preds_model_b_cpu_ph,-1,dtype=torch.long)
                        for i,m_idx_ph in enumerate(preds_model_b_cpu_ph.tolist()):
                            lbl_s_ph=idx2label_model_phase.get(m_idx_ph);
                            if lbl_s_ph and lbl_s_ph in label2idx_eval_map_phase: final_preds_eval_b_tqdm_ph[i]=label2idx_eval_map_phase[lbl_s_ph]
                        for i in range(probs_v_b_cpu_ph.size(0)):
                            if probs_v_b_cpu_ph[i][known_model_indices_phase].max().item() < post_hoc_unk_thresh_cfg_phase: # type: ignore
                                final_preds_eval_b_tqdm_ph[i]=unk_eval_assign_idx_phase
                        epoch_val_corrects_posthoc_ph += (final_preds_eval_b_tqdm_ph==true_labels_eval_idx_batch_cpu_ph).sum().item()
                        tqdm_pf_ph['avg_acc_post_hoc']=f"{epoch_val_corrects_posthoc_ph/epoch_val_seen_samples_ph:.4f}" if epoch_val_seen_samples_ph>0 else "N/A"
                    val_pbar_phase.set_postfix(**tqdm_pf_ph)
            val_pbar_phase.close()

            all_val_logits_cat_ph = torch.cat(all_val_logits_list_ph); all_val_true_labels_eval_cat_ph = torch.cat(all_val_true_labels_eval_list_ph)
            all_val_probs_cat_cpu_ph = F.softmax(all_val_logits_cat_ph,dim=1)

            current_epoch_metrics_val_dict_phase = {}
            current_epoch_metrics_val_dict_phase[f"Loss/val_epoch_{phase_name}"] = epoch_val_loss_sum_ph/epoch_val_loss_samples_ph if epoch_val_loss_samples_ph > 0 else 0
            current_epoch_metrics_val_dict_phase[f"Accuracy/val_epoch_original_argmax_{phase_name}"] = epoch_val_corrects_orig_ph/epoch_val_orig_samples_ph if epoch_val_orig_samples_ph > 0 else 0

            preds_original_modelspace_epoch_ph_cpu = all_val_probs_cat_cpu_ph.argmax(dim=1)

            true_lbls_known_model_list_ph, known_f1_mask_ph = [], torch.zeros_like(all_val_true_labels_eval_cat_ph, dtype=torch.bool)
            for i, true_eval_idx_s_ph in enumerate(all_val_true_labels_eval_cat_ph.tolist()):
                true_lbl_s_ph=idx2label_eval_phase.get(true_eval_idx_s_ph)
                if true_lbl_s_ph and true_lbl_s_ph in label2idx_model_map_phase:
                    true_lbls_known_model_list_ph.append(label2idx_model_map_phase[true_lbl_s_ph]); known_f1_mask_ph[i]=True
            if true_lbls_known_model_list_ph:
                true_lbls_known_model_tensor_ph = torch.tensor(true_lbls_known_model_list_ph,dtype=torch.long)
                preds_for_known_f1_ph = preds_original_modelspace_epoch_ph_cpu[known_f1_mask_ph]
                if preds_for_known_f1_ph.size(0)==true_lbls_known_model_tensor_ph.size(0) and preds_for_known_f1_ph.size(0)>0:
                    current_epoch_metrics_val_dict_phase[f"F1Score/val_macro_known_classes_{phase_name}"] = F1Score(task="multiclass", num_classes=num_model_classes_this_phase, average="macro")(preds_for_known_f1_ph, true_lbls_known_model_tensor_ph).item()
                else: current_epoch_metrics_val_dict_phase[f"F1Score/val_macro_known_classes_{phase_name}"]=0.0
            else: current_epoch_metrics_val_dict_phase[f"F1Score/val_macro_known_classes_{phase_name}"]=0.0

            final_preds_eval_epoch_ph = torch.full_like(preds_original_modelspace_epoch_ph_cpu,-1,dtype=torch.long)
            for i,m_idx_ph_e in enumerate(preds_original_modelspace_epoch_ph_cpu.tolist()):
                lbl_s_ph_e=idx2label_model_phase.get(m_idx_ph_e);
                if lbl_s_ph_e and lbl_s_ph_e in label2idx_eval_map_phase: final_preds_eval_epoch_ph[i]=label2idx_eval_map_phase[lbl_s_ph_e]
            if post_hoc_active_this_phase:
                for i in range(all_val_probs_cat_cpu_ph.size(0)):
                    if all_val_probs_cat_cpu_ph[i][known_model_indices_phase].max().item() < post_hoc_unk_thresh_cfg_phase: # type: ignore
                        final_preds_eval_epoch_ph[i]=unk_eval_assign_idx_phase
            current_epoch_metrics_val_dict_phase[f"Accuracy/val_epoch_post_hoc_unk_{phase_name}"] = (final_preds_eval_epoch_ph == all_val_true_labels_eval_cat_ph).sum().item() / epoch_val_seen_samples_ph if epoch_val_seen_samples_ph > 0 else 0.0 # Corrected calculation based on posthoc preds
            current_epoch_metrics_val_dict_phase[f"F1Score/val_macro_post_hoc_unk_{phase_name}"] = F1Score(task="multiclass", num_classes=num_eval_classes_this_phase, average="macro")(final_preds_eval_epoch_ph, all_val_true_labels_eval_cat_ph).item()

            true_labels_auroc_model_list_ph, auroc_sample_mask_ph = [], torch.zeros(all_val_true_labels_eval_cat_ph.size(0),dtype=torch.bool)
            for i,eval_idx_s_ph in enumerate(all_val_true_labels_eval_cat_ph.tolist()):
                lbl_s_ph=idx2label_eval_phase.get(eval_idx_s_ph);
                if lbl_s_ph and lbl_s_ph in label2idx_model_map_phase: true_labels_auroc_model_list_ph.append(label2idx_model_map_phase[lbl_s_ph]); auroc_sample_mask_ph[i]=True
            if true_labels_auroc_model_list_ph and auroc_sample_mask_ph.any():
                true_lbls_auroc_tensor_ph = torch.tensor(true_labels_auroc_model_list_ph,dtype=torch.long)
                probs_auroc_ph = all_val_probs_cat_cpu_ph[auroc_sample_mask_ph]
                if probs_auroc_ph.size(0)==true_lbls_auroc_tensor_ph.size(0) and probs_auroc_ph.size(0)>0:
                    current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=AUROC(task="multiclass",num_classes=num_model_classes_this_phase,average="macro")(probs_auroc_ph,true_lbls_auroc_tensor_ph).item()
                    # pAUC
                    pauc_max_fpr_cfg = main_training_cfg.get("pauc_max_fpr", 0.2) # Get from config
                    current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{pauc_max_fpr_cfg}_{phase_name}"] = AUROC(task="multiclass",num_classes=num_model_classes_this_phase,average="macro", max_fpr=pauc_max_fpr_cfg)(probs_auroc_ph, true_lbls_auroc_tensor_ph).item()
                else:
                    current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=0.0
                    current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{main_training_cfg.get('pauc_max_fpr', 0.2)}_{phase_name}"]=0.0
            else:
                current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=0.0
                current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{main_training_cfg.get('pauc_max_fpr', 0.2)}_{phase_name}"]=0.0


            optimal_thresholds_for_saving_epoch_ph = {}
            heavy_metrics_were_run_ph = False
            if epoch_global_phase % val_metrics_heavy_interval_phase == 0 or epoch_offset_phase == (num_epochs_this_phase - 1):
                heavy_metrics_were_run_ph = True; logger.info(f"F{fold_run_id_str} E{epoch_global_phase} {phase_name}: Calculating HEAVY validation metrics (PR-curves).")
                if num_model_classes_this_phase > 1:
                    pr_true_model_list_ph, pr_mask_ph = [], torch.zeros(all_val_true_labels_eval_cat_ph.size(0),dtype=torch.bool)
                    for i,eval_idx_s_ph in enumerate(all_val_true_labels_eval_cat_ph.tolist()):
                        lbl_s_ph=idx2label_eval_phase.get(eval_idx_s_ph)
                        if lbl_s_ph and lbl_s_ph in label2idx_model_map_phase:
                            pr_true_model_list_ph.append(label2idx_model_map_phase[lbl_s_ph]); pr_mask_ph[i]=True

                    if pr_true_model_list_ph:
                        pr_true_tensor_ph = torch.tensor(pr_true_model_list_ph,dtype=torch.long)
                        pr_probs_np_ph = all_val_probs_cat_cpu_ph[pr_mask_ph].numpy()
                        pr_lbls_oh_np_ph = F.one_hot(pr_true_tensor_ph, num_model_classes_this_phase).numpy()

                        opt_f1s_ph, opt_sens_ph = [],[]
                        for cls_i_ph in range(num_model_classes_this_phase):
                            f1_ph,thr_ph,sens_ph = 0.0,0.5,0.0
                            try:
                                p_rc_ph,r_rc_ph,t_rc_ph = precision_recall_curve(pr_lbls_oh_np_ph[:,cls_i_ph],pr_probs_np_ph[:,cls_i_ph])
                                if len(p_rc_ph)>1 and len(r_rc_ph)>1 and len(t_rc_ph)>0:
                                    f1_curve_ph=(2*p_rc_ph*r_rc_ph)/(p_rc_ph+r_rc_ph+1e-8)
                                    rel_f1_ph=f1_curve_ph[1:]; rel_r_ph=r_rc_ph[1:]
                                    valid_idx_ph=np.where(np.isfinite(rel_f1_ph)&(p_rc_ph[1:]+r_rc_ph[1:]>0))[0]
                                    if len(valid_idx_ph)>0:
                                        best_i_ph=valid_idx_ph[np.argmax(rel_f1_ph[valid_idx_ph])]
                                        f1_ph,thr_ph,sens_ph=float(rel_f1_ph[best_i_ph]),float(t_rc_ph[best_i_ph]),float(rel_r_ph[best_i_ph])
                            except Exception as e_pr_ph: logger.warning(f"PR curve failed for {phase_name} cls {cls_i_ph} E{epoch_global_phase}: {e_pr_ph}")
                            opt_f1s_ph.append(f1_ph); opt_sens_ph.append(sens_ph); optimal_thresholds_for_saving_epoch_ph[cls_i_ph]=thr_ph
                        if opt_f1s_ph: current_epoch_metrics_val_dict_phase[f"F1Score/val_mean_optimal_per_class_from_PR_{phase_name}"]=np.mean(opt_f1s_ph)
                        if opt_sens_ph: current_epoch_metrics_val_dict_phase[f"Sensitivity/val_mean_optimal_per_class_from_PR_{phase_name}"]=np.mean(opt_sens_ph)
                    else: logger.warning(f"{phase_name} PR Curves: No valid samples after filtering for model's known classes.")
                else: logger.warning(f"{phase_name} PR Curves: Not enough model output classes ({num_model_classes_this_phase}) for PR curves.")


            current_epoch_metrics_combined_phase.update(current_epoch_metrics_val_dict_phase)

            metric_key_map_for_phase = {
                "macro_auc": f"AUROC/val_macro_{phase_name}",
                "f1_macro_known_classes": f"F1Score/val_macro_known_classes_{phase_name}",
                "f1_macro_post_hoc_unk": f"F1Score/val_macro_post_hoc_unk_{phase_name}",
                "accuracy_post_hoc_unk": f"Accuracy/val_epoch_post_hoc_unk_{phase_name}",
                "mean_optimal_f1": f"F1Score/val_mean_optimal_per_class_from_PR_{phase_name}",
                "mean_optimal_sensitivity": f"Sensitivity/val_mean_optimal_per_class_from_PR_{phase_name}",
            }
            primary_metric_key_for_phase = metric_key_map_for_phase.get(model_selection_metric_name_phase, f"AUROC/val_macro_{phase_name}")
            current_primary_metric_val_phase = current_epoch_metrics_val_dict_phase.get(primary_metric_key_for_phase, -float('inf'))

            log_line_ph = (f"F{fold_run_id_str} E{epoch_global_phase} {phase_name} Val -> "
                           f"Loss={current_epoch_metrics_val_dict_phase.get(f'Loss/val_epoch_{phase_name}', -1.0):.4f} "
                           f"AccOrig={current_epoch_metrics_val_dict_phase.get(f'Accuracy/val_epoch_original_argmax_{phase_name}', -1.0):.4f} ")
            if post_hoc_active_this_phase: log_line_ph += f"AccPostHoc={current_epoch_metrics_val_dict_phase.get(f'Accuracy/val_epoch_post_hoc_unk_{phase_name}', -1.0):.4f} "
            log_line_ph += f"Selected ({model_selection_metric_name_phase}->{primary_metric_key_for_phase.replace(f'_{phase_name}','') if primary_metric_key_for_phase else 'N/A'}): {current_primary_metric_val_phase:.4f}"
            logger.info(log_line_ph)


            if current_primary_metric_val_phase > best_metric_val_current_phase:
                best_metric_val_current_phase = current_primary_metric_val_phase
                best_epoch_for_metric_current_phase = epoch_global_phase
                patience_counter_current_phase = 0
                best_ckpt_path_current_phase = str(run_ckpt_dir_phase / f"{exp_name_for_files_phase}_fold{fold_run_id_str}_{phase_name}_best_E{epoch_global_phase}.pt")
                logger.info(f"New best for {phase_name} ({model_selection_metric_name_phase}): {best_metric_val_current_phase:.4f}. Saving to {best_ckpt_path_current_phase}")

                ckpt_data_ph = {
                    'epoch': epoch_global_phase,
                    'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config_runtime': cfg,
                    'label2idx_model': label2idx_model_map_phase,
                    'label2idx_eval': label2idx_eval_map_phase,
                    'phase_name': phase_name,
                    f'best_{model_selection_metric_name_phase}': best_metric_val_current_phase,
                    'metadata_dim': model.metadata_mlp.fc1.in_features # Save the runtime metadata dim
                }
                if ema_model_phase: ckpt_data_ph['ema_model_state_dict'] = getattr(ema_model_phase, '_orig_mod', ema_model_phase).state_dict()
                if post_hoc_active_this_phase: ckpt_data_ph['post_hoc_unk_threshold_used'] = post_hoc_unk_thresh_cfg_phase
                
                save_pr_th_cfg = main_training_cfg.get("save_optimal_thresholds_from_pr", False)
                is_pr_metric = model_selection_metric_name_phase in ["mean_optimal_f1", "mean_optimal_sensitivity"]
                if (save_pr_th_cfg or is_pr_metric) and heavy_metrics_were_run_ph and optimal_thresholds_for_saving_epoch_ph:
                    ckpt_data_ph['optimal_thresholds_val_from_pr'] = optimal_thresholds_for_saving_epoch_ph
                    optimal_thresholds_best_ckpt_current_phase = optimal_thresholds_for_saving_epoch_ph
                    logger.info(f"Saved optimal PR thresholds with best {phase_name} checkpoint.")
                torch.save(ckpt_data_ph, best_ckpt_path_current_phase)
            else:
                patience_counter_current_phase += 1
            if patience_counter_current_phase >= early_stopping_patience_phase:
                logger.info(f"Early stopping for {phase_name} at E{epoch_global_phase}."); break

        tb_logger.log_epoch_summary(current_epoch_metrics_combined_phase, epoch_global_phase)
        if patience_counter_current_phase >= early_stopping_patience_phase: break

    last_ckpt_path_phase = run_ckpt_dir_phase / f"{exp_name_for_files_phase}_fold{fold_run_id_str}_{phase_name}_last_E{epoch_global_phase}.pt"
    last_ckpt_data_ph = {
        'epoch': epoch_global_phase,
        'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config_runtime': cfg,
        'label2idx_model': label2idx_model_map_phase,
        'label2idx_eval': label2idx_eval_map_phase,
        'phase_name': phase_name,
        'current_primary_metric': current_epoch_metrics_val_dict_phase.get(primary_metric_key_for_phase, -float('inf')) if 'current_epoch_metrics_val_dict_phase' in locals() and primary_metric_key_for_phase in current_epoch_metrics_val_dict_phase else -float('inf'),
        'metadata_dim': model.metadata_mlp.fc1.in_features
    }
    if ema_model_phase: last_ckpt_data_ph['ema_model_state_dict'] = getattr(ema_model_phase, '_orig_mod', ema_model_phase).state_dict()
    if post_hoc_active_this_phase: last_ckpt_data_ph['post_hoc_unk_threshold_used'] = post_hoc_unk_thresh_cfg_phase
    if 'optimal_thresholds_for_saving_epoch_ph' in locals() and optimal_thresholds_for_saving_epoch_ph : # type: ignore
         last_ckpt_data_ph['optimal_thresholds_val_from_pr'] = optimal_thresholds_for_saving_epoch_ph # type: ignore
    torch.save(last_ckpt_data_ph, str(last_ckpt_path_phase))
    logger.info(f"Saved last {phase_name} model to {last_ckpt_path_phase}")

    logger.info(f"Finished {phase_name} for F{fold_run_id_str}. Best {model_selection_metric_name_phase}: {best_metric_val_current_phase:.4f} at E{best_epoch_for_metric_current_phase}")
    return best_metric_val_current_phase, best_epoch_for_metric_current_phase, best_ckpt_path_current_phase, optimal_thresholds_best_ckpt_current_phase


# --- Main Training Orchestrator for a Single Fold with Metadata ---
def train_one_fold_with_meta(
    fold_run_id: int | str, train_df_fold_selected: pd.DataFrame,
    val_df_fold_selected: pd.DataFrame, meta_df_full_data: pd.DataFrame,
    cfg: dict, label2idx_model_map_runtime: dict[str,int], label2idx_eval_map_runtime: dict[str,int],
    image_root_path: Path, run_log_dir_main_fold: Path, run_ckpt_dir_main_fold: Path,
    exp_name_cli_arg: str, device_obj_runtime: torch.device,
) -> float | None:

    fold_run_id_str_main = str(fold_run_id)
    main_train_cfg_fold = cfg.get("training", {})
    meta_tune_cfg_fold = cfg.get("meta_tuning", {}) # May not exist
    data_cfg_fold = cfg.get("data", {})
    model_arch_cfg_fold = cfg.get("model", {})

    _bs_tb_fold = main_train_cfg_fold.get("batch_size", 32)
    tb_train_len_fold = (len(train_df_fold_selected) + _bs_tb_fold - 1) // _bs_tb_fold
    try:
        tb_logger_fold = TensorBoardLogger(log_dir=run_log_dir_main_fold, experiment_config=cfg, train_loader_len=tb_train_len_fold)
    except Exception as e: logger.error(f"TBLogger init fail fold {fold_run_id_str_main}: {e}", exc_info=True); return None

    tf_train_fold = build_transform(data_cfg_fold.get("cpu_augmentations",{}), train=True)
    tf_val_fold = build_transform(data_cfg_fold.get("cpu_augmentations",{}), train=False)

    dataset_args_from_cfg = data_cfg_fold.get("dataset_args", {})
    dataset_args_from_cfg.update({ # Pass specific args for FlatDatasetWithMeta
        "meta_features_names": data_cfg_fold.get("meta_features_names"),
        "meta_augmentation_p": data_cfg_fold.get("meta_augmentation_p", 0.0),
        "meta_nan_fill_value": data_cfg_fold.get("meta_nan_fill_value", 0.0),
        "image_loader": data_cfg_fold.get("image_loader", "pil"),
        "enable_ram_cache": data_cfg_fold.get("enable_ram_cache", False)
    })

    train_ds_obj = FlatDatasetWithMeta(df=train_df_fold_selected, meta_df=meta_df_full_data, root=image_root_path,
                                    label2idx=label2idx_model_map_runtime, tf=tf_train_fold,
                                    **dataset_args_from_cfg) # type: ignore
    val_ds_obj = FlatDatasetWithMeta(df=val_df_fold_selected, meta_df=meta_df_full_data, root=image_root_path,
                                  label2idx=label2idx_eval_map_runtime, tf=tf_val_fold, # eval uses eval map for labels
                                  **dataset_args_from_cfg) # type: ignore

    metadata_dim_runtime = train_ds_obj.metadata_dim
    logger.info(f"Fold {fold_run_id_str_main}: Runtime metadata dimension: {metadata_dim_runtime}")
    cfg['model']['metadata_input_dim_runtime'] = metadata_dim_runtime # Save for potential ckpt loading

    base_cnn_params = {"MODEL_TYPE": model_arch_cfg_fold.get("base_cnn_type"),
                       "numClasses": len(label2idx_model_map_runtime),
                       "pretrained": model_arch_cfg_fold.get("pretrained_cnn", True)}
    base_cnn_instance = get_base_cnn_model(base_cnn_params)

    meta_head_args_cfg = model_arch_cfg_fold.get("meta_head_args", {}).copy()
    meta_head_args_cfg.pop('num_classes', None)
    meta_head_args_cfg.pop('metadata_input_dim', None)

    full_model_instance = CNNWithMetadata(
        base_cnn_model=base_cnn_instance,
        num_classes=len(label2idx_model_map_runtime),
        metadata_input_dim=metadata_dim_runtime,
        **meta_head_args_cfg
    ).to(device_obj_runtime)
    ema_model_instance = None
    if main_train_cfg_fold.get("ema_decay", 0.0) > 0:
        ema_model_instance = copy.deepcopy(full_model_instance).to(device_obj_runtime)
        if ema_model_instance: [p.requires_grad_(False) for p in ema_model_instance.parameters()] # type: ignore

    class_counts_for_loss = get_class_counts(train_df_fold_selected, label2idx_model_map_runtime)
    criterion_main, loss_type_main = None, main_train_cfg_fold.get("loss", {}).get("type", "cross_entropy").lower()
    loss_params_cfg = main_train_cfg_fold.get("loss", {})

    if loss_type_main == "focal_ce_loss":
        criterion_main = focal_ce_loss # This is a function, not a class instance here
        logger.info(f"Using Focal CE Loss function.")
    elif loss_type_main == "cross_entropy":
        criterion_main = nn.CrossEntropyLoss()
        logger.info("Using nn.CrossEntropyLoss.")
    elif loss_type_main == "ldam_loss":
        ldam_specific_params = loss_params_cfg.get("ldam_params", {})
        criterion_main = LDAMLoss(class_counts=class_counts_for_loss, **ldam_specific_params).to(device_obj_runtime) # type: ignore
        logger.info(f"Using LDAM Loss with params: {ldam_specific_params}")
    else:
        criterion_main = nn.CrossEntropyLoss()
        logger.warning(f"Unknown loss type '{loss_type_main}', defaulting to nn.CrossEntropyLoss.")


    scaler_main = GradScaler(device_obj_runtime.type if device_obj_runtime.type in ['cuda','mps'] else 'cpu',
                             enabled=(device_obj_runtime.type == 'cuda' and main_train_cfg_fold.get("amp_enabled", True)))

    overall_best_metric_for_fold = -float('inf')
    path_to_overall_best_ckpt_fold = ""
    final_optimal_thresholds_for_fold = None
    final_epoch_completed_for_tb_table = 0

    # ========================= PHASE 1: Joint Training =========================
    if main_train_cfg_fold.get("num_epochs", 0) > 0 :
        logger.info(f"===== Fold {fold_run_id_str_main}: Starting Phase 1: Joint Training =====")
        full_model_instance.set_base_cnn_trainable(True)

        dl_args_p1 = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": True, "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if dl_args_p1["persistent_workers"] and dl_args_p1["num_workers"] > 0:
             dl_args_p1["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)

        train_sampler_p1 = ClassBalancedSampler(train_ds_obj, len(train_ds_obj)) if data_cfg_fold.get("sampler",{}).get("type")=="class_balanced_sqrt" else None
        train_loader_p1 = DataLoader(train_ds_obj, batch_size=main_train_cfg_fold.get("batch_size",32), sampler=train_sampler_p1, shuffle=(train_sampler_p1 is None), **dl_args_p1) # type: ignore
        val_loader_p1 = DataLoader(val_ds_obj, batch_size=main_train_cfg_fold.get("batch_size",32), shuffle=False, **dl_args_p1) # type: ignore

        opt_cfg_p1 = main_train_cfg_fold.get("optimizer", {})
        optimizer_p1 = AdamW(full_model_instance.parameters(), lr=opt_cfg_p1.get("lr",1e-3), weight_decay=opt_cfg_p1.get("weight_decay",1e-4))
        sched_cfg_p1 = main_train_cfg_fold.get("scheduler", {})
        scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=main_train_cfg_fold.get("num_epochs",10), eta_min=sched_cfg_p1.get("min_lr",1e-6))

        best_metric_p1, best_epoch_p1, best_ckpt_p1_path, best_thresh_p1 = run_training_phase(
            phase_name="P1_Joint", model=full_model_instance, train_ld=train_loader_p1, val_ld=val_loader_p1,
            criterion=criterion_main, optimizer=optimizer_p1, scheduler=scheduler_p1, scaler=scaler_main,
            device=device_obj_runtime, cfg=cfg, phase_cfg=main_train_cfg_fold, fold_run_id_str=fold_run_id_str_main,
            label2idx_model_map_phase=label2idx_model_map_runtime, label2idx_eval_map_phase=label2idx_eval_map_runtime,
            tb_logger=tb_logger_fold, run_ckpt_dir_phase=run_ckpt_dir_main_fold, exp_name_for_files_phase=exp_name_cli_arg,
            start_epoch=0, ema_model_phase=ema_model_instance, initial_best_metric_val_phase=-float('inf')
        )
        final_epoch_completed_for_tb_table = max(final_epoch_completed_for_tb_table, best_epoch_p1 if best_epoch_p1 != -1 else (main_train_cfg_fold.get("num_epochs",0) -1))
        if best_metric_p1 > overall_best_metric_for_fold:
            overall_best_metric_for_fold = best_metric_p1
            path_to_overall_best_ckpt_fold = best_ckpt_p1_path
            final_optimal_thresholds_for_fold = best_thresh_p1

        if meta_tune_cfg_fold.get("enable", False) and best_epoch_p1 != -1 and Path(best_ckpt_p1_path).exists():
            logger.info(f"Loading best P1 model (E{best_epoch_p1}) for P2: {best_ckpt_p1_path}")
            ckpt_p1 = torch.load(best_ckpt_p1_path, map_location=device_obj_runtime)
            full_model_instance.load_state_dict(ckpt_p1['model_state_dict'])
            if ema_model_instance and ckpt_p1.get('ema_model_state_dict'):
                ema_model_instance.load_state_dict(ckpt_p1['ema_model_state_dict'])
    else:
        logger.info(f"Fold {fold_run_id_str_main}: Phase 1 (Joint Training) skipped.")

    # ========================= PHASE 2: Metadata Head Fine-tuning =========================
    if meta_tune_cfg_fold.get("enable", False) and meta_tune_cfg_fold.get("num_epochs",0) > 0:
        logger.info(f"===== Fold {fold_run_id_str_main}: Starting Phase 2: Meta Head Fine-tuning =====")
        full_model_instance.set_base_cnn_trainable(False); logger.info("CNN backbone frozen for Phase 2.")

        dl_args_p2 = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": True, "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if dl_args_p2["persistent_workers"] and dl_args_p2["num_workers"] > 0:
             dl_args_p2["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)

        train_sampler_p2 = ClassBalancedSampler(train_ds_obj, len(train_ds_obj)) if data_cfg_fold.get("sampler",{}).get("type")=="class_balanced_sqrt" else None
        train_loader_p2 = DataLoader(train_ds_obj, batch_size=meta_tune_cfg_fold.get("batch_size",32), sampler=train_sampler_p2, shuffle=(train_sampler_p2 is None), **dl_args_p2) # type: ignore
        val_loader_p2 = DataLoader(val_ds_obj, batch_size=meta_tune_cfg_fold.get("batch_size",32), shuffle=False, **dl_args_p2) # type: ignore

        params_to_tune_p2 = [p for p in full_model_instance.parameters() if p.requires_grad]
        if not params_to_tune_p2:
            logger.error("Phase 2: No parameters found for tuning (all require_grad=False). Skipping Phase 2.")
        else:
            logger.info(f"P2: Num params to tune: {sum(p.numel() for p in params_to_tune_p2)}")

            opt_cfg_p2 = meta_tune_cfg_fold.get("optimizer", {})
            optimizer_p2 = AdamW(params_to_tune_p2, lr=opt_cfg_p2.get("lr",1e-4), weight_decay=opt_cfg_p2.get("weight_decay",1e-5))
            sched_cfg_p2 = meta_tune_cfg_fold.get("scheduler", {})
            scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=meta_tune_cfg_fold.get("num_epochs",10), eta_min=sched_cfg_p2.get("min_lr",1e-7))

            phase1_num_epochs = main_train_cfg_fold.get("num_epochs",0)

            best_metric_p2, best_epoch_p2, best_ckpt_p2_path, best_thresh_p2 = run_training_phase(
                phase_name="P2_MetaTune", model=full_model_instance, train_ld=train_loader_p2, val_ld=val_loader_p2,
                criterion=criterion_main, optimizer=optimizer_p2, scheduler=scheduler_p2, scaler=scaler_main,
                device=device_obj_runtime, cfg=cfg, phase_cfg=meta_tune_cfg_fold, fold_run_id_str=str(fold_run_id),
                label2idx_model_map_phase=label2idx_model_map_runtime, label2idx_eval_map_phase=label2idx_eval_map_runtime,
                tb_logger=tb_logger_fold, run_ckpt_dir_phase=run_ckpt_dir_main_fold, exp_name_for_files_phase=exp_name_cli_arg,
                start_epoch=phase1_num_epochs, ema_model_phase=ema_model_instance,
                initial_best_metric_val_phase=-float('inf')
            )
            final_epoch_completed_for_tb_table = max(final_epoch_completed_for_tb_table, best_epoch_p2 if best_epoch_p2 != -1 else (phase1_num_epochs + meta_tune_cfg_fold.get("num_epochs",0) -1) )
            if best_metric_p2 > overall_best_metric_for_fold:
                overall_best_metric_for_fold = best_metric_p2
                path_to_overall_best_ckpt_fold = best_ckpt_p2_path
                final_optimal_thresholds_for_fold = best_thresh_p2
    else:
        logger.info(f"Fold {fold_run_id_str_main}: Phase 2 Meta-Tuning skipped.")

    if overall_best_metric_for_fold > -float('inf') and Path(path_to_overall_best_ckpt_fold).exists():
        logger.info(f"Fold {fold_run_id_str_main}: Calculating final metrics using ckpt: {path_to_overall_best_ckpt_fold}")
        best_ckpt_loaded_final = torch.load(path_to_overall_best_ckpt_fold, map_location=device_obj_runtime)

        cfg_final_eval = best_ckpt_loaded_final.get('config_runtime', cfg)
        final_val_label2idx_model = best_ckpt_loaded_final.get('label2idx_model', label2idx_model_map_runtime)
        final_val_label2idx_eval = best_ckpt_loaded_final.get('label2idx_eval', label2idx_eval_map_runtime)
        final_val_idx2label_model = {v: k for k, v in final_val_label2idx_model.items()}
        final_val_idx2label_eval = {v: k for k, v in final_val_label2idx_eval.items()}
        final_val_num_model_classes = len(final_val_label2idx_model)
        final_val_num_eval_classes = len(final_val_label2idx_eval)

        final_model_cfg_from_ckpt = cfg_final_eval.get("model",{})
        final_base_cnn_params = {"MODEL_TYPE": final_model_cfg_from_ckpt.get("base_cnn_type"),
                                 "numClasses": final_val_num_model_classes,
                                 "pretrained": final_model_cfg_from_ckpt.get("pretrained_cnn", False)}
        final_base_cnn_instance = get_base_cnn_model(final_base_cnn_params)

        final_metadata_dim = best_ckpt_loaded_final.get('metadata_dim', cfg_final_eval.get("model",{}).get('metadata_input_dim_runtime', metadata_dim_runtime) )
        if 'metadata_dim' not in best_ckpt_loaded_final: logger.warning("metadata_dim not directly in checkpoint, using runtime/config value.")

        final_meta_head_args = final_model_cfg_from_ckpt.get("meta_head_args", {}).copy()
        final_meta_head_args.pop('num_classes', None)
        final_meta_head_args.pop('metadata_input_dim', None)

        model_for_final_eval = CNNWithMetadata(base_cnn_model=final_base_cnn_instance,
                                               num_classes=final_val_num_model_classes,
                                               metadata_input_dim=final_metadata_dim,
                                               **final_meta_head_args).to(device_obj_runtime) # type: ignore

        final_weights_key = 'ema_model_state_dict' if cfg_final_eval.get("training",{}).get("use_ema_for_val",True) and \
                                                       'ema_model_state_dict' in best_ckpt_loaded_final and \
                                                       best_ckpt_loaded_final['ema_model_state_dict'] else 'model_state_dict'
        model_for_final_eval.load_state_dict(best_ckpt_loaded_final[final_weights_key])
        model_for_final_eval.eval()

        final_eval_dl_args = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": True,
                              "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if final_eval_dl_args["persistent_workers"] and final_eval_dl_args["num_workers"] > 0:
             final_eval_dl_args["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)
        final_val_loader = DataLoader(val_ds_obj, batch_size=cfg_final_eval.get("training",{}).get("batch_size",32), shuffle=False, **final_eval_dl_args) # type: ignore

        all_final_val_probs_list, all_final_val_true_labels_list = [], []
        with torch.no_grad():
            for inputs_final, labels_final_eval_cpu in tqdm_iterator(final_val_loader, desc=f"F{fold_run_id_str_main} Final Table Preds"):
                img_f_cpu, meta_f_cpu = inputs_final
                img_f_dev, meta_f_dev = img_f_cpu.to(device_obj_runtime), meta_f_cpu.to(device_obj_runtime)
                logits_f = model_for_final_eval(img_f_dev, meta_f_dev)
                all_final_val_probs_list.append(F.softmax(logits_f, dim=1).cpu())
                all_final_val_true_labels_list.append(labels_final_eval_cpu.cpu())

        all_final_val_probs_cat = torch.cat(all_final_val_probs_list)
        all_final_val_true_labels_eval_cat = torch.cat(all_final_val_true_labels_list)

        f_post_hoc_thresh = best_ckpt_loaded_final.get('post_hoc_unk_threshold_used', cfg_final_eval.get("training",{}).get("post_hoc_unk_threshold"))
        f_unk_str = cfg_final_eval.get("training",{}).get("unk_label_string", "UNK")
        f_known_model_indices = [idx for lbl,idx in final_val_label2idx_model.items() if lbl!=f_unk_str and idx!=final_val_label2idx_model.get(f_unk_str,-1)]
        f_unk_eval_assign_idx = final_val_label2idx_eval.get(f_unk_str, -1)
        f_post_hoc_active = (f_post_hoc_thresh is not None and bool(f_known_model_indices) and f_unk_eval_assign_idx != -1)

        f_preds_modelspace = all_final_val_probs_cat.argmax(dim=1)
        f_final_preds_evalspace = torch.full_like(f_preds_modelspace, -1, dtype=torch.long)
        for i, f_model_idx in enumerate(f_preds_modelspace.tolist()):
            f_label_str = final_val_idx2label_model.get(f_model_idx)
            if f_label_str and f_label_str in final_val_label2idx_eval:
                f_final_preds_evalspace[i] = final_val_label2idx_eval[f_label_str]
        if f_post_hoc_active:
            logger.info(f"Applying post-hoc UNK (Th={f_post_hoc_thresh}) for final metrics table.")
            for i in range(all_final_val_probs_cat.size(0)):
                if all_final_val_probs_cat[i][f_known_model_indices].max().item() < f_post_hoc_thresh : # type: ignore
                    f_final_preds_evalspace[i] = f_unk_eval_assign_idx
        
        # --- Generate Confusion Matrix Figure ---
        cm_disp_labels = [lbl for lbl, idx in sorted(final_val_idx2label_eval.items(), key=lambda item: item[1])]
        cm_fig_title = f"F{fold_run_id_str_main} Val CM (Best E{best_ckpt_loaded_final['epoch']})"
        if f_post_hoc_active: cm_fig_title += f" UNK Th={f_post_hoc_thresh:.2f}" # type: ignore
        cm_figure_obj = generate_confusion_matrix_figure(all_final_val_true_labels_eval_cat.numpy(), f_final_preds_evalspace.numpy(), cm_disp_labels, cm_fig_title)
        if tb_logger_fold.writer: tb_logger_fold.writer.add_figure(f"Run_{fold_run_id_str_main}/ConfusionMatrix_BestModel_ValSet", cm_figure_obj, global_step=final_epoch_completed_for_tb_table)
        plt.close(cm_figure_obj)


        # --- Calculate and Store Metrics for Table ---
        metrics_table_rows = []
        f_eval_class_names = [final_val_idx2label_eval.get(i, f"UnknownIdx{i}") for i in range(final_val_num_eval_classes)]

        acc_overall = (f_final_preds_evalspace == all_final_val_true_labels_eval_cat).sum().item() / len(all_final_val_true_labels_eval_cat)
        metrics_table_rows.append(["Overall Accuracy", f"{acc_overall:.4f}"])
        f1_macro_eval = F1Score(task="multiclass", num_classes=final_val_num_eval_classes, average="macro")(f_final_preds_evalspace, all_final_val_true_labels_eval_cat).item()
        metrics_table_rows.append(["Macro F1-Score (Eval Space)", f"{f1_macro_eval:.4f}"])

        true_labels_for_known_auroc_list_final, known_auroc_sample_mask_final = [], torch.zeros_like(all_final_val_true_labels_eval_cat, dtype=torch.bool)
        for i, true_eval_idx_s_final in enumerate(all_final_val_true_labels_eval_cat.tolist()):
            true_label_s_final = final_val_idx2label_eval.get(true_eval_idx_s_final)
            if true_label_s_final and true_label_s_final in final_val_label2idx_model:
                true_labels_for_known_auroc_list_final.append(final_val_label2idx_model[true_label_s_final])
                known_auroc_sample_mask_final[i] = True
        if true_labels_for_known_auroc_list_final:
            true_labels_known_auroc_tensor_final = torch.tensor(true_labels_for_known_auroc_list_final, dtype=torch.long)
            probs_for_known_auroc_final = all_final_val_probs_cat[known_auroc_sample_mask_final]
            if probs_for_known_auroc_final.size(0) == true_labels_known_auroc_tensor_final.size(0) and probs_for_known_auroc_final.size(0) > 0:
                macro_auroc_known_final = AUROC(task="multiclass", num_classes=final_val_num_model_classes, average="macro")(probs_for_known_auroc_final, true_labels_known_auroc_tensor_final).item()
                metrics_table_rows.append(["Macro AUROC (Model Space)", f"{macro_auroc_known_final:.4f}"])
                pauc_fpr_cfg_final = cfg_final_eval.get("training",{}).get("pauc_max_fpr", 0.2)
                pauc_known_maxfpr_final = AUROC(task="multiclass", num_classes=final_val_num_model_classes, average="macro", max_fpr=pauc_fpr_cfg_final)(probs_for_known_auroc_final, true_labels_known_auroc_tensor_final).item()
                metrics_table_rows.append([f"Macro pAUROC@FPR{pauc_fpr_cfg_final} (Model Space)", f"{pauc_known_maxfpr_final:.4f}"])
            else:
                metrics_table_rows.append(["Macro AUROC (Model Space)", "N/A (error)"])
                metrics_table_rows.append([f"Macro pAUROC@FPR{cfg_final_eval.get('training',{}).get('pauc_max_fpr',0.2)} (Model Space)", "N/A (error)"])
        else:
            metrics_table_rows.append(["Macro AUROC (Model Space)", "N/A (no known)"])
            metrics_table_rows.append([f"Macro pAUROC@FPR{cfg_final_eval.get('training',{}).get('pauc_max_fpr',0.2)} (Model Space)", "N/A (no known)"])

        all_sens, all_spec, all_ppv, all_npv, all_f1_cls, all_ap_cls = [],[],[],[],[],[]
        for i in range(final_val_num_eval_classes):
            true_bin = (all_final_val_true_labels_eval_cat == i).int(); pred_bin = (f_final_preds_evalspace == i).int()
            cm_bin = confusion_matrix(true_bin.numpy(), pred_bin.numpy(), labels=[0,1]) # Ensure labels=[0,1] for binary CM
            if cm_bin.size == 4: tn, fp, fn, tp = cm_bin.ravel()
            elif true_bin.sum() == 0 and pred_bin.sum() == 0 : tn, fp, fn, tp = len(true_bin),0,0,0 # All true neg, all pred neg
            elif true_bin.sum() == len(true_bin) and pred_bin.sum() == len(pred_bin) : tn, fp, fn, tp = 0,0,0,len(true_bin) # All true pos, all pred pos
            elif true_bin.sum() == 0 : tn, fp, fn, tp = cm_bin[0,0] if cm_bin.shape==(1,1) and pred_bin.sum()==0 else len(true_bin)-pred_bin.sum().item(), pred_bin.sum().item(), 0,0 # No true positives
            elif true_bin.sum() == len(true_bin) : tn, fp, fn, tp = 0,0, len(true_bin)-pred_bin.sum().item(), pred_bin.sum().item() # No true negatives
            else: # fallback for other edge cases, though previous should cover most
                logger.warning(f"Unexpected binary CM shape for class {i}: {cm_bin.shape}. True sum: {true_bin.sum()}, Pred sum: {pred_bin.sum()}. Using 0 for TP/FP/FN/TN.")
                tn, fp, fn, tp = 0,0,0,0


            sens=tp/(tp+fn) if (tp+fn)>0 else 0.0; all_sens.append(sens)
            spec=tn/(tn+fp) if (tn+fp)>0 else 0.0; all_spec.append(spec)
            ppv=tp/(tp+fp) if (tp+fp)>0 else 0.0; all_ppv.append(ppv)
            npv=tn/(tn+fn) if (tn+fn)>0 else 0.0; all_npv.append(npv)
            f1c=2*ppv*sens/(ppv+sens) if (ppv+sens)>0 else 0.0; all_f1_cls.append(f1c)

            cls_n=f_eval_class_names[i]; metrics_table_rows.extend([
                [f"{cls_n} - Sens",f"{sens:.4f}"],[f"{cls_n} - Spec",f"{spec:.4f}"],
                [f"{cls_n} - PPV",f"{ppv:.4f}"],[f"{cls_n} - NPV",f"{npv:.4f}"],[f"{cls_n} - F1",f"{f1c:.4f}"]
            ])

            ap_cls_score = 0.0
            if f_eval_class_names[i] == f_unk_str and f_unk_eval_assign_idx == i and f_post_hoc_active: # AP for UNK class if post-hoc active
                # For UNK, score is 1 - max_prob_of_known_classes if prediction is UNK
                # If prediction is not UNK, score is 0 for being UNK.
                # This is a bit heuristic. A more direct way might be needed if UNK is a direct model output.
                # Using max confidence in NON-UNK classes for UNK score:
                scores_unk_ap = torch.zeros(len(f_final_preds_evalspace), dtype=torch.float)
                for k_ap_unk in range(len(f_final_preds_evalspace)):
                    if f_final_preds_evalspace[k_ap_unk] == f_unk_eval_assign_idx : # if classified as UNK by post-hoc
                         if all_final_val_probs_cat[k_ap_unk][f_known_model_indices].numel() > 0 : # Check if known_model_indices is not empty
                            scores_unk_ap[k_ap_unk] = 1.0 - all_final_val_probs_cat[k_ap_unk][f_known_model_indices].max().item()
                         else: # No known classes to compare against, cannot determine confidence for UNK
                            scores_unk_ap[k_ap_unk] = 0.0 # Or some other default.
                    # else: score remains 0, as it wasn't predicted as UNK.
                ap_cls_score = AveragePrecision(task="binary")(scores_unk_ap, true_bin).item()

            elif f_eval_class_names[i] in final_val_label2idx_model: # AP for classes known to model
                model_cls_idx_for_ap = final_val_label2idx_model[f_eval_class_names[i]]
                ap_cls_score = AveragePrecision(task="binary")(all_final_val_probs_cat[:,model_cls_idx_for_ap], true_bin).item()
            # else: AP for classes not in model output and not UNK (e.g. 'other' category) is 0 or not applicable here
            all_ap_cls.append(ap_cls_score); metrics_table_rows.append([f"{cls_n} - AP",f"{ap_cls_score:.4f}"])

        metrics_table_rows.extend([["Macro Avg Sens",f"{np.mean(all_sens):.4f}"],["Macro Avg Spec",f"{np.mean(all_spec):.4f}"],
                                     ["Macro Avg PPV",f"{np.mean(all_ppv):.4f}"],["Macro Avg NPV",f"{np.mean(all_npv):.4f}"],
                                     ["Mean Avg Prec (mAP)",f"{np.mean(all_ap_cls):.4f}"]])

        header_md = "| Metric                        | Value   |\n"; separator_md = "|-------------------------------|---------|\n"
        table_md_str = header_md + separator_md + "".join([f"| {name:<29} | {val:<7} |\n" for name, val in metrics_table_rows])
        if tb_logger_fold.writer:
            tb_logger_fold.writer.add_text(f"Run_{fold_run_id_str_main}/Final_Validation_Metrics_Table", table_md_str, global_step=final_epoch_completed_for_tb_table)
            logger.info(f"\nFinal Validation Metrics Table for Fold {fold_run_id_str_main} (Best Model):\n{table_md_str}")
    else:
        logger.warning(f"Fold {fold_run_id_str_main}: No best checkpoint found or training failed. Skipping final metrics table.")

    tb_logger_fold.close()
    logger.info(f"Fold {fold_run_id_str_main} finished. Overall best val metric for this fold: {overall_best_metric_for_fold:.4f}")
    return overall_best_metric_for_fold


def main():
    ap = argparse.ArgumentParser(description="Train model with metadata for a single specified fold.")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config if set.")
    ap.add_argument("--fold_id_to_run", type=str, required=True, help="The specific fold ID to run.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True) # force=True for re-init if needed
    logger.info(f"Starting single fold experiment with metadata: {args.exp_name}, Fold: {args.fold_id_to_run}")

    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
    if not cfg_path.exists():
        fallback_path = Path(args.config_dir) / "config_metadata_single_fold.yaml" # Default name for meta config
        if not args.config_file and fallback_path.exists(): # Only use fallback if --config_file was not specified
            cfg_path = fallback_path
            logger.warning(f"Experiment config '{args.exp_name}.yaml' not found. Using fallback: {cfg_path}")
        else:
            raise FileNotFoundError(f"Config file {cfg_path} (or specified '{args.config_file}') not found, and fallback '{fallback_path}' not used or not found.")
    cfg = load_config(cfg_path); logger.info(f"Loaded config: {cfg_path}"); cfg = cast_config_values(cfg)

    exp_cfg = cfg.get("experiment_setup", {})
    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    set_seed(seed); logger.info(f"Seed: {seed}"); cfg["experiment_setup"]["seed_runtime"] = seed

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg_main = cfg.get("paths", {})
    proj_root = paths_cfg_main.get("project_root")
    base_path = Path(proj_root).resolve() if proj_root and Path(proj_root).is_dir() else cfg_path.parent

    fold_id_path_str = str(args.fold_id_to_run).replace(" ", "_").replace("/", "-")
    # Use default names if specific keys are missing, as per _get_path_from_config logic
    log_dir_base = _get_path_from_config(cfg, "log_dir", default=f"outputs/tb_{args.exp_name}_meta", base_path=base_path)
    ckpt_dir_base = _get_path_from_config(cfg, "ckpt_dir", default=f"outputs/ckpt_{args.exp_name}_meta", base_path=base_path)
    run_log_dir_fold = log_dir_base / args.exp_name / f"fold_{fold_id_path_str}" / ts # Changed to fold_
    run_ckpt_dir_fold = ckpt_dir_base / args.exp_name / f"fold_{fold_id_path_str}" / ts # Changed to fold_
    run_log_dir_fold.mkdir(parents=True, exist_ok=True); run_ckpt_dir_fold.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs: {run_log_dir_fold}"); logger.info(f"Ckpts: {run_ckpt_dir_fold}")

    labels_csv_file = _get_path_from_config(cfg, "labels_csv", base_path=base_path)
    meta_csv_file = _get_path_from_config(cfg, "meta_csv", base_path=base_path) # Ensure this key is in config
    img_root_dir = _get_path_from_config(cfg, "train_root", base_path=base_path) # Or a more generic "image_root"
    df_labels_all = pd.read_csv(labels_csv_file); df_meta_all = pd.read_csv(meta_csv_file)

    train_cfg = cfg.get("training", {})
    known_lbls_cfg = cfg.get("data",{}).get("known_training_labels")
    exclude_unk_train = train_cfg.get("exclude_unk_from_training_model", False)
    unk_lbl_str = train_cfg.get("unk_label_string", "UNK")
    df_for_model_cls_map = df_labels_all.copy()

    if known_lbls_cfg and isinstance(known_lbls_cfg, list):
        logger.info(f"Model classes from 'known_training_labels': {known_lbls_cfg}")
        df_for_model_cls_map = df_labels_all[df_labels_all['label'].isin(known_lbls_cfg)].copy()
    elif exclude_unk_train and unk_lbl_str:
        logger.info(f"Excluding '{unk_lbl_str}' from model training classes.")
        df_for_model_cls_map = df_labels_all[df_labels_all['label'] != unk_lbl_str].copy()

    unique_model_lbls = sorted(df_for_model_cls_map['label'].unique())
    if not unique_model_lbls: raise ValueError("No model training labels after filtering.")
    label2idx_model_runtime = {name: i for i, name in enumerate(unique_model_lbls)}


    cfg['label2idx_model_runtime'] = label2idx_model_runtime
    logger.info(f"Model trains on {len(label2idx_model_runtime)} classes: {label2idx_model_runtime}")

    unique_eval_lbls = sorted(df_labels_all['label'].unique())
    label2idx_eval_runtime = {name: i for i, name in enumerate(unique_eval_lbls)}
    cfg['label2idx_eval_runtime'] = label2idx_eval_runtime
    logger.info(f"Evaluation uses {len(label2idx_eval_runtime)} classes: {label2idx_eval_runtime}")

    dev_default = "cpu"; dev_str_cfg = exp_cfg.get("device", dev_default) # type: ignore
    if torch.cuda.is_available(): dev_default = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): dev_default = "mps" # type: ignore
    final_dev_str = exp_cfg.get("device", dev_default) # type: ignore
    device = get_device(final_dev_str); logger.info(f"Device: {device}")
    cfg["experiment_setup"]["device_runtime"] = str(device)

    fold_id_arg = args.fold_id_to_run
    fold_col_is_numeric = pd.api.types.is_numeric_dtype(df_labels_all['fold'])
    if fold_col_is_numeric:
        try: fold_id_arg = int(args.fold_id_to_run) # type: ignore
        except ValueError: logger.error(f"Fold ID '{args.fold_id_to_run}' not int for numeric 'fold'."); return

    if fold_id_arg not in df_labels_all['fold'].unique():
        logger.error(f"Fold ID '{fold_id_arg}' not in CSV. Available: {df_labels_all['fold'].unique()}."); return

    train_df_selected_fold = df_for_model_cls_map[df_for_model_cls_map['fold'] != fold_id_arg].reset_index(drop=True)
    val_df_selected_fold = df_labels_all[df_labels_all['fold'] == fold_id_arg].reset_index(drop=True)

    if train_df_selected_fold.empty or val_df_selected_fold.empty:
        logger.error(f"Fold {fold_id_arg}: Empty train ({len(train_df_selected_fold)}) or val ({len(val_df_selected_fold)})."); return
    logger.info(f"Running fold '{fold_id_arg}': Train samples={len(train_df_selected_fold)}, Val samples={len(val_df_selected_fold)}")

    best_metric_val_run = train_one_fold_with_meta(
        fold_run_id=fold_id_arg, # type: ignore
        train_df_fold_selected=train_df_selected_fold,
        val_df_fold_selected=val_df_selected_fold,
        meta_df_full_data=df_meta_all,
        cfg=cfg,
        label2idx_model_map_runtime=label2idx_model_runtime,
        label2idx_eval_map_runtime=label2idx_eval_runtime,
        image_root_path=img_root_dir,
        run_log_dir_main_fold=run_log_dir_fold,
        run_ckpt_dir_main_fold=run_ckpt_dir_fold,
        exp_name_cli_arg=args.exp_name,
        device_obj_runtime=device
    )

    if best_metric_val_run is not None:
        logger.info(f"\n===== Single Fold Run Finished (ID: {args.fold_id_to_run}) =====")
        logger.info(f"Best {cfg.get('training',{}).get('model_selection_metric', 'N/A')}: {best_metric_val_run:.4f}")
    else: logger.warning(f"Single fold run '{args.fold_id_to_run}' did not produce a best metric value.")
    logger.info(f"Experiment {args.exp_name} (run for ID {args.fold_id_to_run}, timestamp: {ts}) finished.")


if __name__ == "__main__":
    main()
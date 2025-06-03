#!/usr/bin/env python
# your_project_name/train_single_meta_effb3_advanced.py
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import cv2 # For Grad-CAM visualization

# --- Grad-CAM Imports ---
try:
    from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image #, preprocess_image # preprocess_image not strictly needed if using original image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("WARNING: pytorch-grad-cam not installed. Grad-CAM functionality will be disabled.")
    print("Install with: pip install grad-cam")
    # Define placeholders if not available to prevent NameErrors later if logic tries to call them
    GradCAM = HiResCAM = GradCAMPlusPlus = EigenCAM = EigenGradCAM = LayerCAM = object 

from data_handling import (
    FlatDatasetWithMeta, build_transform,
    ClassBalancedSampler
)
from models import get_model as get_base_cnn_model, CNNWithMetadata
from losses import focal_ce_loss, LDAMLoss
from utils import (
    set_seed, load_config, cast_config_values,
    update_ema,
    get_device, CudaTimer,
    TensorBoardLogger,
    generate_confusion_matrix_figure # Make sure this is in utils.plot_utils and exported
)
from PIL import Image # For loading original image for Grad-CAM

try:
    from torch.profiler import ProfilerActivity # type: ignore
except ImportError:
    ProfilerActivity = None # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__)

if 'LDAMLoss' not in globals():
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

# --- Grad-CAM Helper ---
def generate_grad_cam_visualizations(
    model_to_use: CNNWithMetadata,
    cfg: dict,
    val_ds_obj: FlatDatasetWithMeta,
    device_obj: torch.device,
    label2idx_model_map: dict,
    label2idx_eval_map: dict,
    tb_logger: TensorBoardLogger,
    current_global_epoch: int, 
    fold_run_id_str: str,
    exp_name_for_files: str,
    output_subfolder_name: str = "intermediate_cams",
    all_val_preds_eval_space: torch.Tensor | None = None,
    all_val_true_labels_eval_space: torch.Tensor | None = None
):
    if not GRAD_CAM_AVAILABLE:
        logger.warning("pytorch-grad-cam is not available. Skipping Grad-CAM generation.")
        return

    grad_cam_cfg = cfg.get("grad_cam", {})
    if not grad_cam_cfg.get("enable", False):
        logger.info("Grad-CAM is disabled in configuration. Skipping.")
        return

    grad_cam_output_dir_base = _get_path_from_config(cfg, "grad_cam_output_dir", default=f"outputs/grad_cam_{exp_name_for_files}", base_path=Path(cfg.get("paths",{}).get("project_root", ".")))
    grad_cam_output_dir_specific = grad_cam_output_dir_base / f"fold_{fold_run_id_str}" / output_subfolder_name
    if output_subfolder_name == "intermediate_cams": 
        grad_cam_output_dir_specific = grad_cam_output_dir_specific / f"epoch_{current_global_epoch}"
    grad_cam_output_dir_specific.mkdir(parents=True, exist_ok=True)

    target_classes_str = grad_cam_cfg.get("target_classes", [])
    num_images_per_class = grad_cam_cfg.get("num_images_per_class", 1)
    target_layer_name_base = grad_cam_cfg.get("target_layer_name_base")

    target_layers_gc = []
    if target_layer_name_base:
        try:
            current_module = model_to_use 
            for part in target_layer_name_base.split('.'):
                if part.endswith(']'): 
                    mod_name, index_str = part[:-1].split('[')
                    index = int(index_str)
                    current_module = getattr(current_module, mod_name)
                    if isinstance(current_module, (torch.nn.ModuleList, torch.nn.Sequential)):
                        current_module = current_module[index]
                    else:
                        logger.warning(f"Attempting indexed access on non-list/sequential module part: {mod_name} for {part}. May fail if not supported.")
                        current_module = current_module[index] 
                else:
                    current_module = getattr(current_module, part)
            target_layers_gc = [current_module]
            logger.info(f"Grad-CAM target layer for '{output_subfolder_name}' at E{current_global_epoch} found: {target_layer_name_base} -> {type(current_module)}")
        except Exception as e_gc_layer:
            logger.error(f"Could not find Grad-CAM target layer '{target_layer_name_base}' for E{current_global_epoch}: {e_gc_layer}", exc_info=True)
            return
    if not target_layers_gc:
        logger.warning(f"Grad-CAM: No target layers specified or found for E{current_global_epoch}. Skipping.")
        return

    cam_algorithm_choice = grad_cam_cfg.get("grad_cam_method", "EigenCAM")
    cam_method_map = {"GradCAM": GradCAM, "HiResCAM": HiResCAM, "GradCAMPlusPlus": GradCAMPlusPlus,
                      "EigenCAM": EigenCAM, "EigenGradCAM": EigenGradCAM, "LayerCAM": LayerCAM}
    SelectedCamAlgorithm = cam_method_map.get(cam_algorithm_choice, EigenCAM)
    if SelectedCamAlgorithm is object: 
        logger.error(f"Selected Grad-CAM algorithm '{cam_algorithm_choice}' is not available or not a valid CAM class. Skipping.")
        return

    idx2label_eval_map = {v: k for k,v in label2idx_eval_map.items()}
    model_to_use.eval() 

    with SelectedCamAlgorithm(model=model_to_use, target_layers=target_layers_gc) as cam:
        cam.batch_size = 1 

        for target_cls_str in target_classes_str:
            target_cls_model_idx = label2idx_model_map.get(target_cls_str)
            target_cls_eval_idx = label2idx_eval_map.get(target_cls_str)

            if target_cls_model_idx is None: 
                logger.warning(f"Grad-CAM (E{current_global_epoch}): Target class '{target_cls_str}' not in model's label map (label2idx_model_map). Skipping.")
                continue
            if target_cls_eval_idx is None: 
                logger.warning(f"Grad-CAM (E{current_global_epoch}): Target class '{target_cls_str}' not in evaluation label map (label2idx_eval_map). Skipping sample search for this class.")
                continue

            candidate_indices = []
            if all_val_preds_eval_space is not None and all_val_true_labels_eval_space is not None:
                # Find true positives for the specific class
                is_target_class_true = (all_val_true_labels_eval_space == target_cls_eval_idx)
                is_target_class_pred = (all_val_preds_eval_space == target_cls_eval_idx)
                true_positive_mask = is_target_class_true & is_target_class_pred
                candidate_indices = torch.where(true_positive_mask)[0].tolist()
                logger.info(f"Grad-CAM for '{target_cls_str}' (best model selection): Found {len(candidate_indices)} true positive samples from validation set.")
            else:
                # Find all instances of the class in the validation set
                candidate_indices = [i for i, sample_info in enumerate(val_ds_obj.samples) if val_ds_obj.get_label_for_idx(i) == target_cls_eval_idx] # Assuming get_label_for_idx returns eval_idx
                logger.info(f"Grad-CAM for '{target_cls_str}' (epoch {current_global_epoch}): Found {len(candidate_indices)} true instances in validation dataset.")


            if not candidate_indices:
                logger.info(f"Grad-CAM (E{current_global_epoch}): No suitable samples found for class '{target_cls_str}'. Skipping.")
                continue

            actual_num_images_to_vis = min(num_images_per_class, len(candidate_indices))
            if actual_num_images_to_vis == 0:
                 logger.info(f"Grad-CAM (E{current_global_epoch}): Not enough samples or num_images_per_class is 0 for '{target_cls_str}'. Skipping visualization.")
                 continue

            selected_indices_in_val_ds = np.random.choice(candidate_indices, size=actual_num_images_to_vis, replace=False)

            for i_img_vis, sample_idx_in_val_ds in enumerate(selected_indices_in_val_ds):
                try:
                    dataset_sample = val_ds_obj[sample_idx_in_val_ds] # This gives ((img, meta), label_eval_idx)
                    (input_img_tensor_gc, input_meta_tensor_gc), _ = dataset_sample 

                    input_img_tensor_gc = input_img_tensor_gc.unsqueeze(0).to(device_obj)
                    input_meta_tensor_gc = input_meta_tensor_gc.unsqueeze(0).to(device_obj)

                    original_img_path = val_ds_obj.samples[sample_idx_in_val_ds]["path"]
                    original_pil_img = Image.open(original_img_path).convert("RGB")

                    target_h, target_w = input_img_tensor_gc.shape[-2], input_img_tensor_gc.shape[-1]
                    vis_image_pil_resized = original_pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    vis_image_np_resized = np.array(vis_image_pil_resized) / 255.0 # HWC, float [0,1]

                    cam_targets = [ClassifierOutputTarget(target_cls_model_idx)]
                    
                    grayscale_cam_frames = cam(input_tensor=(input_img_tensor_gc, input_meta_tensor_gc), targets=cam_targets)

                    if grayscale_cam_frames is None or grayscale_cam_frames.shape[0] == 0:
                        logger.warning(f"Grad-CAM returned no frames for sample_idx {sample_idx_in_val_ds} (orig path: {original_img_path}). Skipping this image.")
                        continue
                    grayscale_cam_frame = grayscale_cam_frames[0, :] # Take the first (and only, for batch_size=1) CAM

                    cam_image_overlay = show_cam_on_image(vis_image_np_resized, grayscale_cam_frame, use_rgb=True) # Returns HWC, uint8 [0,255]

                    img_save_name_prefix = "BESTMODEL" if output_subfolder_name == "BEST_MODEL_CAMS" else f"epoch{current_global_epoch}"
                    out_fname = grad_cam_output_dir_specific / f"{img_save_name_prefix}_{target_cls_str.replace('/','_')}_sample{i_img_vis}_idx{sample_idx_in_val_ds}.jpg"
                    cv2.imwrite(str(out_fname), cv2.cvtColor(cam_image_overlay, cv2.COLOR_RGB2BGR)) # cv2 expects BGR
                    logger.info(f"Saved Grad-CAM to {out_fname}")

                    if tb_logger and tb_logger.writer:
                        tb_tag_suffix = "BestModel" if output_subfolder_name == "BEST_MODEL_CAMS" else f"Epoch{current_global_epoch}"
                        # Convert HWC uint8 to CHW uint8 for add_image
                        tb_image_tensor = torch.from_numpy(cam_image_overlay).permute(2, 0, 1) 
                        tb_logger.writer.add_image(f"GradCAM_{tb_tag_suffix}/{target_cls_str}_Sample{i_img_vis}", tb_image_tensor, global_step=current_global_epoch)

                except Exception as e_gc_img:
                    logger.error(f"Error generating Grad-CAM for sample_idx {sample_idx_in_val_ds} (class {target_cls_str}): {e_gc_img}", exc_info=True)


# --- CORE TRAINING AND VALIDATION FUNCTION FOR A SINGLE PHASE (with DRW for LDAM) ---
def run_training_phase(
    phase_name: str, model: CNNWithMetadata,
    train_ld: DataLoader, val_ld: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler,
    device: torch.device, cfg: dict, phase_cfg: dict, fold_run_id_str: str,
    label2idx_model_map_phase: dict[str, int], label2idx_eval_map_phase: dict[str, int],
    tb_logger: TensorBoardLogger, run_ckpt_dir_phase: Path, exp_name_for_files_phase: str,
    start_epoch: int = 0, ema_model_phase: CNNWithMetadata | None = None,
    initial_best_metric_val_phase: float = -float('inf'),
    class_counts_for_ldam_drw: np.ndarray | None = None
) -> tuple[float, int, str, dict | None]:

    main_training_cfg = cfg.get("training", {})
    loss_params_main_cfg = main_training_cfg.get("loss", {})
    loss_type_for_criterion = loss_params_main_cfg.get("type", "cross_entropy").lower()
    grad_cam_cfg_main = cfg.get("grad_cam", {}) # Get main Grad-CAM config
    log_at_global_epochs_gc = grad_cam_cfg_main.get("log_at_global_epochs", [])


    amp_enabled_phase = main_training_cfg.get("amp_enabled", True)
    ema_decay_phase = main_training_cfg.get("ema_decay", 0.0)
    use_ema_for_val_phase = main_training_cfg.get("use_ema_for_val", True)
    model_selection_metric_name_phase = main_training_cfg.get("model_selection_metric", "macro_auc").lower()
    unk_label_string_cfg_phase = main_training_cfg.get("unk_label_string", "UNK")
    post_hoc_unk_thresh_cfg_phase = main_training_cfg.get("post_hoc_unk_threshold")
    val_metrics_heavy_interval_phase = main_training_cfg.get("val_metrics_heavy_interval", 99999)

    num_epochs_this_phase = phase_cfg.get("num_epochs", 10)
    accum_steps_phase = phase_cfg.get("accum_steps", main_training_cfg.get("accum_steps", 1))
    early_stopping_patience_phase = phase_cfg.get("early_stopping_patience", main_training_cfg.get("early_stopping_patience", 10))

    drw_schedule_epoch_list = main_training_cfg.get("drw_schedule_epochs", [])
    current_drw_stage_val = 0

    known_model_indices_phase = [idx for lbl, idx in label2idx_model_map_phase.items() if lbl != unk_label_string_cfg_phase and idx != label2idx_model_map_phase.get(unk_label_string_cfg_phase, -1)]
    unk_eval_assign_idx_phase = label2idx_eval_map_phase.get(unk_label_string_cfg_phase, -1)
    idx2label_model_phase = {v: k for k, v in label2idx_model_map_phase.items()}
    idx2label_eval_phase = {v: k for k, v in label2idx_eval_map_phase.items()}

    post_hoc_active_this_phase = (post_hoc_unk_thresh_cfg_phase is not None and
                                  bool(known_model_indices_phase) and
                                  unk_eval_assign_idx_phase != -1 and
                                  unk_label_string_cfg_phase in label2idx_eval_map_phase) 
    if post_hoc_unk_thresh_cfg_phase is not None:
        logger.info(f"{phase_name} F{fold_run_id_str}: Post-hoc '{unk_label_string_cfg_phase}' (Th={post_hoc_unk_thresh_cfg_phase}). Active: {post_hoc_active_this_phase}")

    best_metric_val_current_phase = initial_best_metric_val_phase
    best_epoch_for_metric_current_phase = -1
    best_ckpt_path_current_phase = ""
    optimal_thresholds_best_ckpt_current_phase = None
    patience_counter_current_phase = 0

    num_model_classes_this_phase = len(label2idx_model_map_phase)
    focal_alpha_cfg = loss_params_main_cfg.get("focal_alpha",1.0)
    focal_gamma_cfg = loss_params_main_cfg.get("focal_gamma",2.0)
    epoch_global_phase = start_epoch

    logger.info(f"Starting {phase_name} F{fold_run_id_str}: {num_epochs_this_phase} epochs (Global Epochs {start_epoch}-{start_epoch + num_epochs_this_phase -1}).")
    for epoch_offset_phase in range(num_epochs_this_phase):
        epoch_global_phase = start_epoch + epoch_offset_phase

        if loss_type_for_criterion == "ldam_loss" and isinstance(criterion, LDAMLoss) and drw_schedule_epoch_list and class_counts_for_ldam_drw is not None:
            if current_drw_stage_val < len(drw_schedule_epoch_list) and epoch_global_phase >= drw_schedule_epoch_list[current_drw_stage_val]:
                beta_drw = loss_params_main_cfg.get("ldam_params", {}).get("effective_number_beta", 0.999)
                if np.any(class_counts_for_ldam_drw == 0) and num_model_classes_this_phase > 0:
                    drw_weights_np = np.ones(num_model_classes_this_phase, dtype=float)
                elif num_model_classes_this_phase > 0:
                    effective_num_drw = 1.0 - np.power(beta_drw, class_counts_for_ldam_drw)
                    drw_weights_np = (1.0 - beta_drw) / np.maximum(effective_num_drw, 1e-8)
                    drw_weights_np = drw_weights_np / np.sum(drw_weights_np) * num_model_classes_this_phase
                else: drw_weights_np = np.array([])
                if drw_weights_np.size > 0:
                    drw_weights = torch.tensor(drw_weights_np, dtype=torch.float, device=device)
                    criterion.update_weights(drw_weights)
                    logger.info(f"DRW ({phase_name}) E{epoch_global_phase}: Updated LDAMLoss weights (Stage {current_drw_stage_val + 1}). Weights (first 5): {drw_weights_np[:5]}")
                current_drw_stage_val += 1
            elif epoch_global_phase == start_epoch and current_drw_stage_val == 0 and not drw_schedule_epoch_list: # if no DRW schedule, set initial weights once
                 criterion.update_weights(None) # Default: equal weights or no re-weighting based on LDAMLoss internal
                 logger.info(f"DRW ({phase_name}): Initial LDAM weights set for E{epoch_global_phase} (no DRW schedule or pre-DRW stage).")


        if hasattr(train_ld.sampler, 'set_epoch') and train_ld.sampler is not None:
            train_ld.sampler.set_epoch(epoch_global_phase) # type: ignore

        active_profiler_phase = tb_logger.setup_profiler(epoch_global_phase, Path(tb_logger.writer.log_dir if tb_logger.writer else run_ckpt_dir_phase)) # type: ignore
        model.train()
        epoch_gpu_time_ms_phase, epoch_start_time_phase = 0.0, time.time()
        train_loss_sum_pbar, train_corrects_sum_pbar, train_samples_sum_pbar = 0.0,0,0
        optimizer.zero_grad()
        train_pbar_desc_phase = f"F{fold_run_id_str} E{epoch_global_phase} {phase_name} Train"
        train_pbar_phase = tqdm_iterator(train_ld, desc=train_pbar_desc_phase, ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",120), leave=False)
        for batch_idx_train_ph, (inputs_train_ph, labels_train_cpu_ph) in enumerate(train_pbar_phase):
            img_train_cpu_ph, meta_train_cpu_ph = inputs_train_ph
            img_train_dev_ph = img_train_cpu_ph.to(device, non_blocking=True)
            meta_train_dev_ph = meta_train_cpu_ph.to(device, non_blocking=True)
            labels_train_dev_ph = labels_train_cpu_ph.to(device, non_blocking=True)
            batch_gpu_time_ph, timer_active_ph = 0.0, active_profiler_phase is not None or \
                (cfg.get("tensorboard_logging",{}).get("profiler",{}).get("enable_batch_timing_always",False) and \
                 tb_logger._should_log_batch(tb_logger.log_interval_train_batch, batch_idx_train_ph))
            timer_ctx_ph = CudaTimer(device) if timer_active_ph and device.type=='cuda' else contextlib.nullcontext()
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
            batch_loss_val_ph = loss_train_ph.item() * (accum_steps_phase if accum_steps_phase > 1 else 1)
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
        if active_profiler_phase: tb_logger.stop_and_process_profiler(); active_profiler_phase = None

        val_interval_this_phase = phase_cfg.get("val_interval", main_training_cfg.get("val_interval",1))
        is_last_epoch_of_phase = epoch_offset_phase == (num_epochs_this_phase - 1)
        
        if epoch_global_phase % val_interval_this_phase == 0 or is_last_epoch_of_phase:
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
                            if loss_type_for_criterion == "focal_ce_loss": loss_v_b_ph = criterion(logits_for_loss, labels_for_loss, alpha=focal_alpha_cfg, gamma=focal_gamma_cfg)
                            else: loss_v_b_ph = criterion(logits_for_loss, labels_for_loss)
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
                            if len(known_model_indices_phase) > 0 and probs_v_b_cpu_ph[i][known_model_indices_phase].max().item() < post_hoc_unk_thresh_cfg_phase: final_preds_eval_b_tqdm_ph[i]=unk_eval_assign_idx_phase
                            elif not known_model_indices_phase and post_hoc_unk_thresh_cfg_phase is not None:
                                if probs_v_b_cpu_ph[i].max().item() < post_hoc_unk_thresh_cfg_phase: final_preds_eval_b_tqdm_ph[i] = unk_eval_assign_idx_phase
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
                if true_lbl_s_ph and true_lbl_s_ph in label2idx_model_map_phase: true_lbls_known_model_list_ph.append(label2idx_model_map_phase[true_lbl_s_ph]); known_f1_mask_ph[i]=True
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
                    if len(known_model_indices_phase) > 0 and all_val_probs_cat_cpu_ph[i][known_model_indices_phase].max().item() < post_hoc_unk_thresh_cfg_phase: final_preds_eval_epoch_ph[i]=unk_eval_assign_idx_phase
                    elif not known_model_indices_phase and post_hoc_unk_thresh_cfg_phase is not None :
                         if all_val_probs_cat_cpu_ph[i].max().item() < post_hoc_unk_thresh_cfg_phase: final_preds_eval_epoch_ph[i] = unk_eval_assign_idx_phase
            current_epoch_metrics_val_dict_phase[f"Accuracy/val_epoch_post_hoc_unk_{phase_name}"] = (final_preds_eval_epoch_ph == all_val_true_labels_eval_cat_ph).sum().item() / epoch_val_seen_samples_ph if epoch_val_seen_samples_ph > 0 else 0.0
            current_epoch_metrics_val_dict_phase[f"F1Score/val_macro_post_hoc_unk_{phase_name}"] = F1Score(task="multiclass", num_classes=len(label2idx_eval_map_phase), average="macro")(final_preds_eval_epoch_ph, all_val_true_labels_eval_cat_ph).item()
            true_labels_auroc_model_list_ph, auroc_sample_mask_ph = [], torch.zeros(all_val_true_labels_eval_cat_ph.size(0),dtype=torch.bool)
            for i,eval_idx_s_ph in enumerate(all_val_true_labels_eval_cat_ph.tolist()):
                lbl_s_ph=idx2label_eval_phase.get(eval_idx_s_ph);
                if lbl_s_ph and lbl_s_ph in label2idx_model_map_phase: true_labels_auroc_model_list_ph.append(label2idx_model_map_phase[lbl_s_ph]); auroc_sample_mask_ph[i]=True
            if true_labels_auroc_model_list_ph and auroc_sample_mask_ph.any():
                true_lbls_auroc_tensor_ph = torch.tensor(true_labels_auroc_model_list_ph,dtype=torch.long)
                probs_auroc_ph = all_val_probs_cat_cpu_ph[auroc_sample_mask_ph]
                if probs_auroc_ph.size(0)==true_lbls_auroc_tensor_ph.size(0) and probs_auroc_ph.size(0)>0:
                    current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=AUROC(task="multiclass",num_classes=num_model_classes_this_phase,average="macro")(probs_auroc_ph,true_lbls_auroc_tensor_ph).item()
                    pauc_max_fpr_cfg = main_training_cfg.get("pauc_max_fpr", 0.2)
                    current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{pauc_max_fpr_cfg}_{phase_name}"] = AUROC(task="multiclass",num_classes=num_model_classes_this_phase,average="macro", max_fpr=pauc_max_fpr_cfg)(probs_auroc_ph, true_lbls_auroc_tensor_ph).item()
                else: current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=0.0; current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{main_training_cfg.get('pauc_max_fpr', 0.2)}_{phase_name}"]=0.0
            else: current_epoch_metrics_val_dict_phase[f"AUROC/val_macro_{phase_name}"]=0.0; current_epoch_metrics_val_dict_phase[f"pAUROC/val_macro_fpr{main_training_cfg.get('pauc_max_fpr', 0.2)}_{phase_name}"]=0.0
            optimal_thresholds_for_saving_epoch_ph = {}
            heavy_metrics_were_run_ph = False
            if epoch_global_phase % val_metrics_heavy_interval_phase == 0 or is_last_epoch_of_phase: 
                heavy_metrics_were_run_ph = True; logger.info(f"F{fold_run_id_str} E{epoch_global_phase} {phase_name}: Calculating HEAVY validation metrics (PR-curves).")
                if num_model_classes_this_phase > 1:
                    pr_true_model_list_ph, pr_mask_ph = [], torch.zeros(all_val_true_labels_eval_cat_ph.size(0),dtype=torch.bool)
                    for i,eval_idx_s_ph in enumerate(all_val_true_labels_eval_cat_ph.tolist()):
                        lbl_s_ph=idx2label_eval_phase.get(eval_idx_s_ph)
                        if lbl_s_ph and lbl_s_ph in label2idx_model_map_phase: pr_true_model_list_ph.append(label2idx_model_map_phase[lbl_s_ph]); pr_mask_ph[i]=True
                    if pr_true_model_list_ph:
                        pr_true_tensor_ph = torch.tensor(pr_true_model_list_ph,dtype=torch.long)
                        pr_probs_np_ph = all_val_probs_cat_cpu_ph[pr_mask_ph].numpy()
                        pr_lbls_oh_np_ph = F.one_hot(pr_true_tensor_ph, num_classes=num_model_classes_this_phase).numpy() # Added num_classes
                        opt_f1s_ph, opt_sens_ph = [],[]
                        for cls_i_ph in range(num_model_classes_this_phase):
                            f1_ph,thr_ph,sens_ph = 0.0,0.5,0.0
                            try:
                                p_rc_ph,r_rc_ph,t_rc_ph = precision_recall_curve(pr_lbls_oh_np_ph[:,cls_i_ph],pr_probs_np_ph[:,cls_i_ph])
                                if len(p_rc_ph)>1 and len(r_rc_ph)>1 and len(t_rc_ph)>0:
                                    f1_curve_ph=(2*p_rc_ph*r_rc_ph)/(p_rc_ph+r_rc_ph+1e-8); rel_f1_ph=f1_curve_ph[1:]; rel_r_ph=r_rc_ph[1:]
                                    valid_idx_ph=np.where(np.isfinite(rel_f1_ph)&(p_rc_ph[1:]+r_rc_ph[1:]>0))[0]
                                    if len(valid_idx_ph)>0: best_i_ph=valid_idx_ph[np.argmax(rel_f1_ph[valid_idx_ph])]; f1_ph,thr_ph,sens_ph=float(rel_f1_ph[best_i_ph]),float(t_rc_ph[best_i_ph]),float(rel_r_ph[best_i_ph])
                            except Exception as e_pr_ph: logger.warning(f"PR curve failed for {phase_name} cls {cls_i_ph} E{epoch_global_phase}: {e_pr_ph}")
                            opt_f1s_ph.append(f1_ph); opt_sens_ph.append(sens_ph); optimal_thresholds_for_saving_epoch_ph[cls_i_ph]=thr_ph
                        if opt_f1s_ph: current_epoch_metrics_val_dict_phase[f"F1Score/val_mean_optimal_per_class_from_PR_{phase_name}"]=np.mean(opt_f1s_ph)
                        if opt_sens_ph: current_epoch_metrics_val_dict_phase[f"Sensitivity/val_mean_optimal_per_class_from_PR_{phase_name}"]=np.mean(opt_sens_ph)
                    else: logger.warning(f"{phase_name} PR Curves: No valid samples after filtering for model's known classes.")
                else: logger.warning(f"{phase_name} PR Curves: Not enough model output classes ({num_model_classes_this_phase}) for PR curves.")
            current_epoch_metrics_combined_phase.update(current_epoch_metrics_val_dict_phase)
            metric_key_map_for_phase = {"macro_auc": f"AUROC/val_macro_{phase_name}", "f1_macro_known_classes": f"F1Score/val_macro_known_classes_{phase_name}", "f1_macro_post_hoc_unk": f"F1Score/val_macro_post_hoc_unk_{phase_name}", "accuracy_post_hoc_unk": f"Accuracy/val_epoch_post_hoc_unk_{phase_name}", "mean_optimal_f1": f"F1Score/val_mean_optimal_per_class_from_PR_{phase_name}", "mean_optimal_sensitivity": f"Sensitivity/val_mean_optimal_per_class_from_PR_{phase_name}"}
            primary_metric_key_for_phase = metric_key_map_for_phase.get(model_selection_metric_name_phase, f"AUROC/val_macro_{phase_name}")
            current_primary_metric_val_phase = current_epoch_metrics_val_dict_phase.get(primary_metric_key_for_phase, -float('inf'))
            log_line_ph = (f"F{fold_run_id_str} E{epoch_global_phase} {phase_name} Val -> Loss={current_epoch_metrics_val_dict_phase.get(f'Loss/val_epoch_{phase_name}', -1.0):.4f} AccOrig={current_epoch_metrics_val_dict_phase.get(f'Accuracy/val_epoch_original_argmax_{phase_name}', -1.0):.4f} ")
            if post_hoc_active_this_phase: log_line_ph += f"AccPostHoc={current_epoch_metrics_val_dict_phase.get(f'Accuracy/val_epoch_post_hoc_unk_{phase_name}', -1.0):.4f} "
            log_line_ph += f"Selected ({model_selection_metric_name_phase}->{primary_metric_key_for_phase.replace(f'_{phase_name}','') if primary_metric_key_for_phase else 'N/A'}): {current_primary_metric_val_phase:.4f}"
            logger.info(log_line_ph)
            if current_primary_metric_val_phase > best_metric_val_current_phase:
                best_metric_val_current_phase = current_primary_metric_val_phase; best_epoch_for_metric_current_phase = epoch_global_phase; patience_counter_current_phase = 0
                best_ckpt_path_current_phase = str(run_ckpt_dir_phase / f"{exp_name_for_files_phase}_fold{fold_run_id_str}_{phase_name}_best_E{epoch_global_phase}.pt")
                logger.info(f"New best for {phase_name} ({model_selection_metric_name_phase}): {best_metric_val_current_phase:.4f}. Saving to {best_ckpt_path_current_phase}")
                ckpt_data_ph = {'epoch': epoch_global_phase, 'model_state_dict': getattr(model, '_orig_mod', model).state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'config_runtime': cfg, 'label2idx_model': label2idx_model_map_phase, 'label2idx_eval': label2idx_eval_map_phase, 'phase_name': phase_name, f'best_{model_selection_metric_name_phase}': best_metric_val_current_phase, 'metadata_dim': model.metadata_mlp.fc1.in_features if hasattr(model, 'metadata_mlp') and model.metadata_mlp else None } # Safegaurd metadata_dim
                if ema_model_phase: ckpt_data_ph['ema_model_state_dict'] = getattr(ema_model_phase, '_orig_mod', ema_model_phase).state_dict()
                if post_hoc_active_this_phase: ckpt_data_ph['post_hoc_unk_threshold_used'] = post_hoc_unk_thresh_cfg_phase
                save_pr_th_cfg = main_training_cfg.get("save_optimal_thresholds_from_pr", False); is_pr_metric = model_selection_metric_name_phase in ["mean_optimal_f1", "mean_optimal_sensitivity"]
                if (save_pr_th_cfg or is_pr_metric) and heavy_metrics_were_run_ph and optimal_thresholds_for_saving_epoch_ph:
                    ckpt_data_ph['optimal_thresholds_val_from_pr'] = optimal_thresholds_for_saving_epoch_ph; optimal_thresholds_best_ckpt_current_phase = optimal_thresholds_for_saving_epoch_ph
                    logger.info(f"Saved optimal PR thresholds with best {phase_name} checkpoint.")
                torch.save(ckpt_data_ph, best_ckpt_path_current_phase)
            else: patience_counter_current_phase += 1
            
            if patience_counter_current_phase >= early_stopping_patience_phase: 
                logger.info(f"Early stopping for {phase_name} at E{epoch_global_phase} (Patience: {early_stopping_patience_phase}).")
                break # Break from epoch loop
            
            # --- INTERMEDIATE Grad-CAM Logging (during training epochs) ---
            if GRAD_CAM_AVAILABLE and grad_cam_cfg_main.get("enable", False) and \
               epoch_global_phase in log_at_global_epochs_gc:
                logger.info(f"Generating INTERMEDIATE Grad-CAM overlays for epoch {epoch_global_phase} ({phase_name})...")
                # Ensure val_ld.dataset is FlatDatasetWithMeta
                val_dataset_for_cam = val_ld.dataset
                if not isinstance(val_dataset_for_cam, FlatDatasetWithMeta):
                    logger.warning(f"Validation dataset for Grad-CAM is not FlatDatasetWithMeta type. Type: {type(val_dataset_for_cam)}. Skipping Grad-CAM.")
                else:
                    generate_grad_cam_visualizations(
                        model_to_use=eval_model_to_use_phase,
                        cfg=cfg, 
                        val_ds_obj=val_dataset_for_cam, 
                        device_obj=device,
                        label2idx_model_map=label2idx_model_map_phase,
                        label2idx_eval_map=label2idx_eval_map_phase,
                        tb_logger=tb_logger,
                        current_global_epoch=epoch_global_phase,
                        fold_run_id_str=fold_run_id_str,
                        exp_name_for_files=exp_name_for_files_phase,
                        output_subfolder_name="intermediate_cams"
                        # all_val_preds_eval_space and all_val_true_labels_eval_space are None here, so it will pick random class samples
                    )

        tb_logger.log_epoch_summary(current_epoch_metrics_combined_phase, epoch_global_phase)
        if patience_counter_current_phase >= early_stopping_patience_phase: break # Ensure breaking from outer loop as well
    
    # Save last checkpoint if training completed or early stopped
    last_ckpt_path_phase = run_ckpt_dir_phase / f"{exp_name_for_files_phase}_fold{fold_run_id_str}_{phase_name}_last_E{epoch_global_phase}.pt"
    last_ckpt_data = {
        'epoch': epoch_global_phase,
        'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config_runtime': cfg,
        'label2idx_model': label2idx_model_map_phase,
        'label2idx_eval': label2idx_eval_map_phase,
        'phase_name': phase_name,
        'metadata_dim': model.metadata_mlp.fc1.in_features if hasattr(model, 'metadata_mlp') and model.metadata_mlp else None
    }
    if ema_model_phase:
        last_ckpt_data['ema_model_state_dict'] = getattr(ema_model_phase, '_orig_mod', ema_model_phase).state_dict()
    torch.save(last_ckpt_data, last_ckpt_path_phase)
    logger.info(f"Saved last checkpoint for {phase_name} to {last_ckpt_path_phase}")


    logger.info(f"Finished {phase_name} for F{fold_run_id_str}. Best {model_selection_metric_name_phase}: {best_metric_val_current_phase:.4f} at E{best_epoch_for_metric_current_phase}")
    return best_metric_val_current_phase, best_epoch_for_metric_current_phase, best_ckpt_path_current_phase, optimal_thresholds_best_ckpt_current_phase


# --- Main Training Orchestrator ---
def train_one_fold_with_meta(
    fold_run_id: int | str, train_df_fold_selected: pd.DataFrame,
    val_df_fold_selected: pd.DataFrame, meta_df_full_data: pd.DataFrame,
    cfg: dict, label2idx_model_map_runtime: dict[str,int], label2idx_eval_map_runtime: dict[str,int],
    image_root_path: Path, run_log_dir_main_fold: Path, run_ckpt_dir_main_fold: Path,
    exp_name_cli_arg: str, device_obj_runtime: torch.device,
) -> float | None:

    fold_run_id_str_main = str(fold_run_id)
    main_train_cfg_fold = cfg.get("training", {})
    meta_tune_cfg_fold = cfg.get("meta_tuning", {})
    data_cfg_fold = cfg.get("data", {})
    model_arch_cfg_fold = cfg.get("model", {})
    grad_cam_main_cfg = cfg.get("grad_cam", {}) # Main Grad-CAM config for final eval
    
    _bs_tb_fold = main_train_cfg_fold.get("batch_size", 32)
    # Calculate based on actual train_df_fold_selected, not full df
    tb_train_len_fold = (len(train_df_fold_selected) + _bs_tb_fold -1) // _bs_tb_fold if _bs_tb_fold > 0 else 0

    try: tb_logger_fold = TensorBoardLogger(log_dir=run_log_dir_main_fold, experiment_config=cfg, train_loader_len=tb_train_len_fold)
    except Exception as e: logger.error(f"TBLogger init fail fold {fold_run_id_str_main}: {e}", exc_info=True); return None
    
    tf_train_fold = build_transform(data_cfg_fold.get("cpu_augmentations",{}), train=True)
    tf_val_fold = build_transform(data_cfg_fold.get("cpu_augmentations",{}), train=False)
    
    dataset_args_from_cfg = data_cfg_fold.get("dataset_args", {}).copy() # Use .copy()
    dataset_args_from_cfg.update({
        "meta_features_names": data_cfg_fold.get("meta_features_names"), 
        "meta_augmentation_p": data_cfg_fold.get("meta_augmentation_p", 0.0), 
        "meta_nan_fill_value": data_cfg_fold.get("meta_nan_fill_value", 0.0), 
        "image_loader": data_cfg_fold.get("image_loader", "pil"), 
        "enable_ram_cache": data_cfg_fold.get("enable_ram_cache", False)
    })

    train_ds_obj = FlatDatasetWithMeta(df=train_df_fold_selected, meta_df=meta_df_full_data, root=image_root_path, label2idx=label2idx_model_map_runtime, tf=tf_train_fold, **dataset_args_from_cfg)
    # For val_ds_obj, labels are in eval space. label2idx should be label2idx_eval_map_runtime
    val_ds_obj = FlatDatasetWithMeta(df=val_df_fold_selected, meta_df=meta_df_full_data, root=image_root_path, label2idx=label2idx_eval_map_runtime, tf=tf_val_fold, **dataset_args_from_cfg)
    train_ds_obj.training = True; val_ds_obj.training = False # Set training flag
    
    metadata_dim_runtime = train_ds_obj.metadata_dim
    logger.info(f"Fold {fold_run_id_str_main}: Runtime metadata dimension: {metadata_dim_runtime}")
    cfg['model']['metadata_input_dim_runtime'] = metadata_dim_runtime 
    
    base_cnn_params = {"MODEL_TYPE": model_arch_cfg_fold.get("base_cnn_type"), "numClasses": len(label2idx_model_map_runtime), "pretrained": model_arch_cfg_fold.get("pretrained_cnn", True)}
    base_cnn_instance = get_base_cnn_model(base_cnn_params)
    
    meta_head_args_cfg = model_arch_cfg_fold.get("meta_head_args", {}).copy()
    meta_head_args_cfg.pop('num_classes', None); meta_head_args_cfg.pop('metadata_input_dim', None)
    
    full_model_instance = CNNWithMetadata(base_cnn_model=base_cnn_instance, num_classes=len(label2idx_model_map_runtime), metadata_input_dim=metadata_dim_runtime, **meta_head_args_cfg).to(device_obj_runtime)
    
    ema_model_instance = None
    if main_train_cfg_fold.get("ema_decay", 0.0) > 0:
        ema_model_instance = copy.deepcopy(full_model_instance).to(device_obj_runtime)
        if ema_model_instance: [p.requires_grad_(False) for p in ema_model_instance.parameters()]
        
    class_counts_for_loss = get_class_counts(train_df_fold_selected, label2idx_model_map_runtime)
    criterion_main, loss_type_main = None, main_train_cfg_fold.get("loss", {}).get("type", "cross_entropy").lower()
    loss_params_cfg = main_train_cfg_fold.get("loss", {})
    
    if loss_type_main == "focal_ce_loss": 
        criterion_main = focal_ce_loss # This is a function, not a class instance
        logger.info(f"Using Focal CE Loss function.")
    elif loss_type_main == "cross_entropy": 
        criterion_main = nn.CrossEntropyLoss(); logger.info("Using nn.CrossEntropyLoss.")
    elif loss_type_main == "ldam_loss":
        ldam_specific_params = loss_params_cfg.get("ldam_params", {})
        criterion_main = LDAMLoss(class_counts=class_counts_for_loss, **ldam_specific_params).to(device_obj_runtime); logger.info(f"Using LDAM Loss with params: {ldam_specific_params}")
    else: 
        criterion_main = nn.CrossEntropyLoss(); logger.warning(f"Unknown loss type '{loss_type_main}', defaulting to nn.CrossEntropyLoss.")
        
    scaler_main = GradScaler(device_type=device_obj_runtime.type if device_obj_runtime.type in ['cuda','mps'] else 'cpu', enabled=(device_obj_runtime.type == 'cuda' and main_train_cfg_fold.get("amp_enabled", True))) # Corrected device_type usage
    
    overall_best_metric_for_fold = -float('inf'); path_to_overall_best_ckpt_fold = ""; final_optimal_thresholds_for_fold = None; final_epoch_completed_for_tb_table = 0; phase1_epochs_run = 0

    if main_train_cfg_fold.get("num_epochs", 0) > 0 :
        logger.info(f"===== Fold {fold_run_id_str_main}: Starting Phase 1: Joint Training =====")
        full_model_instance.set_base_cnn_trainable(True)
        
        dl_args_p1 = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": data_cfg_fold.get("pin_memory", True), "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if dl_args_p1["persistent_workers"] and dl_args_p1["num_workers"] > 0: dl_args_p1["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)
        
        train_sampler_p1_type = data_cfg_fold.get("sampler",{}).get("type")
        train_sampler_p1 = None
        if train_sampler_p1_type == "class_balanced_sqrt" and len(train_ds_obj) > 0 : # Ensure dataset not empty
             train_sampler_p1 = ClassBalancedSampler(train_ds_obj, len(train_ds_obj)) # Assumes train_ds_obj.get_labels() exists for sampler
        
        train_loader_p1 = DataLoader(train_ds_obj, batch_size=main_train_cfg_fold.get("batch_size",32), sampler=train_sampler_p1, shuffle=(train_sampler_p1 is None), **dl_args_p1) 
        val_loader_p1 = DataLoader(val_ds_obj, batch_size=main_train_cfg_fold.get("batch_size",32), shuffle=False, **dl_args_p1) 
        
        opt_cfg_p1 = main_train_cfg_fold.get("optimizer", {}); optimizer_p1 = AdamW(full_model_instance.parameters(), lr=opt_cfg_p1.get("lr",1e-3), weight_decay=opt_cfg_p1.get("weight_decay",1e-4))
        sched_cfg_p1 = main_train_cfg_fold.get("scheduler", {}); scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=main_train_cfg_fold.get("num_epochs",10), eta_min=sched_cfg_p1.get("min_lr",1e-6))
        
        best_metric_p1, best_epoch_p1, best_ckpt_p1_path, best_thresh_p1 = run_training_phase(
            phase_name="P1_Joint", model=full_model_instance, train_ld=train_loader_p1, val_ld=val_loader_p1, criterion=criterion_main, optimizer=optimizer_p1, scheduler=scheduler_p1, scaler=scaler_main, device=device_obj_runtime, cfg=cfg, phase_cfg=main_train_cfg_fold, fold_run_id_str=fold_run_id_str_main, label2idx_model_map_phase=label2idx_model_map_runtime, label2idx_eval_map_phase=label2idx_eval_map_runtime, tb_logger=tb_logger_fold, run_ckpt_dir_phase=run_ckpt_dir_main_fold, exp_name_for_files_phase=exp_name_cli_arg, start_epoch=0, ema_model_phase=ema_model_instance, initial_best_metric_val_phase=-float('inf'), class_counts_for_ldam_drw=class_counts_for_loss if loss_type_main == "ldam_loss" else None)
        
        phase1_epochs_run = best_epoch_p1 if best_epoch_p1 != -1 else (main_train_cfg_fold.get("num_epochs", 0) -1 if main_train_cfg_fold.get("num_epochs", 0) > 0 else 0) # Approximation if no best epoch found
        final_epoch_completed_for_tb_table = max(final_epoch_completed_for_tb_table, phase1_epochs_run)

        if best_metric_p1 > overall_best_metric_for_fold: 
            overall_best_metric_for_fold = best_metric_p1; path_to_overall_best_ckpt_fold = best_ckpt_p1_path; final_optimal_thresholds_for_fold = best_thresh_p1
        
        if meta_tune_cfg_fold.get("enable", False) and best_epoch_p1 != -1 and Path(best_ckpt_p1_path).exists():
            logger.info(f"Loading best P1 model (E{best_epoch_p1}) for P2: {best_ckpt_p1_path}")
            ckpt_p1 = torch.load(best_ckpt_p1_path, map_location=device_obj_runtime)
            full_model_instance.load_state_dict(ckpt_p1['model_state_dict'])
            if ema_model_instance and ckpt_p1.get('ema_model_state_dict'): ema_model_instance.load_state_dict(ckpt_p1['ema_model_state_dict'])
    else: logger.info(f"Fold {fold_run_id_str_main}: Phase 1 (Joint Training) skipped.")

    if meta_tune_cfg_fold.get("enable", False) and meta_tune_cfg_fold.get("num_epochs",0) > 0:
        logger.info(f"===== Fold {fold_run_id_str_main}: Starting Phase 2: Meta Head Fine-tuning =====")
        full_model_instance.set_base_cnn_trainable(False); logger.info("CNN backbone frozen for Phase 2.")
        
        dl_args_p2 = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": data_cfg_fold.get("pin_memory", True), "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if dl_args_p2["persistent_workers"] and dl_args_p2["num_workers"] > 0: dl_args_p2["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)
        
        train_sampler_p2_type = data_cfg_fold.get("sampler",{}).get("type") # P2 might use different sampler settings from config if specified under meta_tuning
        train_sampler_p2 = None
        if train_sampler_p2_type == "class_balanced_sqrt" and len(train_ds_obj) > 0:
             train_sampler_p2 = ClassBalancedSampler(train_ds_obj, len(train_ds_obj))
        
        train_loader_p2 = DataLoader(train_ds_obj, batch_size=meta_tune_cfg_fold.get("batch_size",32), sampler=train_sampler_p2, shuffle=(train_sampler_p2 is None), **dl_args_p2) 
        val_loader_p2 = DataLoader(val_ds_obj, batch_size=meta_tune_cfg_fold.get("batch_size",32), shuffle=False, **dl_args_p2) 
        
        params_to_tune_p2 = [p for p in full_model_instance.parameters() if p.requires_grad]
        if not params_to_tune_p2: logger.error("Phase 2: No parameters found for tuning. Skipping Phase 2.")
        else:
            logger.info(f"P2: Num params to tune: {sum(p.numel() for p in params_to_tune_p2)}")
            opt_cfg_p2 = meta_tune_cfg_fold.get("optimizer", {}); optimizer_p2 = AdamW(params_to_tune_p2, lr=opt_cfg_p2.get("lr",1e-4), weight_decay=opt_cfg_p2.get("weight_decay",1e-5))
            sched_cfg_p2 = meta_tune_cfg_fold.get("scheduler", {}); scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=meta_tune_cfg_fold.get("num_epochs",10), eta_min=sched_cfg_p2.get("min_lr",1e-7))
            
            best_metric_p2, best_epoch_p2, best_ckpt_p2_path, best_thresh_p2 = run_training_phase(
                phase_name="P2_MetaTune", model=full_model_instance, train_ld=train_loader_p2, val_ld=val_loader_p2, criterion=criterion_main, optimizer=optimizer_p2, scheduler=scheduler_p2, scaler=scaler_main, device=device_obj_runtime, cfg=cfg, phase_cfg=meta_tune_cfg_fold, fold_run_id_str=str(fold_run_id), label2idx_model_map_phase=label2idx_model_map_runtime, label2idx_eval_map_phase=label2idx_eval_map_runtime, tb_logger=tb_logger_fold, run_ckpt_dir_phase=run_ckpt_dir_main_fold, exp_name_for_files_phase=exp_name_cli_arg, start_epoch=phase1_epochs_run + 1, ema_model_phase=ema_model_instance, initial_best_metric_val_phase=-float('inf'), class_counts_for_ldam_drw=class_counts_for_loss if loss_type_main == "ldam_loss" and meta_tune_cfg_fold.get("use_drw_in_phase2", False) else None)
            
            phase2_epochs_run = best_epoch_p2 if best_epoch_p2 != -1 else (phase1_epochs_run + 1 + meta_tune_cfg_fold.get("num_epochs",0) -1 if meta_tune_cfg_fold.get("num_epochs",0) > 0 else phase1_epochs_run +1)
            final_epoch_completed_for_tb_table = max(final_epoch_completed_for_tb_table, phase2_epochs_run)

            if best_metric_p2 > overall_best_metric_for_fold: 
                overall_best_metric_for_fold = best_metric_p2; path_to_overall_best_ckpt_fold = best_ckpt_p2_path; final_optimal_thresholds_for_fold = best_thresh_p2
    else: logger.info(f"Fold {fold_run_id_str_main}: Phase 2 Meta-Tuning skipped.")


    if overall_best_metric_for_fold > -float('inf') and path_to_overall_best_ckpt_fold and Path(path_to_overall_best_ckpt_fold).exists():
        logger.info(f"Fold {fold_run_id_str_main}: Calculating final metrics using best overall ckpt: {path_to_overall_best_ckpt_fold}")
        best_ckpt_loaded_final = torch.load(path_to_overall_best_ckpt_fold, map_location=device_obj_runtime)
        cfg_final_eval = best_ckpt_loaded_final.get('config_runtime', cfg) # Use config from checkpoint if available
        
        final_val_label2idx_model = best_ckpt_loaded_final.get('label2idx_model', label2idx_model_map_runtime)
        final_val_label2idx_eval = best_ckpt_loaded_final.get('label2idx_eval', label2idx_eval_map_runtime)
        final_val_idx2label_model = {v: k for k, v in final_val_label2idx_model.items()}
        final_val_idx2label_eval = {v: k for k, v in final_val_label2idx_eval.items()}
        final_val_num_model_classes = len(final_val_label2idx_model)
        final_val_num_eval_classes = len(final_val_label2idx_eval)

        final_model_cfg_from_ckpt = cfg_final_eval.get("model",{})
        final_base_cnn_params = {"MODEL_TYPE": final_model_cfg_from_ckpt.get("base_cnn_type"), "numClasses": final_val_num_model_classes, "pretrained": False} # Pretrained False for loading weights
        final_base_cnn_instance = get_base_cnn_model(final_base_cnn_params)
        
        final_metadata_dim_fallback = cfg_final_eval.get("model",{}).get('metadata_input_dim_runtime', metadata_dim_runtime)
        final_metadata_dim = best_ckpt_loaded_final.get('metadata_dim', final_metadata_dim_fallback )
        if final_metadata_dim is None: # Further fallback if metadata_dim was None in checkpoint and runtime
            logger.warning("Metadata dimension for final eval model is None, attempting to use val_ds_obj.metadata_dim")
            final_metadata_dim = val_ds_obj.metadata_dim

        final_meta_head_args = final_model_cfg_from_ckpt.get("meta_head_args", {}).copy()
        final_meta_head_args.pop('num_classes', None); final_meta_head_args.pop('metadata_input_dim', None)
        
        model_for_final_eval = CNNWithMetadata(base_cnn_model=final_base_cnn_instance, num_classes=final_val_num_model_classes, metadata_input_dim=final_metadata_dim, **final_meta_head_args).to(device_obj_runtime)
        
        final_weights_key = 'ema_model_state_dict' if cfg_final_eval.get("training",{}).get("use_ema_for_val",True) and 'ema_model_state_dict' in best_ckpt_loaded_final and best_ckpt_loaded_final['ema_model_state_dict'] else 'model_state_dict'
        model_for_final_eval.load_state_dict(best_ckpt_loaded_final[final_weights_key]); model_for_final_eval.eval()
        
        final_eval_dl_args = {"num_workers": data_cfg_fold.get("num_workers",0), "pin_memory": data_cfg_fold.get("pin_memory", True), "persistent_workers": data_cfg_fold.get("num_workers",0)>0 and data_cfg_fold.get("persistent_workers", False), "drop_last": False}
        if final_eval_dl_args["persistent_workers"] and final_eval_dl_args["num_workers"] > 0: final_eval_dl_args["prefetch_factor"] = data_cfg_fold.get("prefetch_factor",2)
        final_val_loader = DataLoader(val_ds_obj, batch_size=cfg_final_eval.get("training",{}).get("batch_size",32), shuffle=False, **final_eval_dl_args) 
        
        all_final_val_probs_list, all_final_val_true_labels_list = [], []
        with torch.no_grad():
            for inputs_final, labels_final_eval_cpu in tqdm_iterator(final_val_loader, desc=f"F{fold_run_id_str_main} Final Table Preds", ncols=cfg.get("experiment_setup",{}).get("TQDM_NCOLS",120)):
                img_f_cpu, meta_f_cpu = inputs_final; img_f_dev, meta_f_dev = img_f_cpu.to(device_obj_runtime, non_blocking=True), meta_f_cpu.to(device_obj_runtime, non_blocking=True)
                with autocast(device_type=device_obj_runtime.type, enabled=(device_obj_runtime.type == 'cuda' and cfg_final_eval.get("training",{}).get("amp_enabled",True))): # Use AMP for consistency if enabled
                    logits_f = model_for_final_eval(img_f_dev, meta_f_dev)
                all_final_val_probs_list.append(F.softmax(logits_f, dim=1).cpu()); all_final_val_true_labels_list.append(labels_final_eval_cpu.cpu())
        all_final_val_probs_cat = torch.cat(all_final_val_probs_list); all_final_val_true_labels_eval_cat = torch.cat(all_final_val_true_labels_list)
        
        f_post_hoc_thresh_cfg = cfg_final_eval.get("training",{}).get("post_hoc_unk_threshold")
        f_post_hoc_thresh = best_ckpt_loaded_final.get('post_hoc_unk_threshold_used', f_post_hoc_thresh_cfg) # Prefer threshold from ckpt if saved
        f_unk_str = cfg_final_eval.get("training",{}).get("unk_label_string", "UNK")
        f_known_model_indices = [idx for lbl,idx in final_val_label2idx_model.items() if lbl!=f_unk_str and idx!=final_val_label2idx_model.get(f_unk_str,-1)]
        f_unk_eval_assign_idx = final_val_label2idx_eval.get(f_unk_str, -1)
        f_post_hoc_active = (f_post_hoc_thresh is not None and 
                             (bool(f_known_model_indices) or not final_val_label2idx_model) and # active if known indices exist OR if model map is empty (e.g. only UNK)
                             f_unk_eval_assign_idx != -1 and 
                             f_unk_str in final_val_label2idx_eval)

        f_preds_modelspace = all_final_val_probs_cat.argmax(dim=1); f_final_preds_evalspace = torch.full_like(f_preds_modelspace, -1, dtype=torch.long)
        for i, f_model_idx in enumerate(f_preds_modelspace.tolist()):
            f_label_str = final_val_idx2label_model.get(f_model_idx)
            if f_label_str and f_label_str in final_val_label2idx_eval: f_final_preds_evalspace[i] = final_val_label2idx_eval[f_label_str]
        
        if f_post_hoc_active:
            logger.info(f"Applying post-hoc UNK (Th={f_post_hoc_thresh:.3f}) for final metrics table.")
            for i in range(all_final_val_probs_cat.size(0)):
                max_prob_known = -1.0
                if f_known_model_indices: # Standard case: there are known classes
                    max_prob_known = all_final_val_probs_cat[i][f_known_model_indices].max().item()
                elif not final_val_label2idx_model: # Edge case: model output is empty (e.g. only trained on UNK, no specific known classes)
                    max_prob_known = all_final_val_probs_cat[i].max().item() # Max over all outputs
                
                if max_prob_known < f_post_hoc_thresh : f_final_preds_evalspace[i] = f_unk_eval_assign_idx
        
        # --- Confusion Matrix ---
        cm_disp_labels_unsorted = {idx: lbl for lbl, idx in final_val_label2idx_eval.items()}
        cm_disp_labels = [cm_disp_labels_unsorted[i] for i in sorted(cm_disp_labels_unsorted.keys())]

        cm_fig_title = f"F{fold_run_id_str_main} Val CM (Best E{best_ckpt_loaded_final.get('epoch', 'N/A')})"
        valid_preds_mask_cm = f_final_preds_evalspace != -1 # Exclude samples mapped to -1 (error)
        
        if f_post_hoc_active: cm_fig_title += f" UNK Th={f_post_hoc_thresh:.2f}"
        
        cm_true_labels = all_final_val_true_labels_eval_cat[valid_preds_mask_cm].numpy()
        cm_pred_labels = f_final_preds_evalspace[valid_preds_mask_cm].numpy()
        
        if len(cm_true_labels) > 0 and len(cm_pred_labels) > 0:
            cm_figure_obj = generate_confusion_matrix_figure(cm_true_labels, cm_pred_labels, cm_disp_labels, cm_fig_title)
            if tb_logger_fold.writer: tb_logger_fold.writer.add_figure(f"Run_{fold_run_id_str_main}/ConfusionMatrix_BestModel_ValSet", cm_figure_obj, global_step=final_epoch_completed_for_tb_table)
            plt.close(cm_figure_obj) # Close figure to free memory
        else: logger.warning("Not enough valid predictions to generate confusion matrix for best model.")

        # --- Metrics Table ---
        metrics_table_rows = []
        # Overall Accuracy (post-hoc)
        acc_overall_ph = (f_final_preds_evalspace == all_final_val_true_labels_eval_cat).float().mean().item() if len(all_final_val_true_labels_eval_cat) > 0 else 0.0
        metrics_table_rows.append(("Overall Accuracy (Post-Hoc)", f"{acc_overall_ph:.4f}"))
        
        # Macro F1-Score (post-hoc) over all eval classes
        f1_macro_ph_all_eval = F1Score(task="multiclass", num_classes=final_val_num_eval_classes, average="macro")(f_final_preds_evalspace, all_final_val_true_labels_eval_cat).item()
        metrics_table_rows.append(("Macro F1-Score (Post-Hoc, All Eval)", f"{f1_macro_ph_all_eval:.4f}"))

        # AUROC and pAUROC (model space, for classes known to model)
        true_labels_auroc_model_list_f, auroc_sample_mask_f = [], torch.zeros(all_final_val_true_labels_eval_cat.size(0),dtype=torch.bool)
        for i,eval_idx_s_f in enumerate(all_final_val_true_labels_eval_cat.tolist()):
            lbl_s_f = final_val_idx2label_eval.get(eval_idx_s_f)
            if lbl_s_f and lbl_s_f in final_val_label2idx_model: 
                true_labels_auroc_model_list_f.append(final_val_label2idx_model[lbl_s_f]); auroc_sample_mask_f[i]=True
        
        if true_labels_auroc_model_list_f and auroc_sample_mask_f.any():
            true_lbls_auroc_tensor_f = torch.tensor(true_labels_auroc_model_list_f,dtype=torch.long)
            probs_auroc_f = all_final_val_probs_cat[auroc_sample_mask_f]
            if probs_auroc_f.size(0)==true_lbls_auroc_tensor_f.size(0) and probs_auroc_f.size(0)>0 and final_val_num_model_classes > 0:
                auroc_val_f = AUROC(task="multiclass",num_classes=final_val_num_model_classes,average="macro")(probs_auroc_f,true_lbls_auroc_tensor_f).item()
                pauc_max_fpr_cfg_f = cfg_final_eval.get("training", {}).get("pauc_max_fpr", 0.2)
                pauroc_val_f = AUROC(task="multiclass",num_classes=final_val_num_model_classes,average="macro", max_fpr=pauc_max_fpr_cfg_f)(probs_auroc_f, true_lbls_auroc_tensor_f).item()
                metrics_table_rows.append(("Macro AUROC (Model Space)", f"{auroc_val_f:.4f}"))
                metrics_table_rows.append((f"Macro pAUROC (Model Space, FPR@{pauc_max_fpr_cfg_f})", f"{pauroc_val_f:.4f}"))
            else:
                metrics_table_rows.append(("Macro AUROC (Model Space)", "N/A"))
                metrics_table_rows.append((f"Macro pAUROC (Model Space, FPR@{cfg_final_eval.get('training', {}).get('pauc_max_fpr', 0.2)})", "N/A"))
        else:
            metrics_table_rows.append(("Macro AUROC (Model Space)", "N/A"))
            metrics_table_rows.append((f"Macro pAUROC (Model Space, FPR@{cfg_final_eval.get('training', {}).get('pauc_max_fpr', 0.2)})", "N/A"))

        header_md = "| Metric                        | Value   |\n"; separator_md = "|-------------------------------|---------|\n"
        table_md_str = header_md + separator_md + "".join([f"| {name:<29} | {val:<7} |\n" for name, val in metrics_table_rows])
        if tb_logger_fold.writer: tb_logger_fold.writer.add_text(f"Run_{fold_run_id_str_main}/Final_Validation_Metrics_Table", table_md_str, global_step=final_epoch_completed_for_tb_table)
        logger.info(f"\nFinal Validation Metrics Table for Fold {fold_run_id_str_main} (Best Model):\n{table_md_str}")

        # --- FINAL Grad-CAM Generation (for the BEST model) ---
        if GRAD_CAM_AVAILABLE and grad_cam_main_cfg.get("enable", False): 
            logger.info("Generating FINAL Grad-CAM overlays for the BEST model...")
            generate_grad_cam_visualizations(
                model_to_use=model_for_final_eval,
                cfg=cfg_final_eval, # Use config from checkpoint
                val_ds_obj=val_ds_obj, # The original val_ds_obj for this fold
                device_obj=device_obj_runtime,
                label2idx_model_map=final_val_label2idx_model,
                label2idx_eval_map=final_val_label2idx_eval,
                tb_logger=tb_logger_fold,
                current_global_epoch=final_epoch_completed_for_tb_table, 
                fold_run_id_str=fold_run_id_str_main,
                exp_name_for_files=exp_name_cli_arg,
                output_subfolder_name="BEST_MODEL_CAMS",
                all_val_preds_eval_space=f_final_preds_evalspace, 
                all_val_true_labels_eval_space=all_final_val_true_labels_eval_cat
            )
    else:
        logger.warning(f"Fold {fold_run_id_str_main}: No best checkpoint found or path invalid. Skipping final metrics table and Grad-CAM.")

    tb_logger_fold.close()
    logger.info(f"Fold {fold_run_id_str_main} finished. Overall best val metric for this fold: {overall_best_metric_for_fold:.4f}")
    return overall_best_metric_for_fold

def main():
    ap = argparse.ArgumentParser(description="Train EffNet-B3+Meta with LDAM-DRW, Warm-Cold, Grad-CAM.")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config if set.")
    ap.add_argument("--fold_id_to_run", type=str, required=True, help="The specific fold ID to run.")
    args = ap.parse_args()

    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
    
    if not cfg_path.exists():
        # Fallback for users who might name it config_meta_effb3_advanced.yaml directly
        fallback_path = Path(args.config_dir) / "config_meta_effb3_advanced.yaml" 
        if not args.config_file and fallback_path.exists() and args.exp_name == "config_meta_effb3_advanced": # only use if exp_name matches
            cfg_path = fallback_path
            logger.warning(f"Experiment config '{args.exp_name}.yaml' not found. Using fallback: {cfg_path}")
        else: # Default behavior if specific exp_name.yaml not found and not using the direct fallback name
            default_config_name = "config_meta_effb3_advanced.yaml" # A generic default if exp_name.yaml is missing
            potential_default_path = Path(args.config_dir) / default_config_name
            if potential_default_path.exists():
                 logger.warning(f"Config file {cfg_path} not found. Trying general default '{default_config_name}' from {args.config_dir}.")
                 cfg_path = potential_default_path
            else:
                 raise FileNotFoundError(f"Config file {cfg_path} (or specified '{args.config_file}') not found. General default '{potential_default_path}' also not found.")


    cfg = load_config(cfg_path); logger.info(f"Loaded config: {cfg_path}"); cfg = cast_config_values(cfg)
    exp_cfg = cfg.get("experiment_setup", {})
    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    set_seed(seed); logger.info(f"Seed: {seed}"); cfg["experiment_setup"]["seed_runtime"] = seed
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg_main = cfg.get("paths", {}); proj_root = paths_cfg_main.get("project_root")
    # Determine base_path for resolving relative paths in config
    if proj_root and Path(proj_root).is_absolute():
        base_path = Path(proj_root)
    elif proj_root: # proj_root is relative
        base_path = (Path.cwd() / proj_root).resolve() # Assuming relative to CWD if not absolute
    else: # No project_root, assume relative to config file's directory
        base_path = cfg_path.parent.resolve()
    logger.info(f"Project base path for resolving config paths: {base_path}")

    fold_id_path_str = str(args.fold_id_to_run).replace(" ", "_").replace("/", "-")
    
    log_dir_base_cfg = paths_cfg_main.get("log_dir", f"outputs/logs/{args.exp_name}") # Default with exp_name
    ckpt_dir_base_cfg = paths_cfg_main.get("ckpt_dir", f"outputs/checkpoints/{args.exp_name}") # Default with exp_name

    log_dir_base = base_path / log_dir_base_cfg if not Path(log_dir_base_cfg).is_absolute() else Path(log_dir_base_cfg)
    ckpt_dir_base = base_path / ckpt_dir_base_cfg if not Path(ckpt_dir_base_cfg).is_absolute() else Path(ckpt_dir_base_cfg)
    
    run_log_dir_fold = log_dir_base / f"fold_{fold_id_path_str}" / ts
    run_ckpt_dir_fold = ckpt_dir_base / f"fold_{fold_id_path_str}" / ts
    run_log_dir_fold.mkdir(parents=True, exist_ok=True); run_ckpt_dir_fold.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs: {run_log_dir_fold}"); logger.info(f"Ckpts: {run_ckpt_dir_fold}")
    
    labels_csv_file = _get_path_from_config(cfg, "labels_csv", base_path=base_path)
    meta_csv_file = _get_path_from_config(cfg, "meta_csv", base_path=base_path)
    img_root_dir = _get_path_from_config(cfg, "train_root", base_path=base_path) # Ensure train_root is also resolved
    
    df_labels_all = pd.read_csv(labels_csv_file); df_meta_all = pd.read_csv(meta_csv_file)
    
    train_cfg = cfg.get("training", {}); known_lbls_cfg = cfg.get("data",{}).get("known_training_labels")
    exclude_unk_train = train_cfg.get("exclude_unk_from_training_model", False)
    unk_lbl_str = train_cfg.get("unk_label_string", "UNK"); df_for_model_cls_map = df_labels_all.copy()
    
    if known_lbls_cfg and isinstance(known_lbls_cfg, list): 
        df_for_model_cls_map = df_labels_all[df_labels_all['label'].isin(known_lbls_cfg)].copy()
    elif exclude_unk_train and unk_lbl_str: 
        df_for_model_cls_map = df_labels_all[df_labels_all['label'] != unk_lbl_str].copy()
    
    unique_model_lbls = sorted(df_for_model_cls_map['label'].unique())
    if not unique_model_lbls: raise ValueError("No model training labels after filtering. Check 'known_training_labels' or 'exclude_unk_from_training_model' in config.")
    label2idx_model_runtime = {name: i for i, name in enumerate(unique_model_lbls)}
    cfg['label2idx_model_runtime'] = label2idx_model_runtime; logger.info(f"Model trains on {len(label2idx_model_runtime)} classes: {label2idx_model_runtime}")
    
    unique_eval_lbls = sorted(df_labels_all['label'].unique())
    label2idx_eval_runtime = {name: i for i, name in enumerate(unique_eval_lbls)}
    cfg['label2idx_eval_runtime'] = label2idx_eval_runtime; logger.info(f"Evaluation uses {len(label2idx_eval_runtime)} classes: {label2idx_eval_runtime}")
    
    dev_default = "cpu";
    if torch.cuda.is_available(): dev_default = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): dev_default = "mps"
    final_dev_str = exp_cfg.get("device", dev_default); device = get_device(final_dev_str); logger.info(f"Device: {device}")
    cfg["experiment_setup"]["device_runtime"] = str(device)
    
    fold_id_arg_str = str(args.fold_id_to_run) # Keep as string initially for flexible fold column types
    fold_id_to_run_for_df = fold_id_arg_str # Default to string
    if 'fold' not in df_labels_all.columns:
        logger.error(f"'fold' column not found in {labels_csv_file}. Cannot perform K-fold validation.")
        return

    # Try to convert fold_id_to_run_for_df to the dtype of the 'fold' column in DataFrame
    fold_column_dtype = df_labels_all['fold'].dtype
    try:
        if pd.api.types.is_integer_dtype(fold_column_dtype):
            fold_id_to_run_for_df = int(fold_id_arg_str)
        elif pd.api.types.is_float_dtype(fold_column_dtype): # Less common for folds, but possible
            fold_id_to_run_for_df = float(fold_id_arg_str)
        # If string, no conversion needed. If other types, might need specific handling or rely on direct comparison.
    except ValueError:
        logger.error(f"Could not convert provided fold ID '{fold_id_arg_str}' to match 'fold' column type ({fold_column_dtype}). Proceeding with string comparison.")
        # fold_id_to_run_for_df remains string

    if fold_id_to_run_for_df not in df_labels_all['fold'].unique(): 
        logger.error(f"Fold ID '{fold_id_to_run_for_df}' not in CSV 'fold' column. Available: {df_labels_all['fold'].unique()}."); return
    
    train_df_selected_fold = df_for_model_cls_map[df_for_model_cls_map['fold'] != fold_id_to_run_for_df].reset_index(drop=True)
    val_df_selected_fold = df_labels_all[df_labels_all['fold'] == fold_id_to_run_for_df].reset_index(drop=True)
    
    if train_df_selected_fold.empty or val_df_selected_fold.empty: 
        logger.error(f"Fold {fold_id_to_run_for_df}: Empty train ({len(train_df_selected_fold)}) or val ({len(val_df_selected_fold)}). Check fold splits and label filtering."); return
    
    logger.info(f"Running fold '{fold_id_to_run_for_df}': Train samples (model scope)={len(train_df_selected_fold)}, Val samples (eval scope)={len(val_df_selected_fold)}")

    best_metric_val_run = train_one_fold_with_meta(
        fold_run_id=fold_id_to_run_for_df, 
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
        logger.info(f"\n===== Single Fold Run Finished (ID: {args.fold_id_to_run}) =====\nBest {cfg.get('training',{}).get('model_selection_metric', 'N/A')}: {best_metric_val_run:.4f}")
    else: 
        logger.warning(f"Single fold run '{args.fold_id_to_run}' did not produce a best metric value or encountered an error.")
    
    logger.info(f"Experiment {args.exp_name} (run for ID {args.fold_id_to_run}, timestamp: {ts}) finished.")

if __name__ == "__main__":
    main()
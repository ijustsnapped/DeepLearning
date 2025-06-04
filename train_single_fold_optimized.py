#!/usr/bin/env python
# your_project_name/train_single_fold_optimized.py
# Trains a single specified fold with optimized logic and corrected error handling.
from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Mitigate potential MKL/Intel conflicts

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
    if 'IPython' in sys.modules and hasattr(sys.modules['IPython'], 'core') and hasattr(sys.modules['IPython'].core, 'getipython') and sys.modules['IPython'].core.getipython.get_ipython() is not None: # type: ignore
            tqdm_iterator = tqdm_notebook
    else:
        from tqdm import tqdm as tqdm_cli
        tqdm_iterator = tqdm_cli
except ImportError:
    from tqdm import tqdm as tqdm_cli
    tqdm_iterator = tqdm_cli


from torchmetrics import AUROC, F1Score, AveragePrecision, Recall # Added Recall
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # Logger might not be configured yet at this top level, will log warning inside function if needed


# --- Custom Module Imports ---
try:
    from data_handling import (
        FlatDataset, build_transform, build_gpu_transform_pipeline,
        ClassBalancedSampler
    )
    from models import get_model
    from models.factory import DinoClassifier # Explicitly import for isinstance check
    from losses import focal_ce_loss, LDAMLoss
    from utils import (
        set_seed, load_config, cast_config_values,
        update_ema,
        get_device, CudaTimer, reset_cuda_peak_memory_stats, empty_cuda_cache,
        TensorBoardLogger
    )
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Details: {e}")
    print("Please ensure the script is run from your project's root directory or that necessary modules are in PYTHONPATH.")
    sys.exit(1)


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


def generate_confusion_matrix_figure(true_labels: np.ndarray, pred_labels: np.ndarray, display_labels: list[str], title: str):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig_s_base, fig_s_factor = 8, 0.6
    fig_w = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig_h = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d'); ax.set_title(title); plt.tight_layout()
    return fig

# --- Grad-CAM Helper Functions ---
def denormalize_image(tensor, mean, std):
    """Denormalizes an image tensor."""
    if tensor.ndim == 3: # C, H, W
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    elif tensor.ndim == 4: # B, C, H, W
        for i in range(tensor.size(0)):
            for t_b, m, s in zip(tensor[i], mean, std):
                t_b.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

cam_activations = {}
cam_gradients = {}

def get_cam_activations_hook(name):
    def hook(model, input, output):
        cam_activations[name] = output.detach()
    return hook

def get_cam_gradients_hook(name):
    def hook(model, grad_input, grad_output):
        cam_gradients[name] = grad_output[0].detach() # grad_output is a tuple
    return hook

def generate_grad_cam_overlay(model, target_layer_name_str, input_tensor_transformed,
                              original_image_for_overlay_tensor, target_class_idx, device):
    if not CV2_AVAILABLE:
        logger.warning("cv2 is not available. Skipping Grad-CAM generation.")
        return None

    global cam_activations, cam_gradients
    cam_activations.clear(); cam_gradients.clear()

    target_layer = None
    try:
        module_path = target_layer_name_str.split('.')
        current_module = model
        for m_name in module_path:
            if '[' in m_name and m_name.endswith(']'): # e.g. blocks[-1]
                base_name, index_str = m_name[:-1].split('[')
                index = int(index_str)
                current_module = getattr(current_module, base_name)[index]
            else:
                current_module = getattr(current_module, m_name)
        target_layer = current_module
    except (AttributeError, IndexError, ValueError) as e:
        logger.error(f"Target layer '{target_layer_name_str}' not found or invalid in model: {e}")
        return None

    if not isinstance(target_layer, nn.Module):
        logger.error(f"Resolved target '{target_layer_name_str}' is not an nn.Module.")
        return None

    handle_fwd = target_layer.register_forward_hook(get_cam_activations_hook(target_layer_name_str))
    handle_bwd = target_layer.register_full_backward_hook(get_cam_gradients_hook(target_layer_name_str))

    model.eval()
    input_tensor_on_device = input_tensor_transformed.to(device)
    if input_tensor_on_device.ndim == 3: input_tensor_on_device = input_tensor_on_device.unsqueeze(0)

    logits = model(input_tensor_on_device)
    
    if not (0 <= target_class_idx < logits.size(1)):
        logger.error(f"Grad-CAM: target_class_idx {target_class_idx} out of bounds for logits shape {logits.shape}.")
        handle_fwd.remove(); handle_bwd.remove()
        return None

    score = logits[:, target_class_idx]
    model.zero_grad()
    score.backward(retain_graph=False)
    handle_fwd.remove(); handle_bwd.remove()

    if target_layer_name_str not in cam_activations or target_layer_name_str not in cam_gradients:
        logger.error("Failed to capture activations or gradients for Grad-CAM."); return None
        
    activations = cam_activations[target_layer_name_str]
    gradients = cam_gradients[target_layer_name_str]

    if activations.ndim < 3 or gradients.ndim < 3 or activations.numel() == 0 or gradients.numel() == 0:
        logger.error(f"Activations or gradients have unexpected shape or are empty. Act: {activations.shape}, Grad: {gradients.shape}"); return None
    if activations.size(0) != 1 or gradients.size(0) != 1: # Ensure batch size 1
        activations = activations[0].unsqueeze(0); gradients = gradients[0].unsqueeze(0)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.size(1)): activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze(); heatmap = F.relu(heatmap)
    if heatmap.numel() == 0: logger.error("Heatmap empty after ReLU/mean."); return None
    if heatmap.max() > 0: heatmap /= torch.max(heatmap)
    heatmap_np = heatmap.cpu().numpy()

    img_for_overlay_np = original_image_for_overlay_tensor.cpu().permute(1, 2, 0).numpy()
    img_for_overlay_np = (img_for_overlay_np * 255).astype(np.uint8)
    
    heatmap_resized = cv2.resize(heatmap_np, (img_for_overlay_np.shape[1], img_for_overlay_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlayed_image_rgb = cv2.addWeighted(cv2.cvtColor(img_for_overlay_np, cv2.COLOR_BGR2RGB) if img_for_overlay_np.shape[2]==3 else img_for_overlay_np, 0.6, heatmap_colored, 0.4, 0)
    # Return RGB for direct use with add_image which expects CHW or HWC (RGB)
    return overlayed_image_rgb


# --- Main Training Function for a Single Fold ---
def train_one_fold(
    fold_run_id: int | str, train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict,
    label2idx_model: dict[str,int], label2idx_eval: dict[str,int],
    train_root_path: Path, run_log_dir: Path, run_ckpt_dir: Path,
    exp_name_for_files: str, device: torch.device,
) -> float | None:

    experiment_setup_cfg = cfg.get("experiment_setup", {})
    model_yaml_cfg = cfg.get("model", {})
    training_loop_cfg = cfg.get("training", {})
    data_handling_cfg = cfg.get("data", {})
    cpu_augmentations_cfg = data_handling_cfg.get("cpu_augmentations", {})
    gpu_augmentations_cfg = data_handling_cfg.get("gpu_augmentations", {})
    loss_cfg = training_loop_cfg.get("loss", {})
    optimizer_yaml_cfg = training_loop_cfg.get("optimizer", {})
    scheduler_yaml_cfg = training_loop_cfg.get("scheduler", {})
    tb_logging_cfg = cfg.get("tensorboard_logging", {})

    logger.info(f"Starting training for run '{fold_run_id}' of experiment '{exp_name_for_files}'. TB logs: {run_log_dir}")
    try:
        _bs = training_loop_cfg.get("batch_size", 32); _bs = 1 if _bs <= 0 else _bs
        _train_len = (len(train_df) + _bs - 1) // _bs
        tb_logger = TensorBoardLogger(log_dir=run_log_dir, experiment_config=cfg, train_loader_len=_train_len)
    except Exception as e: logger.error(f"Failed to init TBLogger: {e}", exc_info=True); return None

    dataset_args_cfg = data_handling_cfg.get("dataset_args", {}).copy() # Use copy
    image_loader_cfg = data_handling_cfg.get("image_loader", "pil")
    enable_ram_cache_cfg = data_handling_cfg.get("enable_ram_cache", False)

    tf_train_cpu = build_transform(cpu_augmentations_cfg, train=True)
    tf_val_cpu = build_transform(cpu_augmentations_cfg, train=False)

    # Ensure args passed to FlatDataset are correct based on its definition
    train_dataset_args = {
        "df": train_df, "root": train_root_path, "label2idx": label2idx_model, "tf": tf_train_cpu,
        "image_loader": image_loader_cfg, "enable_ram_cache": enable_ram_cache_cfg, **dataset_args_cfg
    }
    val_dataset_args = {
        "df": val_df, "root": train_root_path, "label2idx": label2idx_eval, "tf": tf_val_cpu,
        "image_loader": image_loader_cfg, "enable_ram_cache": enable_ram_cache_cfg, **dataset_args_cfg
    }
    
    train_ds = FlatDataset(**train_dataset_args)
    val_ds = FlatDataset(**val_dataset_args)


    train_sampler = None
    if data_handling_cfg.get("sampler", {}).get("type") == "class_balanced_sqrt":
        train_sampler = ClassBalancedSampler(train_ds, num_samples=len(train_ds)); logger.info("Using ClassBalancedSampler.")

    _nw = data_handling_cfg.get("num_workers", 0)
    dl_args = {"num_workers": _nw,
               "pin_memory": (device.type == 'cuda' and _nw > 0),
               "persistent_workers": (data_handling_cfg.get("persistent_workers", False) and _nw > 0),
               "drop_last": False}
    if _nw > 0 and dl_args["persistent_workers"]:
        dl_args["prefetch_factor"] = data_handling_cfg.get("prefetch_factor", 2)


    train_ld = DataLoader(train_ds, batch_size=training_loop_cfg.get("batch_size",32),
                          sampler=train_sampler, shuffle=(train_sampler is None), **dl_args) # type: ignore
    val_ld = DataLoader(val_ds, batch_size=training_loop_cfg.get("batch_size",32), shuffle=False, **dl_args) # type: ignore

    gpu_aug_train = build_gpu_transform_pipeline(gpu_augmentations_cfg, device) if gpu_augmentations_cfg.get("enable", False) else None
    if gpu_aug_train: logger.info("GPU augmentation enabled for training.")

    model_factory_cfg = {**model_yaml_cfg, "numClasses": len(label2idx_model)}
    if "type" in model_factory_cfg: model_factory_cfg["MODEL_TYPE"] = model_factory_cfg.pop("type")
    model = get_model(model_factory_cfg).to(device)

    ema_decay_val = training_loop_cfg.get("ema_decay", 0.0)
    ema_model = copy.deepcopy(model).to(device) if ema_decay_val > 0 else None
    if ema_model: [p.requires_grad_(False) for p in ema_model.parameters()]; logger.info(f"EMA enabled with decay {ema_decay_val}.")

    freeze_epochs = training_loop_cfg.get("freeze_epochs", 0)
    backbone_lr_mult = training_loop_cfg.get("backbone_lr_mult", 1.0)
    initial_lr = optimizer_yaml_cfg.get("lr", 1e-3)
    opt_params = []
    head_param_prefixes = []
    if isinstance(model, DinoClassifier): head_param_prefixes.append("classifier.")
    elif hasattr(model, 'default_cfg') and 'classifier' in model.default_cfg and hasattr(model, model.default_cfg['classifier']):
        head_param_prefixes.append(f"{model.default_cfg['classifier']}.")
    else:
        for common_head_name in ['classifier', 'fc', 'head']:
            if hasattr(model, common_head_name) and isinstance(getattr(model, common_head_name), (nn.Linear, nn.Sequential)):
                head_param_prefixes.append(f"{common_head_name}.")
    if not head_param_prefixes: logger.warning("Could not determine head prefixes for diff LR/freezing.")

    if freeze_epochs > 0:
        trainable_head_params, frozen_params_count = [], 0
        if not head_param_prefixes:
            logger.warning("Freezing: No head prefixes. Trying to freeze all but last Linear layer.")
            last_linear = next((m for _, m in reversed(list(model.named_modules())) if isinstance(m, nn.Linear)), None)
            if last_linear:
                for param in model.parameters(): param.requires_grad = False
                for param in last_linear.parameters(): param.requires_grad = True; trainable_head_params.append(param)
                frozen_params_count = sum(1 for p in model.parameters() if not p.requires_grad)
                logger.info(f"Last linear layer '{[name for name, mod in model.named_modules() if mod is last_linear][0]}' set as trainable.")
            else: logger.error("Freezing: No head prefixes AND no final Linear. Training all."); trainable_head_params = list(model.parameters())
        else:
            logger.info(f"Freezing: Head prefixes: {head_param_prefixes}")
            for name, param in model.named_parameters():
                if any(name.startswith(p) for p in head_param_prefixes): param.requires_grad = True; trainable_head_params.append(param)
                else: param.requires_grad = False; frozen_params_count +=1
        if not trainable_head_params and frozen_params_count > 0:
            logger.warning("Attempted freeze, but no head params trainable. Unfreezing all."); [p.requires_grad_(True) for p in model.parameters()]; trainable_head_params = list(model.parameters())
        logger.info(f"Backbone frozen for {freeze_epochs} epochs. {frozen_params_count} params frozen.")
        opt_params = [{'params': trainable_head_params, 'lr': initial_lr}] if trainable_head_params else [{'params': model.parameters(), 'lr': initial_lr}]
    elif backbone_lr_mult != 1.0 and head_param_prefixes:
        logger.info(f"Diff LR: backbone_lr_mult={backbone_lr_mult}, head_prefixes={head_param_prefixes}")
        head_p, backbone_p = [], []; [(head_p if any(n.startswith(pfx) for pfx in head_param_prefixes) else backbone_p).append(p) for n,p in model.named_parameters() if p.requires_grad]
        if head_p: opt_params.append({'params': head_p, 'lr': initial_lr})
        if backbone_p: opt_params.append({'params': backbone_p, 'lr': initial_lr * backbone_lr_mult})
    else: opt_params = [{'params': filter(lambda p:p.requires_grad, model.parameters()), 'lr': initial_lr}]
    if not opt_params or not any(pg.get('params') for pg in opt_params):
        logger.warning("Optimizer params empty. Defaulting to all model params."); opt_params = [{'params': model.parameters(), 'lr': initial_lr}]

    optimizer = AdamW(opt_params, weight_decay=optimizer_yaml_cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=training_loop_cfg.get("num_epochs"), eta_min=scheduler_yaml_cfg.get("min_lr", 1e-6))
    amp_active = (device.type == 'cuda' and training_loop_cfg.get("amp_enabled", True))
    scaler = GradScaler(device.type if device.type in ['cuda','mps'] else 'cpu', enabled=amp_active)

    accum_steps = training_loop_cfg.get("accum_steps", 1)
    post_hoc_thresh_val = training_loop_cfg.get("post_hoc_unk_threshold") # Can be None
    unk_str_cfg = training_loop_cfg.get("unk_label_string") # Can be None from YAML (null)
    
    known_model_output_indices, unk_eval_assignment_idx_val, post_hoc_is_active = [], -1, False
    if unk_str_cfg is not None: # Only setup for explicit UNK string
        known_model_output_indices = [idx for lbl, idx in label2idx_model.items() if lbl != unk_str_cfg and idx != label2idx_model.get(unk_str_cfg, -1)]
        unk_eval_assignment_idx_val = label2idx_eval.get(unk_str_cfg, -1)
        post_hoc_is_active = (post_hoc_thresh_val is not None and bool(known_model_output_indices) and unk_eval_assignment_idx_val != -1)
        if post_hoc_thresh_val is not None: logger.info(f"R{fold_run_id}: Post-hoc '{unk_str_cfg}' (Th={post_hoc_thresh_val}). Active: {post_hoc_is_active}")
    else: logger.info(f"R{fold_run_id}: unk_label_string is null. Post-hoc UNK assignment for a specific UNK class is disabled.")


    idx2label_model_map = {v: k for k, v in label2idx_model.items()}
    idx2label_eval_map = {v: k for k, v in label2idx_eval.items()}
    model_selection_metric_str = training_loop_cfg.get("model_selection_metric", "macro_auc").lower()
    best_metric_value, patience_counter_val, last_ckpt_metric_value = -float('inf'), 0, -float('inf')
    num_model_output_classes, num_eval_log_classes = len(label2idx_model), len(label2idx_eval)
    class_counts_for_train = get_class_counts(train_df, label2idx_model)

    current_loss_type, current_criterion = loss_cfg.get("type", "cross_entropy").lower(), None
    focal_loss_alpha, focal_loss_gamma = loss_cfg.get("focal_alpha",1.0), loss_cfg.get("focal_gamma",2.0)

    if current_loss_type == "focal_ce_loss": current_criterion = focal_ce_loss; logger.info(f"Using FocalCE (alpha={focal_loss_alpha}, gamma={focal_loss_gamma}).")
    elif current_loss_type == "cross_entropy": current_criterion = nn.CrossEntropyLoss(); logger.info("Using CrossEntropyLoss.")
    elif current_loss_type == "weighted_cross_entropy":
        wce_k = loss_cfg.get("wce_k_factor",1.0); w_np = np.ones(num_model_output_classes,dtype=float)
        if wce_k>0 and np.sum(class_counts_for_train)>0: safe_c = np.maximum(class_counts_for_train,1e-8); raw_w=(np.sum(class_counts_for_train)/safe_c)**wce_k; w_np = raw_w/np.sum(raw_w)*num_model_output_classes if np.sum(raw_w)>1e-12 else w_np
        current_criterion = nn.CrossEntropyLoss(weight=torch.tensor(w_np,dtype=torch.float,device=device)); logger.info(f"Using WCE (k={wce_k}). W (first 5): {w_np[:5]}")
    elif current_loss_type == "ldam_loss":
        ldam_params = loss_cfg.get("ldam_params",{}); current_criterion = LDAMLoss(class_counts=class_counts_for_train,**ldam_params).to(device); logger.info(f"Using LDAM: {ldam_params}") # type: ignore
    else: raise ValueError(f"Unsupported loss: {current_loss_type}")

    val_metrics_heavy_run_interval = training_loop_cfg.get("val_metrics_heavy_interval", 99999)
    drw_schedule_epoch_list = training_loop_cfg.get("drw_schedule_epochs", []); current_drw_stage_val = 0
    final_epoch_num_completed = 0

    logger.info(f"Starting training: {training_loop_cfg.get('num_epochs')} epochs for R'{fold_run_id}'.")
    for epoch_num in range(training_loop_cfg.get('num_epochs')):
        final_epoch_num_completed = epoch_num
        # --- Freeze/Unfreeze Logic ---
        if freeze_epochs > 0 and epoch_num == freeze_epochs:
            logger.info(f"E{epoch_num}: Unfreezing backbone. Resetting optimizer/scheduler.")
            new_unfrozen = sum(1 for p in model.parameters() if not p.requires_grad and p.requires_grad_(True))
            logger.info(f"Unfroze {new_unfrozen} parameters.")
            opt_params_unfrozen = []
            if backbone_lr_mult != 1.0 and head_param_prefixes:
                h_p, b_p = [], []; [(h_p if any(n.startswith(pfx) for pfx in head_param_prefixes) else b_p).append(p) for n,p in model.named_parameters() if p.requires_grad]
                if h_p: opt_params_unfrozen.append({'params': h_p, 'lr': initial_lr})
                if b_p: opt_params_unfrozen.append({'params': b_p, 'lr': initial_lr * backbone_lr_mult})
            else: unfrozen_lr = initial_lr * backbone_lr_mult if backbone_lr_mult != 1.0 else initial_lr; logger.info(f"Uniform LR for unfrozen: {unfrozen_lr}"); opt_params_unfrozen = [{'params': model.parameters(), 'lr': unfrozen_lr}]
            if not opt_params_unfrozen or not any(pg.get('params') for pg in opt_params_unfrozen): logger.error("CRIT: Optimizer params empty post-unfreeze."); opt_params_unfrozen = [{'params':model.parameters(), 'lr':initial_lr}]
            optimizer = AdamW(opt_params_unfrozen, weight_decay=optimizer_yaml_cfg.get("weight_decay",1e-4))
            rem_epochs = training_loop_cfg.get("num_epochs")-freeze_epochs; scheduler = CosineAnnealingLR(optimizer,T_max=rem_epochs if rem_epochs>0 else 1,eta_min=scheduler_yaml_cfg.get("min_lr",1e-6)); logger.info(f"Opt/Sched re-init. Rem epochs for sched: {rem_epochs if rem_epochs > 0 else 1}")
        # --- DRW for LDAM Loss ---
        if current_loss_type=="ldam_loss" and hasattr(current_criterion,'update_weights') and drw_schedule_epoch_list:
            if current_drw_stage_val < len(drw_schedule_epoch_list) and epoch_num >= drw_schedule_epoch_list[current_drw_stage_val]:
                beta_drw = loss_cfg.get("ldam_params",{}).get("effective_number_beta",0.999); drw_w_np = np.ones(num_model_output_classes,dtype=float)
                if num_model_output_classes>0: eff_num = 1.0-np.power(beta_drw,class_counts_for_train); drw_w_np = (1.0-beta_drw)/np.maximum(eff_num,1e-8); drw_w_np = drw_w_np/np.sum(drw_w_np)*num_model_output_classes
                if drw_w_np.size>0: current_criterion.update_weights(torch.tensor(drw_w_np,dtype=torch.float,device=device)); logger.info(f"DRW: Updated LDAM weights E{epoch_num}. W (first 5): {drw_w_np[:5]}") # type: ignore
                current_drw_stage_val+=1
            elif epoch_num==0 and current_drw_stage_val==0 : current_criterion.update_weights(None) # type: ignore
        # --- Training Loop ---
        if train_sampler and hasattr(train_sampler,'set_epoch'): train_sampler.set_epoch(epoch_num) # type: ignore
        active_profiler = tb_logger.setup_profiler(epoch_num, run_log_dir); model.train();
        if gpu_aug_train: gpu_aug_train.train()
        epoch_gpu_time,epoch_wall_start,train_loss,train_ok,train_n = 0.0,time.time(),0.0,0,0; optimizer.zero_grad()
        pbar_train = tqdm_iterator(train_ld,desc=f"R{fold_run_id} E{epoch_num} Train",ncols=experiment_setup_cfg.get("TQDM_NCOLS",120),leave=False)
        for batch_idx, (imgs_cpu,lbls_cpu) in enumerate(pbar_train):
            imgs_dev,lbls_dev = imgs_cpu.to(device,non_blocking=True),lbls_cpu.to(device,non_blocking=True)
            if gpu_aug_train: imgs_dev=gpu_aug_train(imgs_dev)
            tmr_act = active_profiler is not None or (tb_logging_cfg.get("profiler",{}).get("enable_batch_timing_always",False) and tb_logger._should_log_batch(tb_logger.log_interval_train_batch,batch_idx)) # type: ignore
            timer_ctx = CudaTimer(device) if tmr_act and device.type=='cuda' else contextlib.nullcontext() # type: ignore
            with timer_ctx as batch_tmr: # type: ignore
                with autocast(device.type,enabled=amp_active):
                    logits = model(imgs_dev); loss = current_criterion(logits.float(),lbls_dev,alpha=focal_loss_alpha,gamma=focal_loss_gamma) if current_loss_type=="focal_ce_loss" else current_criterion(logits.float(),lbls_dev)
                    if accum_steps>1: loss/=accum_steps
                scaler.scale(loss).backward()
                if (batch_idx+1)%accum_steps==0 or (batch_idx+1)==len(train_ld): scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); 
                if ema_model and ema_decay_val > 0: update_ema(ema_model,model,ema_decay_val) # type: ignore
            batch_gpu_t = batch_tmr.get_elapsed_time_ms() if tmr_act and device.type=='cuda' and batch_tmr else 0.0; epoch_gpu_time+=batch_gpu_t # type: ignore
            b_loss,b_preds = loss.item()*(accum_steps if accum_steps>1 else 1),logits.argmax(dim=1); b_ok,b_n=(b_preds==lbls_dev).sum().item(),imgs_dev.size(0)
            train_loss+=b_loss*b_n; train_ok+=b_ok; train_n+=b_n; pbar_train.set_postfix(loss=f"{train_loss/train_n:.4f}",acc=f"{train_ok/train_n:.4f}")
            tb_logger.log_train_batch_metrics(loss=b_loss,acc=(b_ok/b_n if b_n>0 else 0.0),lr=optimizer.param_groups[0]['lr'],epoch=epoch_num,batch_idx=batch_idx,batch_gpu_time_ms=batch_gpu_t if tmr_act else None)
            if active_profiler: active_profiler.step()
        pbar_train.close(); epoch_wall_dur=time.time()-epoch_wall_start; scheduler.step()
        ep_metrics = {"Loss/train_epoch":train_loss/train_n if train_n>0 else 0, "Accuracy/train_epoch":train_ok/train_n if train_n>0 else 0,"LearningRate/epoch":optimizer.param_groups[0]['lr'],"Time/Train_epoch_duration_sec":epoch_wall_dur,"Throughput/train_samples_per_sec":train_n/epoch_wall_dur if epoch_wall_dur>0 else 0}
        if device.type=='cuda' and epoch_gpu_time>0: ep_metrics["Time/GPU_ms_per_train_epoch"]=epoch_gpu_time
        if active_profiler: tb_logger.stop_and_process_profiler(); active_profiler=None
        # --- Validation Loop ---
        if epoch_num%training_loop_cfg.get("val_interval",1)==0 or epoch_num==training_loop_cfg.get("num_epochs")-1:
            val_prep_start=time.time(); eval_model = ema_model if ema_model and training_loop_cfg.get("use_ema_for_val",True) else model; eval_model.eval(); 
            if gpu_aug_train: gpu_aug_train.eval() # type: ignore
            logger.info(f"R{fold_run_id} E{epoch_num} ValPrep: {time.time()-val_prep_start:.2f}s {'EMA' if eval_model is ema_model else 'Primary'}")
            val_logits_all,val_true_all = [],[]; val_loss,val_seen,val_loss_n,val_ok_orig,val_orig_n,val_ok_posthoc = 0.0,0,0,0,0,0
            pbar_val = tqdm_iterator(val_ld,desc=f"R{fold_run_id} E{epoch_num} Val",ncols=experiment_setup_cfg.get("TQDM_NCOLS",120),leave=False)
            with torch.no_grad():
                for bval_idx, (imgs_v_cpu,lbls_v_eval_cpu) in enumerate(pbar_val):
                    imgs_v_dev,lbls_v_eval_cpu_t = imgs_v_cpu.to(device,non_blocking=True),lbls_v_eval_cpu.cpu()
                    lbls_v_model_l,loss_mask_l = [],[]; 
                    for true_eval_s in lbls_v_eval_cpu_t.tolist(): lbl_s=idx2label_eval_map.get(true_eval_s,"ERR"); model_idx=label2idx_model.get(lbl_s,-1) if lbl_s!="ERR" else -1; lbls_v_model_l.append(model_idx); loss_mask_l.append(model_idx!=-1)
                    lbls_v_model_dev,loss_mask_dev = torch.tensor(lbls_v_model_l,dtype=torch.long,device=device),torch.tensor(loss_mask_l,dtype=torch.bool,device=device)
                    with autocast(device.type,enabled=amp_active):
                        logits_v = eval_model(imgs_v_dev)
                        if loss_mask_dev.any():
                            n_loss_b=imgs_v_dev[loss_mask_dev].size(0); l_for_loss,t_for_loss = logits_v[loss_mask_dev].float(),lbls_v_model_dev[loss_mask_dev]
                            loss_v_b = current_criterion(l_for_loss,t_for_loss,alpha=focal_loss_alpha,gamma=focal_loss_gamma) if current_loss_type=="focal_ce_loss" else current_criterion(l_for_loss,t_for_loss)
                            val_loss+=loss_v_b.item()*n_loss_b; val_loss_n+=n_loss_b
                        else: loss_v_b=torch.tensor(0.0,device=device)
                    preds_orig_model_b = logits_v.argmax(dim=1)
                    if loss_mask_dev.any(): ok_orig_b=(preds_orig_model_b[loss_mask_dev]==lbls_v_model_dev[loss_mask_dev]).sum().item(); val_ok_orig+=ok_orig_b; val_orig_n+=loss_mask_dev.sum().item()
                    val_seen+=imgs_v_dev.size(0); val_logits_all.append(logits_v.cpu()); val_true_all.append(lbls_v_eval_cpu_t)
                    tqdm_pf = {'avg_loss':f"{val_loss/val_loss_n:.4f}" if val_loss_n>0 else "N/A", 'avg_acc_orig':f"{val_ok_orig/val_orig_n:.4f}" if val_orig_n>0 else "N/A"}
                    if post_hoc_is_active: # This post_hoc_is_active respects if unk_label_string was null for the run
                        probs_v_b_cpu=F.softmax(logits_v.cpu(),dim=1); preds_m_b_cpu=probs_v_b_cpu.argmax(dim=1); final_preds_b_tqdm=torch.full_like(preds_m_b_cpu,-1,dtype=torch.long)
                        for i,m_idx in enumerate(preds_m_b_cpu.tolist()): lbl_s=idx2label_model_map.get(m_idx); 
                        if lbl_s and lbl_s in label2idx_eval: final_preds_b_tqdm[i]=label2idx_eval[lbl_s]
                        for i in range(probs_v_b_cpu.size(0)): # Apply threshold
                            max_prob_known = 0.0
                            if known_model_output_indices: max_prob_known = probs_v_b_cpu[i][known_model_output_indices].max().item()
                            # No 'elif not known_model_output_indices and post_hoc_thresh_val is not None' needed here if unk_str_cfg is None,
                            # as post_hoc_is_active would be false. If unk_str_cfg is not None, known_model_output_indices is used.
                            if max_prob_known < post_hoc_thresh_val: final_preds_b_tqdm[i]=unk_eval_assignment_idx_val
                        val_ok_posthoc+=(final_preds_b_tqdm==lbls_v_eval_cpu_t).sum().item(); tqdm_pf['avg_acc_post_hoc']=f"{val_ok_posthoc/val_seen:.4f}" if val_seen>0 else "N/A"
                    pbar_val.set_postfix(**tqdm_pf)
            pbar_val.close()
            val_logits_cat,val_true_eval_cat = torch.cat(val_logits_all),torch.cat(val_true_all); val_probs_cat_cpu=F.softmax(val_logits_cat,dim=1)
            ep_metrics_val = {"Loss/val_epoch":val_loss/val_loss_n if val_loss_n>0 else 0, "Accuracy/val_epoch_original_argmax":val_ok_orig/val_orig_n if val_orig_n>0 else 0}
            preds_orig_model_ep_cpu = val_probs_cat_cpu.argmax(dim=1)
            true_lbls_known_m_l,known_f1_mask=[],torch.zeros(val_true_eval_cat.size(0),dtype=torch.bool)
            for i,true_eval_s in enumerate(val_true_eval_cat.tolist()): lbl_s=idx2label_eval_map.get(true_eval_s); 
            if lbl_s and lbl_s in label2idx_model: true_lbls_known_m_l.append(label2idx_model[lbl_s]); known_f1_mask[i]=True
            if true_lbls_known_m_l:
                true_lbls_known_m_t=torch.tensor(true_lbls_known_m_l,dtype=torch.long); preds_known_f1=preds_orig_model_ep_cpu[known_f1_mask]
                if preds_known_f1.size(0)==true_lbls_known_m_t.size(0) and preds_known_f1.size(0)>0: ep_metrics_val["F1Score/val_macro_known_classes"]=F1Score(task="multiclass",num_classes=num_model_output_classes,average="macro")(preds_known_f1,true_lbls_known_m_t).item()
                else: ep_metrics_val["F1Score/val_macro_known_classes"]=0.0
            else: ep_metrics_val["F1Score/val_macro_known_classes"]=0.0
            
            final_preds_eval_ep=torch.full_like(preds_orig_model_ep_cpu,-1,dtype=torch.long)
            for i,m_idx in enumerate(preds_orig_model_ep_cpu.tolist()): lbl_s=idx2label_model_map.get(m_idx); 
            if lbl_s and lbl_s in label2idx_eval: final_preds_eval_ep[i]=label2idx_eval[lbl_s]
            if post_hoc_is_active: # Apply threshold for final metrics
                for i in range(val_probs_cat_cpu.size(0)):
                    max_prob_known = 0.0
                    if known_model_output_indices: max_prob_known = val_probs_cat_cpu[i][known_model_output_indices].max().item()
                    if max_prob_known < post_hoc_thresh_val : final_preds_eval_ep[i]=unk_eval_assignment_idx_val
            ep_metrics_val["Accuracy/val_epoch_post_hoc_unk"]=(final_preds_eval_ep==val_true_eval_cat).sum().item()/val_seen if val_seen>0 else 0.0
            ep_metrics_val["F1Score/val_macro_post_hoc_unk"]=F1Score(task="multiclass",num_classes=num_eval_log_classes,average="macro")(final_preds_eval_ep,val_true_eval_cat).item()

            true_auroc_m_l,auroc_mask=[],torch.zeros(val_true_eval_cat.size(0),dtype=torch.bool)
            for i,eval_s in enumerate(val_true_eval_cat.tolist()): lbl_s=idx2label_eval_map.get(eval_s); 
            if lbl_s and lbl_s in label2idx_model: true_auroc_m_l.append(label2idx_model[lbl_s]); auroc_mask[i]=True
            if true_auroc_m_l and auroc_mask.any():
                true_auroc_t=torch.tensor(true_auroc_m_l,dtype=torch.long); probs_auroc=val_probs_cat_cpu[auroc_mask]
                if probs_auroc.size(0)==true_auroc_t.size(0) and probs_auroc.size(0)>0: ep_metrics_val["AUROC/val_macro"]=AUROC(task="multiclass",num_classes=num_model_output_classes,average="macro")(probs_auroc,true_auroc_t).item()
                else: ep_metrics_val["AUROC/val_macro"]=0.0
            else: ep_metrics_val["AUROC/val_macro"]=0.0
            
            opt_thresh_save={}; heavy_ran=False
            if epoch_num%val_metrics_heavy_run_interval==0 or epoch_num==training_loop_cfg.get("num_epochs")-1:
                heavy_ran=True; logger.info(f"R{fold_run_id} E{epoch_num}: Heavy val metrics (PR-curves).")
                if num_model_output_classes>1:
                    pr_true_m_l,pr_mask=[],torch.zeros(val_true_eval_cat.size(0),dtype=torch.bool)
                    for i,eval_s in enumerate(val_true_eval_cat.tolist()): lbl_s=idx2label_eval_map.get(eval_s); 
                    if lbl_s and lbl_s in label2idx_model: pr_true_m_l.append(label2idx_model[lbl_s]); pr_mask[i]=True
                    if pr_true_m_l:
                        pr_true_t=torch.tensor(pr_true_m_l,dtype=torch.long); pr_probs_np=val_probs_cat_cpu[pr_mask].numpy(); pr_lbls_oh=F.one_hot(pr_true_t,num_model_output_classes).numpy()
                        opt_f1s,opt_sens=[],[]
                        for cls_i in range(num_model_output_classes):
                            f1,thr,sens=0.0,0.5,0.0
                            try:
                                p,r,t=precision_recall_curve(pr_lbls_oh[:,cls_i],pr_probs_np[:,cls_i])
                                if len(p)>1 and len(r)>1 and len(t)>0: f1c=(2*p*r)/(p+r+1e-8); rf1,rr=f1c[1:],r[1:]; vidx=np.where(np.isfinite(rf1)&(p[1:]+r[1:]>0))[0]; 
                                if len(vidx)>0:bi=vidx[np.argmax(rf1[vidx])];f1,thr,sens=float(rf1[bi]),float(t[bi]),float(rr[bi])
                            except Exception as e_pr: logger.warning(f"PR fail C{cls_i} E{epoch_num}: {e_pr}")
                            opt_f1s.append(f1);opt_sens.append(sens);opt_thresh_save[cls_i]=thr
                        if opt_f1s: ep_metrics_val["F1Score/val_mean_optimal_per_class_from_PR"]=np.mean(opt_f1s)
                        if opt_sens: ep_metrics_val["Sensitivity/val_mean_optimal_per_class_from_PR"]=np.mean(opt_sens)
            ep_metrics.update(ep_metrics_val)
            metric_map={"macro_auc":"AUROC/val_macro","mean_optimal_f1":"F1Score/val_mean_optimal_per_class_from_PR","mean_optimal_sensitivity":"Sensitivity/val_mean_optimal_per_class_from_PR","f1_macro_known_classes":"F1Score/val_macro_known_classes","f1_macro_post_hoc_unk":"F1Score/val_macro_post_hoc_unk","accuracy_post_hoc_unk":"Accuracy/val_epoch_post_hoc_unk"}
            primary_metric_key = metric_map.get(model_selection_metric_str,"AUROC/val_macro")
            current_metric_val = ep_metrics_val.get(primary_metric_key,-float('inf')); last_ckpt_metric_value=current_metric_val
            log_l=f"R{fold_run_id} E{epoch_num} Val -> Loss={ep_metrics_val.get('Loss/val_epoch',-1):.4f} AccOrig={ep_metrics_val.get('Accuracy/val_epoch_original_argmax',-1):.4f} ";
            if post_hoc_is_active: log_l+=f"AccPostHoc={ep_metrics_val.get('Accuracy/val_epoch_post_hoc_unk',-1):.4f} "
            log_l+=f"Sel({model_selection_metric_str}->{primary_metric_key})={current_metric_val:.4f}"; logger.info(log_l)
            if current_metric_val > best_metric_value:
                best_metric_value=current_metric_val;patience_counter_val=0; logger.info(f"New best {model_selection_metric_str}: {best_metric_value:.4f} E{epoch_num}. Saving.")
                best_ckpt_path = run_ckpt_dir/f"{exp_name_for_files}_run_{fold_run_id}_best.pt"
                ckpt_data = {'epoch':epoch_num,'model_state_dict':getattr(model,'_orig_mod',model).state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),f'best_{model_selection_metric_str}':best_metric_value,'config_runtime':cfg,'label2idx_model':label2idx_model,'label2idx_eval':label2idx_eval}
                if ema_model: ckpt_data['ema_model_state_dict']=getattr(ema_model,'_orig_mod',ema_model).state_dict()
                if post_hoc_is_active: ckpt_data['post_hoc_unk_threshold_used']=post_hoc_thresh_val
                if (training_loop_cfg.get("save_optimal_thresholds_from_pr",False) or model_selection_metric_str in ["mean_optimal_f1","mean_optimal_sensitivity"]) and heavy_ran and opt_thresh_save: ckpt_data['optimal_thresholds_val_from_pr']=opt_thresh_save; logger.info("Saved PR thresholds.")
                torch.save(ckpt_data,str(best_ckpt_path))
            else: patience_counter_val+=1
            if patience_counter_val>=training_loop_cfg.get("early_stopping_patience",10): logger.info(f"Early stopping E{epoch_num} R{fold_run_id}."); break
            tb_logger.log_epoch_summary(ep_metrics,epoch_num)
        if patience_counter_val>=training_loop_cfg.get("early_stopping_patience",10): break
    # --- End of Training Loop ---

    # --- Final Evaluation, CM, Grad-CAM on Best Model ---
    if best_metric_value > -float('inf'):
        final_best_ckpt_path = run_ckpt_dir / f"{exp_name_for_files}_run_{fold_run_id}_best.pt"
        if final_best_ckpt_path.exists():
            logger.info(f"R{fold_run_id}: Loading best model ({final_best_ckpt_path}) for final reports.")
            best_ckpt_data = torch.load(final_best_ckpt_path, map_location=device)
            cfg_from_ckpt = best_ckpt_data.get('config_runtime', cfg)
            final_l2i_model = best_ckpt_data.get('label2idx_model', label2idx_model)
            final_l2i_eval = best_ckpt_data.get('label2idx_eval', label2idx_eval)
            final_i2l_model = {v:k for k,v in final_l2i_model.items()}; final_i2l_eval={v:k for k,v in final_l2i_eval.items()}
            final_n_model_cls,final_n_eval_cls = len(final_l2i_model),len(final_l2i_eval)
            
            m_cfg_final = cfg_from_ckpt.get('model',{}); m_cfg_final["numClasses"]=final_n_model_cls
            if "type" in m_cfg_final and "MODEL_TYPE" not in m_cfg_final: m_cfg_final["MODEL_TYPE"]=m_cfg_final.pop("type")
            model_final_eval = get_model(m_cfg_final).to(device)
            w_key_final = 'ema_model_state_dict' if cfg_from_ckpt.get("training",{}).get("use_ema_for_val",True) and 'ema_model_state_dict' in best_ckpt_data and best_ckpt_data['ema_model_state_dict'] else 'model_state_dict'
            model_final_eval.load_state_dict(best_ckpt_data[w_key_final]); model_final_eval.eval()

            f_probs_all,f_true_all=[],[]
            with torch.no_grad():
                for f_imgs_cpu,f_lbls_cpu in tqdm_iterator(val_ld,desc=f"R{fold_run_id} FinalReportsPreds"):
                    f_logits=model_final_eval(f_imgs_cpu.to(device,non_blocking=True)); f_probs_all.append(F.softmax(f_logits,dim=1).cpu()); f_true_all.append(f_lbls_cpu.cpu())
            f_probs_cat,f_true_eval_cat = torch.cat(f_probs_all),torch.cat(f_true_all)

            f_post_hoc_thresh_final = best_ckpt_data.get('post_hoc_unk_threshold_used', training_loop_cfg.get("post_hoc_unk_threshold"))
            f_unk_str_final = cfg_from_ckpt.get("training",{}).get("unk_label_string") # Can be None from YAML
            
            f_known_m_indices_final, f_unk_eval_idx_final, f_post_hoc_act_final = [], -1, False
            if f_unk_str_final is not None:
                f_known_m_indices_final = [idx for lbl,idx in final_l2i_model.items() if lbl!=f_unk_str_final and idx!=final_l2i_model.get(f_unk_str_final,-1)]
                f_unk_eval_idx_final = final_l2i_eval.get(f_unk_str_final,-1)
                f_post_hoc_act_final = (f_post_hoc_thresh_final is not None and bool(f_known_m_indices_final) and f_unk_eval_idx_final!=-1)
            
            f_preds_m_space = f_probs_cat.argmax(dim=1)
            f_preds_eval_space_final = torch.full_like(f_preds_m_space,-1,dtype=torch.long)
            for i,f_m_idx in enumerate(f_preds_m_space.tolist()): f_lbl_s=final_i2l_model.get(f_m_idx); 
            if f_lbl_s and f_lbl_s in final_l2i_eval: f_preds_eval_space_final[i]=final_l2i_eval[f_lbl_s]
            if f_post_hoc_act_final:
                logger.info(f"Applying post-hoc UNK (Th={f_post_hoc_thresh_final}) for final CM/GradCAM.")
                for i in range(f_probs_cat.size(0)):
                    max_prob_known = 0.0
                    if f_known_m_indices_final: max_prob_known = f_probs_cat[i][f_known_m_indices_final].max().item()
                    if max_prob_known < f_post_hoc_thresh_final: f_preds_eval_space_final[i]=f_unk_eval_idx_final
            
            if tb_logger.writer:
                cm_labels=[lbl for lbl,idx in sorted(final_i2l_eval.items(),key=lambda x:x[1])]; cm_title=f"R{fold_run_id} ValCM (BestE{best_ckpt_data['epoch']})"
                if f_post_hoc_act_final: cm_title+=f" UNKTh={f_post_hoc_thresh_final:.2f}"
                valid_cm_mask = f_preds_eval_space_final != -1
                if valid_cm_mask.sum() > 0:
                    cm_fig = generate_confusion_matrix_figure(f_true_eval_cat[valid_cm_mask].numpy(),f_preds_eval_space_final[valid_cm_mask].numpy(),cm_labels,cm_title)
                    tb_logger.writer.add_figure(f"Run_{fold_run_id}/ZZ_ConfusionMatrix_BestModel",cm_fig,final_epoch_num_completed); plt.close(cm_fig)
                else: logger.warning("No valid preds for CM.")
                
                if CV2_AVAILABLE:
                    logger.info("Generating Grad-CAMs for MEL & BCC...")
                    grad_cam_targets = {"MEL": final_l2i_model.get("MEL"), "BCC": final_l2i_model.get("BCC")}
                    grad_cam_layer = 'conv_head' # For EfficientNet
                    denorm_m,denorm_s = cpu_augmentations_cfg.get("norm_mean"),cpu_augmentations_cfg.get("norm_std") # From current run's config
                    
                    for cls_name_gc,model_cls_idx_gc in grad_cam_targets.items():
                        if model_cls_idx_gc is None: logger.warning(f"GC: Class '{cls_name_gc}' not in model outputs."); continue
                        eval_cls_idx_gc = final_l2i_eval.get(cls_name_gc)
                        if eval_cls_idx_gc is None: logger.warning(f"GC: Class '{cls_name_gc}' not in eval map."); continue
                        
                        found_gc_samples=0
                        for i_gc in range(len(val_ds)): # val_ds is from outer scope of train_one_fold
                            img_gc_transformed, true_lbl_gc_eval = val_ds[i_gc] # val_ds labels are in eval_idx space
                            if true_lbl_gc_eval == eval_cls_idx_gc and found_gc_samples < 2:
                                img_gc_denorm = denormalize_image(img_gc_transformed.clone(),denorm_m,denorm_s)
                                overlay_rgb = generate_grad_cam_overlay(model_final_eval,grad_cam_layer,img_gc_transformed,img_gc_denorm,model_cls_idx_gc,device)
                                if overlay_rgb is not None:
                                    overlay_tensor = torch.from_numpy(overlay_rgb).permute(2,0,1).float()/255.0 # HWC,RGB to CHW,Float
                                    tb_logger.writer.add_image(f"Run_{fold_run_id}/ZZ_GradCAM_{cls_name_gc}_Sample{found_gc_samples+1}",overlay_tensor,final_epoch_num_completed)
                                found_gc_samples+=1
                            if found_gc_samples>=2: break
                        if found_gc_samples==0: logger.warning(f"GC: No '{cls_name_gc}' samples in val_ds.")
                else: logger.warning("CV2 not available. Skipping Grad-CAMs.")

            metrics_rows=[] # For the text table in TB
            # (Populate metrics_rows similar to evaluate_fold_summary.py, using f_preds_eval_space_final and f_true_eval_cat)
            # This part is duplicated from the original script's logic for final metrics table.
            # For brevity, I'll add placeholders, ensure you fill this similarly to how it was done before.
            valid_final_mask = f_preds_eval_space_final != -1
            f_true_final = f_true_eval_cat[valid_final_mask]
            f_pred_final = f_preds_eval_space_final[valid_final_mask]

            if len(f_true_final) > 0:
                acc_overall = (f_pred_final == f_true_final).sum().item() / len(f_true_final)
                metrics_rows.append(["Overall Accuracy", f"{acc_overall:.4f}"])
                if final_n_eval_cls > 0:
                    f1_macro_eval = F1Score(task="multiclass", num_classes=final_n_eval_cls, average="macro", ignore_index=-1)(f_pred_final, f_true_final).item()
                    metrics_rows.append(["Macro F1-Score (Eval Space)", f"{f1_macro_eval:.4f}"])
                    mean_sens_eval = Recall(task="multiclass", num_classes=final_n_eval_cls, average="macro", ignore_index=-1, zero_division=0)(f_pred_final, f_true_final).item()
                    metrics_rows.append(["Mean Sensitivity (Eval Space)", f"{mean_sens_eval:.4f}"])
            else:
                metrics_rows.extend([["Overall Accuracy", "N/A"], ["Macro F1-Score (Eval Space)", "N/A"], ["Mean Sensitivity (Eval Space)", "N/A"]])

            # ... (add AUROC, pAUROC, per-class metrics to metrics_rows as before) ...
            true_auc_m_l_f, auc_mask_f = [], torch.zeros_like(f_true_eval_cat, dtype=torch.bool)
            for i, true_eval_s_f in enumerate(f_true_eval_cat.tolist()):
                lbl_s_f = final_i2l_eval.get(true_eval_s_f)
                if lbl_s_f and lbl_s_f in final_l2i_model: true_auc_m_l_f.append(final_l2i_model[lbl_s_f]); auc_mask_f[i] = True
            if true_auc_m_l_f:
                true_auc_t_f = torch.tensor(true_auc_m_l_f, dtype=torch.long); probs_auc_f = f_probs_cat[auc_mask_f]
                if probs_auc_f.size(0) == true_auc_t_f.size(0) and probs_auc_f.size(0) > 0:
                    metrics_rows.append(["Macro AUROC (Model Space)", f"{AUROC(task='multiclass', num_classes=final_n_model_cls, average='macro')(probs_auc_f, true_auc_t_f).item():.4f}"])
                    pauc_fpr_cfg_f = cfg_from_ckpt.get("training",{}).get("pauc_max_fpr", 0.2)
                    metrics_rows.append([f"Macro pAUROC@FPR{pauc_fpr_cfg_f} (Model Space)", f"{AUROC(task='multiclass', num_classes=final_n_model_cls, average='macro', max_fpr=pauc_fpr_cfg_f)(probs_auc_f, true_auc_t_f).item():.4f}"])
                else: metrics_rows.extend([["Macro AUROC (Model Space)", "N/A"], [f"Macro pAUROC@FPR{cfg_from_ckpt.get('training',{}).get('pauc_max_fpr',0.2)} (Model Space)", "N/A"]])
            else: metrics_rows.extend([["Macro AUROC (Model Space)", "N/A"], [f"Macro pAUROC@FPR{cfg_from_ckpt.get('training',{}).get('pauc_max_fpr',0.2)} (Model Space)", "N/A"]])


            if tb_logger.writer and metrics_rows:
                hdr_md="| Metric                        | Value   |\n"; sep_md="|-------------------------------|---------|\n"; table_str=hdr_md+sep_md+"".join([f"| {n:<29} | {v:<7} |\n" for n,v in metrics_rows])
                tb_logger.writer.add_text(f"Run_{fold_run_id}/ZZ_Final_Metrics_Table",table_str,final_epoch_num_completed)
        else: logger.warning(f"R{fold_run_id}: Best ckpt not found. Skipping final reports.")
    else: logger.warning(f"R{fold_run_id}: No valid training. Skipping final reports.")
    # --- Save Last Checkpoint ---
    last_ckpt_path=run_ckpt_dir/f"{exp_name_for_files}_run_{fold_run_id}_last_E{final_epoch_num_completed}.pt"
    last_ckpt_data = {'epoch':final_epoch_num_completed,'model_state_dict':getattr(model,'_orig_mod',model).state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'current_primary_metric':last_ckpt_metric_value,'config_runtime':cfg,'label2idx_model':label2idx_model,'label2idx_eval':label2idx_eval}
    if ema_model: last_ckpt_data['ema_model_state_dict']=getattr(ema_model,'_orig_mod',ema_model).state_dict()
    if post_hoc_is_active: last_ckpt_data['post_hoc_unk_threshold_used']=post_hoc_thresh_val # type: ignore
    torch.save(last_ckpt_data,str(last_ckpt_path)); logger.info(f"Saved last model to {last_ckpt_path}")
    tb_logger.close(); logger.info(f"Finished R'{fold_run_id}'. Best {model_selection_metric_str}: {best_metric_value:.4f}")
    return best_metric_value if best_metric_value > -float('inf') else None


def main():
    ap = argparse.ArgumentParser(description="Train a single specified fold with optimized logic.")
    ap.add_argument("exp_name", help="Experiment name for config and output naming.")
    ap.add_argument("--config_file", default=None, help="Path to specific YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config if set.")
    ap.add_argument("--fold_id_to_run", type=str, required=True,
                        help="The specific fold ID to run for training and validation.")
    args = ap.parse_args()

    if args.config_file: cfg_path = Path(args.config_file)
    else: cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"

    if not cfg_path.exists():
        fallback_cfg_path = Path(args.config_dir) / "config_single_fold_optimized.yaml" # Default name
        if not args.config_file and fallback_cfg_path.exists():
            logger.warning(f"Config {cfg_path} not found. Using fallback {fallback_cfg_path}"); cfg_path = fallback_cfg_path
        else:
            raise FileNotFoundError(f"Config file {cfg_path} not found (and fallback not used/found).")
    cfg = load_config(cfg_path); logger.info(f"Loaded config from {cfg_path}"); cfg = cast_config_values(cfg)

    exp_setup_cfg = cfg.get("experiment_setup", {})
    current_seed = args.seed if args.seed is not None else exp_setup_cfg.get("seed", 42)
    set_seed(current_seed); logger.info(f"Seed set to {current_seed}")
    cfg["experiment_setup"]["seed_runtime"] = current_seed

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths_cfg = cfg.get("paths", {})
    config_proj_root = paths_cfg.get("project_root")
    base_path_for_config_paths = Path(config_proj_root).resolve() if config_proj_root and Path(config_proj_root).is_dir() else cfg_path.parent

    fold_id_for_paths = str(args.fold_id_to_run).replace(" ", "_").replace("/", "-")

    base_log_dir = _get_path_from_config(cfg, "log_dir", default=f"outputs/tb_{args.exp_name}", base_path=base_path_for_config_paths)
    base_ckpt_dir = _get_path_from_config(cfg, "ckpt_dir", default=f"outputs/ckpt_{args.exp_name}", base_path=base_path_for_config_paths)

    run_specific_log_dir = base_log_dir / args.exp_name / f"run_{fold_id_for_paths}" / timestamp
    run_specific_ckpt_dir = base_ckpt_dir / args.exp_name / f"run_{fold_id_for_paths}" / timestamp
    run_specific_log_dir.mkdir(parents=True, exist_ok=True); run_specific_ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log directory for this run: {run_specific_log_dir}")
    logger.info(f"Checkpoint directory for this run: {run_specific_ckpt_dir}")

    labels_csv_p = _get_path_from_config(cfg, "labels_csv", base_path=base_path_for_config_paths)
    # In FlatDataset, paths are constructed as root/dataset_col/filename_col
    # So, train_root_path should be the directory *containing* the dataset subdirectories (like ISIC2019, PAD20 etc.)
    # If your labels_csv 'dataset' column correctly names these subdirectories,
    # and 'filename' is just the image file name, then train_root_p should point to the parent of these dataset folders.
    # From your YAML, train_root is "splits/training". Assuming "splits/training/ISIC2019", "splits/training/PAD20" etc. exist.
    train_root_p = _get_path_from_config(cfg, "train_root", base_path=base_path_for_config_paths) # This should be "splits/training"
    
    if not labels_csv_p.exists(): raise FileNotFoundError(f"Labels CSV not found: {labels_csv_p}")
    if not train_root_p.is_dir(): raise FileNotFoundError(f"Train root dir not found or not a directory: {train_root_p}")

    df_full = pd.read_csv(labels_csv_p)

    train_cfg_main = cfg.get("training", {})
    known_training_labels_cfg = cfg.get("data",{}).get("known_training_labels", None)
    exclude_unk_from_training = train_cfg_main.get("exclude_unk_from_training_model", False)
    unk_label_string_for_model_exclusion = train_cfg_main.get("unk_label_string") # This can be None (null in YAML)
    
    df_for_model_classes = df_full.copy()

    if known_training_labels_cfg and isinstance(known_training_labels_cfg, list):
        logger.info(f"Model training classes explicitly defined by 'known_training_labels': {known_training_labels_cfg}")
        df_for_model_classes = df_full[df_full['label'].isin(known_training_labels_cfg)].copy()
    elif exclude_unk_from_training and unk_label_string_for_model_exclusion is not None: # Only exclude if string is provided
        logger.info(f"Excluding label '{unk_label_string_for_model_exclusion}' from model training classes based on config.")
        df_for_model_classes = df_full[df_full['label'] != unk_label_string_for_model_exclusion].copy()
    elif exclude_unk_from_training and unk_label_string_for_model_exclusion is None:
        logger.info("'exclude_unk_from_training_model' is true, but 'unk_label_string' is null. No classes will be excluded by this rule.")


    labels_unique_model = sorted(df_for_model_classes['label'].unique())
    if not labels_unique_model: raise ValueError("No unique labels for model training after filtering.")
    label2idx_model = {name: i for i, name in enumerate(labels_unique_model)}

    cfg.setdefault("model", {})["numClasses"] = len(label2idx_model) # Ensure model config has numClasses
    cfg['label2idx_model_runtime'] = label2idx_model
    logger.info(f"Model trains on {len(label2idx_model)} classes: {label2idx_model}")

    all_labels_in_csv = sorted(df_full['label'].unique())
    label2idx_eval = {name: i for i, name in enumerate(all_labels_in_csv)}
    cfg['label2idx_eval_runtime'] = label2idx_eval
    logger.info(f"Evaluation uses {len(label2idx_eval)} classes (from CSV): {label2idx_eval}")

    default_device_val = "cpu"
    if torch.cuda.is_available(): default_device_val = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): default_device_val = "mps"
    device_str = exp_setup_cfg.get("device", default_device_val); device = get_device(device_str)
    cfg["experiment_setup"]["device_runtime"] = str(device)

    fold_id_from_arg_typed = args.fold_id_to_run
    if pd.api.types.is_numeric_dtype(df_full['fold']):
        try: fold_id_from_arg_typed = int(args.fold_id_to_run)
        except ValueError: logger.info(f"Fold ID '{args.fold_id_to_run}' kept as str for numeric 'fold' col.")
    if fold_id_from_arg_typed not in df_full['fold'].unique():
        logger.error(f"Fold ID '{fold_id_from_arg_typed}' not in CSV. Has: {df_full['fold'].unique().tolist()}"); return

    val_fold_identifier = fold_id_from_arg_typed
    train_df_selected = df_for_model_classes[df_for_model_classes['fold'] != val_fold_identifier].reset_index(drop=True)
    val_df_selected = df_full[df_full['fold'] == val_fold_identifier].reset_index(drop=True)

    if train_df_selected.empty or val_df_selected.empty:
        logger.error(f"Fold {args.fold_id_to_run}: empty train ({len(train_df_selected)}) or val ({len(val_df_selected)})."); return
    logger.info(f"Fold '{args.fold_id_to_run}': Train={len(train_df_selected)}, Val={len(val_df_selected)}")

    best_metric_for_run_val = train_one_fold(
        fold_run_id=args.fold_id_to_run, train_df=train_df_selected, val_df=val_df_selected, cfg=cfg,
        label2idx_model=label2idx_model, label2idx_eval=label2idx_eval, train_root_path=train_root_p,
        run_log_dir=run_specific_log_dir, run_ckpt_dir=run_specific_ckpt_dir,
        exp_name_for_files=args.exp_name, device=device
    )

    if best_metric_for_run_val is not None:
        logger.info(f"\n===== Single Fold Run Finished (ID: {args.fold_id_to_run}) =====")
        logger.info(f"Best {cfg.get('training',{}).get('model_selection_metric', 'N/A')}: {best_metric_for_run_val:.4f}")
    else: logger.warning(f"Fold run '{args.fold_id_to_run}' did not produce a best metric value.")
    logger.info(f"Experiment {args.exp_name} (ID {args.fold_id_to_run}, ts: {timestamp}) finished.")

if __name__ == "__main__":
    # Check for CV2 availability at the start of main if Grad-CAM is a critical feature.
    if not CV2_AVAILABLE:
        logger.warning("OpenCV (cv2) is not available. Grad-CAM functionality will be skipped.")
    main()
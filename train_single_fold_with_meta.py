#!/usr/bin/env python
# train_single_fold_with_meta.py
#
# Single‐fold trainer with two phases:
#   Phase 1: Joint training of CNN + meta‐head (full CNN logic: freeze, backbone LR, AMP, EMA, etc.)
#   Phase 2: Freeze CNN backbone; fine‐tune metadata head only
#
# Usage:
#   python train_single_fold_with_meta.py exp_name --fold_id_to_run FOLD_ID [--config_file CONFIG_FILE] [--seed SEED]

from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import time
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path
import contextlib

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from torchmetrics import AUROC, F1Score, Recall
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
    get_device, CudaTimer, TensorBoardLogger
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)


def _get_path_from_config(
    cfg: dict, key: str, default: str | None = None, base_path: Path | None = None
) -> Path:
    paths_cfg = cfg.get("paths", {})
    path_str = paths_cfg.get(key)
    if path_str is None:
        if default is not None:
            logger.warning(f"Path for '{key}' not in config. Using default: '{default}'")
            path_str = default
        else:
            logger.error(f"Required path for '{key}' not found. Config paths: {paths_cfg}")
            raise ValueError(f"Missing path for '{key}'")
    path = Path(path_str)
    if base_path and not path.is_absolute():
        path = base_path / path
    return path.resolve()


def generate_confusion_matrix_figure(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    display_labels: list[str],
    title: str
) -> plt.Figure:
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig_s_base, fig_s_factor = 8, 0.6
    fig_w = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig_h = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def get_class_counts(df: pd.DataFrame, label2idx: dict[str, int]) -> np.ndarray:
    num_classes = len(label2idx)
    counts = np.zeros(num_classes, dtype=int)
    mapped = df['label'].map(lambda x: label2idx.get(x, -1))
    valid = mapped[mapped != -1]
    vc = valid.value_counts()
    for idx, cnt in vc.items():
        counts[int(idx)] = int(cnt)
    return counts


class SimpleGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _find_module(self, name: str) -> torch.nn.Module:
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"Could not find layer '{name}' in model.")

    def _hook_activations(self, module, input, output):
        self.activations = output.detach()

    def _hook_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        target_module = self._find_module(self.target_layer_name)
        target_module.register_forward_hook(self._hook_activations)
        target_module.register_full_backward_hook(self._hook_gradients)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.activations = None
        self.gradients = None

        self.model.zero_grad()
        preds = self.model(input_tensor)
        score = preds[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0]   # [C, h, w]
        acts = self.activations[0]  # [C, h, w]

        weights = grads.view(grads.size(0), -1).mean(dim=1)  # [C]
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)     # [h, w]
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.cpu().numpy()


def run_training_phase(
    phase_name: str,
    model: CNNWithMetadata,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module | callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    phase_cfg: dict,
    fold_str: str,
    label2idx_model: dict[str, int],
    label2idx_eval: dict[str, int],
    tb_logger: TensorBoardLogger,
    ckpt_dir_phase: Path,
    exp_name: str,
    start_epoch: int = 0,
    ema_model: CNNWithMetadata | None = None,
    initial_best_metric: float = -float('inf'),
) -> tuple[float, int, str]:
    """
    Single‐phase training loop. Used for both:
      • Phase 1 (joint CNN+meta)
      • Phase 2 (meta‐only, with CNN frozen)

    Returns:
      best_metric (float), best_epoch (int), best_ckpt_path (str)
    """
    main_cfg = cfg.get("training", {})
    amp_enabled = main_cfg.get("amp_enabled", True)
    use_ema_for_val = main_cfg.get("use_ema_for_val", True)
    model_sel_metric = main_cfg.get("model_selection_metric", "macro_auc").lower()

    # Phase-specific params
    num_epochs_phase = phase_cfg.get("num_epochs", phase_cfg.get("epochs", 1))
    accum_steps_phase = phase_cfg.get("accum_steps", main_cfg.get("accum_steps", 1))
    early_stop_patience_phase = phase_cfg.get("early_stopping_patience", main_cfg.get("early_stopping_patience", 10))
    freeze_epochs = phase_cfg.get("freeze_epochs", 0)
    backbone_lr_mult = phase_cfg.get("backbone_lr_mult", 1.0)
    drw_epochs = phase_cfg.get("drw_schedule_epochs", [])

    best_metric = initial_best_metric
    best_epoch = -1
    best_ckpt_path = ""
    epoch_global = start_epoch

    # Precompute class_counts & LDAM params if necessary
    use_ldam = False
    if hasattr(criterion, "update_weights"):
        use_ldam = True

    logger.info(f"Starting {phase_name} (F{fold_str}) for {num_epochs_phase} epochs: global epochs [{start_epoch}..{start_epoch+num_epochs_phase-1}].")

    for epoch_offset in range(num_epochs_phase):
        epoch = start_epoch + epoch_offset
        epoch_global = epoch

        # If using a sampler with set_epoch (e.g., distributed)
        if hasattr(train_loader.sampler, 'set_epoch') and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        # Handle freeze/backbone-LR only in Phase 1
        if phase_name == "P1_Joint" and (freeze_epochs > 0) and (epoch == freeze_epochs):
            # Unfreeze entire model after freeze_epochs
            for p in model.parameters():
                p.requires_grad_(True)

            # Rebuild optimizer param groups
            base_lr = phase_cfg.get("optimizer", {}).get("lr", 1e-3)
            head_prefixes: list[str] = []
            for name, mod in model.named_modules():
                if name.endswith("metadata_mlp") or name.endswith("post_concat") or name.endswith("classifier"):
                    head_prefixes.append(f"{name}.")
            head_params, back_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(n.startswith(pfx) for pfx in head_prefixes):
                    head_params.append(p)
                else:
                    back_params.append(p)
            groups = []
            if head_params:
                groups.append({"params": head_params, "lr": base_lr})
            if back_params:
                groups.append({"params": back_params, "lr": base_lr * backbone_lr_mult})
            if not groups:
                groups = [{"params": model.parameters(), "lr": base_lr}]
            optimizer = AdamW(groups, lr=groups[0]["lr"], weight_decay=phase_cfg.get("optimizer", {}).get("weight_decay", 1e-4))
            rem = num_epochs_phase - freeze_epochs
            if phase_cfg.get("scheduler", {}).get("type", "").lower() == "steplr":
                step_size = phase_cfg.get("scheduler", {}).get("step_size", 10)
                gamma = phase_cfg.get("scheduler", {}).get("gamma", 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=(rem if rem > 0 else 1), eta_min=phase_cfg.get("scheduler", {}).get("min_lr", 0.0))
            logger.info(f"[{phase_name}] Unfroze at epoch {epoch}, reinitialized optimizer/scheduler.")

        # If using LDAM+DRW, update weights at specified epochs
        if phase_name == "P1_Joint" and use_ldam and (drw_epochs is not None) and (len(drw_epochs) > 0):
            if (epoch_offset < len(drw_epochs)) and (epoch == drw_epochs[epoch_offset]):
                class_counts = phase_cfg.get("_class_counts", None)
                beta = phase_cfg.get("loss", {}).get("ldam_params", {}).get("effective_number_beta", 0.999)
                eff_num = 1.0 - np.power(beta, class_counts)
                drw_w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
                drw_w = drw_w / drw_w.sum() * len(class_counts)
                w_tensor = torch.tensor(drw_w, dtype=torch.float32, device=device)
                criterion.update_weights(w_tensor)
                logger.info(f"[{phase_name}] DRW at E{epoch} → weights (first5): {drw_w[:5]}")
            elif (epoch == 0):
                criterion.update_weights(None)

        # ── Train Loop ──
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"F{fold_str} E{epoch} {phase_name} Train", ncols=cfg.get("experiment_setup", {}).get("TQDM_NCOLS", 120), leave=False)
        epoch_gpu_time_ms = 0.0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(pbar):
            # Unpack either 3‐tuple (imgs, meta, labels)
            # or 2‐tuple ( (imgs, meta), labels ), or (imgs, labels)
            if len(batch) == 3:
                imgs_cpu, meta_cpu, labels_cpu = batch
            elif len(batch) == 2:
                first, second = batch
                # if the first element is a pair (imgs, meta)
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    imgs_cpu, meta_cpu = first
                    labels_cpu = second
                else:
                    imgs_cpu, labels_cpu = batch
                    meta_dim = model.metadata_mlp.fc1.in_features
                    meta_cpu = torch.zeros((labels_cpu.size(0), meta_dim), dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected batch structure: len(batch)={len(batch)}")

            # ─── Ensure img/meta/label are Tensors ───
            if isinstance(imgs_cpu, (list, tuple)):
                imgs_cpu = torch.stack(imgs_cpu, dim=0)
            if isinstance(meta_cpu, (list, tuple)):
                meta_cpu = torch.stack(meta_cpu, dim=0)
            if isinstance(labels_cpu, (list, tuple)):
                labels_cpu = torch.tensor(labels_cpu, dtype=torch.long)
            # ─── End conversion ───

            imgs = imgs_cpu.to(device, non_blocking=True)
            meta = meta_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)

            # Simple metadata augmentation: zero‐out each feature with p=0.1 (only during training)
            if phase_name in ("P1_Joint", "P2_MetaTune"):
                prob_mask = (torch.rand_like(meta) < 0.1).float()
                meta = meta * (1.0 - prob_mask)

            timer_active = False
            if cfg.get("tensorboard_logging", {}).get("profiler", {}).get("enable_batch_timing_always", False):
                timer_active = True
            if timer_active and device.type == 'cuda':
                batch_timer = CudaTimer(device)
                batch_timer.__enter__()
            else:
                batch_timer = contextlib.nullcontext()

            with batch_timer:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(imgs, meta)
                    loss = criterion(logits, labels)
                    if accum_steps_phase > 1:
                        loss = loss / accum_steps_phase

                scaler.scale(loss).backward()
                if ((batch_idx + 1) % accum_steps_phase == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema_model is not None and (phase_name == "P1_Joint"):
                        update_ema(ema_model, model, main_cfg.get("ema_decay", 0.0))

            if timer_active and device.type == 'cuda':
                epoch_gpu_time_ms += batch_timer.get_elapsed_time_ms()

            batch_loss = loss.item() * (accum_steps_phase if accum_steps_phase > 1 else 1)
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            running_loss += batch_loss * bs
            running_correct += correct
            running_total += bs

            avg_loss = running_loss / running_total if running_total > 0 else 0.0
            avg_acc = running_correct / running_total if running_total > 0 else 0.0

            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

            # Log batch‐level metrics
            if tb_logger and tb_logger.writer:
                interval = getattr(tb_logger, "log_interval_batches_train", None)
                if interval is not None and (batch_idx % interval == 0):
                    step = epoch * len(train_loader) + batch_idx
                    tb_logger.writer.add_scalar(f"Train/Loss_batch_{phase_name}", avg_loss, step)
                    tb_logger.writer.add_scalar(f"Train/Acc_batch_{phase_name}", avg_acc, step)
                    tb_logger.writer.add_scalar(f"Train/LR_{phase_name}", optimizer.param_groups[0]['lr'], step)

        pbar.close()
        epoch_duration = time.time() - epoch_start
        scheduler.step()

        epoch_train_loss = running_loss / running_total if running_total > 0 else 0.0
        epoch_train_acc = running_correct / running_total if running_total > 0 else 0.0

        # Log epoch‐level metrics
        if tb_logger and tb_logger.writer:
            tb_logger.writer.add_scalar(f"Train/Loss_epoch_{phase_name}", epoch_train_loss, epoch)
            tb_logger.writer.add_scalar(f"Train/Acc_epoch_{phase_name}", epoch_train_acc, epoch)
            tb_logger.writer.add_scalar(f"Train/LR_epoch_{phase_name}", optimizer.param_groups[0]['lr'], epoch)
            tb_logger.writer.add_scalar(f"Time/Train_epoch_duration_sec_{phase_name}", epoch_duration, epoch)
            if device.type == 'cuda':
                tb_logger.writer.add_scalar(f"Time/GPU_ms_per_train_epoch_{phase_name}", epoch_gpu_time_ms, epoch)

        # ── Validation Loop ──
        val_loss_sum = 0.0
        val_correct = 0
        val_seen = 0
        all_logits_list: list[torch.Tensor] = []
        all_true_list: list[torch.Tensor] = []

        do_val = ((epoch % phase_cfg.get("val_interval", main_cfg.get("val_interval", 1)) == 0) or
                  (epoch_offset == num_epochs_phase - 1))
        if do_val:
            eval_model = ema_model if (ema_model is not None and use_ema_for_val and phase_name == "P1_Joint") else model
            eval_model.eval()
            pbar_val = tqdm(val_loader, desc=f"F{fold_str} E{epoch} {phase_name} Val", ncols=cfg.get("experiment_setup", {}).get("TQDM_NCOLS", 120), leave=False)
            with torch.no_grad():
                for batch in pbar_val:
                    if len(batch) == 3:
                        imgs_cpu, meta_cpu, labels_cpu = batch
                    elif len(batch) == 2:
                        first, second = batch
                        if isinstance(first, (list, tuple)) and len(first) == 2:
                            imgs_cpu, meta_cpu = first
                            labels_cpu = second
                        else:
                            imgs_cpu, labels_cpu = batch
                            meta_dim = model.metadata_mlp.fc1.in_features
                            meta_cpu = torch.zeros((labels_cpu.size(0), meta_dim), dtype=torch.float32)
                    else:
                        raise ValueError(f"Unexpected batch structure: len(batch)={len(batch)}")

                    if isinstance(imgs_cpu, (list, tuple)):
                        imgs_cpu = torch.stack(imgs_cpu, dim=0)
                    if isinstance(meta_cpu, (list, tuple)):
                        meta_cpu = torch.stack(meta_cpu, dim=0)
                    if isinstance(labels_cpu, (list, tuple)):
                        labels_cpu = torch.tensor(labels_cpu, dtype=torch.long)

                    imgs = imgs_cpu.to(device, non_blocking=True)
                    meta = meta_cpu.to(device, non_blocking=True)
                    labels = labels_cpu.to(device, non_blocking=True)

                    with autocast(device_type=device.type, enabled=amp_enabled):
                        logits = eval_model(imgs, meta)
                        loss_v = criterion(logits, labels)
                        probs = F.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    correct = (preds == labels).sum().item()
                    bs = labels.size(0)

                    val_loss_sum += loss_v.item() * bs
                    val_correct += correct
                    val_seen += bs

                    all_logits_list.append(logits.cpu())
                    all_true_list.append(labels.cpu())

                    avg_val_loss = val_loss_sum / val_seen if val_seen > 0 else 0.0
                    avg_val_acc = val_correct / val_seen if val_seen > 0 else 0.0
                    pbar_val.set_postfix(val_loss=f"{avg_val_loss:.4f}", val_acc=f"{avg_val_acc:.4f}")
                pbar_val.close()

            avg_val_loss = val_loss_sum / val_seen if val_seen > 0 else 0.0
            avg_val_acc = val_correct / val_seen if val_seen > 0 else 0.0

            all_logits_cat = torch.cat(all_logits_list, dim=0)
            all_probs = F.softmax(all_logits_cat, dim=1)
            all_true_cat = torch.cat(all_true_list, dim=0)

            # Compute metrics
            f1_macro = F1Score(task="multiclass", num_classes=len(label2idx_model), average="macro")(
                all_probs.argmax(dim=1), all_true_cat
            ).item()
            try:
                auroc_macro = AUROC(task="multiclass", num_classes=len(label2idx_model), average="macro")(
                    all_probs, all_true_cat
                ).item()
            except Exception:
                auroc_macro = float("nan")
            try:
                pauc_macro = AUROC(task="multiclass", num_classes=len(label2idx_model), average="macro", max_fpr=main_cfg.get("pauc_max_fpr", 0.2))(
                    all_probs, all_true_cat
                ).item()
            except Exception:
                pauc_macro = float("nan")
            sens_macro = Recall(task="multiclass", num_classes=len(label2idx_model), average="macro", zero_division=0)(
                all_probs.argmax(dim=1), all_true_cat
            ).item()

            if tb_logger and tb_logger.writer:
                tb_logger.writer.add_scalar(f"Val/Loss_epoch_{phase_name}", avg_val_loss, epoch)
                tb_logger.writer.add_scalar(f"Val/Acc_epoch_{phase_name}", avg_val_acc, epoch)
                tb_logger.writer.add_scalar(f"Val/F1_macro_{phase_name}", f1_macro, epoch)
                tb_logger.writer.add_scalar(f"Val/AUROC_macro_{phase_name}", auroc_macro, epoch)
                tb_logger.writer.add_scalar(f"Val/pAUROC@{main_cfg.get('pauc_max_fpr', 0.2)}_{phase_name}", pauc_macro, epoch)
                tb_logger.writer.add_scalar(f"Val/Sensitivity_macro_{phase_name}", sens_macro, epoch)

            logger.info(
                f"[F{fold_str}][E{epoch}][{phase_name}] Val → "
                f"Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f} "
                f"F1={f1_macro:.4f} AUROC={auroc_macro:.4f} Sens={sens_macro:.4f}"
            )

            # Optional: save PR‐curve thresholds
            save_pr = main_cfg.get("save_optimal_thresholds_from_pr", False)
            opt_thresholds: dict[int, float] = {}
            if save_pr:
                n_cls = len(label2idx_model)
                pr_true = all_true_cat.numpy()
                pr_probs_np = all_probs.numpy()
                for cls_i in range(n_cls):
                    onehot = (pr_true == cls_i).astype(int)
                    try:
                        p_arr, r_arr, t_arr = precision_recall_curve(onehot, pr_probs_np[:, cls_i])
                        f1_arr = (2 * p_arr * r_arr) / (p_arr + r_arr + 1e-8)
                        if len(f1_arr) > 1:
                            best_idx = int(np.nanargmax(f1_arr[1:])) + 1
                            opt_thresholds[cls_i] = float(t_arr[best_idx])
                        else:
                            opt_thresholds[cls_i] = 0.5
                    except Exception:
                        opt_thresholds[cls_i] = 0.5
                if tb_logger and tb_logger.writer:
                    tb_logger.writer.add_text(f"{phase_name}/optimal_thresholds", str(opt_thresholds), epoch)

            # Primary metric selection
            metric_map = {
                "macro_auc": auroc_macro,
                "mean_optimal_f1": f1_macro if save_pr else f1_macro,
                "mean_optimal_sensitivity": sens_macro if save_pr else sens_macro,
            }
            current_primary = metric_map.get(model_sel_metric, sens_macro)

            if current_primary > best_metric:
                best_metric = current_primary
                best_epoch = epoch
                ckpt_name = f"{exp_name}_fold{fold_str}_{phase_name}_best.pt"
                best_ckpt_path = str(ckpt_dir_phase / ckpt_name)
                logger.info(f"New best [{phase_name}] ({model_sel_metric}) = {best_metric:.4f} at E{epoch}. Saving → {best_ckpt_path}")

                ckpt_data: dict = {
                    "epoch": epoch,
                    "model_state_dict": getattr(eval_model, '_orig_mod', eval_model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": cfg,
                    "label2idx_model": label2idx_model,
                    "label2idx_eval": label2idx_eval,
                    "phase_name": phase_name,
                    f"best_{model_sel_metric}": best_metric,
                    "metadata_dim": model.metadata_mlp.fc1.in_features
                }
                if ema_model is not None and (phase_name == "P1_Joint"):
                    ckpt_data["ema_model_state_dict"] = getattr(ema_model, '_orig_mod', ema_model).state_dict()
                if save_pr and opt_thresholds:
                    ckpt_data["optimal_thresholds"] = opt_thresholds

                torch.save(ckpt_data, best_ckpt_path)
            else:
                early_stop_patience_phase -= 1

            if early_stop_patience_phase <= 0:
                logger.info(f"[{phase_name}] Early stopping at E{epoch}.")
                break

            model.train()

        # End epoch
        if tb_logger:
            tb_logger.log_epoch_summary({
                f"Loss/train_epoch_{phase_name}": epoch_train_loss,
                f"Accuracy/train_epoch_{phase_name}": epoch_train_acc,
                f"LearningRate/epoch_{phase_name}": optimizer.param_groups[0]['lr'],
                f"Time/Train_epoch_duration_sec_{phase_name}": epoch_duration
            }, epoch)
        if early_stop_patience_phase <= 0:
            break

    # Save “last” checkpoint for this phase
    last_ckpt_name = f"{exp_name}_fold{fold_str}_{phase_name}_last.pt"
    last_ckpt_path = ckpt_dir_phase / last_ckpt_name
    last_data: dict = {
        "epoch": epoch_global,
        "model_state_dict": getattr(model, '_orig_mod', model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg,
        "label2idx_model": label2idx_model,
        "label2idx_eval": label2idx_eval,
        "phase_name": phase_name,
        "current_primary_metric": best_metric,
        "metadata_dim": model.metadata_mlp.fc1.in_features
    }
    if ema_model is not None and (phase_name == "P1_Joint"):
        last_data["ema_model_state_dict"] = getattr(ema_model, '_orig_mod', ema_model).state_dict()
    torch.save(last_data, str(last_ckpt_path))
    logger.info(f"[{phase_name}] Saved last checkpoint → {last_ckpt_path}")

    return best_metric, best_epoch, best_ckpt_path


def train_one_fold_with_meta(
    fold_id: int | str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: dict,
    label2idx_model: dict[str, int],
    label2idx_eval: dict[str, int],
    image_root: Path,
    log_dir: Path,
    ckpt_dir: Path,
    exp_name: str,
    device: torch.device,
) -> float | None:
    """
    Orchestrates:
      • Phase 1: Joint CNN+Meta training (full CNN logic)
      • Phase 2: Freeze CNN, fine‐tune Meta head only

    Returns overall best validation metric across both phases.
    """
    fold_str = str(fold_id)
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    meta_tune_cfg = cfg.get("meta_tuning", {})

    # Build TensorBoard logger
    tb_train_len = (len(train_df) + training_cfg.get("batch_size", 32) - 1) // training_cfg.get("batch_size", 32)
    tb_logger = TensorBoardLogger(log_dir=log_dir, experiment_config=cfg, train_loader_len=tb_train_len)

    # Transforms
    tf_train = build_transform(data_cfg.get("cpu_augmentations", {}), train=True)
    tf_val = build_transform(data_cfg.get("cpu_augmentations", {}), train=False)

    # Datasets (joint CNN+Meta uses FlatDatasetWithMeta)
    dataset_args = {
        "meta_features_names": data_cfg.get("meta_features_names"),
        "meta_augmentation_p": data_cfg.get("meta_augmentation_p", 0.0),
        "meta_nan_fill_value": data_cfg.get("meta_nan_fill_value", 0.0),
        "image_loader": data_cfg.get("image_loader", "pil"),
        "enable_ram_cache": data_cfg.get("enable_ram_cache", False)
    }

    train_ds = FlatDatasetWithMeta(
        df=train_df,
        meta_df=meta_df,
        root=image_root,
        label2idx=label2idx_model,
        tf=tf_train,
        **dataset_args
    )
    val_ds = FlatDatasetWithMeta(
        df=val_df,
        meta_df=meta_df,
        root=image_root,
        label2idx=label2idx_eval,
        tf=tf_val,
        **dataset_args
    )

    # Record metadata_dim for model creation
    metadata_dim = train_ds.metadata_dim
    cfg['model']['metadata_input_dim_runtime'] = metadata_dim
    logger.info(f"Fold {fold_str}: detected metadata_dim = {metadata_dim}")

    # Instantiate CNN + Metadata model
    model_cfg = cfg.get("model", {})
    # Allow either base_cnn_type or type
    base_cnn_type = model_cfg.get("base_cnn_type", model_cfg.get("type"))
    base_cnn_params = {
        "MODEL_TYPE": base_cnn_type,
        "numClasses": len(label2idx_model),
        "pretrained": model_cfg.get("pretrained_cnn", True)
    }
    base_cnn = get_base_cnn_model(base_cnn_params)
    meta_head_args = model_cfg.get("meta_head_args", {}).copy()
    model = CNNWithMetadata(
        base_cnn_model=base_cnn,
        num_classes=len(label2idx_model),
        metadata_input_dim=metadata_dim,
        **meta_head_args
    ).to(device)

    # Set up EMA
    ema_model = None
    if training_cfg.get("ema_decay", 0.0) > 0:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA enabled, decay = {training_cfg.get('ema_decay')}")

    # ─── Build Phase 1 DataLoaders ───
    dl_kwargs_p1 = {
        "batch_size": training_cfg.get("batch_size", 32),
        "num_workers": data_cfg.get("num_workers", 0),
        "pin_memory": True,
        "drop_last": False
    }
    if dl_kwargs_p1["num_workers"] > 0:
        dl_kwargs_p1["persistent_workers"] = data_cfg.get("persistent_workers", False)
        dl_kwargs_p1["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)

    sampler_p1 = ClassBalancedSampler(train_ds, len(train_ds)) \
                 if data_cfg.get("sampler", {}).get("type") == "class_balanced_sqrt" else None

    train_loader_p1 = DataLoader(
        train_ds,
        sampler=sampler_p1,
        shuffle=(sampler_p1 is None),
        **dl_kwargs_p1
    )
    val_loader_p1 = DataLoader(
        val_ds,
        shuffle=False,
        **dl_kwargs_p1
    )

    # Phase 1: Build criterion
    loss_type = training_cfg.get("loss", {}).get("type", "cross_entropy").lower()
    if loss_type == "focal_ce_loss":
        criterion_p1 = focal_ce_loss
        logger.info(f"[F{fold_str}] Phase 1: Using focal_ce_loss.")
    elif loss_type == "ldam_loss":
        class_counts = get_class_counts(train_df, label2idx_model)
        ldam_params = training_cfg.get("loss", {}).get("ldam_params", {})
        criterion_p1 = LDAMLoss(
            class_counts=class_counts,
            max_margin=ldam_params.get("max_margin", 0.5),
            use_effective_number_margin=ldam_params.get("use_effective_number_margin", True),
            effective_number_beta=ldam_params.get("effective_number_beta", 0.999),
            scale=training_cfg.get("loss", {}).get("ldam_params", {}).get("scale", 30.0)
        ).to(device)
        training_cfg["_class_counts"] = class_counts
        logger.info(f"[F{fold_str}] Phase 1: Using LDAMLoss.")
    else:
        label_smoothing = training_cfg.get("loss", {}).get("label_smoothing", 0.0)
        criterion_p1 = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        logger.info(f"[F{fold_str}] Phase 1: Using CrossEntropyLoss (label_smoothing={label_smoothing}).")

    # Phase 1: Build optimizer + scheduler
    opt_cfg_p1 = training_cfg.get("optimizer", {})
    optimizer_p1 = AdamW(
        model.parameters(),
        lr=opt_cfg_p1.get("lr", 1e-3),
        weight_decay=opt_cfg_p1.get("weight_decay", 1e-4)
    )
    sched_cfg_p1 = training_cfg.get("scheduler", {})
    scheduler_p1 = CosineAnnealingLR(
        optimizer_p1,
        T_max=training_cfg.get("num_epochs", 1),
        eta_min=sched_cfg_p1.get("min_lr", 0.0)
    )

    scaler_p1 = GradScaler(enabled=(device.type == "cuda" and training_cfg.get("amp_enabled", True)))

    # Run Phase 1
    best_metric_p1, best_epoch_p1, best_ckpt_p1 = run_training_phase(
        phase_name="P1_Joint",
        model=model,
        train_loader=train_loader_p1,
        val_loader=val_loader_p1,
        criterion=criterion_p1,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        scaler=scaler_p1,
        device=device,
        cfg=cfg,
        phase_cfg=training_cfg,
        fold_str=fold_str,
        label2idx_model=label2idx_model,
        label2idx_eval=label2idx_eval,
        tb_logger=tb_logger,
        ckpt_dir_phase=ckpt_dir,
        exp_name=exp_name,
        start_epoch=0,
        ema_model=ema_model,
        initial_best_metric=-float('inf')
    )

    overall_best_metric = best_metric_p1
    overall_best_ckpt = best_ckpt_p1

    # Phase 2: Meta‐only tuning
    if meta_tune_cfg.get("enable", False) and best_epoch_p1 >= 0 and Path(best_ckpt_p1).exists():
        logger.info(f"[F{fold_str}] Loading best Phase 1 checkpoint for Phase 2: {best_ckpt_p1}")
        ckpt_data = torch.load(best_ckpt_p1, map_location=device)
        model.load_state_dict(ckpt_data['model_state_dict'])
        if ema_model is not None and ckpt_data.get('ema_model_state_dict') is not None:
            ema_model.load_state_dict(ckpt_data['ema_model_state_dict'])

        # Freeze CNN backbone; leave meta MLP + classifier trainable
        model.set_base_cnn_trainable(False)
        logger.info(f"[F{fold_str}] Phase 2: CNN backbone frozen; training metadata head only.")

        # Phase 2 DataLoaders (smaller batch size)
        dl_kwargs_p2 = {
            "batch_size": meta_tune_cfg.get("batch_size", 20),
            "num_workers": data_cfg.get("num_workers", 0),
            "pin_memory": True,
            "drop_last": False
        }
        if dl_kwargs_p2["num_workers"] > 0:
            dl_kwargs_p2["persistent_workers"] = data_cfg.get("persistent_workers", False)
            dl_kwargs_p2["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)

        train_loader_p2 = DataLoader(
            train_ds,
            shuffle=True,
            **dl_kwargs_p2
        )
        val_loader_p2 = DataLoader(
            val_ds,
            shuffle=False,
            **dl_kwargs_p2
        )

        # Phase 2: Optimizer on only params with requires_grad=True
        params_to_tune = [p for p in model.parameters() if p.requires_grad]
        if len(params_to_tune) == 0:
            logger.error(f"[F{fold_str}] Phase 2: No trainable parameters found. Skipping Phase 2.")
        else:
            opt_cfg_p2 = meta_tune_cfg.get("optimizer", {})
            optimizer_p2 = AdamW(
                params_to_tune,
                lr=opt_cfg_p2.get("lr", 1e-5),
                weight_decay=opt_cfg_p2.get("weight_decay", 1e-5)
            )
            sched_cfg_p2 = meta_tune_cfg.get("scheduler", {})
            scheduler_p2 = CosineAnnealingLR(
                optimizer_p2,
                T_max=meta_tune_cfg.get("num_epochs", 1),
                eta_min=sched_cfg_p2.get("min_lr", 0.0)
            )

            # Phase 2: criterion (CrossEntropyLoss by default, or focal_ce if desired)
            loss_type_p2 = training_cfg.get("loss", {}).get("type", "cross_entropy").lower()
            if loss_type_p2 == "focal_ce_loss":
                criterion_p2 = focal_ce_loss
                logger.info(f"[F{fold_str}] Phase 2: Using focal_ce_loss.")
            else:
                label_smoothing_p2 = training_cfg.get("loss", {}).get("label_smoothing", 0.0)
                criterion_p2 = nn.CrossEntropyLoss(label_smoothing=label_smoothing_p2).to(device)
                logger.info(f"[F{fold_str}] Phase 2: Using CrossEntropyLoss (label_smoothing={label_smoothing_p2}).")

            scaler_p2 = GradScaler(enabled=(device.type == "cuda" and training_cfg.get("amp_enabled", True)))

            best_metric_p2, best_epoch_p2, best_ckpt_p2 = run_training_phase(
                phase_name="P2_MetaTune",
                model=model,
                train_loader=train_loader_p2,
                val_loader=val_loader_p2,
                criterion=criterion_p2,
                optimizer=optimizer_p2,
                scheduler=scheduler_p2,
                scaler=scaler_p2,
                device=device,
                cfg=cfg,
                phase_cfg=meta_tune_cfg,
                fold_str=fold_str,
                label2idx_model=label2idx_model,
                label2idx_eval=label2idx_eval,
                tb_logger=tb_logger,
                ckpt_dir_phase=ckpt_dir,
                exp_name=exp_name,
                start_epoch=best_epoch_p1 + 1,
                ema_model=ema_model,
                initial_best_metric=best_metric_p1
            )

            if best_metric_p2 > overall_best_metric:
                overall_best_metric = best_metric_p2
                overall_best_ckpt = best_ckpt_p2
    else:
        logger.info(f"[F{fold_str}] Phase 2 disabled or Phase 1 failed → skipping Phase 2.")

    tb_logger.close()
    logger.info(f"[F{fold_str}] Complete. Overall best metric = {overall_best_metric:.4f}")
    return overall_best_metric


def main():
    ap = argparse.ArgumentParser(description="Train a single fold with CNN+metadata (two‐phase).")
    ap.add_argument("exp_name", help="Experiment name (locates `<config_dir>/<exp_name>.yaml`).")
    ap.add_argument("--config_file", default=None, help="Path to YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs if --config_file not set.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed.")
    ap.add_argument("--fold_id_to_run", type=str, required=True, help="Fold ID to run.")
    args = ap.parse_args()

    if args.config_file:
        cfg_path = Path(args.config_file)
    else:
        cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"

    if not cfg_path.exists():
        fallback = Path(args.config_dir) / "config_metadata_single_fold.yaml"
        if not args.config_file and fallback.exists():
            cfg_path = fallback
            logger.warning(f"Config not found at {cfg_path}; using fallback {fallback}")
        else:
            raise FileNotFoundError(f"Could not find config at {cfg_path}")

    cfg = load_config(cfg_path)
    cfg = cast_config_values(cfg)
    logger.info(f"Loaded config from {cfg_path}")

    exp_setup = cfg.get("experiment_setup", {})
    seed = args.seed if (args.seed is not None) else exp_setup.get("seed", 42)
    set_seed(seed)
    cfg.setdefault("experiment_setup", {})["seed_runtime"] = seed
    logger.info(f"Seed set → {seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths_cfg = cfg.get("paths", {})
    proj_root = paths_cfg.get("project_root")
    if proj_root and Path(proj_root).is_dir():
        base_path = Path(proj_root).resolve()
    else:
        base_path = cfg_path.parent

    fold_str = str(args.fold_id_to_run).replace(" ", "_").replace("/", "-")
    exp = args.exp_name

    base_log = _get_path_from_config(cfg, "log_dir", default=f"outputs/tensorboard_meta", base_path=base_path)
    base_ckpt = _get_path_from_config(cfg, "ckpt_dir", default=f"outputs/checkpoints_meta", base_path=base_path)

    log_dir = base_log / exp / f"fold_{fold_str}" / timestamp
    ckpt_dir = base_ckpt / exp / f"fold_{fold_str}" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs dir: {log_dir}")
    logger.info(f"Checkpoints dir: {ckpt_dir}")

    labels_csv = _get_path_from_config(cfg, "labels_csv", base_path=base_path)
    meta_csv = _get_path_from_config(cfg, "meta_csv", base_path=base_path)
    image_root = _get_path_from_config(cfg, "train_root", base_path=base_path)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_csv}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Train root (image directory) invalid: {image_root}")

    df_labels = pd.read_csv(labels_csv)
    df_meta = pd.read_csv(meta_csv)

    # Build label2idx for model vs. eval
    unique_model_labels = sorted(df_labels['label'].unique())
    label2idx_model = {lbl: i for i, lbl in enumerate(unique_model_labels)}
    logger.info(f"Model will train on {len(label2idx_model)} classes: {label2idx_model}")

    unique_eval_labels = sorted(df_labels['label'].unique())
    label2idx_eval = {lbl: i for i, lbl in enumerate(unique_eval_labels)}
    logger.info(f"Evaluation labels: {label2idx_eval}")

    dev_default = "cpu"
    if torch.cuda.is_available():
        dev_default = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev_default = "mps"
    final_dev = exp_setup.get("device", dev_default)
    device = get_device(final_dev)
    logger.info(f"Using device: {device}")
    cfg["experiment_setup"]["device_runtime"] = str(device)

    fold_arg = args.fold_id_to_run
    if pd.api.types.is_numeric_dtype(df_labels['fold']):
        try:
            fold_arg = int(args.fold_id_to_run)
        except ValueError:
            logger.error(f"Fold ID '{args.fold_id_to_run}' is not an int but fold column is numeric.")
            return

    if fold_arg not in df_labels['fold'].unique():
        logger.error(f"Fold '{fold_arg}' not found in CSV. Available: {df_labels['fold'].unique().tolist()}")
        return

    train_df = df_labels[df_labels['fold'] != fold_arg].reset_index(drop=True)
    val_df = df_labels[df_labels['fold'] == fold_arg].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        logger.error(f"Fold {fold_arg}: train or val split is empty (train={len(train_df)}, val={len(val_df)})")
        return
    logger.info(f"Fold {fold_arg}: train={len(train_df)}, val={len(val_df)}")

    best_metric = train_one_fold_with_meta(
        fold_id=fold_arg,
        train_df=train_df,
        val_df=val_df,
        meta_df=df_meta,
        cfg=cfg,
        label2idx_model=label2idx_model,
        label2idx_eval=label2idx_eval,
        image_root=image_root,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        exp_name=exp,
        device=device
    )

    if best_metric is not None:
        logger.info(f"Fold {fold_arg} finished. Best metric = {best_metric:.4f}")
    else:
        logger.warning(f"Fold {fold_arg} did not produce a valid best metric.")


if __name__ == "__main__":
    main()

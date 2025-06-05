#!/usr/bin/env python
# train_single_fold.py
#
# Single-fold trainer with full YAML-driven data & training options,
# including optional ClassBalancedSampler, LDAM+DRW, EMA, AMP, TensorBoard, etc.

"""
usage: train_single_fold.py [-h] [--config_file CONFIG_FILE] [--config_dir CONFIG_DIR]
                            [--seed SEED] --fold_id FOLD_ID
                            exp_name
"""

from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from torchmetrics import F1Score, AUROC, Recall

# Utilities
from utils.general_utils import set_seed, load_config, cast_config_values
from utils.torch_utils import get_device
from utils.ema import update_ema

# Data handling
from data_handling.datasets import FlatDataset
from data_handling.transforms import build_transform
from data_handling.custom_samplers import ClassBalancedSampler
from data_handling.gpu_transforms import build_gpu_transform_pipeline

# Model & losses
from models.factory import get_model, DinoClassifier
from losses.focal_loss import focal_ce_loss
from losses.custom_losses import LDAMLoss

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def overlay_heatmap_on_image(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a single-channel heatmap onto a color image.
    
    image: [H, W, 3] float32 in [0..1]
    heatmap: [h, w] float32 in [0..1]
    alpha: blending factor
    Returns: [H, W, 3] float32 in [0..1]
    """
    # Resize heatmap to match image size
    h_img, w_img = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w_img, h_img))
    # Convert to uint8 and apply a colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0  # [H, W, 3] in [0..1]
    # Blend heatmap with the original image
    overlay = heatmap_color * alpha + image
    # Re-normalize if necessary
    max_val = overlay.max()
    if max_val > 1.0:
        overlay = overlay / max_val
    return overlay


class SimpleGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        model: your CNN (or ViT) whose final convolutional layer is named `target_layer_name`.
        target_layer_name: a string like "features.4" or "blocks.11.norm1" depending on architecture.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _find_module(self, name: str) -> torch.nn.Module:
        """ Recursively walk model.named_modules() to find the submodule with given name. """
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"Could not find layer '{name}' in model.")

    def _hook_activations(self, module, input, output):
        # output is feature‐map: shape [B, C, H, W]
        self.activations = output.detach()

    def _hook_gradients(self, module, grad_in, grad_out):
        # grad_out[0] has gradient of the activation: shape [B, C, H, W]
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        target_module = self._find_module(self.target_layer_name)
        # forward hook to grab activations
        target_module.register_forward_hook(self._hook_activations)
        # backward hook to grab gradients (using full backward hook to avoid deprecation warning)
        target_module.register_full_backward_hook(self._hook_gradients)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        input_tensor: [1, 3, H, W], a single image (unsqueezed to batch=1).
        class_idx: which output class to generate CAM for.
        Returns: a heatmap of shape [H, W] in [0..1].
        """
        # Ensure previous activations/gradients are cleared
        self.activations = None
        self.gradients = None

        self.model.zero_grad()
        preds = self.model(input_tensor)  # [1, num_classes]
        score = preds[0, class_idx]
        score.backward(retain_graph=True)

        # activations: [1, C, h, w]; gradients: [1, C, h, w]
        grads = self.gradients[0]            # [C, h, w]
        acts = self.activations[0]           # [C, h, w]

        # global‐average‐pool the gradients over (h, w)
        weights = grads.view(grads.size(0), -1).mean(dim=1)  # [C]

        # weighted combination of feature maps
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)     # [h, w]
        cam = F.relu(cam)                                    # zero out negatives

        # normalize heatmap to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()  # numpy array [h, w] in [0,1]


def _get_path_from_config(
    cfg: dict, key: str, default: str | None = None, base_path: Path | None = None
) -> Path:
    paths_cfg = cfg.get("paths", {})
    path_str = paths_cfg.get(key)
    if path_str is None:
        if default is not None:
            logger.warning(f"Config.paths.{key} missing → using default '{default}'")
            path_str = default
        else:
            logger.error(f"Config.paths.{key} missing and no default provided. Available keys: {list(paths_cfg.keys())}")
            raise ValueError(f"Missing path '{key}' in config.")
    p = Path(path_str)
    if base_path and not p.is_absolute():
        p = base_path / p
    return p.resolve()


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
    disp.plot(ax=ax, xticks_rotation="vertical", cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def train_one_fold(
    fold_id: int | str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    label2idx: dict[str, int],
    train_root: Path,
    log_dir: Path,
    ckpt_dir: Path,
    exp_name: str,
    device: torch.device,
) -> float | None:
    """
    Trains a single fold according to base_experiment.yaml specification:
      • full data options (sampler, cpu/gpu aug),
      • training options (optimizer, scheduler, AMP, accum, LR multipliers, freeze, EMA, LDAM+DRW, etc.),
      • TensorBoard logging,
      • early stopping and model selection.
    """
    # ── Unpack configs ──
    exp_setup_cfg = cfg.get("experiment_setup", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    tb_cfg = cfg.get("tensorboard_logging", {})

    # Loss config
    loss_cfg = training_cfg.get("loss", {})
    drw_epochs = training_cfg.get("drw_schedule_epochs", [])

    # Optimizer & scheduler config
    optim_cfg = training_cfg.get("optimizer", {})
    sched_cfg = training_cfg.get("scheduler", {})

    # Logging setup
    logger.info(f"[Fold {fold_id}] Starting. Logs → {log_dir}, Ckpts → {ckpt_dir}")
    writer = None
    if tb_cfg.get("enable", False):
        writer = SummaryWriter(str(log_dir))
        logger.info(f"[Fold {fold_id}] TensorBoard enabled at {log_dir}")

    # ── Transforms & Datasets ──
    cpu_aug = data_cfg.get("cpu_augmentations", {})
    tf_train = build_transform(cpu_aug, train=True)
    tf_val = build_transform(cpu_aug, train=False)

    train_ds = FlatDataset(
        df=train_df,
        root=train_root,
        label2idx=label2idx,
        tf=tf_train,
        image_loader=data_cfg.get("image_loader", "pil"),
        enable_ram_cache=data_cfg.get("enable_ram_cache", False),
    )
    val_ds = FlatDataset(
        df=val_df,
        root=train_root,
        label2idx=label2idx,
        tf=tf_val,
        image_loader=data_cfg.get("image_loader", "pil"),
        enable_ram_cache=data_cfg.get("enable_ram_cache", False),
    )

    # Optional sampler
    train_sampler = None
    sampler_cfg = data_cfg.get("sampler", {}).get("type", "default")
    if sampler_cfg == "class_balanced_sqrt":
        train_sampler = ClassBalancedSampler(train_ds, num_samples=len(train_ds))
        logger.info(f"[Fold {fold_id}] Using ClassBalancedSampler (1/sqrt(Nc)).")

    # GPU aug pipeline (if any)
    gpu_aug_cfg = data_cfg.get("gpu_augmentations", {})
    gpu_aug_train = None
    if gpu_aug_cfg.get("enable", False):
        gpu_aug_train = build_gpu_transform_pipeline(gpu_aug_cfg, device)
        if gpu_aug_train:
            logger.info(f"[Fold {fold_id}] GPU augmentations enabled.")

    # DataLoader parameters
    batch_size = training_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    dl_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda" and num_workers > 0),
        "drop_last": False,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = data_cfg.get("persistent_workers", False)
        dl_kwargs["prefetch_factor"] = prefetch_factor

    if train_sampler:
        train_loader = DataLoader(train_ds, sampler=train_sampler, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # ── Model + EMA ──
    model_cfg = model_cfg.copy()
    model_cfg["numClasses"] = len(label2idx)
    if "type" in model_cfg and "MODEL_TYPE" not in model_cfg:
        model_cfg["MODEL_TYPE"] = model_cfg.pop("type")
    model = get_model(model_cfg).to(device)

    ema_decay = training_cfg.get("ema_decay", 0.0)
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"[Fold {fold_id}] EMA enabled (decay={ema_decay})")

    # ── Freeze / backbone-LR logic ──
    freeze_epochs = training_cfg.get("freeze_epochs", 0)
    backbone_lr_mult = training_cfg.get("backbone_lr_mult", 1.0)
    base_lr = optim_cfg.get("lr", 1e-3)

    # Identify head prefixes
    head_prefixes: list[str] = []
    if isinstance(model, DinoClassifier):
        head_prefixes.append("classifier.")
    elif hasattr(model, "default_cfg") and "classifier" in model.default_cfg and hasattr(model, model.default_cfg["classifier"]):
        head_prefixes.append(f"{model.default_cfg['classifier']}.")
    else:
        for name, mod in model.named_modules():
            if name.endswith("classifier") or name.endswith("fc") or name.endswith("head"):
                if isinstance(mod, (nn.Linear, nn.Sequential)):
                    head_prefixes.append(f"{name}.")

    # Build optimizer param groups
    opt_groups: list[dict] = []
    if freeze_epochs > 0:
        # Only train head for first freeze_epochs
        trainable = []
        frozen_count = 0
        for n, p in model.named_parameters():
            if any(n.startswith(pfx) for pfx in head_prefixes):
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)
                frozen_count += 1

        if not trainable:
            # fallback: unfreeze all
            for p in model.parameters():
                p.requires_grad_(True)
            trainable = list(model.parameters())
            logger.warning(f"[Fold {fold_id}] No head params detected; training all for first {freeze_epochs} epochs.")

        opt_groups = [{"params": trainable, "lr": base_lr}]
        logger.info(f"[Fold {fold_id}] Freezing backbone for first {freeze_epochs} epochs ({frozen_count} params frozen).")
    elif (backbone_lr_mult != 1.0) and head_prefixes:
        head_params, back_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(n.startswith(pfx) for pfx in head_prefixes):
                head_params.append(p)
            else:
                back_params.append(p)
        if head_params:
            opt_groups.append({"params": head_params, "lr": base_lr})
        if back_params:
            opt_groups.append({"params": back_params, "lr": base_lr * backbone_lr_mult})
        logger.info(f"[Fold {fold_id}] Diff LR: head@{base_lr}, backbone@{base_lr * backbone_lr_mult}")
    else:
        opt_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    if not any(pg.get("params") for pg in opt_groups):
        opt_groups = [{"params": model.parameters(), "lr": base_lr}]
        logger.warning(f"[Fold {fold_id}] No trainable params found; defaulting to all.")

    # Instantiate optimizer
    optim_type = optim_cfg.get("type", "AdamW").lower()
    if optim_type == "sgd":
        optimizer = SGD(opt_groups, lr=base_lr, weight_decay=optim_cfg.get("weight_decay", 1e-4), momentum=optim_cfg.get("momentum", 0.9))
    else:
        optimizer = AdamW(opt_groups, lr=base_lr, weight_decay=optim_cfg.get("weight_decay", 1e-4))

    # Scheduler
    sched_type = sched_cfg.get("type", "cosineannealinglr").lower()
    if sched_type == "steplr":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:  # CosineAnnealingLR
        t_max = sched_cfg.get("t_max", training_cfg.get("num_epochs", 1))
        min_lr = sched_cfg.get("min_lr", 0.0)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

    # ── Choose loss (LDAM, focal CE, or vanilla CE) ──
    loss_type = loss_cfg.get("type", "cross_entropy").lower()
    if loss_type == "ldam_loss":
        # Compute class counts on train_df→label2idx
        counts = np.zeros(len(label2idx), dtype=int)
        mapped = train_df["label"].map(label2idx)
        vc = mapped.value_counts()
        for idx, cnt in vc.items():
            counts[idx] = int(cnt)
        ldam_params = {
            "max_margin": loss_cfg.get("ldam_max_margin", 0.5),
            "use_effective_number_margin": loss_cfg.get("ldam_use_effective_number_margin", True),
            "effective_number_beta": loss_cfg.get("ldam_effective_number_beta", 0.999),
        }
        criterion = LDAMLoss(class_counts=counts, **ldam_params).to(device)
        # DRW schedule will update `criterion.weight` later
    elif loss_type == "focal_ce_loss":
        alpha = loss_cfg.get("focal_alpha", 1.0)
        gamma = loss_cfg.get("focal_gamma", 2.0)
        criterion = lambda logits, targets: focal_ce_loss(logits, targets, alpha=alpha, gamma=gamma)
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError(f"Unsupported loss type '{loss_type}'")

    drw_stage = 0

    # ── AMP setup ──
    use_amp = (device.type == "cuda" and training_cfg.get("amp_enabled", True))
    scaler = GradScaler(enabled=use_amp)

    # ── Training bookkeeping ──
    num_epochs = training_cfg.get("num_epochs", 1)
    accum_steps = training_cfg.get("accum_steps", 1)
    val_interval = training_cfg.get("val_interval", 1)
    early_stop_patience = training_cfg.get("early_stopping_patience", 10)
    save_thresh = training_cfg.get("save_optimal_thresholds", False)
    model_sel_metric = training_cfg.get("model_selection_metric", "mean_optimal_sensitivity").lower()
    pauc_max_fpr = training_cfg.get("pauc_max_fpr", 0.2)

    best_metric = -float("inf")
    patience_counter = 0
    best_epoch = -1

    # ── Epoch Loop ──
    for epoch in range(num_epochs):
        model.train()
        # Unfreeze at freeze_epochs
        if (freeze_epochs > 0) and (epoch == freeze_epochs):
            for p in model.parameters():
                p.requires_grad_(True)
            # Rebuild optimizer & scheduler after unfreeze
            if (backbone_lr_mult != 1.0) and head_prefixes:
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
            else:
                groups = [{"params": model.parameters(), "lr": base_lr * (backbone_lr_mult if backbone_lr_mult != 1.0 else 1.0)}]

            optimizer = AdamW(groups, lr=groups[0]["lr"], weight_decay=optim_cfg.get("weight_decay", 1e-4))
            rem = num_epochs - freeze_epochs
            if sched_type == "steplr":
                step_size = sched_cfg.get("step_size", 10)
                gamma = sched_cfg.get("gamma", 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=(rem if rem > 0 else 1), eta_min=sched_cfg.get("min_lr", 0.0))
            logger.info(f"[Fold {fold_id}] Unfroze at epoch {epoch}, reinitialized optimizer/scheduler.")

        # DRW Weight update (for LDAMLoss)
        if loss_type == "ldam_loss" and (drw_stage < len(drw_epochs)) and (epoch >= drw_epochs[drw_stage]):
            beta = ldam_params["effective_number_beta"]
            eff_num = 1.0 - np.power(beta, counts)
            drw_w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
            drw_w = drw_w / drw_w.sum() * len(counts)
            w_tensor = torch.tensor(drw_w, dtype=torch.float32, device=device)
            criterion.update_weights(w_tensor)
            logger.info(f"[Fold {fold_id}] DRW at E{epoch} → weights (first5): {drw_w[:5]}")
            drw_stage += 1
        elif (epoch == 0) and (drw_stage == 0) and loss_type == "ldam_loss":
            criterion.update_weights(None)

        # ── Train Loop ──
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Fold {fold_id} E{epoch} Train", ncols=exp_setup_cfg.get("TQDM_NCOLS", 100), leave=False)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if gpu_aug_train:
                imgs = gpu_aug_train(imgs)

            with autocast(enabled=use_amp, device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)
                if accum_steps > 1:
                    loss = loss / accum_steps

                # Convert to probabilities for metrics
                probs = F.softmax(logits, dim=1)

            scaler.scale(loss).backward()

            if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema_model is not None:
                    update_ema(ema_model, model, ema_decay)

            preds = probs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)
            # Since CrossEntropyLoss is mean-reduced, multiply by batch size to accumulate total
            scaling = accum_steps if accum_steps > 1 else 1
            running_loss += loss.item() * bs * scaling
            running_correct += correct
            running_total += bs

            avg_loss = running_loss / running_total if running_total > 0 else 0.0
            avg_acc = running_correct / running_total if running_total > 0 else 0.0

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.4f}"
            })

            # Batch-level TensorBoard logging
            if writer and (tb_cfg.get("log_interval_batches_train", 0) > 0):
                interval = tb_cfg["log_interval_batches_train"]
                if (batch_idx + 1) % interval == 0:
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar("Train/Loss_batch", avg_loss, step)
                    writer.add_scalar("Train/Acc_batch", avg_acc, step)

        pbar.close()
        epoch_loss = running_loss / running_total if running_total > 0 else 0.0
        epoch_acc = running_correct / running_total if running_total > 0 else 0.0
        scheduler.step()

        if writer:
            writer.add_scalar("Train/Loss_epoch", epoch_loss, epoch)
            writer.add_scalar("Train/Acc_epoch", epoch_acc, epoch)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)

        # ── Validation Loop ──
        do_val = ((epoch % val_interval) == 0) or (epoch == num_epochs - 1)
        if do_val:
            model.eval()
            if ema_model is not None:
                ema_model.eval()
            eval_model = ema_model if (ema_model is not None and training_cfg.get("use_ema_for_val", True)) else model

            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_logits: list[torch.Tensor] = []
            all_true: list[torch.Tensor] = []

            with torch.no_grad():
                pbar_v = tqdm(val_loader, desc=f"Fold {fold_id} E{epoch} Val", ncols=exp_setup_cfg.get("TQDM_NCOLS", 100), leave=False)
                for imgs, labels in pbar_v:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast(enabled=use_amp, device_type=device.type):
                        logits = eval_model(imgs)
                        loss = criterion(logits, labels)
                        probs = F.softmax(logits, dim=1)

                    preds = probs.argmax(dim=1)
                    correct = (preds == labels).sum().item()
                    bs = labels.size(0)

                    val_loss += loss.item() * bs
                    val_correct += correct
                    val_total += bs

                    all_logits.append(logits.cpu())
                    all_true.append(labels.cpu())

                    avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
                    avg_val_acc = val_correct / val_total if val_total > 0 else 0.0

                    pbar_v.set_postfix({
                        "val_loss": f"{avg_val_loss:.4f}",
                        "val_acc": f"{avg_val_acc:.4f}"
                    })
                pbar_v.close()

            avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
            avg_val_acc = val_correct / val_total if val_total > 0 else 0.0

            all_logits_cat = torch.cat(all_logits, dim=0)
            all_probs = F.softmax(all_logits_cat, dim=1)
            all_true_cat = torch.cat(all_true, dim=0)

            # F1 (macro) on hard preds
            f1_macro = F1Score(task="multiclass", num_classes=len(label2idx), average="macro")(
                all_probs.argmax(dim=1), all_true_cat
            ).item()

            try:
                auroc_macro = AUROC(task="multiclass", num_classes=len(label2idx), average="macro")(
                    all_probs, all_true_cat
                ).item()
            except Exception:
                auroc_macro = float("nan")

            try:
                # pAUROC @ max_fpr
                pauc = AUROC(task="multiclass", num_classes=len(label2idx), average="macro", max_fpr=pauc_max_fpr)(
                    all_probs, all_true_cat
                ).item()
            except Exception:
                pauc = float("nan")

            sens_macro = Recall(task="multiclass", num_classes=len(label2idx), average="macro", zero_division=0)(
                all_probs.argmax(dim=1), all_true_cat
            ).item()

            if writer:
                writer.add_scalar("Val/Loss_epoch", avg_val_loss, epoch)
                writer.add_scalar("Val/Acc_epoch", avg_val_acc, epoch)
                writer.add_scalar("Val/F1_macro", f1_macro, epoch)
                writer.add_scalar("Val/AUROC_macro", auroc_macro, epoch)
                writer.add_scalar(f"Val/pAUROC@{pauc_max_fpr}", pauc, epoch)
                writer.add_scalar("Val/Sensitivity_macro", sens_macro, epoch)

            logger.info(
                f"[Fold {fold_id}][Epoch {epoch}] Val → "
                f"Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f} "
                f"F1={f1_macro:.4f} AUROC={auroc_macro:.4f} Sens={sens_macro:.4f}"
            )

            # ── IMAGE LOGGING with Grad-CAM ──
            img_cfg = tb_cfg.get("image_logging", {})
            if writer and img_cfg.get("enable", False) and (epoch in img_cfg.get("log_at_epochs", [])):
                # a) Grab a small batch from val_loader
                samples = next(iter(val_loader))
                imgs_sample, labels_sample = (
                    samples[0][: img_cfg.get("num_samples", 4)],
                    samples[1][: img_cfg.get("num_samples", 4)],
                )
                imgs_sample = imgs_sample.to(device)  # [B, 3, H, W]
                B = imgs_sample.size(0)

                # b) Forward‐pass to get logits and pick predicted class (for computing Grad-CAM)
                model_to_use = ema_model if (ema_model is not None and training_cfg.get("use_ema_for_val", True)) else model
                model_to_use.eval()
                with torch.no_grad():
                    logits_sample = model_to_use(imgs_sample)      # [B, num_classes]
                    probs_sample = F.softmax(logits_sample, dim=1) # [B, num_classes]
                    preds_sample = probs_sample.argmax(dim=1)      # [B]

                # c) Denormalize for visualization
                if img_cfg.get("denormalize", True):
                    mean = torch.tensor(cpu_aug["norm_mean"]).view(1, 3, 1, 1).to(device)
                    std  = torch.tensor(cpu_aug["norm_std"]).view(1, 3, 1, 1).to(device)
                    imgs_denorm = imgs_sample * std + mean        # still on GPU
                else:
                    imgs_denorm = imgs_sample

                imgs_denorm = imgs_denorm.clamp(0, 1).cpu()  # move to CPU, shape [B,3,H,W]

                # d) Prepare Grad-CAM helper (hook on “last_conv” layer)
                #    — Make sure `"last_conv_layer"` matches your model’s layer name.
                #    For EfficientNet-B0, you might do: target_layer = "blocks.6" or similar.
                target_layer = training_cfg.get("gradcam_layer", "blocks.6")
                gradcam = SimpleGradCAM(model_to_use, target_layer)

                # e) For each sample, generate heatmap and overlay
                overlays: list[np.ndarray] = []
                for i in range(B):
                    img_i = imgs_denorm[i].permute(1, 2, 0).numpy()     # [H, W, 3] in [0..1]
                    class_idx = preds_sample[i].item()               # predicted class
                    # Compute CAM for this one image (unsqueeze to [1,3,H,W])
                    single = imgs_denorm[i].unsqueeze(0).to(device)
                    heatmap = gradcam(single, class_idx)              # [h, w] in [0..1], np.ndarray

                    # Overlay heatmap onto original
                    overlay_i = overlay_heatmap_on_image(img_i, heatmap, alpha=0.4)  # [H,W,3]
                    overlays.append(overlay_i)

                # f) Stack overlays into a tensor [B, 3, H, W] for TensorBoard
                overlays_np = np.stack(overlays, axis=0)            # [B, H, W, 3]
                overlays_t = torch.from_numpy(overlays_np).permute(0, 3, 1, 2)  # [B,3,H,W]

                # g) Log overlays under tag "Val/GradCAM_E{epoch}"
                writer.add_images(f"Val/GradCAM_E{epoch}", overlays_t, global_step=epoch)

                # h) (Optional) also write the raw inputs or text labels as before
                #     If you still want the plain inputs:
                writer.add_images(f"Val/RawInputs_E{epoch}", imgs_denorm, global_step=epoch)

                # Optionally log predicted vs. true labels as text
                for idx, (gt, pred) in enumerate(zip(labels_sample.cpu().tolist(), preds_sample.cpu().tolist())):
                    writer.add_text(
                        f"Val/Pred_E{epoch}",
                        f"Index {idx}: GT={gt}, Pred={pred}",
                        epoch * B + idx
                    )
            # ── end IMAGE LOGGING ──

            # Choose primary metric
            metric_map = {
                "macro_auc": auroc_macro,
                "mean_optimal_f1": f1_macro if save_thresh else f1_macro,
                "mean_optimal_sensitivity": sens_macro,
            }
            current_metric = metric_map.get(model_sel_metric, sens_macro)

            # Save thresholds from PR if requested and metric is F1 or sensitivity
            opt_thresholds: dict[int, float] = {}
            if save_thresh and (model_sel_metric in ["mean_optimal_f1", "mean_optimal_sensitivity"]):
                n_cls = len(label2idx)
                pr_true = all_true_cat.numpy()
                pr_probs = all_probs.numpy()
                opt_vals: list[float] = []
                opt_sens_list: list[float] = []
                for cls_i in range(n_cls):
                    onehot = (pr_true == cls_i).astype(int)
                    try:
                        p, r, t = precision_recall_curve(onehot, pr_probs[:, cls_i])
                        f1_scores = (2 * p * r) / (p + r + 1e-8)
                        if len(f1_scores) > 1:
                            best_idx = np.nanargmax(f1_scores[1:])
                            opt_thresholds[cls_i] = float(t[best_idx])
                            opt_vals.append(f1_scores[best_idx])
                            opt_sens_list.append(r[best_idx])
                        else:
                            opt_thresholds[cls_i] = 0.5
                            opt_vals.append(0.0)
                            opt_sens_list.append(0.0)
                    except Exception:
                        opt_thresholds[cls_i] = 0.5
                        opt_vals.append(0.0)
                        opt_sens_list.append(0.0)
                if model_sel_metric == "mean_optimal_sensitivity" and len(opt_sens_list) > 0:
                    current_metric = float(np.nanmean(opt_sens_list))
                if writer:
                    writer.add_text(f"Val/optimal_thresholds", str(opt_thresholds), epoch)

            # Save best model
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_path = ckpt_dir / f"{exp_name}_fold{fold_id}_best.pt"
                data_to_save: dict = {
                    "epoch": epoch,
                    "model_state_dict": (ema_model if (ema_model is not None and training_cfg.get("use_ema_for_val", True)) else model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    f"best_{model_sel_metric}": best_metric,
                    "config": cfg,
                    "label2idx": label2idx,
                }
                if ema_model is not None:
                    data_to_save["ema_state_dict"] = ema_model.state_dict()
                if save_thresh and opt_thresholds:
                    data_to_save["optimal_thresholds"] = opt_thresholds
                torch.save(data_to_save, str(best_path))
                logger.info(f"[Fold {fold_id}] New best ({model_sel_metric})={best_metric:.4f} at E{epoch}. Saved → {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                logger.info(f"[Fold {fold_id}] Early stopping at E{epoch}.")
                break

            model.train()

    # ── Save last checkpoint ──
    last_path = ckpt_dir / f"{exp_name}_fold{fold_id}_last.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "last_metric": best_metric,
            "config": cfg,
            "label2idx": label2idx,
        },
        str(last_path),
    )
    logger.info(f"[Fold {fold_id}] Saved last checkpoint → {last_path}")

    if writer:
        writer.close()

    return best_metric if best_epoch >= 0 else None


def main():
    ap = argparse.ArgumentParser(description="Single-fold trainer (LDAM+DRW, EMA, full config).")
    ap.add_argument("exp_name", help="Experiment name (locates `<config_dir>/<exp_name>.yaml`).")
    ap.add_argument("--config_file", default=None, help="Path to YAML config.")
    ap.add_argument("--config_dir", default="configs", help="Dir for YAML configs.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed.")
    ap.add_argument("--fold_id", type=str, required=True, help="Fold identifier (matches 'fold' column).")
    args = ap.parse_args()

    # ── Load config ──
    if args.config_file:
        cfg_path = Path(args.config_file)
    else:
        cfg_path = Path(args.config_dir) / f"{args.exp_name}.yaml"

    if not cfg_path.exists():
        fallback = Path(args.config_dir) / "config_single_fold.yaml"
        if not args.config_file and fallback.exists():
            logger.warning(f"Config not found at {cfg_path}; using fallback {fallback}")
            cfg_path = fallback
        else:
            raise FileNotFoundError(f"Could not find config at {cfg_path}")

    cfg = load_config(cfg_path)
    cfg = cast_config_values(cfg)
    logger.info(f"Loaded config from {cfg_path}")

    # ── Seed & timestamp ──
    exp_setup = cfg.get("experiment_setup", {})
    main_seed = args.seed if (args.seed is not None) else exp_setup.get("seed", 42)
    set_seed(main_seed)
    cfg.setdefault("experiment_setup", {})["seed_runtime"] = main_seed
    logger.info(f"Seed set to {main_seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Paths ──
    paths_cfg = cfg.get("paths", {})
    proj_root = paths_cfg.get("project_root")
    if proj_root and Path(proj_root).is_dir():
        base_path = Path(proj_root).resolve()
    else:
        base_path = cfg_path.parent

    fold_str = str(args.fold_id).replace(" ", "_").replace("/", "-")
    exp = args.exp_name

    base_log = _get_path_from_config(cfg, "log_dir", default=f"outputs/tensorboard", base_path=base_path)
    base_ckpt = _get_path_from_config(cfg, "ckpt_dir", default=f"outputs/checkpoints", base_path=base_path)

    log_dir = base_log / exp / f"fold_{fold_str}" / timestamp
    ckpt_dir = base_ckpt / exp / f"fold_{fold_str}" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs directory: {log_dir}")
    logger.info(f"Checkpoints directory: {ckpt_dir}")

    # ── CSV & label2idx ──
    labels_csv_p = _get_path_from_config(cfg, "labels_csv", base_path=base_path)
    train_root_p = _get_path_from_config(cfg, "train_root", base_path=base_path)

    if not labels_csv_p.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv_p}")
    if not train_root_p.is_dir():
        raise FileNotFoundError(f"Train root invalid: {train_root_p}")

    df_full = pd.read_csv(labels_csv_p)
    all_labels = sorted(df_full["label"].unique())
    if not all_labels:
        raise ValueError("No labels found in CSV.")
    label2idx = {lbl: i for i, lbl in enumerate(all_labels)}
    logger.info(f"Built label2idx with {len(label2idx)} classes.")

    # ── Device ──
    device_str = cfg.get("experiment_setup", {}).get("device", None)
    device = get_device(device_str)
    logger.info(f"Using device: {device}")

    # ── Fold splitting ──
    fold_val = args.fold_id
    if pd.api.types.is_numeric_dtype(df_full["fold"].dtype):
        try:
            fold_val = int(args.fold_id)
        except ValueError:
            logger.info(f"Fold ID '{args.fold_id}' kept as string for numeric fold column.")
    if fold_val not in df_full["fold"].unique():
        logger.error(f"Fold '{fold_val}' not in CSV. Available: {df_full['fold'].unique().tolist()}")
        sys.exit(1)

    train_df = df_full[df_full["fold"] != fold_val].reset_index(drop=True)
    val_df = df_full[df_full["fold"] == fold_val].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        logger.error(f"Fold {fold_val}: train ({len(train_df)}) or val ({len(val_df)}) is empty.")
        sys.exit(1)
    logger.info(f"Fold {fold_val}: {len(train_df)} train samples, {len(val_df)} val samples.")

    best_metric = train_one_fold(
        fold_id=args.fold_id,
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        label2idx=label2idx,
        train_root=train_root_p,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        exp_name=exp,
        device=device,
    )

    if best_metric is not None:
        logger.info(f"▸ Fold {fold_val} completed. Best metric = {best_metric:.4f}")
    else:
        logger.warning(f"▸ Fold {fold_val} ended without a saved best model (best_metric=None).")


if __name__ == "__main__":
    main()

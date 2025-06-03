#!/usr/bin/env python
# evaluate_fold_summary.py
from __future__ import annotations

import argparse
import contextlib # Not actively used, but kept from original structure
import logging
import sys
import time # Not actively used, but kept
from pathlib import Path

import matplotlib.pyplot as plt # Keep for potential future use
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

# --- Adjusted Imports to match your project structure ---
try:
    from data_handling.datasets import FlatDataset, FlatDatasetWithMeta
    from data_handling.transforms import build_transform
    from models.factory import get_model as get_core_model
    from models.meta_models import CNNWithMetadata
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules. Details: {e}")
    print("Ensure that 'data_handling' and 'models' are packages in your PYTHONPATH,")
    print("or that this script is run from a location where they can be found (e.g., project root).")
    print("Also ensure __init__.py files exist in 'data_handling' and 'models' directories,")
    print("and that they correctly export the required classes/functions.")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {Path.cwd()}")
    sys.exit(1)

from torchmetrics import AUROC, F1Score, Recall 

# Minimal utility functions
def load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def cast_config_values(config_dict):
    for k, v in config_dict.items():
        if isinstance(v, dict):
            cast_config_values(v)
        elif isinstance(v, str):
            try:
                if '.' in v or 'e' in v.lower(): config_dict[k] = float(v)
                else: config_dict[k] = int(v)
            except ValueError:
                if v.lower() == 'true': config_dict[k] = True
                elif v.lower() == 'false': config_dict[k] = False
                elif v.lower() == 'none' or v == '': config_dict[k] = None
    return config_dict

def get_device(device_str: str | None = None) -> torch.device:
    if device_str: return torch.device(device_str)
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

try:
    from tqdm.notebook import tqdm as tqdm_notebook
    if 'IPython' in sys.modules and hasattr(sys.modules['IPython'], 'core') and \
       hasattr(sys.modules['IPython'].core, 'getipython') and \
       sys.modules['IPython'].core.getipython.get_ipython() is not None:
            tqdm_iterator = tqdm_notebook
    else:
        from tqdm import tqdm as tqdm_cli
        tqdm_iterator = tqdm_cli
except ImportError:
    from tqdm import tqdm as tqdm_cli
    tqdm_iterator = tqdm_cli

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__)

def get_experiment_name_from_path(ckpt_path: Path) -> str:
    try:
        return ckpt_path.parent.parent.parent.name
    except Exception:
        try: return ckpt_path.parent.parent.name
        except: return ckpt_path.stem

def evaluate_single_model_on_fold(
    ckpt_path: Path,
    fold_id: int | str,
    full_labels_df: pd.DataFrame,
    image_dir: Path,
    meta_features_df: pd.DataFrame | None,
    device: torch.device,
    global_base_cfg: dict
) -> dict | None:
    logger.info(f"Evaluating checkpoint: {ckpt_path} on fold {fold_id}")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}"); return None
    try:
        ckpt_data = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint {ckpt_path}: {e}", exc_info=True); return None
    
    cfg_runtime = ckpt_data.get('config_runtime', global_base_cfg)
    if not isinstance(cfg_runtime, dict):
        logger.error(f"Config invalid. Cfg: {cfg_runtime}"); cfg_runtime = {}
    cfg_runtime = cast_config_values(cfg_runtime)

    if 'label2idx_model' not in ckpt_data or 'label2idx_eval' not in ckpt_data:
        logger.error(f"Ckpt {ckpt_path} missing label2idx_model/eval."); return None
        
    label2idx_model_ckpt = ckpt_data['label2idx_model']
    label2idx_eval_ckpt = ckpt_data['label2idx_eval']
    idx2label_model_ckpt = {v: k for k, v in label2idx_model_ckpt.items()}
    idx2label_eval_ckpt = {v: k for k, v in label2idx_eval_ckpt.items()}

    num_model_classes = len(label2idx_model_ckpt)
    num_eval_classes = len(label2idx_eval_ckpt)

    is_meta_model = ckpt_data.get('metadata_dim') is not None or \
                    cfg_runtime.get("model",{}).get("base_cnn_type") is not None
    metadata_dim_model = ckpt_data.get('metadata_dim', 0)
    if is_meta_model and metadata_dim_model == 0:
        metadata_dim_model = cfg_runtime.get("model", {}).get('metadata_input_dim_runtime', 0)

    model_cfg_from_ckpt = cfg_runtime.get("model", {})
    model_to_eval: torch.nn.Module

    try:
        if is_meta_model:
            if meta_features_df is None: logger.error(f"Meta model {ckpt_path} needs meta_df."); return None
            if metadata_dim_model == 0: logger.error(f"Meta model {ckpt_path}, but meta_dim is 0."); return None
            base_cnn_type = model_cfg_from_ckpt.get("base_cnn_type")
            if not base_cnn_type: logger.error(f"Meta model {ckpt_path} missing 'base_cnn_type'."); return None
            base_cnn_cfg = {"MODEL_TYPE":base_cnn_type,"numClasses":num_model_classes,"pretrained":model_cfg_from_ckpt.get("pretrained_cnn",False)}
            base_cnn = get_core_model(base_cnn_cfg)
            meta_head_args = model_cfg_from_ckpt.get("meta_head_args",{}).copy()
            meta_head_args.pop('num_classes',None); meta_head_args.pop('metadata_input_dim',None)
            model_to_eval = CNNWithMetadata(base_cnn,num_model_classes,metadata_dim_model,**meta_head_args).to(device)
            logger.info(f"Constructed CNNWithMetadata. Meta_dim: {metadata_dim_model}")
        else:
            std_model_cfg = model_cfg_from_ckpt.copy()
            std_model_cfg["numClasses"] = num_model_classes
            if "type" in std_model_cfg and "MODEL_TYPE" not in std_model_cfg: std_model_cfg["MODEL_TYPE"] = std_model_cfg.pop("type")
            if "MODEL_TYPE" not in std_model_cfg: logger.error(f"Std model {ckpt_path} missing 'MODEL_TYPE'."); return None
            model_to_eval = get_core_model(std_model_cfg).to(device)
            logger.info(f"Constructed standard model (type: {std_model_cfg['MODEL_TYPE']}).")

        training_cfg_ckpt = cfg_runtime.get("training", {})
        use_ema = training_cfg_ckpt.get("use_ema_for_val", True)
        weights_key = 'ema_model_state_dict' if use_ema and 'ema_model_state_dict' in ckpt_data and ckpt_data['ema_model_state_dict'] else 'model_state_dict'
        if weights_key not in ckpt_data or not ckpt_data[weights_key]:
            logger.error(f"Weights key '{weights_key}' invalid in {ckpt_path}. Keys: {list(ckpt_data.keys())}"); return None
        logger.info(f"Using '{weights_key}' from {ckpt_path.name}.")
        model_to_eval.load_state_dict(ckpt_data[weights_key])
        model_to_eval.eval()
    except Exception as e:
        logger.error(f"Model construction/load error for {ckpt_path}: {e}", exc_info=True); return None

    val_df_fold = full_labels_df[full_labels_df['fold'] == fold_id].reset_index(drop=True)
    if val_df_fold.empty: logger.error(f"No data for fold {fold_id}."); return None

    data_cfg_ckpt = cfg_runtime.get("data", {})
    cpu_augs_cfg = data_cfg_ckpt.get("cpu_augmentations", {})
    if not cpu_augs_cfg or not isinstance(cpu_augs_cfg, dict):
        logger.warning(f"cpu_augs invalid for {ckpt_path}. Using minimal default.")
        cpu_augs_cfg = {"resize":256,"crop_size":224,"norm_mean":[0.485,0.456,0.406],"norm_std":[0.229,0.224,0.225]}
    try:
        transform_val = build_transform(cpu_augs_cfg, train=False)
    except KeyError as e:
        logger.error(f"KeyError in cpu_augs for build_transform: {e}. Config: {cpu_augs_cfg}"); return None

    dataset_args = data_cfg_ckpt.get("dataset_args", {}).copy()
    try:
        if is_meta_model:
            meta_args = {"meta_features_names":data_cfg_ckpt.get("meta_features_names"),"meta_augmentation_p":0.0,"meta_nan_fill_value":data_cfg_ckpt.get("meta_nan_fill_value",0.0)}
            for k in list(meta_args.keys())+['root','label2idx','tf']: dataset_args.pop(k,None)
            val_ds = FlatDatasetWithMeta(val_df_fold,meta_features_df,image_dir,label2idx_eval_ckpt,transform_val,data_cfg_ckpt.get("image_loader","pil"),data_cfg_ckpt.get("enable_ram_cache",False),**meta_args,**dataset_args)
        else:
            for k in ['root','root_path','label2idx','label2idx_map','tf','transform']: dataset_args.pop(k,None)
            val_ds = FlatDataset(val_df_fold,image_dir,label2idx_eval_ckpt,transform_val,data_cfg_ckpt.get("image_loader","pil"),data_cfg_ckpt.get("enable_ram_cache",False),**dataset_args)
    except Exception as e:
        logger.error(f"Dataset init error for {ckpt_path}: {e}", exc_info=True); return None

    val_loader = DataLoader(val_ds,batch_size=training_cfg_ckpt.get("batch_size",32),shuffle=False,num_workers=data_cfg_ckpt.get("num_workers",0),pin_memory=(device.type=='cuda' and data_cfg_ckpt.get("num_workers",0)>0))

    all_probs, all_true_lbls = [], []
    amp_eval = device.type == 'cuda' and training_cfg_ckpt.get("amp_enabled",True)
    with torch.no_grad():
        for batch_data in tqdm_iterator(val_loader,desc=f"Eval {ckpt_path.stem} F{fold_id}",leave=False):
            imgs_dev, meta_dev_opt = None, None
            if is_meta_model: (imgs_cpu,meta_cpu),lbls_cpu=batch_data; imgs_dev,meta_dev_opt=imgs_cpu.to(device),meta_cpu.to(device)
            else: imgs_cpu,lbls_cpu=batch_data; imgs_dev=imgs_cpu.to(device)
            with autocast(device_type=device.type,enabled=amp_eval):
                logits = model_to_eval(imgs_dev,meta_dev_opt) if is_meta_model else model_to_eval(imgs_dev)
            all_probs.append(F.softmax(logits,dim=1).cpu()); all_true_lbls.append(lbls_cpu.cpu())
    if not all_probs: logger.error(f"No preds for {ckpt_path} F{fold_id}."); return None
    all_probs_cat = torch.cat(all_probs); all_true_eval_cat = torch.cat(all_true_lbls)

    post_hoc_thresh = ckpt_data.get('post_hoc_unk_threshold_used',training_cfg_ckpt.get("post_hoc_unk_threshold"))
    unk_str = training_cfg_ckpt.get("unk_label_string","UNK")
    known_model_idxs = [idx for lbl,idx in label2idx_model_ckpt.items() if lbl!=unk_str and idx!=label2idx_model_ckpt.get(unk_str,-1)]
    unk_eval_idx = label2idx_eval_ckpt.get(unk_str,-1)
    post_hoc_on = (post_hoc_thresh is not None and (bool(known_model_idxs) or not label2idx_model_ckpt.get(unk_str)) and unk_eval_idx!=-1)

    preds_model_raw = all_probs_cat.argmax(dim=1)
    final_preds_eval = torch.full_like(preds_model_raw,-1,dtype=torch.long)
    for i,m_idx in enumerate(preds_model_raw.tolist()):
        lbl_s = idx2label_model_ckpt.get(m_idx)
        if lbl_s and lbl_s in label2idx_eval_ckpt: final_preds_eval[i]=label2idx_eval_ckpt[lbl_s]
    if post_hoc_on:
        logger.info(f"Post-hoc UNK (Th={post_hoc_thresh}, UNK eval idx={unk_eval_idx}) for {ckpt_path.name}")
        for i in range(all_probs_cat.size(0)):
            max_prob_k = 0.0
            if known_model_idxs: max_prob_k = all_probs_cat[i][known_model_idxs].max().item()
            elif not label2idx_model_ckpt.get(unk_str): max_prob_k = all_probs_cat[i].max().item()
            if max_prob_k < post_hoc_thresh: final_preds_eval[i]=unk_eval_idx
    
    metrics = {}
    valid_mask = final_preds_eval!=-1
    
    if len(all_true_eval_cat[valid_mask]) > 0:
        metrics["Overall Accuracy (Eval)"] = (final_preds_eval[valid_mask]==all_true_eval_cat[valid_mask]).sum().item()/len(all_true_eval_cat[valid_mask])
    else: metrics["Overall Accuracy (Eval)"]=0.0

    if num_eval_classes>0 and valid_mask.sum()>0:
        metrics["Macro F1 (Eval)"] = F1Score(task="multiclass",num_classes=num_eval_classes,average="macro",ignore_index=-1)(final_preds_eval[valid_mask],all_true_eval_cat[valid_mask]).item()
        metrics["Mean Sensitivity (Eval)"] = Recall(task="multiclass",num_classes=num_eval_classes,average="macro",ignore_index=-1,zero_division=0)(final_preds_eval[valid_mask],all_true_eval_cat[valid_mask]).item()
    else: metrics["Macro F1 (Eval)"]=0.0; metrics["Mean Sensitivity (Eval)"]=0.0; logger.warning(f"No MacroF1/MeanSens for {ckpt_path.name}.")
    
    true_lbls_model_auc, probs_model_auc_list = [], []
    for i,eval_idx in enumerate(all_true_eval_cat.tolist()):
        lbl_s = idx2label_eval_ckpt.get(eval_idx)
        if lbl_s and lbl_s in label2idx_model_ckpt:
            true_lbls_model_auc.append(label2idx_model_ckpt[lbl_s]); probs_model_auc_list.append(all_probs_cat[i])
    
    if true_lbls_model_auc and num_model_classes > 0:
        true_t_auc = torch.tensor(true_lbls_model_auc,dtype=torch.long)
        probs_t_auc = torch.stack(probs_model_auc_list)
        
        # Per-class Full AUROC
        per_class_auroc_full = AUROC(task="multiclass",num_classes=num_model_classes,average=None)(probs_t_auc,true_t_auc)
        for i in range(num_model_classes):
            class_name = idx2label_model_ckpt.get(i, f"ClassIdx{i}")
            metrics[f"AUROC_{class_name} (Model)"] = per_class_auroc_full[i].item() if not torch.isnan(per_class_auroc_full[i]) else 0.0
        metrics["Macro AUROC (Model)"] = torch.nanmean(per_class_auroc_full).item() if not torch.all(torch.isnan(per_class_auroc_full)) else 0.0
        
        # Per-class Partial AUROC
        pauc_fpr_val = training_cfg_ckpt.get("pauc_max_fpr",0.2)
        per_class_pauroc = AUROC(task="multiclass",num_classes=num_model_classes,average=None,max_fpr=pauc_fpr_val)(probs_t_auc,true_t_auc)
        for i in range(num_model_classes):
            class_name = idx2label_model_ckpt.get(i, f"ClassIdx{i}")
            metrics[f"pAUROC_{class_name}@FPR{pauc_fpr_val:.1f} (Model)"] = per_class_pauroc[i].item() if not torch.isnan(per_class_pauroc[i]) else 0.0
        metrics[f"Macro pAUROC@FPR{pauc_fpr_val:.1f} (Model)"] = torch.nanmean(per_class_pauroc).item() if not torch.all(torch.isnan(per_class_pauroc)) else 0.0
    else:
        logger.warning(f"Not enough valid samples for AUROC/pAUROC for {ckpt_path.name}.")
        metrics["Macro AUROC (Model)"]=0.0
        pauc_fpr_val = training_cfg_ckpt.get("pauc_max_fpr",0.2)
        metrics[f"Macro pAUROC@FPR{pauc_fpr_val:.1f} (Model)"]=0.0
        # Add NaN or 0.0 for per-class if no samples
        for i in range(num_model_classes):
            class_name = idx2label_model_ckpt.get(i, f"ClassIdx{i}")
            metrics[f"AUROC_{class_name} (Model)"] = 0.0
            metrics[f"pAUROC_{class_name}@FPR{pauc_fpr_val:.1f} (Model)"] = 0.0

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints and summarize performance.")
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path, help="Model checkpoint paths.")
    parser.add_argument("--fold_id", required=True, help="Fold ID for validation.")
    parser.add_argument("--labels_csv", required=True, type=Path, help="Main CSV with labels and folds.")
    parser.add_argument("--image_dir", required=True, type=Path, help="Root directory of images.")
    parser.add_argument("--meta_csv", type=Path, default=None, help="Metadata CSV (if any meta models).")
    parser.add_argument("--base_config", type=Path, default=None, help="Base YAML config for fallbacks.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Save summary table CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed); device = get_device(); logger.info(f"Using device: {device}")
    if not args.labels_csv.exists(): logger.error(f"Labels CSV missing: {args.labels_csv}"); sys.exit(1)
    if not args.image_dir.is_dir(): logger.error(f"Image dir missing or not a dir: {args.image_dir}"); sys.exit(1)
    if args.meta_csv and not args.meta_csv.exists(): logger.warning(f"Meta CSV {args.meta_csv} specified but not found.")

    full_labels_df = pd.read_csv(args.labels_csv)
    meta_features_df = pd.read_csv(args.meta_csv) if args.meta_csv and args.meta_csv.exists() else None
    base_cfg = load_config(args.base_config) if args.base_config and args.base_config.exists() else {}
    if args.base_config and not base_cfg: logger.warning(f"Base config {args.base_config} specified but not loaded.")

    fold_id_to_eval = args.fold_id 
    if 'fold' in full_labels_df.columns and pd.api.types.is_numeric_dtype(full_labels_df['fold']):
        try: fold_id_to_eval = int(args.fold_id)
        except ValueError: logger.warning(f"Fold ID '{args.fold_id}' is not int, but 'fold' column is numeric. Using as string.")
    if fold_id_to_eval not in full_labels_df['fold'].unique():
        logger.error(f"Fold ID '{fold_id_to_eval}' not in CSV. Available: {full_labels_df['fold'].unique().tolist()}"); sys.exit(1)

    results_data_for_df = []
    for ckpt_p in args.checkpoints:
        exp_name = get_experiment_name_from_path(ckpt_p)
        metrics_dict = evaluate_single_model_on_fold(
            ckpt_path=ckpt_p, fold_id=fold_id_to_eval, full_labels_df=full_labels_df,
            image_dir=args.image_dir, meta_features_df=meta_features_df,
            device=device, global_base_cfg=base_cfg
        )
        if metrics_dict:
            results_data_for_df.append({"Experiment": exp_name, "Checkpoint": ckpt_p.name, **metrics_dict})
        else:
            logger.warning(f"Evaluation failed for checkpoint: {ckpt_p}")

    if not results_data_for_df:
        logger.info("No results to summarize."); return
    
    summary_df = pd.DataFrame(results_data_for_df)
    if summary_df.empty:
        logger.info("Summary DataFrame is empty. No metrics to show."); return

    metric_cols = [col for col in summary_df.columns if col not in ["Experiment", "Checkpoint"]]
    ordered_cols = ["Experiment", "Checkpoint"] + sorted(
        metric_cols, 
        key=lambda x: (
            "accuracy" not in x.lower(),        # Accuracy first
            "sensitivity" not in x.lower(),     # Then sensitivity
            "f1" not in x.lower(),              # Then F1
            "auroc" not in x.lower(),           # Then AUROC variants
            "macro" not in x.lower(),           # Group macro metrics after per-class of same type
            "pauroc" in x.lower(),              # Ensure pAUROC comes after AUROC for same class
            x # Alphabetical for remaining/ties
        )
    )
    summary_df = summary_df[ordered_cols]
    
    logger.info("\n--- Evaluation Summary Table ---")
    try:
        from tabulate import tabulate
        print(tabulate(summary_df, headers='keys', tablefmt='pipe', floatfmt=".4f"))
    except ImportError:
        print(summary_df.to_markdown(index=False, floatfmt=".4f"))

    if args.output_csv:
        try:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(args.output_csv, index=False, float_format="%.4f")
            logger.info(f"Summary table saved to: {args.output_csv}")
        except Exception as e:
            logger.error(f"Failed to save summary CSV to {args.output_csv}: {e}")

if __name__ == "__main__":
    main()
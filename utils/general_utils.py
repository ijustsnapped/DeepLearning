# your_project_name/utils/general_utils.py
import random
import numpy as np
import torch
import yaml
from pathlib import Path

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Can impact performance, set to True if input sizes don't vary

def load_config(config_path: str | Path) -> dict:
    """Loads a YAML configuration file safely."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def cast_config_values(cfg: dict) -> dict:
    """Casts specific configuration values to their correct types."""
    for key in [
        "BATCH_SIZE", "NUM_WORKERS", "STEP_SIZE", "NUM_EPOCHS",
        "VAL_INTERVAL", "ACCUM_STEPS", "FREEZE_EPOCHS",
        "EARLY_STOPPING_PATIENCE", "RAND_AUG_N", "RAND_AUG_M",
        "RESIZE", "CROP_SIZE"
    ]:
        if key in cfg:
            cfg[key] = int(cfg[key])
    for key in [
        "LEARNING_RATE", "WEIGHT_DECAY", "EMA_DECAY", "GAMMA",
        "BACKBONE_LR_MULT", "MIXUP_ALPHA", "CUTMIX_ALPHA",
        "HFLIP_P", "ROTATE_DEGREES", "MIN_LR"
        # For COLOR_JITTER, ensure it's a list/tuple of floats if specified that way
    ]:
        if key in cfg:
            cfg[key] = float(cfg[key])
    if "COLOR_JITTER" in cfg and isinstance(cfg["COLOR_JITTER"], list):
         cfg["COLOR_JITTER"] = [float(x) for x in cfg["COLOR_JITTER"]]

    return cfg
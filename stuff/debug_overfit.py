#!/usr/bin/env python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # If you still need this

import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from pathlib import Path

# --- Adjusted Imports based on your directory structure ---
try:
    from data_handling.datasets import FlatDataset # Assuming FlatDataset is in datasets.py
    from data_handling.transforms import build_transform # Assuming build_transform is in transforms.py
    from models.factory import get_model
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Details: {e}")
    print("Please ensure the script is run from your project's root directory,")
    print("and that __init__.py files are present in 'data_handling' and 'models' directories.")
    print("Current working directory:", os.getcwd())
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def overfit_small_batch(cfg: dict):
    """
    Attempts to overfit the model on a very small batch of data.
    """
    general_cfg = cfg.get("general", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {}) # data_cfg will contain the flat cpu_augmentations
    optimizer_cfg = cfg.get("optimizer", {})
    overfit_test_cfg = cfg.get("overfit_test", {})

    # --- 1. Setup Device ---
    device_str = general_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # --- 2. Prepare a Tiny Dataset ---
    labels_csv_path = Path(data_cfg.get("labels_csv_path", "path/to/your/labels.csv"))
    train_root_path = Path(data_cfg.get("train_root_path", "path/to/your/images"))
    num_samples_per_class = overfit_test_cfg.get("num_samples_per_class", 2)
    num_target_classes = model_cfg.get("numClasses", 2)

    if not labels_csv_path.exists():
        logger.error(f"Labels CSV not found at: {labels_csv_path}")
        return False
    if not train_root_path.exists():
        logger.error(f"Train root image directory not found at: {train_root_path}")
        return False

    df_all = pd.read_csv(labels_csv_path)
    if 'filename' not in df_all.columns or 'label' not in df_all.columns:
        logger.error(f"Labels CSV '{labels_csv_path}' must contain 'filename' and 'label' columns.")
        return False

    labels_unique = sorted(df_all['label'].unique())
    if num_target_classes > len(labels_unique):
        logger.warning(f"Requested numClasses ({num_target_classes}) is more than unique labels in CSV ({len(labels_unique)}). Using {len(labels_unique)} classes.")
        num_target_classes = len(labels_unique)
    
    model_cfg["numClasses"] = num_target_classes
    
    selected_labels = labels_unique[:num_target_classes]
    label2idx = {name: i for i, name in enumerate(selected_labels)}
    logger.info(f"Using classes for test: {selected_labels} with mapping: {label2idx}")

    tiny_df_list = []
    for label_name, label_idx in label2idx.items():
        class_samples = df_all[df_all['label'] == label_name].head(num_samples_per_class)
        if len(class_samples) < num_samples_per_class:
            logger.warning(f"Could only find {len(class_samples)} samples for class '{label_name}', requested {num_samples_per_class}.")
        if not class_samples.empty:
            tiny_df_list.append(class_samples)

    if not tiny_df_list:
        logger.error("Could not prepare any samples for the tiny dataset. Check CSV and class names.")
        return False

    tiny_df = pd.concat(tiny_df_list).reset_index(drop=True)
    if tiny_df.empty:
        logger.error("Tiny DataFrame is empty after trying to select samples.")
        return False

    logger.info(f"Created a tiny dataset with {len(tiny_df)} samples:")
    logger.info(tiny_df.head())

    # --- Get CPU Augmentation config directly from data_cfg ---
    # This dictionary should now have the flat keys: 'resize', 'crop_size', 'norm_mean', 'norm_std'
    cpu_augs_config_for_overfit = data_cfg.get("cpu_augmentations")
    if not cpu_augs_config_for_overfit:
        logger.error("'cpu_augmentations' dictionary not found in the 'data' section of the test_config.")
        return False
    if not all(k in cpu_augs_config_for_overfit for k in ['resize', 'crop_size', 'norm_mean', 'norm_std']):
        logger.error(f"cpu_augmentations is missing one of the required keys: 'resize', 'crop_size', 'norm_mean', 'norm_std'. Found: {list(cpu_augs_config_for_overfit.keys())}")
        return False

    logger.info(f"Using CPU augmentations for overfit test (passed to build_transform): {cpu_augs_config_for_overfit}")
    transform_tiny = build_transform(cpu_augs_config_for_overfit, train=False) # train=False for minimal processing

    tiny_dataset = FlatDataset(df=tiny_df, root=train_root_path, label2idx=label2idx, tf=transform_tiny)
    
    if len(tiny_dataset) == 0:
        logger.error("FlatDataset is empty. Check if filenames in the tiny_df correctly map to images in train_root_path.")
        return False

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=len(tiny_dataset), # One single batch containing all tiny_dataset samples
        shuffle=True,
        num_workers=0 # Simpler for debugging
    )

    # --- 3. Initialize Model ---
    if "type" in model_cfg and "MODEL_TYPE" not in model_cfg:
        model_cfg["MODEL_TYPE"] = model_cfg.pop("type")
    if "MODEL_TYPE" not in model_cfg:
        logger.error("Missing 'MODEL_TYPE' in model configuration for get_model.")
        return False

    model = get_model(model_cfg).to(device)
    model.train() # Set to train mode
    logger.info(f"Model '{model_cfg['MODEL_TYPE']}' loaded on {device}.")

    # --- 4. Optimizer and Loss ---
    learning_rate = optimizer_cfg.get("lr", 1e-3)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Optimizer: AdamW (lr={learning_rate}), Loss: CrossEntropyLoss")

    # --- 5. Training Loop to Overfit ---
    num_epochs_overfit = overfit_test_cfg.get("num_epochs", 300)
    target_accuracy = overfit_test_cfg.get("target_accuracy", 1.0)
    print_interval = overfit_test_cfg.get("print_interval", 10)

    logger.info(f"Attempting to overfit for {num_epochs_overfit} epochs, aiming for {target_accuracy*100}% accuracy.")

    try:
        imgs, labels = next(iter(tiny_loader))
    except StopIteration:
        logger.error("DataLoader returned no data. Tiny dataset might be effectively empty or batch_size issue.")
        return False
        
    imgs = imgs.to(device)
    labels = labels.to(device)

    for epoch in range(num_epochs_overfit):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)

        if (epoch + 1) % print_interval == 0 or accuracy >= target_accuracy:
            logger.info(f"Epoch [{epoch+1}/{num_epochs_overfit}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f} ({correct}/{len(labels)})")

        if accuracy >= target_accuracy:
            logger.info(f"Successfully overfit the small batch! Reached {accuracy*100:.2f}% accuracy.")
            return True

    logger.warning(f"Failed to reach {target_accuracy*100}% accuracy after {num_epochs_overfit} epochs. "
                   f"Final accuracy: {accuracy:.4f}. There might be a fundamental issue.")
    return False

if __name__ == "__main__":
    # --- Configuration for the Overfit Test ---
    # !!! YOU MUST UPDATE THESE PATHS AND MODEL SETTINGS !!!
    test_config = {
        "general": {
            "device": "cuda" # or "cpu"
        },
        "model": {
            # 1. CHOOSE YOUR MODEL_TYPE from models/factory.py (e.g., "efficientnet_b0", "efficientnet_b3")
            "MODEL_TYPE": "efficientnet_b0", # <--- EDIT THIS
            "numClasses": 2,
            "pretrained": True,
            # This img_size will be used to populate 'resize' and 'crop_size' in cpu_augmentations below
            "img_size": 224 # <--- EDIT THIS if different for your model
        },
        "data": {
            # 2. SET PATH TO YOUR LABELS CSV (e.g., "splits/training/labels.csv")
            "labels_csv_path": "splits/training/labels.csv", # <--- EDIT THIS
            # 3. SET PATH TO THE ROOT OF YOUR TRAINING IMAGES
            #    Example: If labels.csv has "ISIC_0000000.jpg" and it's in "splits/training/ISIC2019/ISIC_0000000.jpg",
            #    then train_root_path should be "splits/training/ISIC2019"
            "train_root_path": "splits/training", # <--- !! VERY IMPORTANT - EDIT THIS !!

            # cpu_augmentations is now a flat dictionary with keys expected by your build_transform
            "cpu_augmentations": {
                # These values will be updated by model.img_size just below this config block
                "resize": 224,
                "crop_size": 224,
                # 4. VERIFY/EDIT NORMALIZATION STATS if different from ImageNet defaults
                "norm_mean": [0.485, 0.456, 0.406], # <--- EDIT THESE if needed
                "norm_std": [0.229, 0.224, 0.225]   # <--- EDIT THESE if needed
                # Add any other flat keys your build_transform might require for train=False
            }
        },
        "optimizer": {
            "lr": 1e-3
        },
        "overfit_test": {
            "num_samples_per_class": 2,
            "num_epochs": 300,
            "target_accuracy": 1.0,
            "print_interval": 10
        }
    }

    # Dynamically set resize and crop_size in cpu_augmentations from model.img_size
    # This ensures consistency for the test.
    img_s_from_model_cfg = test_config["model"].get("img_size", 224)
    test_config["data"]["cpu_augmentations"]["resize"] = img_s_from_model_cfg
    test_config["data"]["cpu_augmentations"]["crop_size"] = img_s_from_model_cfg
    # Note: Your `build_transform` might expect `resize` to be a list/tuple, e.g., [img_s, img_s].
    # If it errors saying it expected a sequence for resize, change the above to:
    # test_config["data"]["cpu_augmentations"]["resize"] = [img_s_from_model_cfg, img_s_from_model_cfg]

    logger.info("Starting small batch overfitting test...")
    success = overfit_small_batch(test_config)

    if success:
        logger.info("Overfitting test PASSED.")
    else:
        logger.error("Overfitting test FAILED. Please check logs for errors and review your configuration.")
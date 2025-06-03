# your_project_name/data_handling/datasets.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class FlatDataset(Dataset): # Original FlatDataset
    def __init__(self, df: pd.DataFrame, root: Path, label2idx: dict[str,int],
                 tf: transforms.Compose, image_loader: str = "pil", enable_ram_cache: bool = False):
        self.samples = [(root/row.dataset/row.filename, label2idx[row.label])
                        for row in df.itertuples(index=False)]
        self.tf = tf
        self.image_loader = image_loader.lower()
        self.enable_ram_cache = enable_ram_cache
        self.image_cache = {}

        if self.image_loader == "opencv" and not CV2_AVAILABLE:
            logger.warning("OpenCV image loader selected in config, but cv2 is not installed. Falling back to PIL.")
            self.image_loader = "pil"

        if self.image_loader == "opencv":
            logger.info("FlatDataset is using OpenCV (cv2) for image loading.")
        else:
            logger.info("FlatDataset is using PIL for image loading.")

        if self.enable_ram_cache:
            logger.info("FlatDataset RAM cache is enabled. This may consume significant memory.")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path_obj, label = self.samples[idx]
        path_str = str(path_obj)

        img_pil = None

        if self.enable_ram_cache and path_str in self.image_cache:
            img_pil = self.image_cache[path_str]
        else:
            try:
                if self.image_loader == "opencv":
                    img_bgr = cv2.imread(path_str)
                    if img_bgr is None:
                        raise RuntimeError(f"cv2.imread failed to load image (returned None): {path_str}")
                    img_rgb_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb_np)
                else:
                    img_pil = Image.open(path_str).convert("RGB")

                if self.enable_ram_cache and img_pil is not None:
                    self.image_cache[path_str] = img_pil

            except UnidentifiedImageError:
                logger.error(f"Failed to load image (UnidentifiedImageError): {path_str}")
                raise
            except Exception as e:
                logger.error(f"Error processing image {path_str} with loader '{self.image_loader}': {e}", exc_info=True)
                raise

        if img_pil is None:
            logger.error(f"Image at {path_str} could not be loaded or retrieved from cache.")
            raise RuntimeError(f"Image at {path_str} resulted in None.")

        return self.tf(img_pil), label

class FlatDatasetWithMeta(Dataset):
    def __init__(self,
                 df: pd.DataFrame, # Main dataframe with labels, image paths
                 meta_df: pd.DataFrame, # Dataframe with metadata
                 root: Path,
                 label2idx: dict[str, int],
                 tf: transforms.Compose,
                 image_loader: str = "pil",
                 enable_ram_cache: bool = False,
                 meta_features_names: list[str] | None = None, # List of metadata column names to use
                 meta_augmentation_p: float = 0.0, # Probability to augment metadata
                 meta_nan_fill_value: float = 0.0 # Value to use for augmented "missing" numerical features
                ):

        self.root = root
        self.label2idx = label2idx
        self.tf = tf
        self.image_loader = image_loader.lower()
        self.enable_ram_cache = enable_ram_cache
        self.image_cache = {}

        self.meta_augmentation_p = meta_augmentation_p
        self.meta_nan_fill_value = meta_nan_fill_value
        self.training = True # Assume training mode for augmentation by default, can be set by user

        df['image_id_merge_key'] = df['filename'].apply(lambda x: Path(x).stem)

        if 'source' in meta_df.columns:
            meta_df_processed = meta_df.drop(columns=['source'])
        else:
            meta_df_processed = meta_df.copy()


        self.merged_df = pd.merge(df, meta_df_processed, left_on='image_id_merge_key', right_on='image', how='left')

        if meta_features_names is None:
            self.meta_features_names = [col for col in meta_df_processed.columns if col != 'image']
        else:
            self.meta_features_names = meta_features_names

        logger.info(f"Using metadata features: {self.meta_features_names}")

        self.meta_feature_groups = {
            "age": ["age_zscore"],
            "site": [col for col in self.meta_features_names if "anatom_site_general" in col],
            "sex": [col for col in self.meta_features_names if "sex" in col]
        }
        self.meta_nan_indicators = {
            "site": "anatom_site_general_nan",
            "sex": "sex_nan"
        }

        for col in self.meta_features_names:
            if self.merged_df[col].isnull().any():
                logger.warning(f"NaNs found in metadata column '{col}' after merge. Filling with {self.meta_nan_fill_value}.")
                is_nan_indicator = False
                for group, nan_col_name in self.meta_nan_indicators.items():
                    if col == nan_col_name:
                        self.merged_df[col] = self.merged_df[col].fillna(True)
                        is_nan_indicator = True
                        break
                if not is_nan_indicator:
                    self.merged_df[col] = self.merged_df[col].fillna(self.meta_nan_fill_value)


        self.samples = []
        self._labels_for_sampler = [] # MODIFIED: Initialize list to store labels

        for row in self.merged_df.itertuples(index=False):
            img_path = self.root / row.dataset / row.filename
            label = self.label2idx[row.label]
            self._labels_for_sampler.append(label) # MODIFIED: Store label for sampler

            meta_values = [getattr(row, f_name, self.meta_nan_fill_value) for f_name in self.meta_features_names]
            meta_values = [float(v) if isinstance(v, bool) else v for v in meta_values]

            self.samples.append({
                "path": img_path,
                "label": label,
                "meta_orig": torch.tensor(meta_values, dtype=torch.float32)
            })

        if self.image_loader == "opencv" and not CV2_AVAILABLE:
            logger.warning("OpenCV image loader selected, but cv2 is not installed. Falling back to PIL.")
            self.image_loader = "pil"

        logger.info(f"FlatDatasetWithMeta initialized with {len(self.samples)} samples.")
        if self.enable_ram_cache: logger.info("RAM cache for images enabled.")
        if self.meta_augmentation_p > 0: logger.info(f"Metadata augmentation enabled with p={self.meta_augmentation_p}")

    def _augment_metadata(self, meta_tensor: torch.Tensor) -> torch.Tensor:
        if self.meta_augmentation_p == 0.0 or not self.training:
            return meta_tensor

        augmented_meta = meta_tensor.clone()
        feature_to_idx = {name: i for i, name in enumerate(self.meta_features_names)}

        for group_name, features_in_group in self.meta_feature_groups.items():
            if np.random.rand() < self.meta_augmentation_p:
                if group_name == "age":
                    if "age_zscore" in feature_to_idx:
                        augmented_meta[feature_to_idx["age_zscore"]] = self.meta_nan_fill_value
                elif group_name in ["site", "sex"]:
                    nan_indicator_col = self.meta_nan_indicators.get(group_name)
                    for f_name in features_in_group:
                        if f_name in feature_to_idx:
                            if f_name == nan_indicator_col :
                                augmented_meta[feature_to_idx[f_name]] = 1.0
                            else:
                                augmented_meta[feature_to_idx[f_name]] = 0.0
        return augmented_meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_data = self.samples[idx]
        path_obj = sample_data["path"]
        label = sample_data["label"]
        meta_tensor_orig = sample_data["meta_orig"]

        path_str = str(path_obj)
        img_pil = None

        if self.enable_ram_cache and path_str in self.image_cache:
            img_pil = self.image_cache[path_str]
        else:
            try:
                if self.image_loader == "opencv":
                    img_bgr = cv2.imread(path_str)
                    if img_bgr is None: raise RuntimeError(f"cv2.imread failed: {path_str}")
                    img_rgb_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb_np)
                else:
                    img_pil = Image.open(path_str).convert("RGB")
                if self.enable_ram_cache and img_pil is not None: self.image_cache[path_str] = img_pil
            except Exception as e:
                logger.error(f"Error loading image {path_str}: {e}", exc_info=True)
                # Return a placeholder or skip? For now, re-raise.
                # Consider returning a dummy item if robustness is critical over correctness here.
                raise

        if img_pil is None: # Should ideally be caught by exceptions above
            logger.error(f"Image at {path_str} resulted in None despite load attempts.")
            raise RuntimeError(f"Image at {path_str} resulted in None.")


        img_tensor = self.tf(img_pil)
        meta_tensor_processed = self._augment_metadata(meta_tensor_orig)

        return (img_tensor, meta_tensor_processed), label

    # MODIFIED: Added get_labels method
    def get_labels(self) -> list[int]:
        """Returns a list of all labels in the dataset for use by samplers."""
        return self._labels_for_sampler

    @property
    def metadata_dim(self):
        return len(self.meta_features_names)
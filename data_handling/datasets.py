# your_project_name/data_handling/datasets.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

class FlatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, label2idx: dict[str,int], tf: transforms.Compose):
        self.samples = [(root/row.dataset/row.filename, label2idx[row.label])
                        for row in df.itertuples(index=False)]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            # Use context manager to ensure file handles are released promptly.
            with Image.open(path) as img:
                img = img.convert("RGB")
        except UnidentifiedImageError:
            # Consider logging this error or returning a placeholder
            # For now, raising an error as in the original script
            raise RuntimeError(f"Failed to load image: {path}")
        except Exception as e:
            # Catch other potential PIL errors
            raise RuntimeError(f"Error processing image {path}: {e}")
        return self.tf(img), label

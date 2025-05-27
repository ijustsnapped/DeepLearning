# your_project_name/data_handling/custom_samplers.py
import logging
import torch
import torch.utils.data
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class ClassBalancedSampler(torch.utils.data.sampler.Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset,
    sampling with replacement from each class with probability proportional
    to 1/sqrt(N_c), where N_c is the number of samples in class c.

    Args:
        dataset: The dataset to sample from. Must have a way to get all labels,
                 e.g., by iterating or having a `get_labels()` method.
                 Assumes `dataset.samples` is a list of (path, label_idx) or similar
                 or `dataset.targets` exists.
        num_samples (int, optional): The total number of samples to draw. If None,
                                     defaults to len(dataset).
    """
    def __init__(self, dataset, num_samples: int | None = None):
        super().__init__(dataset)
        self.dataset = dataset
        self._num_samples = num_samples if num_samples is not None else len(self.dataset)

        logger.info("Initializing ClassBalancedSampler (1/sqrt(N_c) weighting)...")

        # Get all labels from the dataset
        if hasattr(dataset, 'samples') and isinstance(dataset.samples, list) and \
           len(dataset.samples) > 0 and isinstance(dataset.samples[0], tuple) and len(dataset.samples[0]) == 2:
            # Assuming dataset.samples is like [(path, label_idx), ...]
            labels = [s[1] for s in dataset.samples]
        elif hasattr(dataset, 'targets') and isinstance(dataset.targets, (list, torch.Tensor, np.ndarray)):
            labels = np.array(dataset.targets) # Works for list, tensor, ndarray
        elif hasattr(dataset, 'get_labels'): # For datasets with a get_labels method
             labels = dataset.get_labels()
        else:
            # Fallback: iterate through the dataset (can be slow for large datasets)
            logger.warning("Trying to infer labels by iterating through the dataset for ClassBalancedSampler. "
                           "This can be slow. Consider adding a 'samples' or 'targets' attribute or a 'get_labels()' method to your dataset.")
            labels = [dataset[i][1] for i in range(len(dataset))]

        if not labels:
            raise ValueError("Could not extract labels from the dataset for ClassBalancedSampler.")

        class_counts = Counter(labels)
        num_classes = len(class_counts)
        
        if num_classes == 0:
            raise ValueError("No classes found in the dataset for ClassBalancedSampler.")

        min_class_label = min(class_counts.keys())
        max_class_label = max(class_counts.keys())
        
        # Ensure class labels are contiguous from 0 to num_classes-1 for weight indexing
        if not (min_class_label == 0 and max_class_label == num_classes -1 and len(class_counts) == num_classes):
             logger.warning(f"Class labels are not contiguous from 0 to {num_classes-1}. Min: {min_class_label}, Max: {max_class_label}. This might cause issues with weight indexing if not handled carefully.")
             # For robustness, create weights array assuming labels are 0..num_classes-1
             # If labels are e.g. [1, 5, 10], this simple approach might fail or need remapping.
             # For now, assuming labels are already 0-indexed and contiguous from data prep.
             
        # Calculate weights: 1 / sqrt(N_c)
        # Ensure all potential class indices are covered, even if some classes have 0 samples in this subset.
        # This is important if num_classes is known from label2idx but some classes are missing in train_df.
        # For simplicity, we use counts from the current dataset.
        
        # Weights for each class
        class_weights = {
            cls_idx: 1.0 / np.sqrt(count) if count > 0 else 0
            for cls_idx, count in class_counts.items()
        }
        
        # Create weights for each sample in the dataset
        self.weights = torch.zeros(len(labels), dtype=torch.double)
        for i, label_idx in enumerate(labels):
            self.weights[i] = class_weights.get(label_idx, 0) # Use .get for safety if a label is somehow not in class_weights

        if torch.sum(self.weights).item() == 0:
            logger.warning("All sample weights are zero in ClassBalancedSampler. Defaulting to uniform sampling.")
            # Fallback to uniform sampling if all weights are zero (e.g., empty dataset or all classes have 0 count)
            self.weights = torch.ones(len(labels), dtype=torch.double) / len(labels) if len(labels) > 0 else torch.tensor([])


        logger.info(f"ClassBalancedSampler initialized. Num samples to draw: {self.num_samples}. "
                    f"Class counts (first 5): {dict(list(class_counts.items())[:5])}. "
                    f"Sample weights for first 5 samples: {self.weights[:5]}")


    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self):
        if len(self.weights) == 0: # Handle empty dataset case
            return iter([])
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self) -> int:
        return self.num_samples
# E:/DeepLearn/utils/stat_utils.py
import numpy as np
from sklearn.metrics import f1_score, accuracy_score # Add other sklearn metrics as needed
from torchmetrics import AUROC as TorchMetricsAUROC # For AUROC
from torchmetrics import AveragePrecision as TorchMetricsAP # For AP
import torch
import logging # For logging warnings within the util

logger_stat = logging.getLogger(__name__) # Use a specific logger for this util

def bootstrap_metric(y_true: np.ndarray,
                     y_pred: np.ndarray, # Can be hard predictions or probabilities
                     metric_func: callable,
                     metric_kwargs: dict | None = None,
                     n_bootstraps: int = 1000,
                     seed: int = 42) -> np.ndarray:
    """
    Calculates bootstrapped metric scores.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    if n_samples == 0:
        logger_stat.warning("bootstrap_metric called with empty y_true. Returning empty array.")
        return np.array([])
        
    bootstrap_scores = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        indices = rng.choice(n_samples, n_samples, replace=True)
        resampled_true = y_true[indices]
        
        # Handle y_pred shape: if it's probabilities [n_samples, n_classes], it needs to be indexed correctly
        if y_pred.ndim == 2 and y_pred.shape[0] == n_samples: # Assuming y_pred is [n_samples, n_features/n_classes]
            resampled_pred = y_pred[indices, :]
        elif y_pred.ndim == 1 and y_pred.shape[0] == n_samples: # Assuming y_pred is [n_samples] (hard labels)
            resampled_pred = y_pred[indices]
        else:
            logger_stat.error(f"y_pred shape {y_pred.shape} not compatible with y_true shape {y_true.shape} for bootstrapping.")
            return np.array([np.nan] * n_bootstraps)


        # Special handling for torchmetrics that expect tensors
        is_torchmetric_instance = isinstance(metric_func, (TorchMetricsAUROC, TorchMetricsAP))
        is_bound_torchmetric = hasattr(metric_func, '__self__') and isinstance(metric_func.__self__, (TorchMetricsAUROC, TorchMetricsAP))

        if is_torchmetric_instance or is_bound_torchmetric:
            resampled_true_tensor = torch.from_numpy(resampled_true).long().cpu()
            resampled_pred_tensor = torch.from_numpy(resampled_pred).float().cpu()
            try:
                # Ensure the metric object is reset if it accumulates state (though for AUROC/AP passed as func, it's usually stateless per call)
                # If metric_func is an *instance* that accumulates, it should be cloned or reset.
                # For functional calls like this, it's usually okay.
                bootstrap_scores[i] = metric_func(resampled_pred_tensor, resampled_true_tensor, **metric_kwargs).item()
            except Exception as e:
                # logger_stat.warning(f"Bootstrap iteration {i} for torchmetric failed: {e}. Assigning NaN.")
                bootstrap_scores[i] = np.nan
        else: # sklearn-like metrics
            try:
                bootstrap_scores[i] = metric_func(resampled_true, resampled_pred, **metric_kwargs)
            except ValueError as e: # Catch specific errors like "Target is multiclass but average='binary'"
                # logger_stat.warning(f"Bootstrap iteration {i} for {metric_func.__name__} failed (ValueError): {e}. Assigning NaN.")
                bootstrap_scores[i] = np.nan
            except Exception as e:
                # logger_stat.warning(f"Bootstrap iteration {i} for {metric_func.__name__} failed (Other Exception): {e}. Assigning NaN.")
                bootstrap_scores[i] = np.nan

    valid_scores = bootstrap_scores[~np.isnan(bootstrap_scores)]
    if len(valid_scores) < n_bootstraps:
        logger_stat.warning(f"Removed {n_bootstraps - len(valid_scores)} NaN scores from bootstrap results.")
    return valid_scores

def calculate_ci(bootstrap_scores: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Calculates confidence interval from bootstrap scores using percentile method.
    """
    if len(bootstrap_scores) < 2: # Need at least 2 scores to compute percentiles meaningfully
        logger_stat.warning(f"Not enough bootstrap scores ({len(bootstrap_scores)}) to calculate CI. Returning NaN.")
        return (np.nan, np.nan)
    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrap_scores, alpha * 100)
    upper_bound = np.percentile(bootstrap_scores, (1.0 - alpha) * 100)
    return lower_bound, upper_bound
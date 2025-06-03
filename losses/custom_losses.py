# your_project_name/losses/custom_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LDAMLoss(nn.Module):
    """
    LDAMLoss (Label-Distribution-Aware Margin Loss) from the paper:
    "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
    (https://arxiv.org/abs/1906.07413)

    Args:
        class_counts (list or np.ndarray): Number of samples per class.
        max_margin (float): The base maximum margin C.
        use_effective_number_margin (bool): If True, dynamically calculates margin based on effective number.
        effective_number_beta (float): Beta for calculating effective number of samples.
                                       Used only if use_effective_number_margin is True.
        scale (float): Scaling factor s for logits.
        weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                         If None, no re-weighting is applied by default here.
                                         DRW schedule will update this externally.
    """
    def __init__(self,
                 class_counts: list[int] | np.ndarray,
                 max_margin: float = 0.5,
                 use_effective_number_margin: bool = True,
                 effective_number_beta: float = 0.999,
                 scale: float = 30.0,
                 weight: torch.Tensor | None = None):
        super(LDAMLoss, self).__init__()
        if class_counts is None or len(class_counts) == 0:
            raise ValueError("class_counts must be provided for LDAMLoss.")

        num_classes = len(class_counts)
        self.s = scale
        self.weight = weight # This can be updated by DRW

        if use_effective_number_margin:
            if effective_number_beta < 0 or effective_number_beta >= 1:
                logger.warning(f"effective_number_beta should be in [0, 1). Got {effective_number_beta}. Clamping or using default logic.")
            # Calculate effective number of samples
            effective_num = 1.0 - np.power(effective_number_beta, class_counts)
            # per_cls_weights = (1.0 - effective_number_beta) / np.array(effective_num) # This was calculated but not used for margins
            # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            
            # Calculate margin based on effective number
            logger.info(f"LDAM using effective number based margin. Beta: {effective_number_beta}. Max margin: {max_margin}. Effective numbers (first 5): {effective_num[:5]}")
            # Use effective_num for margin calculation. Clip to avoid issues with very small effective_num.
            safe_effective_num = np.maximum(effective_num, 1e-6) 
            margins_raw = safe_effective_num ** (-0.25)
        else:
            # Use fixed margins if not using effective number, based on raw counts
            logger.info(f"LDAM using class count based margin (power -1/4). Max margin: {max_margin}. Class counts (first 5): {np.array(class_counts)[:5]}")
            safe_class_counts = np.maximum(class_counts, 1) # Avoid division by zero if a class has 0 samples
            margins_raw = safe_class_counts ** (-0.25)

        # Normalize margins so the largest calculated margin (for rarest class or smallest effective_num) becomes 1.0
        if np.max(margins_raw) > 1e-12: # Avoid division by zero if all margins are zero
            margins_normalized = margins_raw / np.max(margins_raw)
        else:
            logger.warning("Max of raw margins is close to zero. Using uniform margins of 1.0 before scaling.")
            margins_normalized = np.ones_like(margins_raw, dtype=float)
            
        # Scale by the user-defined max_margin
        margins = margins_normalized * max_margin
        margins = torch.from_numpy(margins.astype(np.float32)).float() # Ensure float32

        self.margins = margins.unsqueeze(0) # Shape [1, num_classes] for broadcasting
        # Log the final calculated margins after normalization and scaling by max_margin
        logger.info(f"LDAM final margins (first 5): {self.margins[0, :5].cpu().numpy()}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.margins.device != logits.device:
            self.margins = self.margins.to(logits.device)

        # Create one-hot targets and subtract margin for target classes
        index = torch.zeros_like(logits, dtype=torch.bool) # MODIFIED: Changed dtype to torch.bool
        index.scatter_(1, targets.data.view(-1, 1), 1)
        
        # Create margin tensor for subtraction
        # margin_sub = self.margins[index] # Direct use of boolean index if margins is [num_classes]
        # We need to subtract m_j from logit_j for the target class j
        # So, construct a batch_margins tensor where only target class has its margin, others 0
        batch_margins = torch.zeros_like(logits)
        # index here is already boolean, no need for .bool()
        batch_margins.scatter_(1, targets.data.view(-1,1), self.margins.repeat(logits.size(0),1)[index].view(-1,1))


        x_m = logits - batch_margins # Subtract margin only from the target class logit
        
        # Standard cross-entropy after adjusting logits
        # Apply scaling factor s
        output = torch.where(index, x_m, logits) # Use adjusted logits for target class, original for others (index is already bool)
        
        log_probs = F.log_softmax(self.s * output, dim=1)
        loss = F.nll_loss(log_probs, targets, weight=self.weight.to(logits.device) if self.weight is not None else None)
        
        return loss

    def update_weights(self, new_weights: torch.Tensor | None):
        """Used by DRW schedule to update class weights."""
        if new_weights is not None:
            logger.info(f"LDAMLoss weights updated. New weights (first 5): {new_weights}")
            self.weight = new_weights.float() # Ensure float
        else:
            logger.info("LDAMLoss weights reset (set to None).")
            self.weight = None
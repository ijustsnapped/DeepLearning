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
            per_cls_weights = (1.0 - effective_number_beta) / np.array(effective_num)
            # Normalize to avoid very large weights if some effective_num are tiny
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            
            # Calculate margin based on effective number: m_j = (1/N_j)^(1/4) or similar scaled by max_margin
            # The original paper suggests m_j = C / (N_j)^(1/4).
            # Here, N_j is class_counts[j].
            # We use a common interpretation: margin proportional to (class_prior)^(-1/4)
            margins = (np.array(class_counts) / sum(class_counts)) ** (-0.25) # (prior)^(-1/4)
            margins = margins / np.max(margins) # Normalize so max margin is 1
            margins = margins * max_margin # Scale by the user-defined max_margin
            margins = torch.from_numpy(margins).float()
            logger.info(f"LDAM using effective number based margin. Beta: {effective_number_beta}. Max margin: {max_margin}. Calculated margins (first 5): {margins[:5]}")
        else:
            # Use fixed margins if not using effective number, or a simpler approach
            # For simplicity, if not using effective number, we can make margins proportional to 1/N_j^0.25
            # This is similar to the above but directly on counts.
            # Or, if max_margin is meant to be a fixed value for all but the majority class,
            # that would be a different interpretation.
            # The paper implies m_j is class-specific.
            margins = (np.array(class_counts)) ** (-0.25)
            margins = margins / np.max(margins)
            margins = margins * max_margin
            margins = torch.from_numpy(margins).float()
            logger.info(f"LDAM using class count based margin (power -1/4). Max margin: {max_margin}. Calculated margins (first 5): {margins[:5]}")

        self.margins = margins.unsqueeze(0) # Shape [1, num_classes] for broadcasting

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.margins.device != logits.device:
            self.margins = self.margins.to(logits.device)

        # Create one-hot targets and subtract margin for target classes
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), True)
        
        # Create margin tensor for subtraction
        # margin_sub = self.margins[index.bool()] # This would be [N], not what we want
        # We need to subtract m_j from logit_j for the target class j
        # So, construct a batch_margins tensor where only target class has its margin, others 0
        batch_margins = torch.zeros_like(logits)
        batch_margins.scatter_(1, targets.data.view(-1, 1),
                               self.margins.repeat(logits.size(0), 1)[index].view(-1, 1))

        x_m = logits - batch_margins # Subtract margin only from the target class logit
        
        # Standard cross-entropy after adjusting logits
        # Apply scaling factor s
        output = torch.where(index, x_m, logits)  # Use adjusted logits for target class, original for others
        
        log_probs = F.log_softmax(self.s * output, dim=1)
        weight = self.weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        loss = F.nll_loss(log_probs, targets, weight=weight)
        
        return loss

    def update_weights(self, new_weights: torch.Tensor | None):
        """Used by DRW schedule to update class weights."""
        if new_weights is not None:
            logger.info(f"LDAMLoss weights updated. New weights (first 5): {new_weights[:5]}")
            self.weight = new_weights.float()
            # Keep weight on same device as margins for efficient forward pass
            if self.margins.device != self.weight.device:
                self.weight = self.weight.to(self.margins.device)
        else:
            logger.info("LDAMLoss weights reset (set to None).")
            self.weight = None

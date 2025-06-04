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
        super().__init__()
        if class_counts is None or len(class_counts) == 0:
            raise ValueError("class_counts must be provided for LDAMLoss.")

        counts = np.array(class_counts, dtype=np.float32)

        if use_effective_number_margin:
            if not (0 <= effective_number_beta < 1):
                logger.warning(f"effective_number_beta should be in [0,1). Got {effective_number_beta}.")
            effective_num = 1.0 - np.power(effective_number_beta, counts)
            safe_num = np.maximum(effective_num, 1e-6)
            margins_raw = safe_num ** (-0.25)
        else:
            safe_counts = np.maximum(counts, 1.0)
            margins_raw = safe_counts ** (-0.25)

        # Normalize & scale margins
        margins = (margins_raw / margins_raw.max()) * max_margin
        m = torch.from_numpy(margins).float()        # shape [num_classes]
        self.register_buffer("margins", m)           # auto-moves with .to(device)
        self.s = scale
        self.weight = weight

        logger.info(f"LDAM final margins (first 5): {self.margins[:5].cpu().numpy()}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # gather per-sample margins
        batch_m = self.margins[targets]               # [batch]
        # subtract margin from true-class logits
        logits_adj = logits.clone()
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_adj[idx, targets] -= batch_m

        # scaled log-softmax + nll loss
        logp = F.log_softmax(self.s * logits_adj, dim=1)
        return F.nll_loss(
            logp,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None
        )

    def update_weights(self, new_weights: torch.Tensor | None):
        if new_weights is not None:
            logger.info(f"LDAMLoss weights updated (first 5): {new_weights[:5]}")
            self.weight = new_weights.float()
        else:
            logger.info("LDAMLoss weights reset to None.")
            self.weight = None
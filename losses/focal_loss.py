# your_project_name/losses/focal_loss.py
import torch
import torch.nn.functional as F
from torch import Tensor

def focal_ce_loss(logits: Tensor, targets: Tensor, *, alpha: float = 1.0, gamma: float = 2.0) -> Tensor:
    """
    Computes the Focal Cross Entropy Loss.
    See: https://arxiv.org/abs/1708.02002
    """
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    focal = (1 - p) ** gamma
    onehot = F.one_hot(targets, logits.size(1)).to(logits.dtype) # Ensure onehot is same dtype as logits
    # Sum over class dimension, then mean over batch dimension
    return (-alpha * focal * onehot * logp).sum(dim=1).mean()
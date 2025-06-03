# your_project_name/utils/ema.py
import torch

def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    """
    Updates the Exponential Moving Average (EMA) of model parameters.
    """
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
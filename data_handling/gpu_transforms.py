# your_project_name/data_handling/gpu_transforms.py
from __future__ import annotations
import logging
import torch
import torch.nn as nn

try:
    import kornia.augmentation as K
    import kornia.geometry.transform as KGT
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    K = None
    KGT = None

logger = logging.getLogger(__name__)

class Lambda(nn.Module):
    """Helper module to apply a lambda function."""
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

def build_gpu_transform_pipeline(cfg_gpu_aug: dict, device: torch.device) -> nn.Sequential | None:
    """
    Builds a pipeline of GPU-accelerated image augmentations using Kornia.

    Args:
        cfg_gpu_aug: Configuration dictionary for GPU augmentations, typically
                     from `experiment_config['gpu_augmentations']`.
        device: The torch device (e.g., 'cuda') where transforms will run.

    Returns:
        An nn.Sequential module containing the Kornia transforms, or None
        if GPU augmentations are disabled or Kornia is not available.
    """
    if not cfg_gpu_aug.get("enable", False):
        logger.info("GPU augmentations are disabled in the configuration.")
        return None

    if not KORNIA_AVAILABLE:
        logger.warning("Kornia library is not installed. GPU augmentations will be skipped. "
                       "Install with: pip install kornia")
        return None

    pipeline_cfg = cfg_gpu_aug.get("pipeline", [])
    if not pipeline_cfg:
        logger.info("GPU augmentation pipeline is empty.")
        return None

    transforms_list = []
    logger.info("Building GPU augmentation pipeline...")

    for transform_entry in pipeline_cfg:
        name = transform_entry.get("name")
        params = transform_entry.get("params", {})
        
        # Ensure all params that should be tensors are on the correct device
        for p_key, p_val in params.items():
            if isinstance(p_val, (list, tuple)) and all(isinstance(x, (int, float)) for x in p_val):
                try: # Try converting to tensor if it looks like numerical data for kornia
                    params[p_key] = torch.tensor(p_val, device=device, dtype=torch.float32)
                except Exception: # If not convertible (e.g. kernel_size as int list) keep as is
                    pass


        transform_module = None
        try:
            if name == "RandomHorizontalFlipGPU":
                transform_module = K.RandomHorizontalFlip(**params)
            elif name == "RandomVerticalFlipGPU":
                transform_module = K.RandomVerticalFlip(**params)
            elif name == "ColorJitterGPU":
                # Kornia's ColorJitter takes ranges as tuples/lists, not torch.tensor for ranges
                # Ensure brightness, contrast, saturation, hue are tuples/lists of floats or single floats
                cj_params = {}
                for k, v_raw in params.items():
                    if isinstance(v_raw, torch.Tensor): # Convert tensor back if we made it one
                        v_list = v_raw.cpu().tolist()
                        if len(v_list) == 1: cj_params[k] = v_list[0]
                        elif len(v_list) == 2: cj_params[k] = tuple(v_list)
                        else: cj_params[k] = v_list # Fallback
                    else: # Already list/tuple/float
                        cj_params[k] = v_raw
                transform_module = K.ColorJitter(**cj_params)
            elif name == "RandomAffineGPU":
                # Degrees can be float or tuple. Others are often ranges.
                # Kornia expects degrees as float or tuple of floats.
                # Translate, scale, shear can be tuples of floats (range) or single floats.
                aff_params = params.copy()
                if 'degrees' in aff_params and isinstance(aff_params['degrees'], torch.Tensor):
                    deg_list = aff_params['degrees'].cpu().tolist()
                    if len(deg_list) == 1: aff_params['degrees'] = deg_list[0]
                    else: aff_params['degrees'] = tuple(deg_list) # Should be (-d, d) or (d1, d2)
                # Similar checks for translate, scale, shear if they were made tensors
                for k_aff in ['translate', 'scale', 'shear']:
                    if k_aff in aff_params and isinstance(aff_params[k_aff], torch.Tensor):
                         aff_params[k_aff] = tuple(aff_params[k_aff].cpu().tolist())


                transform_module = K.RandomAffine(**aff_params)
            elif name == "RandomGaussianBlurGPU":
                # kernel_size should be tuple of ints, sigma tuple of floats
                gb_params = params.copy()
                if 'kernel_size' in gb_params and isinstance(gb_params['kernel_size'], torch.Tensor):
                    gb_params['kernel_size'] = tuple(map(int, gb_params['kernel_size'].cpu().tolist()))
                if 'sigma' in gb_params and isinstance(gb_params['sigma'], torch.Tensor):
                    gb_params['sigma'] = tuple(gb_params['sigma'].cpu().tolist())
                transform_module = K.RandomGaussianBlur(**gb_params)
            elif name == "NormalizeGPU":
                # Mean and std should be tensors
                norm_params = {}
                norm_params['mean'] = torch.tensor(params['mean'], device=device, dtype=torch.float32)
                norm_params['std'] = torch.tensor(params['std'], device=device, dtype=torch.float32)
                transform_module = K.Normalize(**norm_params)
            elif name == "RandomErasingGPU":
                # Kornia's RandomErasing expects scale, ratio as tuples, value as float or tensor
                re_params = params.copy()
                for k_re in ['scale', 'ratio']:
                     if k_re in re_params and isinstance(re_params[k_re], torch.Tensor):
                         re_params[k_re] = tuple(re_params[k_re].cpu().tolist())
                if 'value' in re_params and isinstance(re_params['value'], list): # if value was made a tensor
                    re_params['value'] = re_params['value'][0] # Kornia expects single float or tensor for value

                transform_module = K.RandomErasing(**re_params)
            # Add more Kornia transforms here as needed
            # Example: K.RandomResizedCrop, K.RandomRotation, etc.
            else:
                logger.warning(f"Unknown GPU augmentation name: '{name}'. Skipping.")
                continue

            if transform_module:
                transforms_list.append(transform_module)
                logger.info(f"Added GPU transform: {name} with params: {params}")

        except Exception as e:
            logger.error(f"Failed to initialize GPU transform '{name}' with params {params}: {e}", exc_info=True)

    if not transforms_list:
        logger.info("No valid GPU transforms were added to the pipeline.")
        return None

    # Kornia augmentations are nn.Module, so they can be put in nn.Sequential
    # They also need to be on the correct device, but Kornia handles this internally
    # if the input tensor is on the device.
    gpu_transform_pipeline = nn.Sequential(*transforms_list).to(device)
    # For Kornia DataParallel augmentations (if batch is split across GPUs), use K.AugmentationSequential
    # gpu_transform_pipeline = K.AugmentationSequential(*transforms_list, data_keys=["input"]).to(device)
    # However, for single GPU or DDP, nn.Sequential is usually fine.

    logger.info(f"GPU augmentation pipeline built successfully with {len(transforms_list)} transforms.")
    return gpu_transform_pipeline
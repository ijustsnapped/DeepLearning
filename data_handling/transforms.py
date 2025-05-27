# your_project_name/data_handling/transforms.py
from __future__ import annotations
import logging
from torchvision import transforms
from torchvision.transforms import (
    Resize, CenterCrop, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, ColorJitter,
    RandAugment # Ensure this is imported
)

logger = logging.getLogger(__name__)

def build_transform(cfg_cpu_aug: dict, train: bool) -> transforms.Compose:
    if not all(k in cfg_cpu_aug for k in ["resize", "crop_size", "norm_mean", "norm_std"]):
        missing = [k for k in ["resize", "crop_size", "norm_mean", "norm_std"] if k not in cfg_cpu_aug]
        raise KeyError(f"Missing common CPU params: {missing}. Got: {list(cfg_cpu_aug.keys())}")

    if train:
        # Option: RandomResizedCrop can handle scaling and cropping in one step.
        # tf_list = [
        #     transforms.RandomResizedCrop(
        #         cfg_cpu_aug["crop_size"], 
        #         scale=cfg_cpu_aug.get("train",{}).get("affine_scale_range_for_rrc", (0.08, 1.0)) # Default RRC scale
        #     )
        # ]
        # If using RandomResizedCrop, you might adjust or remove RandomAffine's scaling component.
        # For now, using Resize + RandomCrop as per previous setup.
        tf_list = [
            Resize(cfg_cpu_aug["resize"], antialias=True),
            transforms.RandomCrop(cfg_cpu_aug["crop_size"]) # Ensures final size
        ]
    else: # Validation / Test
        tf_list = [
            Resize(cfg_cpu_aug["resize"], antialias=True),
            CenterCrop(cfg_cpu_aug["crop_size"])
        ]

    if train:
        train_specific_cfg = cfg_cpu_aug.get("train", {})
        logger.info("Building training augmentations (CPU)...")

        # 1. Random Flipping
        hflip_p = train_specific_cfg.get("random_horizontal_flip_p")
        if hflip_p is not None and hflip_p > 0:
            tf_list.append(RandomHorizontalFlip(p=hflip_p))
            logger.info(f"  Added RandomHorizontalFlip (p={hflip_p})")

        vflip_p = train_specific_cfg.get("random_vertical_flip_p")
        if vflip_p is not None and vflip_p > 0:
            tf_list.append(RandomVerticalFlip(p=vflip_p))
            logger.info(f"  Added RandomVerticalFlip (p={vflip_p})")

        # 2. Random Affine (Rotation, Scaling, Shear, Translation)
        affine_degrees = train_specific_cfg.get("affine_degrees")
        affine_translate_cfg = train_specific_cfg.get("affine_translate")
        affine_scale_cfg = train_specific_cfg.get("affine_scale_range")
        affine_shear_cfg = train_specific_cfg.get("affine_shear_degrees")
        affine_fill = train_specific_cfg.get("affine_fill")

        affine_kwargs = {}
        if affine_degrees is not None: affine_kwargs["degrees"] = affine_degrees
        if affine_translate_cfg: affine_kwargs["translate"] = tuple(affine_translate_cfg)
        if affine_scale_cfg: affine_kwargs["scale"] = tuple(affine_scale_cfg)
        if affine_shear_cfg: affine_kwargs["shear"] = affine_shear_cfg
        if affine_fill is not None : affine_kwargs["fill"] = affine_fill
        
        if affine_kwargs:
            tf_list.append(RandomAffine(**affine_kwargs))
            logger.info(f"  Added RandomAffine with kwargs: {affine_kwargs}")

        # 3. ColorJitter
        cj_brightness = train_specific_cfg.get("color_jitter_brightness")
        cj_contrast = train_specific_cfg.get("color_jitter_contrast")
        cj_saturation = train_specific_cfg.get("color_jitter_saturation")
        cj_hue = train_specific_cfg.get("color_jitter_hue")

        cj_kwargs = {}
        # torchvision ColorJitter: if value is x, it's a factor chosen from [max(0, 1-x), 1+x]
        # For hue, it's [-x, x] and x should be <= 0.5
        if cj_brightness is not None: cj_kwargs["brightness"] = cj_brightness
        if cj_contrast is not None: cj_kwargs["contrast"] = cj_contrast
        if cj_saturation is not None: cj_kwargs["saturation"] = cj_saturation
        if cj_hue is not None: cj_kwargs["hue"] = cj_hue
        
        if cj_kwargs:
            tf_list.append(ColorJitter(**cj_kwargs))
            logger.info(f"  Added ColorJitter with kwargs: {cj_kwargs}")

        # 4. RandAugment
        rand_aug_n_cfg = train_specific_cfg.get("rand_aug_n")
        rand_aug_m_cfg = train_specific_cfg.get("rand_aug_m")
        if rand_aug_n_cfg is not None and rand_aug_m_cfg is not None:
            tf_list.append(RandAugment(num_ops=rand_aug_n_cfg, magnitude=rand_aug_m_cfg))
            logger.info(f"  Added RandAugment (N={rand_aug_n_cfg}, M={rand_aug_m_cfg})")

    tf_list += [
        ToTensor(),
        Normalize(mean=cfg_cpu_aug["norm_mean"], std=cfg_cpu_aug["norm_std"])
    ]
    return transforms.Compose(tf_list)
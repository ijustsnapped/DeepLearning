# your_project_name/models/factory.py
"""
Model Factory
-------------
Provides functions to create various neural network models for classification
and segmentation tasks, leveraging libraries like timm and torchvision.
Includes support for EfficientNets, DINO ViT, and DeepLabV3.
"""
from __future__ import annotations # Ensures compatibility with older Python versions for type hints

import torch
import torch.nn as nn
# import torch.nn.functional as F # Not directly used, can be removed if not planned for future use
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import timm
import logging

# Module-specific logger
logger = logging.getLogger(__name__)

# ───────────────────────────── Classification Model Builders ────────────────────────────

def _timm(model_name: str, cfg: dict) -> nn.Module:
    """
    Creates a model using the timm library.

    Args:
        model_name: The name of the model as recognized by timm.
        cfg: Configuration dictionary. Expected to contain 'numClasses' and
             optionally 'pretrained' (bool, default True).

    Returns:
        A PyTorch nn.Module instance.
    """
    num_classes = int(cfg["numClasses"]) # Assumes get_model has prepared cfg
    pretrained = cfg.get("pretrained", True)
    logger.info(f"Creating timm model: '{model_name}' with {num_classes} classes, pretrained={pretrained}")
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

def efficientnet_b0(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B0 model."""
    return _timm("efficientnet_b0", cfg)

def efficientnet_b1(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B1 model."""
    return _timm("efficientnet_b1", cfg)

def efficientnet_b2(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B2 model."""
    return _timm("efficientnet_b2", cfg)

def efficientnet_b3(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B3 model."""
    return _timm("efficientnet_b3", cfg)

def efficientnet_b4(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B3 model."""
    return _timm("efficientnet_b4", cfg)

def efficientnet_b5(cfg: dict) -> nn.Module:
    """Creates an EfficientNet-B3 model."""
    return _timm("efficientnet_b5", cfg)

def dino_vit_s14(cfg: dict) -> nn.Module:
    """
    Creates a DINOv2 ViT-S/14 model with a custom classification head.

    Args:
        cfg: Configuration dictionary. Expected to contain 'numClasses' and
             optionally 'force_reload_hub' (bool, default False).

    Returns:
        A DinoClassifier instance wrapping the DINOv2 backbone.
    """
    logger.info("Loading DINOv2 ViT-S/14 from torch.hub (facebookresearch/dinov2)")
    # force_reload can be useful if the torch.hub cache is corrupted or for development.
    dino_backbone = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vits14',
        force_reload=cfg.get("force_reload_hub", False)
    )
    
    num_classes = int(cfg["numClasses"]) # Assumes get_model has prepared cfg
    logger.info(f"Attaching DinoClassifier with {num_classes} classes to DINOv2 ViT-S/14 backbone.")
    return DinoClassifier(dino_backbone, num_classes)

class DinoClassifier(nn.Module):
    """
    A classifier head for DINO (or similar ViT) backbones.
    It applies a linear layer on top of the features extracted by the backbone.
    For DINOv2 ViT-S/14, it expects patch tokens and applies global average pooling.
    """
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # embed_dim for DINOv2 ViT-S/14 is 384.
        # This might need to be made more dynamic or configurable if supporting
        # other DINO models with different embedding dimensions.
        self.embed_dim = getattr(backbone, 'embed_dim', 384)
        if not hasattr(backbone, 'embed_dim'):
             logger.warning(
                 f"DINO backbone for DinoClassifier does not have 'embed_dim' attribute. "
                 f"Assuming default embed_dim={self.embed_dim}. This might be incorrect for some models."
            )
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x) # Expected to return patch embeddings for dinov2_vits14 hub model
        
        # DINOv2 ViT-S/14 hub model's forward(x) returns patch embeddings of shape (Batch, NumPatches, Dim).
        # For classification, a common approach is to average these patch tokens.
        if features.ndim == 3 and features.shape[1] > 0: # Input is [Batch, NumPatches, Dim]
            logger.debug(f"DinoClassifier input features shape: {features.shape}. Applying mean reduction over patch tokens.")
            # Global Average Pooling over patch tokens
            pooled_features = features.mean(dim=1)
        elif features.ndim == 2: # Input is already [Batch, Dim] (e.g., CLS token already extracted)
            logger.debug(f"DinoClassifier input features shape: {features.shape}. Assuming pre-pooled features.")
            pooled_features = features
        else:
            raise ValueError(
                f"Unexpected feature shape from DINO backbone: {features.shape}. "
                "Expected 2D [Batch, Dim] or 3D [Batch, NumPatches, Dim] tensor."
            )
        return self.classifier(pooled_features)

# ──────────────── Segmentation Model Builder ────────────────
class DeeplabV3_R50(nn.Module):
    """
    DeepLabV3 model with a ResNet-50 backbone, with its classifier head
    modified for a custom number of segmentation classes.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        logger.info("Creating DeeplabV3_ResNet50 model.")
        self.net = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
            progress=cfg.get("progress_bar", True) # Allow controlling download progress bar
        )
        
        num_segmentation_classes = int(cfg["numClasses"]) # Assumes get_model has prepared cfg
            
        # The classifier in torchvision's deeplabv3_resnet50 is a DeepLabHead module.
        # We modify its final convolutional layer to output the desired number of classes.
        # Accessing by index (e.g., `[-1]`) assumes a certain structure. This is common
        # but could be fragile if torchvision's internal DeepLabHead structure changes significantly.
        try:
            old_classifier_final_layer = self.net.classifier[-1] # Typically the last nn.Conv2d
            in_channels = old_classifier_final_layer.in_channels
            
            logger.info(
                f"Replacing DeeplabV3_R50 classifier's last Conv2d (in_channels={in_channels}) "
                f"with a new Conv2d for {num_segmentation_classes} classes."
            )
            self.net.classifier[-1] = nn.Conv2d(in_channels, num_segmentation_classes, kernel_size=1)
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Could not modify DeeplabV3_R50 classifier: {e}. Model structure might have changed.")
            raise RuntimeError(f"Failed to adapt DeeplabV3_R50 classifier: {e}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output of torchvision's DeepLabV3 is an OrderedDict.
        # The segmentation map is typically under the key "out".
        output = self.net(x)
        if isinstance(output, dict) and "out" in output:
            return output["out"]
        else:
            logger.warning("DeeplabV3 output was not a dict with 'out' key. Returning raw output. This might be unexpected.")
            return output # Or raise an error if "out" key is strictly expected.

# ──────────────── Model Registry & Factory Function ────────────────
model_map: dict[str, callable] = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "dino_vit_s14": dino_vit_s14,
    "deeplabv3_r50": DeeplabV3_R50, # Note: This is a class, will be instantiated
}

def get_model(cfg: dict) -> nn.Module:
    """
    Factory function to create a neural network model based on the provided configuration.

    The configuration dictionary `cfg` is expected to contain:
    - 'MODEL_TYPE' or 'model_type': (str) The name of the model to create.
      This should match one of the keys in `model_map`.
    - 'numClasses' or 'num_classes': (int) The number of output classes for the model.
    - Other model-specific parameters like 'pretrained', 'force_reload_hub', etc.,
      can also be included and will be passed to the respective model builders.

    Args:
        cfg: The configuration dictionary.

    Returns:
        An nn.Module instance of the specified model.

    Raises:
        ValueError: If essential configuration keys are missing or if the
                    specified model type is unknown.
    """
    model_type_key = None
    if "MODEL_TYPE" in cfg: # Prefer uppercase if available
        model_type_key = cfg["MODEL_TYPE"]
    elif "model_type" in cfg:
        model_type_key = cfg["model_type"]
    
    if model_type_key is None:
        logger.error(f"'MODEL_TYPE' or 'model_type' not found in config. Keys: {list(cfg.keys())}")
        raise ValueError("Config must contain 'MODEL_TYPE' or 'model_type'.")

    # Standardize numClasses/num_classes in the cfg passed to builders
    # This ensures builders can reliably access cfg["numClasses"]
    if "numClasses" not in cfg and "num_classes" in cfg:
        cfg["numClasses"] = cfg["num_classes"]
    elif "num_classes" not in cfg and "numClasses" in cfg:
        cfg["num_classes"] = cfg["numClasses"]
    elif "numClasses" not in cfg and "num_classes" not in cfg: # Neither exists
        logger.error(f"'numClasses' or 'num_classes' not found in config. Keys: {list(cfg.keys())}")
        raise ValueError("Config must contain 'numClasses' or 'num_classes'.")

    model_identifier = str(model_type_key).lower() # Use lowercase for map lookup

    if model_identifier not in model_map:
        logger.error(
            f"Unknown model type '{model_identifier}' (from config value '{model_type_key}'). "
            f"Available models: {list(model_map.keys())}"
        )
        raise ValueError(f"Unknown model type: '{model_identifier}'.")
    
    logger.info(f"get_model: Creating model for type '{model_identifier}' (from config: '{model_type_key}')")
    
    # Call the builder function or instantiate the class from the map
    builder_or_class = model_map[model_identifier]
    return builder_or_class(cfg) # Pass the (potentially modified) cfg

# ──────────────── Sanity Check Utility ────────────────

def sanity_check(model: nn.Module, cfg: dict, model_key_for_print: str):
    """
    Performs a basic sanity check on a model by passing a random tensor through it.
    Checks input and output shapes.

    Args:
        model: The PyTorch model to check.
        cfg: Configuration dictionary, used for 'numClasses' and image size.
        model_key_for_print: A string identifier for the model for logging.
    """
    model.eval() # Set model to evaluation mode

    num_classes = int(cfg["numClasses"]) # Assumes get_model has prepared cfg

    # Determine input image size from config
    img_sz_config = cfg.get("img_size", cfg.get("CROP_SIZE", 224)) # Fallback for image size
    if isinstance(img_sz_config, (list, tuple)) and len(img_sz_config) == 2:
        img_h, img_w = int(img_sz_config[0]), int(img_sz_config[1])
    else:
        img_h = img_w = int(img_sz_config)

    # Create a dummy input tensor
    # Batch size 1, 3 color channels (RGB), height H, width W
    dummy_input = torch.randn(1, 3, img_h, img_w)
    device = next(model.parameters()).device # Get device from model
    dummy_input = dummy_input.to(device)

    logger.info(
        f"Sanity check for '{model_key_for_print}': "
        f"Input shape {dummy_input.shape}, Target num_classes {num_classes}"
    )

    with torch.no_grad(): # Disable gradient calculations for inference
        output = model(dummy_input)

    logger.info(f"Sanity check for '{model_key_for_print}': Output shape {output.shape}")

    # Check output shape based on expected task (classification or segmentation)
    if output.ndim == 2: # Expected for classification: [BatchSize, NumClasses]
        assert output.shape == (1, num_classes), \
            f"'{model_key_for_print}' classifier output shape mismatch. Expected (1, {num_classes}), Got {output.shape}"
        print(f"[OK] {model_key_for_print} (classification) → Output Shape: {output.shape}")
    elif output.ndim == 4: # Expected for segmentation: [BatchSize, NumClasses, Height, Width]
        _, output_channels, _, _ = output.shape
        assert output_channels == num_classes, \
            f"'{model_key_for_print}' segmenter output channel mismatch. Expected {num_classes} channels, Got {output_channels}. Output shape: {output.shape}"
        print(f"[OK] {model_key_for_print} (segmentation) → Output Shape: {output.shape}")
    else:
        raise ValueError(
            f"'{model_key_for_print}' produced an output with an unexpected number of dimensions: {output.ndim}. "
            f"Shape: {output.shape}"
        )

# ──────────────── CLI Smoke Test ────────────────
if __name__ == "__main__":
    # Configure basic logging ONLY if running this script directly.
    # This prevents interference if imported as a module where the main application configures logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - factory.py (smoke test) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("Running CLI smoke test for models from factory.py...")
    
    # Base configuration for testing. Individual tests can override if needed.
    base_test_config = {
        "num_classes": 2,       # Standardized key for num_classes
        # "numClasses" will be added by get_model or can be set here too
        "img_size": 224,        # For sanity_check input tensor
        "CROP_SIZE": 224,       # Alternative for img_size in sanity_check
        "pretrained": False,    # Avoid downloading weights repeatedly during simple test
        "progress_bar": False,  # Disable progress bars for downloads
        "force_reload_hub": False # Avoid issues during CI/repeated tests if cache is fine
    }

    for model_name_in_map in model_map.keys():
        print(f"\n--- Testing Model: {model_name_in_map} ---")
        
        # Create a fresh config for each model test to avoid cross-test contamination
        test_config = base_test_config.copy()
        test_config["MODEL_TYPE"] = model_name_in_map # Key used by get_model

        try:
            # Use get_model to ensure the factory logic (including cfg preparation) is tested
            model_instance = get_model(test_config)
            # Perform sanity check
            sanity_check(model_instance, test_config, model_name_in_map)
        except Exception as e:
            logger.error(f"[FAIL] Testing '{model_name_in_map}' failed: {e}", exc_info=True) # Log traceback
            print(f"[FAIL] {model_name_in_map}: Encountered an error - {e}")
            
    print("\nCLI smoke test completed.")
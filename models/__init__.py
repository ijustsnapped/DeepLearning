from .factory import get_model # Original
from .meta_models import MetadataMLP, CNNWithMetadata # ADDED

__all__ = ["get_model", "MetadataMLP", "CNNWithMetadata"] # MODIFIED
"""DraftScope. Importing this module never loads a model or initializes CUDA."""

from .config import GenerationConfig, ModelSpec
from .metrics import Timing

__version__ = "0.1.0"
__all__ = ["GenerationConfig", "ModelSpec", "Timing", "__version__"]

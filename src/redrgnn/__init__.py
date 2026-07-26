"""ReDRGNN public training implementation."""

from .config import ExperimentConfig, load_config
from .model import EvidenceDualRouteGNN

__all__ = ["EvidenceDualRouteGNN", "ExperimentConfig", "load_config"]
__version__ = "0.1.0"


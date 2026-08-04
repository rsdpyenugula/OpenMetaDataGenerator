"""OpenMetaDataGenerator: context-grounded, lineage-aware automatic description
generation for data-catalog metadata."""
from __future__ import annotations

from .config import Config, GenerationConfig
from .model import Column, GenerationResult, Table

__version__ = "0.1.0"
__all__ = ["Column", "Config", "GenerationConfig", "GenerationResult", "Table"]

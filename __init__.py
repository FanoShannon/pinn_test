"""Minimal physics-first ProductIntegral model for joint k/gamma inference."""

from .core import MinimalKGModel, load_base_checkpoint, save_base_checkpoint

__all__ = ["MinimalKGModel", "load_base_checkpoint", "save_base_checkpoint"]

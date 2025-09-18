"""
Core components for Flit ML.

This module contains the central registry and shared utilities
for the plugin-based ML model architecture.
"""

from .registry import ModelRegistry, BaseModel

__all__ = ["ModelRegistry", "BaseModel"]
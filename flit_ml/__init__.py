"""
Flit ML - Machine Learning components for the flit ecosystem.

This package provides predictive models for financial risk assessment,
specifically focused on BNPL (Buy Now, Pay Later) risk prediction.
"""

__version__ = "0.1.0"
__author__ = "Kev Waithaka"

from flit_ml.core.registry import ModelRegistry

# Global model registry instance
registry = ModelRegistry()

__all__ = ["registry", "__version__"]
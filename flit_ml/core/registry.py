"""
Model Registry - Plugin Architecture for ML Models

This implements a plugin pattern where models can register themselves
dynamically, allowing for extensible ML model management.
"""

from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the Flit ecosystem.

    This enforces a consistent interface that all models must implement,
    enabling the plugin architecture to work seamlessly.
    """

    @abstractmethod
    def fit(self, X, y) -> None:
        """Train the model on provided data."""
        pass

    @abstractmethod
    def predict(self, X) -> Any:
        """Make predictions on new data."""
        pass

    @abstractmethod
    def predict_proba(self, X) -> Any:
        """Return prediction probabilities."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name for registry."""
        pass

    @property
    def domain(self) -> str:
        """Return business domain (e.g., 'bnpl', 'logistics')."""
        return "shared"  # Default to shared domain


class ModelRegistry:
    """
    Central registry for ML models using plugin architecture.

    Benefits:
    1. Dynamic model loading - add new models without code changes
    2. Consistent interface - all models follow BaseModel contract
    3. Easy A/B testing - swap models by name
    4. Version management - multiple versions of same model
    """

    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}
        self._instances: Dict[str, BaseModel] = {}

    def register(self, model_class: Type[BaseModel]) -> None:
        """
        Register a model class in the registry.

        Example:
            @registry.register
            class XGBoostRiskModel(BaseModel):
                name = "risk_xgboost_v1"
                domain = "bnpl"
                # ... implementation
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model {model_class} must inherit from BaseModel")

        instance = model_class()
        model_name = instance.name
        domain = instance.domain

        # Create domain-scoped key: "domain.model_name"
        registry_key = f"{domain}.{model_name}"

        if registry_key in self._models:
            logger.warning(f"Overriding existing model: {registry_key}")

        self._models[registry_key] = model_class
        logger.info(f"Registered model: {registry_key}")

    def get_model(self, name: str, **kwargs) -> BaseModel:
        """
        Get a model instance by name.

        Args:
            name: Model name as registered
            **kwargs: Arguments passed to model constructor
        """
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")

        # Create new instance each time (stateless models)
        model_class = self._models[name]
        return model_class(**kwargs)

    def get_cached_model(self, name: str, **kwargs) -> BaseModel:
        """
        Get cached model instance (useful for trained models).
        """
        cache_key = f"{name}_{hash(str(sorted(kwargs.items())))}"

        if cache_key not in self._instances:
            self._instances[cache_key] = self.get_model(name, **kwargs)

        return self._instances[cache_key]

    def list_models(self, domain: Optional[str] = None) -> list[str]:
        """
        Return list of registered model names.

        Args:
            domain: Filter by business domain (e.g., 'bnpl', 'logistics')
        """
        if domain is None:
            return list(self._models.keys())

        return [key for key in self._models.keys() if key.startswith(f"{domain}.")]

    def list_domains(self) -> list[str]:
        """Return list of available business domains."""
        domains = {key.split('.')[0] for key in self._models.keys()}
        return sorted(list(domains))

    def get_domain_models(self, domain: str) -> Dict[str, Type[BaseModel]]:
        """Get all models for a specific domain."""
        prefix = f"{domain}."
        return {
            key[len(prefix):]: model_class
            for key, model_class in self._models.items()
            if key.startswith(prefix)
        }

    def decorator(self, model_class: Type[BaseModel]) -> Type[BaseModel]:
        """
        Decorator for registering models.

        Usage:
            @registry.decorator
            class MyModel(BaseModel):
                # ... implementation
        """
        self.register(model_class)
        return model_class
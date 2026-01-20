"""Model factory module.

This module provides a factory for creating model instances based on
configuration.
"""

from typing import Any

from ..constants import MODEL_TYPE_SVM, ModelType
from .base import BaseModel
from .svm import SVMModel


class ModelFactory:
    """Factory for creating model instances.

    Supported model types:
    - "SVM": Support Vector Machine classifier
    """

    _MODELS: dict[str, type[BaseModel]] = {
        MODEL_TYPE_SVM: SVMModel,
    }

    @classmethod
    def create(
        cls,
        model_type: ModelType,
        params: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Model type name.
            params: Model-specific parameters.

        Returns:
            A model instance.

        Raises:
            ValueError: If the model type is not supported.
        """
        model_class = cls._MODELS.get(model_type)

        if model_class is None:
            supported = ", ".join(cls._MODELS.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Supported types: {supported}"
            )

        return model_class(params=params)

    @classmethod
    def supported_types(cls) -> list[str]:
        """Get list of supported model types.

        Returns:
            List of supported model type names.
        """
        return list(cls._MODELS.keys())

"""Support Vector Machine model implementation.

This module provides an SVM classifier for malware detection.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import SVC

from ..constants import DEFAULT_SVM_PARAMS, MODEL_TYPE_SVM, SVM_MODEL_FILENAME
from .base import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine model for malware classification.

    This model uses scikit-learn's SVC with RBF kernel by default.
    """

    # Default SVM parameters
    DEFAULT_PARAMS = DEFAULT_SVM_PARAMS

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SVM model.

        Args:
            params: Model parameters. If not provided, uses defaults.
        """
        # Merge with defaults
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(MODEL_TYPE_SVM, merged_params)

    def train(
        self,
        X: np.ndarray | sparse.csr_matrix,
        y: np.ndarray,
    ) -> None:
        """Train the SVM model.

        Args:
            X: Training features (dense or sparse matrix).
            y: Training labels.
        """
        self._logger.info(
            "training_started",
            params=self.params,
            num_samples=X.shape[0],
            num_features=X.shape[1],
        )

        self._model = SVC(**self.params)
        self._model.fit(X, y)

        self._logger.info("training_completed")

    def save(self, model_folder: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            model_folder: Directory to save the model.

        Raises:
            ValueError: If model has not been trained.
        """
        if self._model is None:
            raise ValueError("Model has not been trained")

        model_folder = Path(model_folder)
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / SVM_MODEL_FILENAME
        joblib.dump(self._model, model_path)

        self._logger.info("model_saved", path=str(model_path))

    def load(self, model_folder: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            model_folder: Directory containing the saved model.

        Raises:
            FileNotFoundError: If model file is not found.
        """
        model_folder = Path(model_folder)
        model_path = model_folder / SVM_MODEL_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)

        self._logger.info("model_loaded", path=str(model_path))

"""Base class for machine learning models.

This module defines the abstract base class that all models must implement,
along with utility classes for label encoding.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from ..constants import LABEL_BENIGNWARE, LABEL_MALWARE


class CustomLabelEncoder(LabelEncoder):
    """Custom label encoder ensuring 'Malware' is encoded as 1.

    For binary classification, this encoder ensures:
    - 'Benignware' -> 0
    - 'Malware' -> 1

    This is important for correctly interpreting precision, recall, and F1
    scores where 'Malware' is the positive class.
    """

    def fit(self, y: np.ndarray | list) -> "CustomLabelEncoder":
        """Fit the encoder to the labels.

        Args:
            y: The labels to encode.

        Returns:
            The fitted encoder.
        """
        super().fit(y)

        # Ensure Malware is encoded as 1 for binary classification
        classes = self.classes_
        if len(classes) == 2 and LABEL_MALWARE in classes and LABEL_BENIGNWARE in classes:
            if self.transform([LABEL_MALWARE])[0] != 1:
                self.classes_ = np.array([LABEL_BENIGNWARE, LABEL_MALWARE])

        return self


class BaseModel(ABC):
    """Abstract base class for all machine learning models.

    All models must implement the following methods:
    - train(): Train the model on data
    - predict(): Make predictions
    - save(): Save model to disk
    - load(): Load model from disk
    """

    def __init__(
        self,
        name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the base model.

        Args:
            name: Model name (e.g., "SVM").
            params: Model-specific parameters.
        """
        self.name = name
        self.params = params or {}
        self._logger = structlog.get_logger().bind(component=self.__class__.__name__)
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Get the underlying sklearn model."""
        return self._model

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._model is not None

    @abstractmethod
    def train(
        self,
        X: np.ndarray | sparse.csr_matrix,
        y: np.ndarray,
    ) -> None:
        """Train the model.

        Args:
            X: Training features (dense or sparse matrix).
            y: Training labels.
        """
        pass

    @abstractmethod
    def save(self, model_folder: str | Path) -> None:
        """Save model to disk.

        Args:
            model_folder: Directory to save the model.
        """
        pass

    @abstractmethod
    def load(self, model_folder: str | Path) -> None:
        """Load model from disk.

        Args:
            model_folder: Directory to load the model from.
        """
        pass

    def predict(
        self,
        X: np.ndarray | sparse.csr_matrix | list,
    ) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Feature matrix (dense, sparse, or list).

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model has not been trained.
        """
        if self._model is None:
            raise ValueError(f"Model {self.name} has not been trained")

        # Handle list input
        if isinstance(X, list):
            if len(X) > 0 and sparse.issparse(X[0]):
                X = sparse.vstack(X)
            else:
                X = np.array(X)

        return self._model.predict(X)

    def evaluate(
        self,
        X: np.ndarray | sparse.csr_matrix,
        y_true: np.ndarray,
        is_binary: bool = True,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Feature matrix.
            y_true: True labels.
            is_binary: Whether this is binary classification.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            ValueError: If model has not been trained.
        """
        if self._model is None:
            raise ValueError(f"Model {self.name} has not been trained")

        y_pred = self._model.predict(X)

        if is_binary:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                ),
                "f1": f1_score(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                ),
            }
        else:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }

        return metrics

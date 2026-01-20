"""Base class for vectorizers.

This module defines the abstract base class that all vectorizers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import structlog


class BaseVectorizer(ABC):
    """Abstract base class for all vectorizers.

    Vectorizers convert raw byte sequences extracted from executable files
    into fixed-size feature vectors suitable for machine learning models.

    All vectorizers must implement the following methods:
    - fit(): Learn vocabulary or parameters from training data
    - transform(): Convert byte sequences to feature vectors
    - save(): Save vectorizer state to disk
    - load(): Load vectorizer state from disk
    """

    def __init__(self) -> None:
        """Initialize the base vectorizer."""
        self._logger = structlog.get_logger().bind(component=self.__class__.__name__)
        self._offset = 0
        self._size_features = 0

    @property
    def offset(self) -> int:
        """Get the byte offset for extraction."""
        return self._offset

    @property
    def size_features(self) -> int:
        """Get the feature size (sequence length)."""
        return self._size_features

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        feature_folder: str | Path,
        section_name: str,
        offset: int,
        size_features: int,
        **kwargs: Any,
    ) -> None:
        """Fit the vectorizer on training data.

        This method is used during training to learn vocabulary or other
        parameters needed for vectorization.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing extracted feature files.
            section_name: Section name to extract bytes from.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            **kwargs: Additional vectorizer-specific parameters.
        """
        pass

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        feature_folder: str | Path,
        vectorize_folder: str | Path,
        section_name: str,
        offset: int,
        size_features: int,
        **kwargs: Any,
    ) -> int:
        """Transform byte sequences into feature vectors.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing extracted feature files.
            vectorize_folder: Directory to save vectorized features.
            section_name: Section name to extract bytes from.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            **kwargs: Additional vectorizer-specific parameters.

        Returns:
            Number of successfully processed files.
        """
        pass

    @abstractmethod
    def save(self, model_folder: str | Path) -> None:
        """Save vectorizer state to disk.

        Args:
            model_folder: Directory to save vectorizer state.
        """
        pass

    @abstractmethod
    def load(self, model_folder: str | Path) -> None:
        """Load vectorizer state from disk.

        Args:
            model_folder: Directory to load vectorizer state from.
        """
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_folder: str | Path,
        vectorize_folder: str | Path,
        model_folder: str | Path,
        section_name: str,
        offset: int,
        size_features: int,
        **kwargs: Any,
    ) -> int:
        """Fit the vectorizer and transform the data in one step.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing extracted feature files.
            vectorize_folder: Directory to save vectorized features.
            model_folder: Directory to save vectorizer state.
            section_name: Section name to extract bytes from.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            **kwargs: Additional vectorizer-specific parameters.

        Returns:
            Number of successfully processed files.
        """
        # Store parameters
        self._offset = offset
        self._size_features = size_features

        # Fit, save, and transform
        self.fit(df, feature_folder, section_name, offset, size_features, **kwargs)
        self.save(model_folder)
        return self.transform(
            df,
            feature_folder,
            vectorize_folder,
            section_name,
            offset,
            size_features,
            **kwargs,
        )

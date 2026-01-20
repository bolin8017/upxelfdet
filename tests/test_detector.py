"""Tests for the main detector module.

This module tests the UpxElfDetector class functionality including
initialization and method availability.
"""

import pytest
from maldet import BaseDetector

from upxelfdet import UpxElfDetector, UpxElfDetectorConfig


class TestUpxElfDetectorInit:
    """Tests for UpxElfDetector initialization."""

    def test_init_with_default_config(self) -> None:
        """Test detector initialization with default config."""
        detector = UpxElfDetector()
        assert detector is not None
        assert detector.config is not None

    def test_init_with_custom_config(self) -> None:
        """Test detector initialization with custom config."""
        config = UpxElfDetectorConfig(
            classify=True,
        )
        detector = UpxElfDetector(config)
        assert detector.config.classify is True

    def test_inherits_from_base_detector(self) -> None:
        """Test that UpxElfDetector inherits from maldet.BaseDetector."""
        detector = UpxElfDetector()
        assert isinstance(detector, BaseDetector)

    def test_config_class_attribute(self) -> None:
        """Test that config_class attribute is set correctly."""
        assert UpxElfDetector.config_class == UpxElfDetectorConfig


class TestUpxElfDetectorMethods:
    """Tests for UpxElfDetector required methods."""

    def test_has_train_method(self) -> None:
        """Test that train method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "train")
        assert callable(detector.train)

    def test_has_evaluate_method(self) -> None:
        """Test that evaluate method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "evaluate")
        assert callable(detector.evaluate)

    def test_has_predict_method(self) -> None:
        """Test that predict method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "predict")
        assert callable(detector.predict)

    def test_has_logger(self) -> None:
        """Test that detector has a logger."""
        detector = UpxElfDetector()
        assert hasattr(detector, "logger")
        assert detector.logger is not None


class TestUpxElfDetectorConfig:
    """Tests for UpxElfDetector configuration access."""

    def test_config_vectorize_access(self) -> None:
        """Test accessing vectorize configuration."""
        detector = UpxElfDetector()
        assert hasattr(detector.config, "vectorize")
        assert detector.config.vectorize.method in {"ngram_numeric", "raw_bytes"}

    def test_config_model_access(self) -> None:
        """Test accessing model configuration."""
        detector = UpxElfDetector()
        assert hasattr(detector.config, "model")
        assert detector.config.model.type == "SVM"

    def test_config_feature_access(self) -> None:
        """Test accessing feature configuration."""
        detector = UpxElfDetector()
        assert hasattr(detector.config, "feature")
        assert detector.config.feature.section_name == ".block_1"

    def test_config_data_access(self) -> None:
        """Test accessing data paths from config (inherited from BaseDetectorConfig)."""
        detector = UpxElfDetector()
        assert hasattr(detector.config, "data")
        assert hasattr(detector.config.data, "train")
        assert hasattr(detector.config.data, "test")
        assert hasattr(detector.config.data, "predict")

    def test_config_output_access(self) -> None:
        """Test accessing output paths from config (inherited from BaseDetectorConfig)."""
        detector = UpxElfDetector()
        assert hasattr(detector.config, "output")
        assert hasattr(detector.config.output, "model")
        assert hasattr(detector.config.output, "feature")
        assert hasattr(detector.config.output, "prediction")


class TestUpxElfDetectorPrivateMethods:
    """Tests for UpxElfDetector private helper methods."""

    def test_has_extract_features_method(self) -> None:
        """Test that _extract_features private method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "_extract_features")
        assert callable(detector._extract_features)

    def test_has_vectorize_method(self) -> None:
        """Test that _vectorize private method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "_vectorize")
        assert callable(detector._vectorize)

    def test_has_train_model_method(self) -> None:
        """Test that _train_model private method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "_train_model")
        assert callable(detector._train_model)

    def test_has_load_model_method(self) -> None:
        """Test that _load_model private method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "_load_model")
        assert callable(detector._load_model)

    def test_has_save_model_method(self) -> None:
        """Test that _save_model private method exists."""
        detector = UpxElfDetector()
        assert hasattr(detector, "_save_model")
        assert callable(detector._save_model)

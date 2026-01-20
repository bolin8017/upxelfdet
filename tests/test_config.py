"""Tests for configuration module.

This module tests the configuration classes and validation functionality.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from upxelfdet.config import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    UpxElfDetectorConfig,
    VectorizeConfig,
)
from upxelfdet.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_OFFSET,
    DEFAULT_SECTION_NAME,
    DEFAULT_SIZE_FEATURES,
    DEFAULT_VECTORIZE_METHOD,
)


class TestDataConfig:
    """Tests for DataConfig class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = DataConfig()
        assert config.dataset == Path("./data/dataset")

    def test_custom_dataset(self) -> None:
        """Test that custom dataset path can be set."""
        config = DataConfig(dataset=Path("/custom/dataset"))
        assert config.dataset == Path("/custom/dataset")

    def test_inherits_base_data_config(self) -> None:
        """Test that DataConfig inherits from BaseDataConfig."""
        config = DataConfig()
        assert hasattr(config, "train")
        assert hasattr(config, "test")
        assert hasattr(config, "predict")
        assert hasattr(config, "dataset")


class TestFeatureConfig:
    """Tests for FeatureConfig class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = FeatureConfig()
        assert config.section_name == DEFAULT_SECTION_NAME

    def test_custom_values(self) -> None:
        """Test that custom values can be set."""
        config = FeatureConfig(section_name=".block_0")
        assert config.section_name == ".block_0"


class TestVectorizeConfig:
    """Tests for VectorizeConfig class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = VectorizeConfig()
        assert config.method == DEFAULT_VECTORIZE_METHOD
        assert config.ngram_size == DEFAULT_NGRAM_SIZE
        assert config.encoding == DEFAULT_ENCODING
        assert config.offset == DEFAULT_OFFSET
        assert config.size_features == DEFAULT_SIZE_FEATURES

    def test_valid_ngram_numeric_config(self) -> None:
        """Test valid n-gram numeric configuration."""
        config = VectorizeConfig(
            method="ngram_numeric",
            ngram_size=3,
            encoding="TFIDF",
        )
        assert config.method == "ngram_numeric"
        assert config.ngram_size == 3
        assert config.encoding == "TFIDF"

    def test_valid_raw_bytes_config(self) -> None:
        """Test valid raw bytes configuration."""
        config = VectorizeConfig(
            method="raw_bytes",
            offset=100,
            size_features=512,
        )
        assert config.method == "raw_bytes"
        assert config.offset == 100
        assert config.size_features == 512

    def test_invalid_method(self) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(method="invalid_method")

    def test_invalid_encoding(self) -> None:
        """Test that invalid encoding raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(encoding="INVALID")

    def test_invalid_ngram_size_too_small(self) -> None:
        """Test that ngram_size < 1 raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(ngram_size=0)

    def test_invalid_ngram_size_too_large(self) -> None:
        """Test that ngram_size > 6 raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(ngram_size=7)

    def test_invalid_negative_offset(self) -> None:
        """Test that negative offset raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(offset=-1)

    def test_invalid_zero_size_features(self) -> None:
        """Test that size_features <= 0 raises error."""
        with pytest.raises(ValidationError):
            VectorizeConfig(size_features=0)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = ModelConfig()
        assert config.type == DEFAULT_MODEL_TYPE
        assert "C" in config.params
        assert "gamma" in config.params
        assert "kernel" in config.params

    def test_custom_params(self) -> None:
        """Test that custom params can be set."""
        config = ModelConfig(
            type="SVM",
            params={"C": 10, "gamma": 0.01, "kernel": "linear"},
        )
        assert config.params["C"] == 10
        assert config.params["gamma"] == 0.01
        assert config.params["kernel"] == "linear"

    def test_invalid_model_type(self) -> None:
        """Test that invalid model type raises error."""
        with pytest.raises(ValidationError):
            ModelConfig(type="INVALID")


class TestUpxElfDetectorConfig:
    """Tests for UpxElfDetectorConfig class."""

    def test_default_config(self) -> None:
        """Test that default config can be created."""
        config = UpxElfDetectorConfig()
        assert config.feature is not None
        assert config.vectorize is not None
        assert config.model is not None
        assert config.classify is False

    def test_has_inherited_attributes(self) -> None:
        """Test that inherited attributes from BaseDetectorConfig exist."""
        config = UpxElfDetectorConfig()
        # From BaseDetectorConfig
        assert hasattr(config, "data")
        assert hasattr(config, "output")
        assert hasattr(config, "log")

    def test_custom_classify_mode(self) -> None:
        """Test that classify mode can be set."""
        config = UpxElfDetectorConfig(classify=True)
        assert config.classify is True

    def test_nested_config_access(self) -> None:
        """Test accessing nested configuration values."""
        config = UpxElfDetectorConfig()
        assert config.vectorize.method == DEFAULT_VECTORIZE_METHOD
        assert config.model.type == DEFAULT_MODEL_TYPE
        assert config.feature.section_name == DEFAULT_SECTION_NAME

    def test_config_immutability(self) -> None:
        """Test that nested configs are frozen (immutable)."""
        config = UpxElfDetectorConfig()
        with pytest.raises(ValidationError):
            config.vectorize.method = "raw_bytes"

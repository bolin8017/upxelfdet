"""Configuration system for UPX ELF Detector.

This module provides configuration classes that extend the base maldet
configuration with UPX-specific settings for feature extraction,
vectorization, and model training.

Typical usage:
    from upxelfdet.config import UpxElfDetectorConfig

    # Load with defaults
    config = UpxElfDetectorConfig()

    # Load from file
    config = UpxElfDetectorConfig.from_file(Path("config.json"))

    # Access settings
    print(config.vectorize.method)
    print(config.model.type)
"""

from pathlib import Path
from typing import Any, Self

from maldet.config import (
    BaseDetectorConfig,
    DataConfig as BaseDataConfig,
    OutputConfig as BaseOutputConfig,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .constants import (
    DEFAULT_CLASSIFY,
    DEFAULT_ENCODING,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_OFFSET,
    DEFAULT_OUTPUT_PATH_VECTORIZE,
    DEFAULT_SECTION_NAME,
    DEFAULT_SIZE_FEATURES,
    DEFAULT_SVM_PARAMS,
    DEFAULT_VECTORIZE_METHOD,
    MAX_NGRAM_SIZE,
    MIN_NGRAM_SIZE,
    VALID_ENCODINGS,
    VALID_MODEL_TYPES,
    VALID_VECTORIZE_METHODS,
    EncodingMethod,
    ModelType,
    VectorizationMethod,
)


class DataConfig(BaseDataConfig):
    """Extended data configuration with dataset path.

    Attributes:
        train: Path to training dataset.
        test: Path to test dataset.
        predict: Path to samples for prediction.
        dataset: Path to the dataset directory containing sample files.
    """

    dataset: Path = Path("./data/dataset")


class OutputConfig(BaseOutputConfig):
    """Extended output configuration with vectorize path.

    Attributes:
        model: Path to save/load trained model.
        feature: Path to save extracted features.
        prediction: Path to save prediction results.
        log: Path to save log files.
        vectorize: Path to save vectorized features.
    """

    vectorize: Path = Path(DEFAULT_OUTPUT_PATH_VECTORIZE)


class FeatureConfig(BaseModel):
    """Feature extraction configuration.

    Attributes:
        section_name: Name of the ELF section to extract features from.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    section_name: str = DEFAULT_SECTION_NAME


class VectorizeConfig(BaseModel):
    """Vectorization configuration.

    Attributes:
        method: Vectorization method to use.
        ngram_size: Size of n-grams for ngram_numeric method.
        encoding: Encoding method for n-gram vectorization.
        offset: Byte offset to start reading from.
        size_features: Number of bytes to read for raw_bytes method.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    method: VectorizationMethod = DEFAULT_VECTORIZE_METHOD
    ngram_size: int = DEFAULT_NGRAM_SIZE
    encoding: EncodingMethod = DEFAULT_ENCODING
    offset: int = DEFAULT_OFFSET
    size_features: int = DEFAULT_SIZE_FEATURES

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate vectorization method."""
        if v not in VALID_VECTORIZE_METHODS:
            raise ValueError(
                f"Invalid vectorization method: {v}. "
                f"Must be one of {list(VALID_VECTORIZE_METHODS)}"
            )
        return v

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, v: str) -> str:
        """Validate encoding method."""
        if v not in VALID_ENCODINGS:
            raise ValueError(
                f"Invalid encoding: {v}. "
                f"Must be one of {list(VALID_ENCODINGS)}"
            )
        return v

    @field_validator("ngram_size")
    @classmethod
    def validate_ngram_size(cls, v: int) -> int:
        """Validate n-gram size is within valid range."""
        if not MIN_NGRAM_SIZE <= v <= MAX_NGRAM_SIZE:
            raise ValueError(
                f"ngram_size must be between {MIN_NGRAM_SIZE} and {MAX_NGRAM_SIZE}"
            )
        return v

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: int) -> int:
        """Validate offset is non-negative."""
        if v < 0:
            raise ValueError("offset must be non-negative")
        return v

    @field_validator("size_features")
    @classmethod
    def validate_size_features(cls, v: int) -> int:
        """Validate size_features is positive."""
        if v <= 0:
            raise ValueError("size_features must be positive")
        return v


class ModelConfig(BaseModel):
    """Model configuration.

    Attributes:
        type: Type of model to use for classification.
        params: Model-specific parameters.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    type: ModelType = DEFAULT_MODEL_TYPE
    params: dict[str, Any] = dict(DEFAULT_SVM_PARAMS)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate model type."""
        if v not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model type: {v}. "
                f"Must be one of {list(VALID_MODEL_TYPES)}"
            )
        return v


class UpxElfDetectorConfig(BaseDetectorConfig):
    """Configuration for UPX ELF Detector.

    Extends BaseDetectorConfig with UPX-specific settings for feature
    extraction, vectorization, and model configuration.

    Attributes:
        data: Data path configuration including dataset directory.
        feature: Feature extraction settings.
        vectorize: Vectorization settings.
        model: Model training settings.
        classify: If True, perform multi-class family classification.
            If False, perform binary malware/benignware classification.

    Example:
        >>> config = UpxElfDetectorConfig()
        >>> config.vectorize.method
        'ngram_numeric'
        >>> config.model.type
        'SVM'

        >>> config = UpxElfDetectorConfig.from_file(Path("config.json"))
        >>> detector = UpxElfDetector(config)
    """

    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    feature: FeatureConfig = FeatureConfig()
    vectorize: VectorizeConfig = VectorizeConfig()
    model: ModelConfig = ModelConfig()
    classify: bool = DEFAULT_CLASSIFY

    @model_validator(mode="after")
    def validate_classification_mode(self) -> Self:
        """Validate configuration consistency for classification mode."""
        # Additional cross-field validation can be added here if needed.
        return self

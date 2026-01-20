"""Custom exceptions for UPX ELF Detector.

This module defines detector-specific exceptions that provide meaningful
error messages for various failure scenarios.
"""

from maldet.exceptions import DetectorError


class UpxElfDetectorError(DetectorError):
    """Base exception for UPX ELF Detector errors."""


class FeatureExtractionError(UpxElfDetectorError):
    """Raised when feature extraction fails."""


class VectorizationError(UpxElfDetectorError):
    """Raised when vectorization fails."""


class ModelError(UpxElfDetectorError):
    """Raised when model training or loading fails."""


class PredictionError(UpxElfDetectorError):
    """Raised when prediction fails."""

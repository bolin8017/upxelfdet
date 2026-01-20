"""UPX ELF Detector - A machine learning detector for UPX-packed ELF malware.

This package provides tools to detect malware in UPX-packed ELF executables
using machine learning techniques. It inherits from the maldet base detector.

Example:
    >>> from upxelfdet import UpxElfDetector, UpxElfDetectorConfig
    >>>
    >>> # Training
    >>> config = UpxElfDetectorConfig.from_file(Path("config.json"))
    >>> detector = UpxElfDetector(config)
    >>> model_path = detector.train()
    >>>
    >>> # Evaluation
    >>> metrics = detector.evaluate()
    >>>
    >>> # Prediction
    >>> output_path = detector.predict()
"""

from .config import UpxElfDetectorConfig
from .detector import UpxElfDetector

__version__ = "0.2.0"
__all__ = [
    "UpxElfDetector",
    "UpxElfDetectorConfig",
]

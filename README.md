# upxelfdet

A machine learning-based detector for identifying UPX-packed ELF malware using n-gram feature extraction and Support Vector Machine (SVM) classification.

## Overview

upxelfdet is a Python tool designed for malware analysis and research. It extracts features from ELF binary sections, vectorizes them using n-gram methods, and applies machine learning models to classify whether binaries are packed with UPX or identify malware families.

**Key Features:**

*   **ELF Binary Analysis**: Extracts features from specific sections of ELF files
*   **N-gram Vectorization**: Converts binary features into numeric vectors using configurable n-gram sizes
*   **SVM Classification**: Trains and evaluates Support Vector Machine models
*   **Flexible Configuration**: JSON-based configuration for easy experimentation
*   **CLI Interface**: Command-line tools for training, evaluation, and prediction
*   **Structured Logging**: Comprehensive logging with both human-readable and JSON formats

## Table of Contents

*   [Installation](#installation)
*   [Quick Start](#quick-start)
*   [Usage](#usage)
    *   [Configuration](#configuration)
    *   [Training](#training)
    *   [Evaluation](#evaluation)
    *   [Prediction](#prediction)
*   [Project Structure](#project-structure)
*   [Architecture](#architecture)
*   [Examples](#examples)
*   [Development](#development)
*   [License](#license)
*   [Citation](#citation)

## Installation

### Requirements

*   Python >= 3.12
*   pip or uv (recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/bolin8017/upxelfdet.git
cd upxelfdet

# Install dependencies (using uv - recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Install from PyPI (Future)

```bash
pip install upxelfdet
```

## Quick Start

1.  **Prepare your dataset**: Organize ELF binaries in `input/dataset/` and create CSV files with labels.

2.  **Configure the detector**: Edit `config.json` to set paths and parameters.

3.  **Train the model**:

    ```bash
    upxelfdet train --config config.json
    ```

4.  **Evaluate performance**:

    ```bash
    upxelfdet evaluate --config config.json
    ```

5.  **Make predictions**:

    ```bash
    upxelfdet predict --config config.json
    ```

## Usage

### Configuration

Create or modify `config.json`:

```json
{
  "data": {
    "train": "./input/train.csv",
    "test": "./input/test.csv",
    "predict": "./input/test.csv",
    "dataset": "./input/dataset"
  },
  "output": {
    "feature": "./output/features",
    "model": "./output/model",
    "prediction": "./output/predictions/predictions.csv",
    "log": "./output/logs"
  },
  "feature": {
    "section_name": ".block_1"
  },
  "vectorize": {
    "method": "ngram_numeric",
    "size_features": 256,
    "offset": 0,
    "ngram_size": 2,
    "encoding": "TF"
  },
  "model": {
    "type": "SVM",
    "params": {
      "C": 100,
      "gamma": 0.001,
      "kernel": "rbf"
    }
  },
  "classify": true,
  "seed": 8017
}
```

**Configuration Options:**

*   `data.train`: Path to training CSV file
*   `data.test`: Path to test CSV file
*   `data.dataset`: Directory containing ELF binary files
*   `feature.section_name`: ELF section to extract features from (e.g., `.block_1`)
*   `vectorize.method`: Vectorization method (`ngram_numeric` or `raw_bytes`)
*   `vectorize.ngram_size`: Size of n-grams (typically 2-4)
*   `vectorize.encoding`: Encoding method (`TF` for term frequency)
*   `model.type`: Model type (currently `SVM`)
*   `classify`: If `true`, performs multi-class classification; if `false`, binary classification

### Training

Train a new model using your dataset:

```bash
upxelfdet train --config config.json
```

**What happens during training:**

1.  Loads training data from CSV
2.  Extracts features from ELF binaries in the dataset directory
3.  Vectorizes features using the specified method
4.  Trains an SVM model with configured parameters
5.  Saves the trained model to `output/model/`

**Output:**

*   Trained model files in `output/model/`
*   Feature extraction results in `output/features/`
*   Vectorization results in `output/vectorize/`
*   Training logs in `output/logs/`

### Evaluation

Evaluate model performance on test data:

```bash
upxelfdet evaluate --config config.json
```

**Metrics reported:**

*   Accuracy
*   Precision
*   Recall
*   F1 Score
*   Confusion Matrix
*   Classification Report (for multi-class)

### Prediction

Make predictions on new samples:

```bash
upxelfdet predict --config config.json
```

Predictions are saved to the path specified in `config.output.prediction`.

### Python API

You can also use the detector programmatically:

```python
from upxelfdet import UpxElfDetector
from upxelfdet.config import UpxElfDetectorConfig

# Load configuration
config = UpxElfDetectorConfig.from_file("config.json")

# Initialize detector
detector = UpxElfDetector(config)

# Train model
model_path = detector.train()

# Evaluate model
metrics = detector.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions_path = detector.predict()
```

See [examples/basic_usage.py](examples/basic_usage.py) for a complete example.

## Project Structure

```
upxelfdet/
├── src/
│   └── upxelfdet/
│       ├── __init__.py
│       ├── cli.py                 # Command-line interface
│       ├── config.py              # Configuration management
│       ├── detector.py            # Main detector class
│       ├── constants.py           # Constants and defaults
│       ├── exceptions.py          # Custom exceptions
│       ├── logging.py             # Logging configuration
│       ├── feature/               # Feature extraction
│       │   ├── __init__.py
│       │   └── extractor.py
│       ├── vectorizer/            # Vectorization methods
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── ngram_numeric.py
│       │   ├── raw_bytes.py
│       │   └── factory.py
│       ├── model/                 # ML models
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── svm.py
│       │   └── factory.py
│       └── predictor/             # Prediction logic
│           ├── __init__.py
│           └── predictor.py
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   └── test_detector.py
├── examples/                      # Usage examples
│   └── basic_usage.py
├── data/                          # Example data (see data/README.md)
│   ├── samples/
│   └── README.md
├── input/                         # Input data (not in repo)
│   ├── dataset/                   # ELF binaries (excluded)
│   ├── train.csv                  # Training labels (excluded)
│   └── test.csv                   # Test labels (excluded)
├── output/                        # Output directories
│   ├── features/                  # Extracted features
│   ├── vectorize/                 # Vectorized features
│   ├── model/                     # Trained models
│   ├── predictions/               # Prediction results
│   └── logs/                      # Log files
├── config.json                    # Configuration file
├── pyproject.toml                 # Project metadata and dependencies
├── LICENSE                        # MIT License
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Architecture

### Feature Extraction Pipeline

1.  **Input**: ELF binary files + CSV with labels
2.  **Feature Extraction**: Extract specified section (e.g., `.block_1`) from ELF
3.  **Vectorization**: Convert binary data to numeric vectors using n-grams
4.  **Model Training**: Train SVM classifier on vectorized features
5.  **Evaluation/Prediction**: Apply trained model to new samples

### Component Overview

*   **FeatureExtractor**: Extracts binary sections from ELF files using `upx-elf-parser`
*   **Vectorizer**: Implements different vectorization strategies (n-gram, raw bytes)
*   **Model**: Wraps scikit-learn models with consistent interface
*   **Predictor**: Handles the complete prediction pipeline
*   **UpxElfDetector**: Main orchestrator class that coordinates all components

## Examples

### Example 1: Basic Training and Evaluation

```python
from upxelfdet import UpxElfDetector
from upxelfdet.config import UpxElfDetectorConfig

config = UpxElfDetectorConfig.from_file("config.json")
detector = UpxElfDetector(config)

# Train and evaluate
detector.train()
metrics = detector.evaluate()
```

### Example 2: Custom Configuration

```python
from upxelfdet.config import (
    UpxElfDetectorConfig,
    DataConfig,
    VectorizeConfig,
    ModelConfig,
)

config = UpxElfDetectorConfig(
    data=DataConfig(
        train="./my_train.csv",
        test="./my_test.csv",
        dataset="./my_dataset",
    ),
    vectorize=VectorizeConfig(
        method="ngram_numeric",
        ngram_size=3,
        size_features=512,
    ),
    model=ModelConfig(
        type="SVM",
        params={"C": 10, "kernel": "linear"},
    ),
)

detector = UpxElfDetector(config)
detector.train()
```

See [examples/basic_usage.py](examples/basic_usage.py) for a complete working example.

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/bolin8017/upxelfdet.git
cd upxelfdet

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Quality

This project uses:

*   **ruff**: For linting and formatting
*   **mypy**: For type checking
*   **pytest**: For testing

```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type check
mypy src/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{upxelfdet,
  author = {bolin8017},
  title = {upxelfdet: Machine Learning-Based Detection for UPX-Packed ELF Malware},
  year = {2025},
  url = {https://github.com/bolin8017/upxelfdet}
}
```

## Acknowledgments

This project builds upon:

*   [islab-malware-detector](https://github.com/yourusername/islab-malware-detector): Base malware detection framework
*   [upx-elf-parser](https://github.com/yourusername/upx-elf-parser): ELF parsing utilities
*   [scikit-learn](https://scikit-learn.org/): Machine learning library

## Security Notice

⚠️ **This tool is intended for security research and educational purposes only.**

*   Do not use this tool for malicious activities
*   Handle malware samples with extreme caution
*   Use isolated environments when analyzing malicious binaries
*   Comply with all applicable laws and regulations

## Contact

For questions, issues, or contributions:

*   **Issues**: [GitHub Issues](https://github.com/bolin8017/upxelfdet/issues)
*   **Repository**: [GitHub](https://github.com/bolin8017/upxelfdet)

---

**Note**: This project is under active development. APIs and features may change.

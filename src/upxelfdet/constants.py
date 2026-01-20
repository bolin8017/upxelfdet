"""Constants for UPX ELF Detector.

This module contains all constant values used throughout the detector.
Type aliases for factory pattern are also defined here for centralized
management and easier extensibility.
"""

from typing import Final, Literal

# =============================================================================
# Configuration Version
# =============================================================================
CONFIG_VERSION: Final[str] = "1.0"

# =============================================================================
# Feature Extraction Configuration
# =============================================================================
DEFAULT_SECTION_NAME: Final[str] = ".block_1"

# =============================================================================
# Vectorization Configuration
# =============================================================================
# Vectorization methods
VECTORIZE_METHOD_NGRAM_NUMERIC: Final[str] = "ngram_numeric"
VECTORIZE_METHOD_RAW_BYTES: Final[str] = "raw_bytes"
DEFAULT_VECTORIZE_METHOD: Final[str] = VECTORIZE_METHOD_NGRAM_NUMERIC

VALID_VECTORIZE_METHODS: Final[frozenset[str]] = frozenset({
    VECTORIZE_METHOD_NGRAM_NUMERIC,
    VECTORIZE_METHOD_RAW_BYTES,
})

# Type alias for vectorization method parameter.
VectorizationMethod = Literal["ngram_numeric", "raw_bytes"]

# Encoding methods for n-gram vectorization
ENCODING_BINARY: Final[str] = "Binary"
ENCODING_TF: Final[str] = "TF"
ENCODING_TFIDF: Final[str] = "TFIDF"
DEFAULT_ENCODING: Final[str] = ENCODING_TF

VALID_ENCODINGS: Final[frozenset[str]] = frozenset({
    ENCODING_BINARY,
    ENCODING_TF,
    ENCODING_TFIDF,
})

# Type alias for encoding method parameter.
EncodingMethod = Literal["Binary", "TF", "TFIDF"]

# Vectorization parameters
DEFAULT_SIZE_FEATURES: Final[int] = 256
DEFAULT_OFFSET: Final[int] = 0
DEFAULT_NGRAM_SIZE: Final[int] = 2
MIN_NGRAM_SIZE: Final[int] = 1
MAX_NGRAM_SIZE: Final[int] = 6

# =============================================================================
# Model Configuration
# =============================================================================
# Model types
MODEL_TYPE_SVM: Final[str] = "SVM"
DEFAULT_MODEL_TYPE: Final[str] = MODEL_TYPE_SVM

VALID_MODEL_TYPES: Final[frozenset[str]] = frozenset({
    MODEL_TYPE_SVM,
})

# Type alias for model type parameter.
ModelType = Literal["SVM"]

# Default SVM parameters
DEFAULT_SVM_PARAMS: Final[dict[str, int | float | str]] = {
    "C": 100,
    "gamma": 0.001,
    "kernel": "rbf",
}

# =============================================================================
# Classification Configuration
# =============================================================================
DEFAULT_CLASSIFY: Final[bool] = False

# Label values
LABEL_MALWARE: Final[str] = "Malware"
LABEL_BENIGNWARE: Final[str] = "Benignware"

# Label encoding (Malware=1, Benignware=0)
LABEL_ENCODING_MALWARE: Final[int] = 1
LABEL_ENCODING_BENIGNWARE: Final[int] = 0

# =============================================================================
# Random Seed
# =============================================================================
DEFAULT_SEED: Final[int] = 8017

# =============================================================================
# File Extensions
# =============================================================================
FEATURE_FILE_EXT: Final[str] = ".pkl"
VECTORIZE_SPARSE_EXT: Final[str] = ".npz"
VECTORIZE_DENSE_EXT: Final[str] = ".npy"
MODEL_FILE_EXT: Final[str] = ".joblib"

# =============================================================================
# Model Artifact Filenames
# =============================================================================
SVM_MODEL_FILENAME: Final[str] = "svm_model.joblib"
LABEL_ENCODER_FILENAME: Final[str] = "label_encoder.pkl"
TRAINING_CONFIG_FILENAME: Final[str] = "training_config.json"
VECTORIZER_METADATA_FILENAME: Final[str] = "vectorizer_metadata.json"

# N-gram vectorizer artifacts
NGRAM_STATE_FILENAME: Final[str] = "ngram_numeric_state.pkl"
NGRAM_VOCAB_FILENAME: Final[str] = "ngram_numeric_vocab.pkl"
NGRAM_TFIDF_FILENAME: Final[str] = "ngram_numeric_tfidf.pkl"

# Raw bytes vectorizer artifacts
RAW_BYTES_STATE_FILENAME: Final[str] = "raw_bytes_state.pkl"
RAW_BYTES_METADATA_FILENAME: Final[str] = "raw_bytes_metadata.json"

# =============================================================================
# CSV Column Names
# =============================================================================
COL_FILE_NAME: Final[str] = "file_name"
COL_LABEL: Final[str] = "label"
COL_FAMILY: Final[str] = "family"
COL_PREDICTED_LABEL: Final[str] = "predicted_label"
COL_PREDICTED_PROBA: Final[str] = "predicted_proba"
COL_CORRECT: Final[str] = "correct"
COL_EXTRACTION_SUCCESS: Final[str] = "extraction_success"

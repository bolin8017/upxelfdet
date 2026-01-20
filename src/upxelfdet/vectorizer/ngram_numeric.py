"""N-gram numeric vectorizer implementation.

This module provides a vectorizer that converts byte sequences into n-gram
numeric representations with vocabulary-based sparse matrix storage.
"""

import json
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from ..constants import (
    COL_FILE_NAME,
    DEFAULT_ENCODING,
    DEFAULT_NGRAM_SIZE,
    ENCODING_BINARY,
    ENCODING_TF,
    ENCODING_TFIDF,
    MAX_NGRAM_SIZE,
    MIN_NGRAM_SIZE,
    NGRAM_STATE_FILENAME,
    NGRAM_TFIDF_FILENAME,
    NGRAM_VOCAB_FILENAME,
    VECTORIZE_METHOD_NGRAM_NUMERIC,
    VECTORIZER_METADATA_FILENAME,
)
from .base import BaseVectorizer


def _bytes_to_ngram_numeric(byte_array: bytes, ngram_size: int) -> np.ndarray:
    """Convert byte array to numeric n-grams.

    For example, with 2-gram: [0x64, 0x65, 0x66] -> [0x6465, 0x6566]

    Args:
        byte_array: Raw bytes to convert.
        ngram_size: Size of n-grams (1-6).

    Returns:
        Array of n-gram values as uint64.
    """
    if len(byte_array) < ngram_size:
        return np.array([], dtype=np.uint64)

    ngrams = []
    for i in range(len(byte_array) - ngram_size + 1):
        ngram_value = 0
        for j in range(ngram_size):
            ngram_value = (ngram_value << 8) | byte_array[i + j]
        ngrams.append(ngram_value)

    return np.array(ngrams, dtype=np.uint64)


def _get_byte_sequence(
    filename: str,
    feature_folder: Path,
    section_name: str,
) -> bytes | None:
    """Get byte sequence for a file from the feature pickle.

    Args:
        filename: File name.
        feature_folder: Directory containing feature files.
        section_name: Section name to extract.

    Returns:
        Byte sequence or None if extraction fails.
    """
    feature_path = feature_folder / filename[:2] / f"{filename}.pkl"

    if not feature_path.exists():
        return None

    try:
        with open(feature_path, "rb") as f:
            byte_sequences = pickle.load(f)
    except Exception:
        return None

    for seq in byte_sequences:
        if seq.get("section_name") == section_name:
            return seq.get("bytes")

    return None


def _load_and_extract_ngrams(
    filename: str,
    feature_folder: str,
    section_name: str,
    offset: int,
    size_features: int,
    ngram_size: int,
) -> np.ndarray | None:
    """Load byte sequence and extract numeric n-grams.

    Args:
        filename: File name to process.
        feature_folder: Directory containing feature files.
        section_name: Section to process.
        offset: Byte offset to start extraction.
        size_features: Number of bytes to extract.
        ngram_size: Size of n-grams.

    Returns:
        Array of n-gram values, or None if failed.
    """
    try:
        feature_folder = Path(feature_folder)
        byte_sequence = _get_byte_sequence(filename, feature_folder, section_name)

        if byte_sequence is None:
            return None

        if len(byte_sequence) < offset:
            return None

        end_pos = min(len(byte_sequence), offset + size_features)
        relevant_bytes = byte_sequence[offset:end_pos]

        if len(relevant_bytes) < ngram_size:
            return None

        return _bytes_to_ngram_numeric(relevant_bytes, ngram_size)

    except Exception:
        return None


def _extract_ngrams_for_vocab(
    filename: str,
    feature_folder: str,
    section_name: str,
    offset: int,
    size_features: int,
    ngram_size: int,
) -> set[int]:
    """Extract unique n-grams from a file for vocabulary building.

    This function is designed for parallel execution with ThreadPoolExecutor.

    Args:
        filename: File name to process.
        feature_folder: Directory containing feature files.
        section_name: Section to process.
        offset: Byte offset to start extraction.
        size_features: Number of bytes to extract.
        ngram_size: Size of n-grams.

    Returns:
        Set of unique n-gram values found in the file.
    """
    ngrams = _load_and_extract_ngrams(
        filename, feature_folder, section_name, offset, size_features, ngram_size
    )
    if ngrams is None or len(ngrams) == 0:
        return set()
    return set(ngrams.tolist())


def _extract_unique_ngrams_for_tfidf(
    filename: str,
    feature_folder: str,
    section_name: str,
    offset: int,
    size_features: int,
    ngram_size: int,
) -> frozenset[int]:
    """Extract unique n-grams from a file for document frequency calculation.

    This function is designed for parallel execution with ThreadPoolExecutor.
    Returns a frozenset for efficient document frequency counting.

    Args:
        filename: File name to process.
        feature_folder: Directory containing feature files.
        section_name: Section to process.
        offset: Byte offset to start extraction.
        size_features: Number of bytes to extract.
        ngram_size: Size of n-grams.

    Returns:
        Frozenset of unique n-gram values found in the file.
    """
    ngrams = _load_and_extract_ngrams(
        filename, feature_folder, section_name, offset, size_features, ngram_size
    )
    if ngrams is None or len(ngrams) == 0:
        return frozenset()
    return frozenset(ngrams.tolist())


def _transform_single_file(
    filename: str,
    feature_folder: str,
    vectorize_folder: str,
    section_name: str,
    offset: int,
    size_features: int,
    ngram_size: int,
    encoding: str,
    vocabulary: dict[int, int] | None,
    idf_values: dict[int, float] | None,
    vocabulary_size: int,
) -> tuple[str, bool, str]:
    """Transform a single file to sparse vector representation.

    This function is designed for parallel execution with ProcessPoolExecutor.

    Args:
        filename: File name to process.
        feature_folder: Directory containing feature files.
        vectorize_folder: Directory to save vectorized features.
        section_name: Section to process.
        offset: Byte offset to start extraction.
        size_features: Number of bytes to extract.
        ngram_size: Size of n-grams.
        encoding: Encoding method.
        vocabulary: N-gram to index mapping.
        idf_values: IDF values for TF-IDF encoding.
        vocabulary_size: Size of the vocabulary.

    Returns:
        Tuple of (filename, success, message).
    """
    try:
        ngrams = _load_and_extract_ngrams(
            filename, feature_folder, section_name, offset, size_features, ngram_size
        )

        if ngrams is None or len(ngrams) == 0:
            return filename, False, "Insufficient bytes"

        # Build sparse vector
        feature_space_size = vocabulary_size if vocabulary else (1 << (8 * ngram_size))
        ngram_counts = Counter(ngrams)
        indices = []
        values = []

        for ngram_value, count in ngram_counts.items():
            if vocabulary is None:
                idx = int(ngram_value)
            else:
                idx = vocabulary.get(ngram_value)
                if idx is None:
                    continue

            if encoding == "BINARY":
                value = 1.0
            elif encoding == "TF":
                value = float(count)
            elif encoding == "TFIDF":
                tf = float(count)
                idf = idf_values.get(ngram_value, 1.0) if idf_values else 1.0
                value = tf * idf
            else:
                value = float(count)

            indices.append(idx)
            values.append(value)

        indices_int32 = np.array(indices, dtype=np.int32)
        sparse_vec = sparse.csr_matrix(
            (values, ([0] * len(indices), indices_int32)),
            shape=(1, feature_space_size),
            dtype=np.float32,
        )

        # Normalize for TF-IDF
        if encoding == "TFIDF":
            norm = sparse.linalg.norm(sparse_vec)
            if norm > 0:
                sparse_vec = sparse_vec / norm

        # Save sparse vector
        vectorize_folder_path = Path(vectorize_folder)
        output_dir = vectorize_folder_path / filename[:2]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.npz"
        sparse.save_npz(output_path, sparse_vec)

        return filename, True, "Success"

    except Exception as e:
        return filename, False, str(e)


class NGramNumericVectorizer(BaseVectorizer):
    """Vectorizer using numeric byte n-grams with vocabulary-based sparse storage.

    This vectorizer extracts n-grams as numeric values (e.g., 2-gram [0x64, 0x65]
    -> 0x6465) and represents them as sparse vectors.

    Supported encoding methods:
    - Binary: 1 if n-gram present, 0 otherwise
    - TF: Term frequency (raw count)
    - TFIDF: TF-IDF (term frequency * inverse document frequency)

    Attributes:
        ngram_size: Size of n-grams (1-6).
        encoding: Encoding method ("Binary", "TF", or "TFIDF").
        vocabulary: Mapping from n-gram values to feature indices.
        vocabulary_size: Number of unique n-grams in vocabulary.
    """

    def __init__(self) -> None:
        """Initialize the n-gram numeric vectorizer."""
        super().__init__()
        self._n_workers = max(1, os.cpu_count() or 1)

        self.ngram_size: int = DEFAULT_NGRAM_SIZE
        self.encoding: str = DEFAULT_ENCODING
        self.vocabulary: dict[int, int] | None = None
        self.vocabulary_size: int = 0
        self.idf_values: dict[int, float] | None = None
        self._n_documents: int = 0

    def _get_feature_space_size(self) -> int:
        """Get the feature space size based on vocabulary."""
        if self.vocabulary is not None:
            return self.vocabulary_size
        return 1 << (8 * self.ngram_size)

    def fit(
        self,
        df: pd.DataFrame,
        feature_folder: str | Path,
        section_name: str,
        offset: int,
        size_features: int,
        ngram_size: int = 2,
        encoding: str = "TF",
        **kwargs: Any,
    ) -> None:
        """Fit the vectorizer on training data.

        Builds vocabulary from training data and computes IDF if needed.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing feature files.
            section_name: Section to process.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            ngram_size: Size of n-grams (1-6).
            encoding: Encoding method ("Binary", "TF", or "TFIDF").
            **kwargs: Additional parameters (ignored).

        Raises:
            ValueError: If ngram_size or encoding is invalid.
        """
        self.ngram_size = ngram_size
        self.encoding = encoding.upper()
        self._n_documents = len(df)

        if not MIN_NGRAM_SIZE <= self.ngram_size <= MAX_NGRAM_SIZE:
            raise ValueError(
                f"ngram_size must be between {MIN_NGRAM_SIZE} and {MAX_NGRAM_SIZE}, "
                f"got {self.ngram_size}"
            )

        valid_encodings = (ENCODING_TF, ENCODING_TFIDF, ENCODING_BINARY.upper())
        if self.encoding not in valid_encodings:
            raise ValueError(
                f"Invalid encoding: {self.encoding}. "
                f"Must be '{ENCODING_TF}', '{ENCODING_TFIDF}', or '{ENCODING_BINARY}'"
            )

        self._logger.info(
            "fitting_vectorizer",
            ngram_size=self.ngram_size,
            encoding=self.encoding,
        )

        feature_folder = Path(feature_folder)

        # Build vocabulary
        self._build_vocabulary(
            df, feature_folder, section_name, offset, size_features
        )

        # Compute IDF values for TF-IDF encoding
        if self.encoding == ENCODING_TFIDF:
            self._fit_tfidf(df, feature_folder, section_name, offset, size_features)

    def _build_vocabulary(
        self,
        df: pd.DataFrame,
        feature_folder: Path,
        section_name: str,
        offset: int,
        size_features: int,
    ) -> None:
        """Build vocabulary from training data using parallel processing.

        Uses ThreadPoolExecutor for I/O-bound file reading operations.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing feature files.
            section_name: Section to process.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
        """
        self._logger.info(
            "building_vocabulary",
            num_threads=self._n_workers,
        )

        all_ngrams: set[int] = set()
        skipped = 0
        filenames = df[COL_FILE_NAME].tolist()

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(
                    _extract_ngrams_for_vocab,
                    filename,
                    str(feature_folder),
                    section_name,
                    offset,
                    size_features,
                    self.ngram_size,
                ): filename
                for filename in filenames
            }

            with tqdm(total=len(filenames), desc="Building vocabulary") as pbar:
                for future in as_completed(futures):
                    ngram_set = future.result()
                    if len(ngram_set) == 0:
                        skipped += 1
                    else:
                        all_ngrams.update(ngram_set)
                    pbar.update(1)

        # Create sorted vocabulary
        sorted_ngrams = sorted(all_ngrams)
        self.vocabulary = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}
        self.vocabulary_size = len(self.vocabulary)

        max_possible = 1 << (8 * self.ngram_size)
        coverage = 100.0 * self.vocabulary_size / max_possible
        self._logger.info(
            "vocabulary_built",
            unique_ngrams=self.vocabulary_size,
            coverage_percent=round(coverage, 6),
            skipped_files=skipped,
        )

    def _fit_tfidf(
        self,
        df: pd.DataFrame,
        feature_folder: Path,
        section_name: str,
        offset: int,
        size_features: int,
    ) -> None:
        """Compute IDF values for TF-IDF encoding using parallel processing.

        Uses ThreadPoolExecutor for I/O-bound file reading operations.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing feature files.
            section_name: Section to process.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
        """
        self._logger.info(
            "computing_tfidf",
            num_threads=self._n_workers,
        )

        document_frequency: Counter[int] = Counter()
        skipped = 0
        filenames = df[COL_FILE_NAME].tolist()

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(
                    _extract_unique_ngrams_for_tfidf,
                    filename,
                    str(feature_folder),
                    section_name,
                    offset,
                    size_features,
                    self.ngram_size,
                ): filename
                for filename in filenames
            }

            with tqdm(total=len(filenames), desc="Computing TF-IDF") as pbar:
                for future in as_completed(futures):
                    unique_ngrams = future.result()
                    if len(unique_ngrams) == 0:
                        skipped += 1
                    else:
                        for ngram in unique_ngrams:
                            document_frequency[ngram] += 1
                    pbar.update(1)

        # Compute IDF: log(n_documents / df) + 1
        self.idf_values = {}
        for ngram, df_count in document_frequency.items():
            idf = np.log((self._n_documents + 1) / (df_count + 1)) + 1
            self.idf_values[ngram] = idf

        idf_vals = list(self.idf_values.values())
        self._logger.info(
            "tfidf_computed",
            min_idf=round(min(idf_vals), 4),
            max_idf=round(max(idf_vals), 4),
        )

    def _ngrams_to_sparse_vector(
        self,
        ngrams: np.ndarray,
        normalize: bool = False,
    ) -> sparse.csr_matrix:
        """Convert n-gram array to sparse feature vector.

        Args:
            ngrams: Array of n-gram values.
            normalize: Whether to normalize to unit length.

        Returns:
            Sparse CSR matrix (1 x vocabulary_size).
        """
        feature_space_size = self._get_feature_space_size()

        if len(ngrams) == 0:
            return sparse.csr_matrix((1, feature_space_size), dtype=np.float32)

        ngram_counts = Counter(ngrams)
        indices = []
        values = []

        for ngram_value, count in ngram_counts.items():
            if self.vocabulary is None:
                idx = int(ngram_value)
            else:
                idx = self.vocabulary.get(ngram_value)
                if idx is None:
                    continue  # Skip unseen n-grams

            if self.encoding == ENCODING_BINARY.upper():
                value = 1.0
            elif self.encoding == ENCODING_TF:
                value = float(count)
            elif self.encoding == ENCODING_TFIDF:
                tf = float(count)
                idf = self.idf_values.get(ngram_value, 1.0) if self.idf_values else 1.0
                value = tf * idf
            else:
                value = float(count)

            indices.append(idx)
            values.append(value)

        indices_int32 = np.array(indices, dtype=np.int32)
        sparse_vec = sparse.csr_matrix(
            (values, ([0] * len(indices), indices_int32)),
            shape=(1, feature_space_size),
            dtype=np.float32,
        )

        if normalize and self.encoding == ENCODING_TFIDF:
            norm = sparse.linalg.norm(sparse_vec)
            if norm > 0:
                sparse_vec = sparse_vec / norm

        return sparse_vec

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
        """Transform byte sequences into n-gram sparse vectors using parallel processing.

        Uses ProcessPoolExecutor for CPU-bound vectorization operations.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing feature files.
            vectorize_folder: Directory to save vectorized features.
            section_name: Section to process.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            **kwargs: Additional parameters (ignored).

        Returns:
            Number of successfully processed files.
        """
        feature_folder = Path(feature_folder)
        vectorize_folder = Path(vectorize_folder)
        vectorize_folder.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "transforming",
            num_files=len(df),
            num_processes=self._n_workers,
        )

        filenames = df[COL_FILE_NAME].tolist()
        successful = 0

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(
                    _transform_single_file,
                    filename,
                    str(feature_folder),
                    str(vectorize_folder),
                    section_name,
                    offset,
                    size_features,
                    self.ngram_size,
                    self.encoding,
                    self.vocabulary,
                    self.idf_values,
                    self.vocabulary_size,
                ): filename
                for filename in filenames
            }

            with tqdm(total=len(filenames), desc="Transforming") as pbar:
                for future in as_completed(futures):
                    filename, success, message = future.result()
                    if success:
                        successful += 1
                    else:
                        self._logger.warning(
                            "transform_skipped",
                            file=filename,
                            reason=message,
                        )
                    pbar.update(1)

        self._logger.info(
            "transform_completed",
            success=successful,
            total=len(df),
        )

        return successful

    def save(self, model_folder: str | Path) -> None:
        """Save vectorizer state to disk.

        Args:
            model_folder: Directory to save vectorizer state.
        """
        model_folder = Path(model_folder)
        model_folder.mkdir(parents=True, exist_ok=True)

        # Save state
        state = {
            "ngram_size": self.ngram_size,
            "encoding": self.encoding,
            "n_documents": self._n_documents,
            "vocabulary_size": self.vocabulary_size,
            "feature_space_size": self._get_feature_space_size(),
        }

        state_path = model_folder / NGRAM_STATE_FILENAME
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        # Save vocabulary
        if self.vocabulary is not None:
            vocab_path = model_folder / NGRAM_VOCAB_FILENAME
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vocabulary, f)

        # Save TF-IDF weights
        if self.idf_values is not None:
            tfidf_path = model_folder / NGRAM_TFIDF_FILENAME
            with open(tfidf_path, "wb") as f:
                pickle.dump(self.idf_values, f)

        # Save metadata JSON
        metadata = {
            "vectorization_method": VECTORIZE_METHOD_NGRAM_NUMERIC,
            "ngram_size": self.ngram_size,
            "encoding": self.encoding,
            "vocabulary_size": self.vocabulary_size,
            "feature_space_size": self._get_feature_space_size(),
            "n_documents": self._n_documents,
            "offset": self._offset,
            "size_features": self._size_features,
            "created_date": datetime.now().isoformat(),
        }

        metadata_path = model_folder / VECTORIZER_METADATA_FILENAME
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._logger.info("vectorizer_saved", path=str(model_folder))

    def load(self, model_folder: str | Path) -> None:
        """Load vectorizer state from disk.

        Args:
            model_folder: Directory to load vectorizer state from.

        Raises:
            FileNotFoundError: If state file is not found.
        """
        model_folder = Path(model_folder)
        state_path = model_folder / NGRAM_STATE_FILENAME

        if not state_path.exists():
            raise FileNotFoundError(f"Vectorizer state not found: {state_path}")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.ngram_size = state["ngram_size"]
        self.encoding = state["encoding"]
        self._n_documents = state.get("n_documents", 0)
        self.vocabulary_size = state.get("vocabulary_size", 0)

        # Load vocabulary
        vocab_path = model_folder / NGRAM_VOCAB_FILENAME
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                self.vocabulary = pickle.load(f)
        else:
            self.vocabulary = None

        # Load TF-IDF weights
        tfidf_path = model_folder / NGRAM_TFIDF_FILENAME
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                self.idf_values = pickle.load(f)
        else:
            self.idf_values = None

        self._logger.info(
            "vectorizer_loaded",
            ngram_size=self.ngram_size,
            encoding=self.encoding,
        )

"""Raw bytes vectorizer implementation.

This module provides a vectorizer that extracts fixed-length raw byte sequences
from executable files.
"""

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..constants import (
    COL_FILE_NAME,
    VECTORIZE_METHOD_RAW_BYTES,
    VECTORIZER_METADATA_FILENAME,
)
from .base import BaseVectorizer


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

    # Find the requested section
    for seq in byte_sequences:
        if seq.get("section_name") == section_name:
            return seq.get("bytes")

    return None


def _extract_and_save_raw_bytes(
    filename: str,
    feature_folder: str,
    vectorize_folder: str,
    section_name: str,
    offset: int,
    size_features: int,
) -> tuple[bool, str]:
    """Extract a fixed-length byte sequence from a file and save it.

    Args:
        filename: File name to process.
        feature_folder: Directory containing feature files.
        vectorize_folder: Directory to save vectorized features.
        section_name: Section to extract.
        offset: Byte offset to start extraction.
        size_features: Number of bytes to extract.

    Returns:
        Tuple of (success_flag, message).
    """
    try:
        feature_folder = Path(feature_folder)
        vectorize_folder = Path(vectorize_folder)

        # Check if output vector already exists
        output_path = vectorize_folder / filename[:2] / f"{filename}.npy"
        if output_path.exists():
            return True, "Already exists"

        # Get byte sequence
        byte_sequence = _get_byte_sequence(filename, feature_folder, section_name)

        if byte_sequence is None:
            return False, "Failed to extract byte sequence"

        # Create feature vector from raw bytes
        if len(byte_sequence) >= offset + size_features:
            vector = np.frombuffer(
                byte_sequence[offset : offset + size_features], dtype=np.uint8
            )
        else:
            # Pad with zeros if not enough bytes
            vector = np.zeros(size_features, dtype=np.uint8)
            available_len = max(0, len(byte_sequence) - offset)
            if available_len > 0:
                vector[:available_len] = np.frombuffer(
                    byte_sequence[offset:], dtype=np.uint8
                )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save vector
        np.save(output_path, vector)
        return True, "Success"

    except Exception as e:
        return False, f"Error: {e}"


class RawBytesVectorizer(BaseVectorizer):
    """Vectorizer that extracts fixed-length raw byte sequences.

    This vectorizer simply takes a contiguous sequence of bytes from the
    specified section and converts them to a numpy array.
    """

    def __init__(self) -> None:
        """Initialize the raw bytes vectorizer."""
        super().__init__()
        self._n_workers = max(1, os.cpu_count() or 1)

    def fit(
        self,
        df: pd.DataFrame,
        feature_folder: str | Path,
        section_name: str,
        offset: int,
        size_features: int,
        **kwargs: Any,
    ) -> None:
        """Raw bytes vectorizer doesn't require fitting.

        Args:
            df: Input DataFrame with file information.
            feature_folder: Directory containing feature files.
            section_name: Section to process.
            offset: Byte offset to start extraction.
            size_features: Number of bytes to extract.
            **kwargs: Additional parameters (ignored).
        """
        self._logger.info("fit_not_required")

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
        """Transform byte sequences into fixed-length raw byte vectors.

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
            "extracting_raw_bytes",
            num_files=len(df),
            size_features=size_features,
        )

        # Prepare arguments for parallel processing
        args_list = [
            (
                row[COL_FILE_NAME],
                str(feature_folder),
                str(vectorize_folder),
                section_name,
                offset,
                size_features,
            )
            for _, row in df.iterrows()
        ]

        successful = 0
        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(_extract_and_save_raw_bytes, *args): args[0]
                for args in args_list
            }

            with tqdm(total=len(args_list), desc="Extracting raw bytes") as pbar:
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        success, msg = future.result()
                        if success:
                            successful += 1
                        else:
                            self._logger.warning(
                                "extraction_failed",
                                file=filename,
                                reason=msg,
                            )
                    except Exception as e:
                        self._logger.error(
                            "extraction_error",
                            file=filename,
                            error=str(e),
                        )
                    pbar.update(1)

        self._logger.info(
            "extraction_completed",
            success=successful,
            total=len(df),
        )

        return successful

    def save(self, model_folder: str | Path) -> None:
        """Save vectorizer metadata.

        Args:
            model_folder: Directory to save vectorizer state.
        """
        model_folder = Path(model_folder)
        model_folder.mkdir(parents=True, exist_ok=True)

        metadata = {
            "vectorization_method": VECTORIZE_METHOD_RAW_BYTES,
            "offset": self._offset,
            "size_features": self._size_features,
            "created_date": datetime.now().isoformat(),
        }

        metadata_path = model_folder / VECTORIZER_METADATA_FILENAME
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._logger.info("metadata_saved", path=str(metadata_path))

    def load(self, model_folder: str | Path) -> None:
        """Load vectorizer metadata.

        Args:
            model_folder: Directory to load vectorizer state from.
        """
        model_folder = Path(model_folder)
        metadata_path = model_folder / VECTORIZER_METADATA_FILENAME

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self._offset = metadata.get("offset", 0)
            self._size_features = metadata.get("size_features", 0)

            self._logger.info("metadata_loaded", path=str(metadata_path))
        else:
            self._logger.warning("metadata_not_found")

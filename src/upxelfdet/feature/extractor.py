"""Feature extraction module using upx-elf-parser.

This module extracts features from UPX-packed ELF files by parsing their
internal structure and saving the extracted byte sequences.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from tqdm import tqdm
from upx_elf_parser import (
    InvalidElfError,
    InvalidUpxStructureError,
    NoLoadSegmentError,
    UnsupportedElfTypeError,
    UpxElfInfo,
    parse_upx_elf,
)

from ..constants import COL_EXTRACTION_SUCCESS, COL_FILE_NAME


@dataclass
class ByteSequence:
    """Data class for storing byte sequence information.

    Attributes:
        addr: Virtual address of the byte sequence.
        bytes: Raw bytes of the sequence.
        section_name: Name of the section this sequence belongs to.
    """

    addr: int
    bytes: bytes
    section_name: str


def _extract_single_file(
    input_path: str,
    output_path: str,
) -> tuple[bool, str]:
    """Extract features from a single ELF file.

    This function is designed to be called in a separate process.

    Args:
        input_path: Path to the input ELF file.
        output_path: Path to save the extracted features.

    Returns:
        A tuple of (success, message).
    """
    try:
        # Parse the UPX-packed ELF file
        result: UpxElfInfo = parse_upx_elf(input_path)

        # Convert to list of byte sequences
        byte_sequences = _convert_to_byte_sequences(result)

        if not byte_sequences:
            return False, "No byte sequences extracted"

        # Save as pickle file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(byte_sequences, f)

        return True, f"Extracted {len(byte_sequences)} sections"

    except FileNotFoundError:
        return False, "File not found"
    except InvalidElfError:
        return False, "Invalid ELF file"
    except UnsupportedElfTypeError:
        return False, "Unsupported ELF type (only ET_EXEC supported)"
    except InvalidUpxStructureError:
        return False, "Invalid UPX structure"
    except NoLoadSegmentError:
        return False, "No suitable PT_LOAD segment found"
    except Exception as e:
        return False, f"Error: {e}"


def _convert_to_byte_sequences(result: UpxElfInfo) -> list[dict[str, Any]]:
    """Convert UpxElfInfo to list of byte sequence dictionaries.

    Args:
        result: Parsed UPX ELF information.

    Returns:
        List of byte sequence dictionaries.
    """
    sequences = []

    # ELF Header
    sequences.append({
        "addr": result.elf_header.address,
        "bytes": result.elf_header.data,
        "section_name": result.elf_header.name,
    })

    # Program Headers
    sequences.append({
        "addr": result.program_headers.address,
        "bytes": result.program_headers.data,
        "section_name": result.program_headers.name,
    })

    # L_info
    sequences.append({
        "addr": result.l_info.section.address,
        "bytes": result.l_info.section.data,
        "section_name": result.l_info.section.name,
    })

    # P_info
    sequences.append({
        "addr": result.p_info.section.address,
        "bytes": result.p_info.section.data,
        "section_name": result.p_info.section.name,
    })

    # Compressed blocks
    for i, block in enumerate(result.compressed_blocks):
        # Block info header
        sequences.append({
            "addr": block.header_section.address,
            "bytes": block.header_section.data,
            "section_name": f".b_info_{i}",
            "block_info": {
                "uncompressed_size": block.block_info.uncompressed_size,
                "compressed_size": block.block_info.compressed_size,
                "method": block.block_info.method,
                "filter_id": block.block_info.filter_id,
            },
        })

        # Compressed data
        sequences.append({
            "addr": block.data_section.address,
            "bytes": block.data_section.data,
            "section_name": f".block_{i}",
            "block_info": {
                "uncompressed_size": block.block_info.uncompressed_size,
                "compressed_size": block.block_info.compressed_size,
                "method": block.block_info.method,
                "filter_id": block.block_info.filter_id,
            },
        })

    # Loader
    if result.loader:
        sequences.append({
            "addr": result.loader.address,
            "bytes": result.loader.data,
            "section_name": result.loader.name,
        })

    # Displacement (if present)
    if result.displacement:
        sequences.append({
            "addr": result.displacement.address,
            "bytes": result.displacement.data,
            "section_name": result.displacement.name,
        })

    return sequences


class FeatureExtractor:
    """Feature extractor for UPX-packed ELF files.

    This class handles the extraction of byte sequences from UPX-packed
    ELF files using the upx-elf-parser library.
    """

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self._logger = structlog.get_logger().bind(component="FeatureExtractor")

    def extract(
        self,
        df: pd.DataFrame,
        dataset_folder: str | Path,
        feature_folder: str | Path,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Extract features from all files in the DataFrame.

        Args:
            df: DataFrame containing file information. Must have columns:
                - file_name: Name of the file.
                - label: "Malware" or "Benignware" (used for logging only).
            dataset_folder: Folder containing the ELF files.
                Files are expected at: dataset_folder/file_name[:2]/file_name
            feature_folder: Folder to save extracted features.
            max_workers: Maximum number of parallel workers.
                If None, uses os.cpu_count().

        Returns:
            DataFrame with extraction results (adds 'extraction_success' column).
        """
        dataset_folder = Path(dataset_folder)
        feature_folder = Path(feature_folder)
        feature_folder.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for parallel processing
        args_list = []
        for _, row in df.iterrows():
            filename = row[COL_FILE_NAME]
            prefix = filename[:2]

            input_path = dataset_folder / prefix / filename
            output_path = feature_folder / prefix / f"{filename}.pkl"

            # Skip if output already exists
            if output_path.exists():
                self._logger.debug(
                    "feature_file_exists",
                    file=filename,
                    path=str(output_path),
                )
                continue

            # Create prefix directory
            (feature_folder / prefix).mkdir(exist_ok=True)

            args_list.append((str(input_path), str(output_path)))

        if not args_list:
            self._logger.info("no_files_to_process")
            df[COL_EXTRACTION_SUCCESS] = True
            return df

        # Process files in parallel
        results = {}
        if max_workers is None:
            max_workers = os.cpu_count()

        self._logger.info(
            "extraction_started",
            num_files=len(args_list),
            max_workers=max_workers,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_extract_single_file, *args): args[0]
                for args in args_list
            }

            with tqdm(total=len(args_list), desc="Extracting features") as pbar:
                for future in as_completed(futures):
                    input_path = futures[future]
                    filename = os.path.basename(input_path)

                    try:
                        success, message = future.result()
                        results[filename] = success

                        if success:
                            self._logger.debug(
                                "extraction_success",
                                file=filename,
                                message=message,
                            )
                        else:
                            self._logger.warning(
                                "extraction_failed",
                                file=filename,
                                reason=message,
                            )

                    except Exception as e:
                        results[filename] = False
                        self._logger.error(
                            "extraction_error",
                            file=filename,
                            error=str(e),
                        )

                    pbar.update(1)

        # Add extraction results to DataFrame
        df = df.copy()
        df[COL_EXTRACTION_SUCCESS] = df[COL_FILE_NAME].map(
            lambda x: results.get(x, True)  # True if already existed
        )

        success_count = df[COL_EXTRACTION_SUCCESS].sum()
        self._logger.info(
            "extraction_completed",
            success=int(success_count),
            total=len(df),
        )

        return df

    @staticmethod
    def load_features(feature_path: str | Path) -> list[dict[str, Any]]:
        """Load extracted features from a pickle file.

        Args:
            feature_path: Path to the feature pickle file.

        Returns:
            List of byte sequence dictionaries.

        Raises:
            FileNotFoundError: If the feature file does not exist.
        """
        with open(feature_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_section_bytes(
        features: list[dict[str, Any]],
        section_name: str,
    ) -> bytes | None:
        """Get bytes from a specific section.

        Args:
            features: List of byte sequence dictionaries.
            section_name: Name of the section to extract.

        Returns:
            Bytes from the section, or None if not found.
        """
        for seq in features:
            if seq.get("section_name") == section_name:
                return seq.get("bytes")
        return None

"""Predictor module for making predictions with trained models.

This module handles the prediction pipeline including loading vectors,
making predictions, and computing evaluation metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from ..constants import (
    COL_CORRECT,
    COL_FAMILY,
    COL_FILE_NAME,
    COL_LABEL,
    COL_PREDICTED_LABEL,
    LABEL_MALWARE,
)
from ..model.base import BaseModel, CustomLabelEncoder


class Predictor:
    """Predictor for making predictions with trained models.

    This class handles:
    - Loading vectorized features
    - Making predictions using trained models
    - Computing evaluation metrics
    - Saving prediction results
    """

    def __init__(self) -> None:
        """Initialize the predictor."""
        self._logger = structlog.get_logger().bind(component="Predictor")

    def predict(
        self,
        df: pd.DataFrame,
        vectorize_folder: str | Path,
        model: BaseModel,
        label_encoder: CustomLabelEncoder,
        is_binary: bool = True,
    ) -> pd.DataFrame:
        """Make predictions for files in the DataFrame.

        Args:
            df: DataFrame containing file information with columns:
                - file_name: Name of the file
                - label: True label (optional, for evaluation)
                - family: Family name (optional, for multi-class)
            vectorize_folder: Folder containing vectorized features.
            model: Trained model to use for prediction.
            label_encoder: Label encoder for decoding predictions.
            is_binary: Whether this is binary classification.

        Returns:
            DataFrame with predictions added (columns: predicted_label,
            predicted_encoded, correct).
        """
        vectorize_folder = Path(vectorize_folder)

        self._logger.info("prediction_started", num_files=len(df))

        # Load features and make predictions
        predictions = []
        skipped = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            filename = row[COL_FILE_NAME]

            # Try loading feature vector
            vector = self._load_vector(vectorize_folder, filename)

            if vector is None:
                skipped.append(filename)
                predictions.append({
                    COL_FILE_NAME: filename,
                    COL_PREDICTED_LABEL: None,
                    "predicted_encoded": None,
                })
                continue

            # Make prediction
            pred_encoded = model.predict(vector)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            predictions.append({
                COL_FILE_NAME: filename,
                COL_PREDICTED_LABEL: pred_label,
                "predicted_encoded": pred_encoded,
            })

        # Create result DataFrame
        result_df = df.copy()
        pred_df = pd.DataFrame(predictions)
        result_df = result_df.merge(pred_df, on=COL_FILE_NAME, how="left")

        # Add correctness column
        # For binary classification, compare with 'label'
        # For multi-class (family classification), compare with 'family'
        if is_binary and COL_LABEL in result_df.columns:
            result_df[COL_CORRECT] = result_df[COL_LABEL] == result_df[COL_PREDICTED_LABEL]
        elif not is_binary and COL_FAMILY in result_df.columns:
            result_df[COL_CORRECT] = result_df[COL_FAMILY] == result_df[COL_PREDICTED_LABEL]

        success_count = len(df) - len(skipped)
        self._logger.info(
            "prediction_completed",
            success=success_count,
            total=len(df),
            skipped=len(skipped),
        )

        return result_df

    def _load_vector(
        self,
        vectorize_folder: Path,
        filename: str,
    ) -> np.ndarray | sparse.csr_matrix | None:
        """Load a feature vector from disk.

        Args:
            vectorize_folder: Folder containing vectorized features.
            filename: Name of the file.

        Returns:
            Feature vector (dense or sparse) or None if not found.
        """
        prefix = filename[:2]

        # Try sparse format first (.npz)
        npz_path = vectorize_folder / prefix / f"{filename}.npz"
        if npz_path.exists():
            try:
                return sparse.load_npz(npz_path)
            except Exception:
                pass

        # Try dense format (.npy)
        npy_path = vectorize_folder / prefix / f"{filename}.npy"
        if npy_path.exists():
            try:
                return np.load(npy_path).reshape(1, -1)
            except Exception:
                pass

        return None

    def evaluate(
        self,
        df: pd.DataFrame,
        is_binary: bool = True,
    ) -> dict[str, Any]:
        """Evaluate prediction results.

        Args:
            df: DataFrame with columns:
                - label: True label
                - predicted_label: Predicted label
                - predicted_encoded: Encoded prediction (optional)
            is_binary: Whether this is binary classification.

        Returns:
            Dictionary containing evaluation metrics.
        """
        # Filter out rows without predictions
        valid_df = df.dropna(subset=[COL_PREDICTED_LABEL])

        if len(valid_df) == 0:
            self._logger.warning("no_valid_predictions")
            return {}

        # For binary, compare with 'label'; for multi-class, compare with 'family'
        if is_binary:
            y_true = valid_df[COL_LABEL].values
        else:
            if COL_FAMILY not in valid_df.columns:
                self._logger.warning(
                    "family_column_required",
                    column=COL_FAMILY,
                )
                return {}
            y_true = valid_df[COL_FAMILY].values

        y_pred = valid_df[COL_PREDICTED_LABEL].values

        # Basic metrics
        if is_binary:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, pos_label=LABEL_MALWARE, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, pos_label=LABEL_MALWARE, zero_division=0
                ),
                "f1": f1_score(
                    y_true, y_pred, pos_label=LABEL_MALWARE, zero_division=0
                ),
            }
        else:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "f1": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
            }

        # Confusion matrix - use all unique labels from both y_true and y_pred
        all_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = all_labels

        # Additional statistics
        metrics["total_samples"] = len(valid_df)
        metrics["correct_predictions"] = int((y_true == y_pred).sum())
        metrics["skipped_samples"] = len(df) - len(valid_df)

        self._logger.info(
            "evaluation_completed",
            accuracy=round(metrics["accuracy"], 4),
            precision=round(metrics["precision"], 4),
            recall=round(metrics["recall"], 4),
            f1=round(metrics["f1"], 4),
        )

        return metrics

    def save_results(
        self,
        df: pd.DataFrame,
        metrics: dict[str, Any],
        output_folder: str | Path,
    ) -> None:
        """Save prediction results to disk.

        Args:
            df: DataFrame with prediction results.
            metrics: Evaluation metrics dictionary.
            output_folder: Folder to save results.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save predictions as CSV
        csv_path = output_folder / "predictions.csv"
        df.to_csv(csv_path, index=False)

        # Save predictions as JSON
        json_path = output_folder / "predictions.json"
        predictions_list = df.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(predictions_list, f, indent=2, default=str)

        # Save metrics
        metrics_path = output_folder / "metrics.json"
        metrics_with_meta = {
            **metrics,
            "created_date": datetime.now().isoformat(),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_with_meta, f, indent=2)

        self._logger.info(
            "results_saved",
            predictions_path=str(csv_path),
            metrics_path=str(metrics_path),
        )

"""Main detector class for UPX ELF malware detection.

This module provides the main entry point for the UPX ELF Detector,
inheriting from maldet.BaseDetector and implementing the required methods.

Typical usage:
    from upxelfdet import UpxElfDetector, UpxElfDetectorConfig

    # Training
    config = UpxElfDetectorConfig.from_file(Path("config.json"))
    detector = UpxElfDetector(config)
    model_path = detector.train()

    # Evaluation
    metrics = detector.evaluate()

    # Prediction
    output_path = detector.predict()
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from maldet import BaseDetector
from scipy import sparse
from tqdm import tqdm

from .config import UpxElfDetectorConfig
from .constants import (
    COL_EXTRACTION_SUCCESS,
    COL_FAMILY,
    COL_FILE_NAME,
    COL_LABEL,
    LABEL_ENCODER_FILENAME,
    TRAINING_CONFIG_FILENAME,
    VECTORIZE_METHOD_NGRAM_NUMERIC,
)
from .exceptions import ModelError, PredictionError
from .feature import FeatureExtractor
from .logging import configure_logging
from .model import BaseModel, CustomLabelEncoder, ModelFactory
from .predictor import Predictor
from .vectorizer import BaseVectorizer, VectorizerFactory


class UpxElfDetector(BaseDetector):
    """Main detector class for UPX-packed ELF malware detection.

    Inherits from maldet.BaseDetector and implements the required methods:
    - train: Extract features, vectorize, and train the model
    - evaluate: Evaluate the model on test data
    - predict: Make predictions on new samples

    Attributes:
        config_class: Configuration class for this detector.
        config: The loaded configuration instance.
        logger: Structured logger bound to this detector.

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

    config_class = UpxElfDetectorConfig

    def __init__(self, config: UpxElfDetectorConfig | None = None) -> None:
        """Initialize the detector with configuration.

        Args:
            config: Configuration instance. If None, creates default
                configuration using config_class.
        """
        super().__init__(config)

        # Setup logging with file output
        configure_logging(
            level=self.config.log.level,
            format=self.config.log.format,
            log_dir=self.config.output.log,
        )

        # Components (initialized lazily)
        self._extractor: FeatureExtractor | None = None
        self._vectorizer: BaseVectorizer | None = None
        self._model: BaseModel | None = None
        self._label_encoder: CustomLabelEncoder | None = None
        self._predictor: Predictor | None = None

        # Log configuration
        self.logger.info(
            "detector_initialized",
            vectorize_method=self.config.vectorize.method,
            model_type=self.config.model.type,
            classify=self.config.classify,
        )

    def train(self) -> Path:
        """Train the detector model.

        This method performs the full training pipeline:
        1. Load training data from config.data.train
        2. Extract features from ELF files
        3. Vectorize the extracted features
        4. Train the classification model
        5. Save the model to config.output.model

        Returns:
            Path to the saved model directory.

        Raises:
            FileNotFoundError: If training data file is not found.
            ModelError: If training fails.
        """
        self.logger.info("training_started", data_path=str(self.config.data.train))

        # Load training data
        df = self._load_data(self.config.data.train)
        self.logger.info("data_loaded", num_samples=len(df))

        # Extract features
        df = self._extract_features(df)

        # Filter successful extractions
        if COL_EXTRACTION_SUCCESS in df.columns:
            df = df[df[COL_EXTRACTION_SUCCESS]]
            self.logger.info("extraction_filtered", num_samples=len(df))

        # Vectorize features (fit mode)
        self._vectorize(df, fit=True)

        # Train model
        self._train_model(df)

        # Save model artifacts
        model_path = self._save_model()

        self.logger.info("training_completed", model_path=str(model_path))
        return model_path

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the detector on test data.

        This method performs:
        1. Load test data from config.data.test
        2. Load the trained model
        3. Extract features and vectorize
        4. Make predictions and compute metrics

        Returns:
            Dictionary containing evaluation metrics including:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - confusion_matrix: Confusion matrix as list

        Raises:
            FileNotFoundError: If test data or model is not found.
            ModelError: If model loading fails.
        """
        self.logger.info("evaluation_started", data_path=str(self.config.data.test))

        # Load model if not already loaded
        if self._model is None:
            self._load_model()

        # Load test data
        df = self._load_data(self.config.data.test)
        self.logger.info("data_loaded", num_samples=len(df))

        # Extract features
        df = self._extract_features(df)

        # Filter successful extractions
        if COL_EXTRACTION_SUCCESS in df.columns:
            df = df[df[COL_EXTRACTION_SUCCESS]]

        # Vectorize features (transform mode)
        self._vectorize(df, fit=False)

        # Make predictions and evaluate
        result_df = self._make_predictions(df)
        metrics = self._compute_metrics(result_df)

        self.logger.info(
            "evaluation_completed",
            accuracy=metrics.get("accuracy", 0),
            f1=metrics.get("f1", 0),
        )

        return metrics

    def predict(self) -> Path:
        """Run prediction on input data.

        This method performs:
        1. Load prediction data from config.data.predict
        2. Load the trained model
        3. Extract features and vectorize
        4. Make predictions
        5. Save results to config.output.prediction

        Returns:
            Path to the prediction output file.

        Raises:
            FileNotFoundError: If prediction data or model is not found.
            PredictionError: If prediction fails.
        """
        self.logger.info("prediction_started", data_path=str(self.config.data.predict))

        # Load model if not already loaded
        if self._model is None:
            self._load_model()

        # Load prediction data
        df = self._load_data(self.config.data.predict)
        self.logger.info("data_loaded", num_samples=len(df))

        # Extract features
        df = self._extract_features(df)

        # Filter successful extractions
        if COL_EXTRACTION_SUCCESS in df.columns:
            original_count = len(df)
            df = df[df[COL_EXTRACTION_SUCCESS]]
            self.logger.info(
                "extraction_filtered",
                original=original_count,
                filtered=len(df),
            )

        # Vectorize features (transform mode)
        self._vectorize(df, fit=False)

        # Make predictions
        result_df = self._make_predictions(df)

        # Save results
        output_path = self._save_predictions(result_df)

        self.logger.info("prediction_completed", output_path=str(output_path))
        return output_path

    def _load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from a CSV file or directory.

        Args:
            data_path: Path to CSV file or directory containing files.

        Returns:
            DataFrame with file information.

        Raises:
            FileNotFoundError: If the data path does not exist.
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if data_path.is_file() and data_path.suffix == ".csv":
            return pd.read_csv(data_path)
        elif data_path.is_dir():
            # Create DataFrame from directory listing
            files = []
            for file_path in data_path.rglob("*"):
                if file_path.is_file():
                    files.append({COL_FILE_NAME: file_path.name})
            return pd.DataFrame(files)
        else:
            raise FileNotFoundError(
                f"Invalid data path: {data_path}. Expected CSV file or directory."
            )

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from ELF files.

        Args:
            df: DataFrame containing file information.

        Returns:
            DataFrame with extraction results added.
        """
        self.logger.info("feature_extraction_started", num_files=len(df))

        if self._extractor is None:
            self._extractor = FeatureExtractor()

        # Ensure output directory exists
        self.ensure_directory_exists(self.config.output.feature)

        result_df = self._extractor.extract(
            df=df,
            dataset_folder=self.config.data.dataset,
            feature_folder=self.config.output.feature,
        )

        if COL_EXTRACTION_SUCCESS in result_df.columns:
            success_count = result_df[COL_EXTRACTION_SUCCESS].sum()
            self.logger.info(
                "feature_extraction_completed",
                success=int(success_count),
                total=len(result_df),
            )

        return result_df

    def _vectorize(self, df: pd.DataFrame, fit: bool = True) -> None:
        """Vectorize the extracted features.

        Args:
            df: DataFrame containing file information.
            fit: If True, fit the vectorizer (training mode).
                If False, only transform (inference mode).
        """
        self.logger.info("vectorization_started", fit=fit, num_files=len(df))

        # Create vectorizer if needed
        if self._vectorizer is None:
            self._vectorizer = VectorizerFactory.create(self.config.vectorize.method)

        # Ensure output directory exists
        vectorize_folder = self.config.output.feature.parent / "vectorize"
        self.ensure_directory_exists(vectorize_folder)

        # Create prefix directories
        for prefix in df[COL_FILE_NAME].str[:2].unique():
            (vectorize_folder / prefix).mkdir(parents=True, exist_ok=True)

        # Prepare kwargs for vectorizer
        kwargs: dict[str, Any] = {}
        if self.config.vectorize.method == VECTORIZE_METHOD_NGRAM_NUMERIC:
            kwargs["ngram_size"] = self.config.vectorize.ngram_size
            kwargs["encoding"] = self.config.vectorize.encoding

        if fit:
            self._vectorizer.fit_transform(
                df=df,
                feature_folder=self.config.output.feature,
                vectorize_folder=vectorize_folder,
                model_folder=self.config.output.model,
                section_name=self.config.feature.section_name,
                offset=self.config.vectorize.offset,
                size_features=self.config.vectorize.size_features,
                **kwargs,
            )
        else:
            self._vectorizer.transform(
                df=df,
                feature_folder=self.config.output.feature,
                vectorize_folder=vectorize_folder,
                section_name=self.config.feature.section_name,
                offset=self.config.vectorize.offset,
                size_features=self.config.vectorize.size_features,
                **kwargs,
            )

        self.logger.info("vectorization_completed")

    def _train_model(self, df: pd.DataFrame) -> None:
        """Train the classification model.

        Args:
            df: DataFrame containing training data with labels.

        Raises:
            ModelError: If training fails.
        """
        self.logger.info("model_training_started")

        # Load training vectors
        X, valid_indices = self._load_vectors(df)

        if X.shape[0] == 0:
            raise ModelError("No valid training samples found")

        self.logger.info("vectors_loaded", num_samples=X.shape[0])

        # Encode labels
        self._label_encoder = CustomLabelEncoder()

        if self.config.classify:
            # Multi-class: use family as label
            if COL_FAMILY not in df.columns:
                raise ModelError(
                    f"'{COL_FAMILY}' column required for classification mode"
                )
            labels = df.loc[valid_indices, COL_FAMILY].values
        else:
            # Binary: use label column
            if COL_LABEL not in df.columns:
                raise ModelError(f"'{COL_LABEL}' column required for binary mode")
            labels = df.loc[valid_indices, COL_LABEL].values

        y = self._label_encoder.fit_transform(labels)

        # Create and train model
        self._model = ModelFactory.create(
            self.config.model.type,
            params=self.config.model.params,
        )
        self._model.train(X, y)

        # Log training metrics
        metrics = self._model.evaluate(X, y, is_binary=not self.config.classify)
        self.logger.info(
            "model_training_completed",
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
        )

    def _load_model(self) -> None:
        """Load a trained model from disk.

        Raises:
            ModelError: If model files are not found.
        """
        model_folder = self.config.output.model

        # Load label encoder
        encoder_path = model_folder / LABEL_ENCODER_FILENAME
        if not encoder_path.exists():
            raise ModelError(f"Label encoder not found: {encoder_path}")

        with open(encoder_path, "rb") as f:
            self._label_encoder = pickle.load(f)

        # Load vectorizer
        self._vectorizer = VectorizerFactory.create(self.config.vectorize.method)
        self._vectorizer.load(model_folder)

        # Load model
        self._model = ModelFactory.create(
            self.config.model.type,
            params=self.config.model.params,
        )
        self._model.load(model_folder)

        self.logger.info("model_loaded", model_folder=str(model_folder))

    def _save_model(self) -> Path:
        """Save the trained model and associated artifacts.

        Returns:
            Path to the model directory.
        """
        model_folder = self.config.output.model
        self.ensure_directory_exists(model_folder)

        # Save model
        self._model.save(model_folder)

        # Save label encoder
        encoder_path = model_folder / LABEL_ENCODER_FILENAME
        with open(encoder_path, "wb") as f:
            pickle.dump(self._label_encoder, f)

        # Save training config
        config_path = model_folder / TRAINING_CONFIG_FILENAME
        self.config.save(config_path)

        self.logger.info("model_saved", model_folder=str(model_folder))
        return model_folder

    def _load_vectors(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray | sparse.csr_matrix, list]:
        """Load vectorized features for the given DataFrame.

        Args:
            df: DataFrame containing file information.

        Returns:
            Tuple of (feature_matrix, valid_indices).
        """
        vectorize_folder = self.config.output.feature.parent / "vectorize"

        vectors = []
        valid_indices = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading vectors"):
            filename = row[COL_FILE_NAME]
            prefix = filename[:2]

            # Try sparse format first
            npz_path = vectorize_folder / prefix / f"{filename}.npz"
            if npz_path.exists():
                try:
                    vec = sparse.load_npz(npz_path)
                    vectors.append(vec)
                    valid_indices.append(idx)
                    continue
                except Exception:
                    pass

            # Try dense format
            npy_path = vectorize_folder / prefix / f"{filename}.npy"
            if npy_path.exists():
                try:
                    vec = np.load(npy_path).reshape(1, -1)
                    vectors.append(vec)
                    valid_indices.append(idx)
                except Exception:
                    pass

        if not vectors:
            return np.array([]), []

        # Stack vectors
        if sparse.issparse(vectors[0]):
            X = sparse.vstack(vectors)
        else:
            X = np.vstack(vectors)

        return X, valid_indices

    def _make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for files in the DataFrame.

        Args:
            df: DataFrame containing file information.

        Returns:
            DataFrame with predictions added.
        """
        if self._predictor is None:
            self._predictor = Predictor()

        vectorize_folder = self.config.output.feature.parent / "vectorize"

        return self._predictor.predict(
            df=df,
            vectorize_folder=vectorize_folder,
            model=self._model,
            label_encoder=self._label_encoder,
            is_binary=not self.config.classify,
        )

    def _compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute evaluation metrics from prediction results.

        Args:
            df: DataFrame with predictions.

        Returns:
            Dictionary containing evaluation metrics.
        """
        if self._predictor is None:
            self._predictor = Predictor()

        return self._predictor.evaluate(df, is_binary=not self.config.classify)

    def _save_predictions(self, df: pd.DataFrame) -> Path:
        """Save prediction results to disk.

        Args:
            df: DataFrame with prediction results.

        Returns:
            Path to the prediction output file.
        """
        output_path = self.config.output.prediction
        self.ensure_directory_exists(output_path)

        # Compute metrics if labels are available
        has_labels = (
            (COL_LABEL in df.columns and not self.config.classify)
            or (COL_FAMILY in df.columns and self.config.classify)
        )

        if has_labels:
            metrics = self._compute_metrics(df)
            self._predictor.save_results(df, metrics, output_path.parent)
        else:
            # Save predictions only
            self._predictor.save_results(df, {}, output_path.parent)

        return output_path

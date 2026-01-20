#!/usr/bin/env python3
"""Example script demonstrating complete usage of UpxElfDetector.

This script shows how to:
1. Load configuration
2. Train a model
3. Evaluate the model
4. Make predictions

Usage:
    python3 examples/basic_usage.py
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from upxelfdet import UpxElfDetector
from upxelfdet.config import UpxElfDetectorConfig


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print(f"\n[{title}]")


def main():
    """Main execution flow demonstrating detector usage."""

    print_separator()
    print("upxelfdet - Complete Usage Example")
    print_separator()

    # Step 1: Load configuration
    print_section("Step 1: Loading Configuration")
    config_path = Path(__file__).parent.parent / "config.json"

    if config_path.exists():
        config = UpxElfDetectorConfig.from_file(config_path)
        print(f"  ✓ Configuration loaded from {config_path}")
    else:
        print(f"  ✗ Configuration file not found: {config_path}")
        print("  Creating default configuration...")
        config = UpxElfDetectorConfig()

    # Display key configuration settings
    print(f"\n  Configuration Summary:")
    print(f"    - Vectorization method: {config.vectorize.method}")
    print(f"    - N-gram size: {config.vectorize.ngram_size}")
    print(f"    - Encoding: {config.vectorize.encoding}")
    print(f"    - Model type: {config.model.type}")
    print(f"    - Model params: {config.model.params}")
    print(f"    - Classification mode: {'Multi-class (family)' if config.classify else 'Binary (malware/benign)'}")
    print(f"    - Dataset folder: {config.data.dataset}")
    print(f"    - Training data: {config.data.train}")
    print(f"    - Test data: {config.data.test}")
    print(f"    - Output folder: {config.output.model}")

    # Step 2: Initialize detector
    print_section("Step 2: Initializing Detector")
    detector = UpxElfDetector(config)
    print("  ✓ Detector initialized with configuration")

    # Step 3: Train the model
    print_section("Step 3: Training the Model")
    print("  This will:")
    print("    1. Load training data from CSV file")
    print("    2. Extract features from ELF files in dataset folder")
    print("    3. Vectorize the extracted features using n-gram method")
    print("    4. Train the SVM classification model")
    print("    5. Save the trained model to disk")
    print()
    print("  Starting training pipeline...")

    try:
        model_path = detector.train()
        print(f"\n  ✓ Training completed successfully!")
        print(f"  ✓ Model saved to: {model_path}")
    except FileNotFoundError as e:
        print(f"\n  ✗ Error: {e}")
        print("  Please ensure:")
        print("    - Training data CSV file exists")
        print("    - Dataset folder contains the sample files")
        print("    - File names in CSV match actual files in dataset folder")
        return
    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Evaluate the model
    print_section("Step 4: Evaluating the Model")
    print("  This will:")
    print("    1. Load test data from CSV file")
    print("    2. Load the trained model from disk")
    print("    3. Extract features and vectorize test samples")
    print("    4. Make predictions on test set")
    print("    5. Compute evaluation metrics")
    print()
    print("  Starting evaluation...")

    try:
        metrics = detector.evaluate()
        print(f"\n  ✓ Evaluation completed successfully!")
        print(f"\n  Evaluation Metrics:")
        print(f"    - Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"    - Precision: {metrics.get('precision', 0):.4f}")
        print(f"    - Recall:    {metrics.get('recall', 0):.4f}")
        print(f"    - F1 Score:  {metrics.get('f1', 0):.4f}")

        if 'confusion_matrix' in metrics:
            print(f"\n  Confusion Matrix:")
            cm = metrics['confusion_matrix']
            if hasattr(cm, 'tolist'):
                cm = cm.tolist()
            for row in cm:
                print(f"    {row}")

        # Print additional metrics if available
        if 'classification_report' in metrics:
            print(f"\n  Classification Report:")
            print(metrics['classification_report'])
    except FileNotFoundError as e:
        print(f"\n  ✗ Error: {e}")
        print("  Please ensure test data exists at the configured path.")
    except Exception as e:
        print(f"\n  ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Make predictions
    print_section("Step 5: Making Predictions")
    print("  This will:")
    print("    1. Load prediction data from CSV file")
    print("    2. Load the trained model from disk")
    print("    3. Extract features and vectorize samples")
    print("    4. Make predictions using the trained model")
    print("    5. Save prediction results to output folder")
    print()
    print("  Starting prediction...")

    try:
        output_path = detector.predict()
        print(f"\n  ✓ Prediction completed successfully!")
        print(f"  ✓ Results saved to: {output_path}")

        # Show sample predictions if file exists
        if output_path.exists():
            print(f"\n  Sample prediction output:")
            with open(output_path, 'r') as f:
                lines = f.readlines()
                for line in lines[:5]:  # Show first 5 lines
                    print(f"    {line.strip()}")
                if len(lines) > 5:
                    print(f"    ... ({len(lines) - 5} more lines)")
    except FileNotFoundError as e:
        print(f"\n  ✗ Error: {e}")
        print("  Please ensure prediction data exists at the configured path.")
    except Exception as e:
        print(f"\n  ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print()
    print_separator()
    print("Example Execution Summary:")
    print_separator()
    print()
    print("This example demonstrated the complete workflow of upxelfdet:")
    print()
    print("1. Configuration: Loaded settings from config.json")
    print("2. Training: Extracted features, vectorized, and trained SVM model")
    print("3. Evaluation: Computed metrics on test set")
    print("4. Prediction: Made predictions on new samples")
    print()
    print("The detector is now ready for use with the trained model saved to disk.")
    print()
    print_separator()


if __name__ == "__main__":
    main()

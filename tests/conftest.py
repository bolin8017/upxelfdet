"""Pytest configuration and fixtures for upxelfdet tests.

This module provides shared fixtures and configuration for all tests.
"""

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the path to config.json."""
    return project_root / "config.json"


@pytest.fixture(scope="session")
def train_csv_path(project_root: Path) -> Path:
    """Return the path to train.csv."""
    return project_root / "train.csv"


@pytest.fixture(scope="session")
def test_csv_path(project_root: Path) -> Path:
    """Return the path to test.csv."""
    return project_root / "test.csv"


@pytest.fixture(scope="session")
def has_train_data(train_csv_path: Path) -> bool:
    """Check if training data exists."""
    return train_csv_path.exists()


@pytest.fixture(scope="session")
def has_test_data(test_csv_path: Path) -> bool:
    """Check if test data exists."""
    return test_csv_path.exists()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

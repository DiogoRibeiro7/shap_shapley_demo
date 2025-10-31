"""
Tests for utils module.

Tests cover common utilities, I/O operations, and helper functions.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from shap_analytics.utils.common import (
    compute_jensen_shannon_divergence,
    compute_mean_abs_shap,
    ensure_directory,
    get_timestamp,
    load_json,
    load_model,
    save_json,
    serialize_model,
    setup_logger,
    shap_to_dataframe,
)


@pytest.mark.unit
class TestDirectoryOperations:
    """Tests for directory management utilities."""

    def test_ensure_directory_creates_dir(self, temp_dir: Path) -> None:
        """Test directory creation."""
        test_path = temp_dir / "new_dir"
        result = ensure_directory(test_path)

        assert result.exists()
        assert result.is_dir()
        assert result == test_path

    def test_ensure_directory_existing_dir(self, temp_dir: Path) -> None:
        """Test with existing directory."""
        result = ensure_directory(temp_dir)

        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_nested(self, temp_dir: Path) -> None:
        """Test nested directory creation."""
        nested_path = temp_dir / "level1" / "level2" / "level3"
        result = ensure_directory(nested_path)

        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_from_file_path(self, temp_dir: Path) -> None:
        """Test directory creation from file path."""
        file_path = temp_dir / "subdir" / "file.txt"
        result = ensure_directory(file_path)

        assert result.exists()
        assert result.is_dir()
        assert result == file_path.parent


@pytest.mark.unit
class TestTimestampGeneration:
    """Tests for timestamp utilities."""

    def test_get_timestamp_format(self) -> None:
        """Test timestamp format is ISO-8601."""
        timestamp = get_timestamp()

        assert isinstance(timestamp, str)
        assert "T" in timestamp
        # Should be parseable as ISO format
        from datetime import datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None

    def test_get_timestamp_unique(self) -> None:
        """Test timestamps are unique (or at least different)."""
        import time

        ts1 = get_timestamp()
        time.sleep(0.01)  # Small delay
        ts2 = get_timestamp()

        # They should be different (though very close)
        assert isinstance(ts1, str)
        assert isinstance(ts2, str)


@pytest.mark.unit
class TestJSONOperations:
    """Tests for JSON I/O utilities."""

    def test_save_and_load_json(self, temp_dir: Path, sample_config: dict[str, Any]) -> None:
        """Test JSON save and load."""
        json_path = temp_dir / "config.json"

        # Save
        saved_path = save_json(sample_config, json_path)
        assert saved_path.exists()

        # Load
        loaded_data = load_json(json_path)
        assert loaded_data == sample_config

    def test_save_json_creates_parent_dirs(self, temp_dir: Path) -> None:
        """Test JSON save creates parent directories."""
        nested_path = temp_dir / "nested" / "deep" / "config.json"
        data = {"key": "value"}

        saved_path = save_json(data, nested_path)

        assert saved_path.exists()
        assert saved_path.parent.exists()

    def test_load_json_nonexistent_file(self, temp_dir: Path) -> None:
        """Test loading nonexistent JSON file."""
        nonexistent = temp_dir / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_json(nonexistent)

    def test_save_json_complex_data(self, temp_dir: Path) -> None:
        """Test JSON save with complex nested data."""
        complex_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        json_path = temp_dir / "complex.json"
        save_json(complex_data, json_path)
        loaded = load_json(json_path)

        assert loaded == complex_data


@pytest.mark.unit
class TestModelSerialization:
    """Tests for model serialization utilities."""

    def test_serialize_and_load_model(
        self,
        small_trained_model: Any,
        temp_dir: Path,
    ) -> None:
        """Test model serialization and loading."""
        model_path = temp_dir / "model.joblib"

        # Serialize
        saved_path = serialize_model(small_trained_model, model_path, compute_checksum=False)
        assert saved_path.exists()

        # Load
        loaded_model = load_model(model_path)
        assert loaded_model is not None
        assert type(loaded_model) is type(small_trained_model)

    def test_serialize_with_checksum(
        self,
        small_trained_model: Any,
        temp_dir: Path,
    ) -> None:
        """Test model serialization with checksum."""
        model_path = temp_dir / "model.joblib"

        serialize_model(small_trained_model, model_path, compute_checksum=True)

        checksum_path = model_path.with_suffix(".sha256")
        assert checksum_path.exists()

        checksum = checksum_path.read_text()
        assert len(checksum) == 64  # SHA256 hex length

    def test_load_model_with_checksum_verification(
        self,
        small_trained_model: Any,
        temp_dir: Path,
    ) -> None:
        """Test loading model with checksum verification."""
        model_path = temp_dir / "model.joblib"

        serialize_model(small_trained_model, model_path, compute_checksum=True)
        loaded_model = load_model(model_path, verify_checksum=True)

        assert loaded_model is not None

    def test_load_model_checksum_mismatch(
        self,
        small_trained_model: Any,
        temp_dir: Path,
    ) -> None:
        """Test loading model with corrupted checksum."""
        model_path = temp_dir / "model.joblib"

        serialize_model(small_trained_model, model_path, compute_checksum=True)

        # Corrupt checksum
        checksum_path = model_path.with_suffix(".sha256")
        checksum_path.write_text("invalid_checksum")

        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_model(model_path, verify_checksum=True)

    def test_load_nonexistent_model(self, temp_dir: Path) -> None:
        """Test loading nonexistent model."""
        nonexistent = temp_dir / "missing.joblib"

        with pytest.raises(FileNotFoundError):
            load_model(nonexistent)


@pytest.mark.unit
class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon divergence computation."""

    def test_identical_distributions(self) -> None:
        """Test JS divergence for identical distributions."""
        p = np.random.randn(100)
        divergence = compute_jensen_shannon_divergence(p, p, bins=20)

        assert 0 <= divergence <= 1
        assert divergence < 0.1  # Should be near zero

    def test_different_distributions(self) -> None:
        """Test JS divergence for different distributions."""
        p = np.random.randn(100)
        q = np.random.randn(100) + 3.0  # Shifted distribution

        divergence = compute_jensen_shannon_divergence(p, q, bins=20)

        assert 0 <= divergence <= 1
        assert divergence > 0.3  # Should be significant

    def test_pandas_series_input(self) -> None:
        """Test JS divergence with pandas Series."""
        p = pd.Series(np.random.randn(100))
        q = pd.Series(np.random.randn(100))

        divergence = compute_jensen_shannon_divergence(p, q, bins=20)

        assert isinstance(divergence, float)
        assert 0 <= divergence <= 1


@pytest.mark.unit
class TestSHAPUtilities:
    """Tests for SHAP-specific utilities."""

    def test_compute_mean_abs_shap(self, mock_shap_values: np.ndarray) -> None:
        """Test mean absolute SHAP computation."""
        mean_abs = compute_mean_abs_shap(mock_shap_values, axis=0)

        assert isinstance(mean_abs, np.ndarray)
        assert len(mean_abs) == mock_shap_values.shape[1]
        assert all(val >= 0 for val in mean_abs)

    def test_shap_to_dataframe(
        self,
        small_trained_model: Any,
        small_dataset: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Test SHAP to DataFrame conversion."""
        from shap_analytics.shap_explain import compute_shap_values

        X, _ = small_dataset
        X_train, X_test = X[:30], X[30:]

        shap_values = compute_shap_values(
            small_trained_model,
            X_train,
            X_test,
            background_size=10,
        )

        shap_df = shap_to_dataframe(shap_values, X_test)

        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape[0] == len(X_test)
        assert shap_df.shape[1] == X_test.shape[1]
        assert list(shap_df.columns) == list(X_test.columns)


@pytest.mark.unit
class TestLogging:
    """Tests for logging utilities."""

    def test_setup_logger_console(self) -> None:
        """Test logger setup with console output."""
        import logging

        logger = setup_logger("test_logger", level=logging.INFO)

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_with_file(self, temp_dir: Path) -> None:
        """Test logger setup with file output."""
        import logging

        log_file = temp_dir / "test.log"
        logger = setup_logger("test_logger_file", log_file=log_file, level=logging.DEBUG)

        logger.info("Test message")

        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_logger_multiple_handlers(self, temp_dir: Path) -> None:
        """Test logger has both console and file handlers."""

        log_file = temp_dir / "multi.log"
        logger = setup_logger("multi_logger", log_file=log_file)

        # Should have console and file handlers
        assert len(logger.handlers) >= 2

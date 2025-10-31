"""
Common utility functions for SHAP analytics.

This module contains shared utilities used across all SHAP modules to avoid
code duplication and ensure consistency.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from scipy.spatial.distance import jensenshannon

__all__ = [
    "compute_jensen_shannon_divergence",
    "compute_mean_abs_shap",
    "ensure_directory",
    "get_timestamp",
    "load_json",
    "load_model",
    "save_json",
    "serialize_model",
    "setup_logger",
    "shap_to_dataframe",
]


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it and parent directories if needed.

    Args:
        path: Directory path or file path (parent directory will be created).

    Returns:
        Path object of the created/existing directory.
    """
    path_obj = Path(path)
    if path_obj.suffix:  # If it's a file path, get parent
        path_obj = path_obj.parent
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_timestamp() -> str:
    """
    Generate ISO-8601 formatted UTC timestamp.

    Returns:
        ISO-8601 formatted timestamp string.
    """
    return datetime.utcnow().isoformat()


def save_json(data: Any, output_path: str | Path, indent: int = 2) -> Path:
    """
    Save data to JSON file with consistent formatting.

    Args:
        data: Data to serialize (must be JSON-serializable).
        output_path: Path where JSON file will be saved.
        indent: JSON indentation level.

    Returns:
        Path object of the saved file.
    """
    path = Path(output_path)
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return path


def load_json(input_path: str | Path) -> dict[str, Any]:
    """
    Load JSON data from file.

    Args:
        input_path: Path to JSON file.

    Returns:
        Parsed JSON data as dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def compute_jensen_shannon_divergence(
    p: npt.NDArray[Any] | pd.Series[Any],
    q: npt.NDArray[Any] | pd.Series[Any],
    bins: int = 20,
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        p: First distribution (array or series).
        q: Second distribution (array or series).
        bins: Number of histogram bins for density estimation.

    Returns:
        Jensen-Shannon divergence score (0-1).
    """
    # Convert to numpy arrays if needed
    p_array = np.asarray(p)
    q_array = np.asarray(q)

    # Create histograms for density estimation
    p_hist, _ = np.histogram(p_array, bins=bins, density=True)
    q_hist, _ = np.histogram(q_array, bins=bins, density=True)

    # Clip to avoid log(0) issues
    p_hist = np.clip(p_hist, 1e-12, None)
    q_hist = np.clip(q_hist, 1e-12, None)

    # Normalize to ensure they sum to 1
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(jensenshannon(p_hist, q_hist))


def serialize_model(
    model: Any,
    model_path: str | Path,
    compute_checksum: bool = True,
) -> Path:
    """
    Serialize a model using joblib with optional checksum validation.

    Args:
        model: Model object to serialize.
        model_path: Path where model will be saved.
        compute_checksum: Whether to compute and save SHA256 checksum.

    Returns:
        Path object of the saved model file.
    """
    path = Path(model_path)
    ensure_directory(path.parent)

    # Save model
    joblib.dump(model, path)

    # Optionally save checksum
    if compute_checksum:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_path = path.with_suffix(".sha256")
        checksum_path.write_text(sha, encoding="utf-8")

    return path


def load_model(
    model_path: str | Path,
    verify_checksum: bool = False,
) -> Any:
    """
    Load a serialized model using joblib with optional checksum verification.

    Args:
        model_path: Path to serialized model file.
        verify_checksum: Whether to verify SHA256 checksum.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If checksum verification fails.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Verify checksum if requested
    if verify_checksum:
        checksum_path = path.with_suffix(".sha256")
        if checksum_path.exists():
            expected_sha = checksum_path.read_text(encoding="utf-8").strip()
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_sha != actual_sha:
                raise ValueError(
                    f"Checksum mismatch for {model_path}. "
                    f"Expected: {expected_sha[:8]}..., Got: {actual_sha[:8]}..."
                )

    return joblib.load(path)


def shap_to_dataframe(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert SHAP Explanation object to pandas DataFrame.

    Args:
        shap_values: SHAP Explanation object from explainer.
        X: Feature DataFrame with column names.

    Returns:
        DataFrame with SHAP values and same column names as X.

    Raises:
        ValueError: If shap_values shape doesn't match X.
    """
    if not hasattr(shap_values, "values"):
        raise ValueError("Invalid SHAP object - missing 'values' attribute")

    values = shap_values.values
    # Handle multi-output case (e.g., binary classification with 2 outputs)
    if values.ndim == 3:
        # Use class 1 for binary classification
        values = values[..., 1]

    if values.shape[1] != len(X.columns):
        raise ValueError(
            f"SHAP values shape {values.shape} doesn't match features {len(X.columns)}"
        )

    return pd.DataFrame(values, columns=X.columns)


def compute_mean_abs_shap(
    shap_values: shap.Explanation | npt.NDArray[Any],
    axis: int = 0,
) -> npt.NDArray[Any]:
    """
    Compute mean absolute SHAP values.

    Args:
        shap_values: SHAP Explanation object or numpy array.
        axis: Axis along which to compute mean (0 for features).

    Returns:
        Array of mean absolute SHAP values.
    """
    values = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Handle multi-output case
    if values.ndim == 3:
        values = values[..., 1]

    return cast(npt.NDArray[Any], np.abs(values).mean(axis=axis))


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of the module).
        log_file: Optional path to log file. If None, logs to console only.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format: Format string for log messages.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        ensure_directory(log_path.parent)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

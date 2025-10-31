"""
I/O utilities for SHAP Analytics.

Advanced file operations, format conversion, and data loading utilities.
"""

import gzip
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "export_shap_report",
    "load_compressed",
    "load_csv_with_types",
    "save_compressed",
]


def load_csv_with_types(
    file_path: str | Path,
    type_map: dict[str, type] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load CSV file with explicit type casting.

    Args:
        file_path: Path to CSV file.
        type_map: Dictionary mapping column names to types.
        **kwargs: Additional arguments for pd.read_csv.

    Returns:
        DataFrame with explicitly typed columns.

    Example:
        >>> type_map = {"age": int, "score": float, "name": str}
        >>> df = load_csv_with_types("data.csv", type_map=type_map)
    """
    df = pd.read_csv(file_path, **kwargs)

    if type_map:
        for col, dtype in type_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    return df


def save_compressed(
    obj: Any,
    file_path: str | Path,
    compression_level: int = 6,
) -> Path:
    """
    Save object to compressed pickle file.

    Args:
        obj: Object to serialize.
        file_path: Output file path (will add .gz if not present).
        compression_level: Gzip compression level (0-9).

    Returns:
        Path to saved file.
    """
    path = Path(file_path)
    if not path.suffix.endswith(".gz"):
        path = path.with_suffix(path.suffix + ".gz")

    with gzip.open(path, "wb", compresslevel=compression_level) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_compressed(file_path: str | Path) -> Any:
    """
    Load object from compressed pickle file.

    Args:
        file_path: Path to compressed pickle file.

    Returns:
        Deserialized object.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Compressed file not found: {file_path}")

    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def export_shap_report(
    shap_values: Any,
    feature_names: list[str],
    output_path: str | Path,
    format: str = "csv",
) -> Path:
    """
    Export SHAP values to various formats.

    Args:
        shap_values: SHAP values array or Explanation object.
        feature_names: List of feature names.
        output_path: Output file path.
        format: Export format ("csv", "parquet", "json").

    Returns:
        Path to exported file.

    Raises:
        ValueError: If format is not supported.
    """

    # Extract values if Explanation object
    values = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Handle multi-output (take class 1 for binary classification)
    if values.ndim == 3:
        values = values[..., 1]

    # Create DataFrame
    df = pd.DataFrame(values, columns=feature_names)

    path = Path(output_path)

    # Export based on format
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'parquet', or 'json'")

    return path

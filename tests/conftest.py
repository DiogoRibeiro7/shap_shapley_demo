"""
Pytest configuration and shared fixtures for SHAP tests.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"feature_{i}" for i in range(5)]
    )


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def trained_model(sample_classification_data) -> RandomForestClassifier:
    """Create and train a simple RandomForest model."""
    X, y = sample_classification_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def shap_values_explanation(
    trained_model, sample_classification_data
) -> shap.Explanation:
    """Create SHAP Explanation object for testing."""
    X, _ = sample_classification_data
    background = X.sample(20, random_state=42)
    explainer = shap.TreeExplainer(trained_model, data=background)
    return explainer(X[:50])


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config_files(temp_dir: Path) -> dict[str, Path]:
    """Create mock configuration files for testing."""
    config = {
        "MODEL_PATH": "model.joblib",
        "DRIFT_THRESHOLD": 0.2,
        "CACHE_TTL_HOURS": 24,
    }

    schema = {
        "type": "object",
        "properties": {
            "MODEL_PATH": {"type": "string"},
            "DRIFT_THRESHOLD": {"type": "number"},
            "CACHE_TTL_HOURS": {"type": "integer"},
        },
        "required": ["MODEL_PATH"]
    }

    config_path = temp_dir / "config.json"
    schema_path = temp_dir / "config_schema.json"

    import json
    config_path.write_text(json.dumps(config))
    schema_path.write_text(json.dumps(schema))

    return {"config": config_path, "schema": schema_path}


@pytest.fixture
def sample_shap_dataframe() -> pd.DataFrame:
    """Create a sample SHAP values DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(50, 5) * 0.5,
        columns=[f"feature_{i}" for i in range(5)]
    )


@pytest.fixture
def trained_rf_model(sample_classification_data) -> RandomForestClassifier:
    """Create and train a RandomForest model (alias for trained_model)."""
    X, y = sample_classification_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def small_trained_model() -> RandomForestClassifier:
    """Create and train a small RandomForest model with minimal samples."""
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small classification dataset for testing."""
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def train_test_split_data(sample_classification_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split from classification data."""
    X, _ = sample_classification_data
    # Simple 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    return X_train, X_test


@pytest.fixture
def sample_config() -> dict[str, object]:
    """Create a sample configuration dictionary."""
    return {
        "MODEL_PATH": "model.joblib",
        "DRIFT_THRESHOLD": 0.2,
        "CACHE_TTL_HOURS": 24,
        "LOG_LEVEL": "INFO",
    }


@pytest.fixture
def mock_shap_values() -> np.ndarray:
    """Create mock SHAP values array for testing."""
    np.random.seed(42)
    return np.random.randn(50, 5) * 0.5


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test."""
    yield
    # Cleanup logic if needed
    import matplotlib.pyplot as plt
    plt.close('all')

# API Reference

Complete API reference for SHAP Analytics modules.

## Core Module: shap_explain

### compute_shap_values

::: shap_analytics.shap_explain.compute_shap_values
    options:
      show_source: true
      heading_level: 3

Computes SHAP values using TreeExplainer with background data sampling.

**Parameters:**
- `model` (Any): Trained tree-based model (RandomForest, XGBoost, etc.)
- `X_train` (pd.DataFrame): Training features for background sampling
- `X_test` (pd.DataFrame): Test features for SHAP computation
- `background_size` (int): Number of background samples (default: 100)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Returns:**
- `shap.Explanation`: SHAP Explanation object with computed values

**Raises:**
- `ValueError`: If model is incompatible with TreeExplainer

**Example:**
```python
from shap_analytics import compute_shap_values
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
shap_values = compute_shap_values(model, X_train, X_test)
```

---

### validate_background_sample

::: shap_analytics.shap_explain.validate_background_sample
    options:
      show_source: true
      heading_level: 3

Validates that background sample is statistically representative of training data.

**Parameters:**
- `X_train` (pd.DataFrame): Training feature set
- `sample_size` (int): Number of samples for validation (default: 100)
- `threshold` (float): Maximum allowed normalized difference (default: 0.1)
- `random_state` (int): Random seed (default: 42)

**Returns:**
- `bool`: True if validation passes, False otherwise

**Warnings:**
- UserWarning if background sample deviates significantly

**Example:**
```python
from shap_analytics import validate_background_sample

is_valid = validate_background_sample(X_train, sample_size=100, threshold=0.1)
if not is_valid:
    print("Warning: Background sample may not be representative")
```

---

### monitor_feature_drift

::: shap_analytics.shap_explain.monitor_feature_drift
    options:
      show_source: true
      heading_level: 3

Monitors feature drift between training and new data using Jensen-Shannon divergence.

**Parameters:**
- `X_train` (pd.DataFrame): Training data (reference distribution)
- `X_new` (pd.DataFrame): New data (current distribution)
- `threshold` (float): Drift threshold for alerting (default: 0.2)
- `bins` (int): Number of histogram bins (default: 20)

**Returns:**
- `Dict[str, float]`: Feature names mapped to drift scores (0-1)

**Example:**
```python
from shap_analytics import monitor_feature_drift

drift_scores = monitor_feature_drift(X_train, X_production, threshold=0.2)
high_drift = {k: v for k, v in drift_scores.items() if v > 0.2}
print(f"Features with high drift: {high_drift}")
```

---

### verify_shap_reconstruction

::: shap_analytics.shap_explain.verify_shap_reconstruction
    options:
      show_source: true
      heading_level: 3

Verifies that SHAP values correctly reconstruct model predictions.

**Parameters:**
- `shap_values` (shap.Explanation): SHAP Explanation object
- `X` (pd.DataFrame): Feature data
- `model` (Any): Trained model with `predict_proba` method
- `sample_index` (int): Index of sample to verify (default: 0)
- `tolerance` (float): Maximum reconstruction error (default: 1e-3)

**Returns:**
- `bool`: True if reconstruction is within tolerance

**Raises:**
- `ValueError`: If SHAP values are incompatible or model lacks `predict_proba`

**Example:**
```python
from shap_analytics import verify_shap_reconstruction

is_valid = verify_shap_reconstruction(shap_values, X_test, model, sample_index=0)
assert is_valid, "SHAP reconstruction failed!"
```

---

## Expansion Module: shap_expansion

### SHAPRegistry

::: shap_analytics.shap_expansion.SHAPRegistry
    options:
      show_source: true
      heading_level: 3

Registry for tracking and managing SHAP explanations with metadata.

**Attributes:**
- `storage_path` (Path): Directory for storing registry data

**Methods:**

#### register

```python
def register(
    self,
    model_name: str,
    model_version: str,
    shap_values: shap.Explanation,
    metadata: Dict[str, Any]
) -> str
```

Register a new SHAP explanation.

**Returns:** Unique explanation ID

#### get

```python
def get(
    self,
    model_name: str,
    model_version: str
) -> Optional[Dict[str, Any]]
```

Retrieve explanation by model name and version.

**Example:**
```python
from shap_analytics import SHAPRegistry

registry = SHAPRegistry(storage_path="./registry")
registry.register("my_model", "v1.0", shap_values, metadata={...})
explanation = registry.get("my_model", "v1.0")
```

---

### create_interactive_dashboard

::: shap_analytics.shap_expansion.create_interactive_dashboard
    options:
      show_source: true
      heading_level: 3

Creates an interactive Plotly dashboard for SHAP visualization.

**Parameters:**
- `shap_values` (shap.Explanation): SHAP values to visualize
- `X` (pd.DataFrame): Feature data
- `model` (Any): Trained model
- `feature_names` (List[str]): Feature names for display

**Returns:**
- `plotly.graph_objects.Figure`: Interactive dashboard figure

**Example:**
```python
from shap_analytics import create_interactive_dashboard

dashboard = create_interactive_dashboard(shap_values, X_test, model, X_test.columns)
dashboard.write_html("shap_dashboard.html")
```

---

### serve_shap_api

::: shap_analytics.shap_expansion.serve_shap_api
    options:
      show_source: true
      heading_level: 3

Creates a FastAPI application for serving SHAP explanations.

**Parameters:**
- `model` (Any): Trained model
- `X_train` (pd.DataFrame): Training data for background sampling
- `background_size` (int): Background sample size (default: 100)

**Returns:**
- `fastapi.FastAPI`: Configured FastAPI application

**Endpoints:**
- `GET /health`: Health check
- `POST /explain`: Compute SHAP values for instances

**Example:**
```python
from shap_analytics import serve_shap_api
import uvicorn

app = serve_shap_api(model, X_train, background_size=100)
uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Utilities Module: utils

### Common Utilities

#### ensure_directory

```python
def ensure_directory(path: str | Path) -> Path
```

Ensures a directory exists, creating it if needed.

---

#### get_timestamp

```python
def get_timestamp() -> str
```

Generates ISO-8601 formatted UTC timestamp.

---

#### save_json

```python
def save_json(data: Any, output_path: str | Path, indent: int = 2) -> Path
```

Saves data to JSON file with consistent formatting.

---

#### load_json

```python
def load_json(input_path: str | Path) -> Dict[str, Any]
```

Loads JSON data from file.

---

#### serialize_model

```python
def serialize_model(
    model: Any,
    model_path: str | Path,
    compute_checksum: bool = True
) -> Path
```

Serializes a model using joblib with optional checksum.

---

#### load_model

```python
def load_model(
    model_path: str | Path,
    verify_checksum: bool = False
) -> Any
```

Loads a serialized model with optional checksum verification.

---

#### shap_to_dataframe

```python
def shap_to_dataframe(
    shap_values: shap.Explanation,
    X: pd.DataFrame
) -> pd.DataFrame
```

Converts SHAP Explanation object to pandas DataFrame.

---

#### compute_mean_abs_shap

```python
def compute_mean_abs_shap(
    shap_values: shap.Explanation | np.ndarray,
    axis: int = 0
) -> np.ndarray
```

Computes mean absolute SHAP values for feature importance.

---

#### setup_logger

```python
def setup_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    log_format: str = "..."
) -> logging.Logger
```

Configures and returns a logger with consistent formatting.

**Example:**
```python
from shap_analytics.utils import setup_logger

logger = setup_logger(__name__, log_file="logs/app.log", level=logging.INFO)
logger.info("Application started")
```

---

#### compute_jensen_shannon_divergence

```python
def compute_jensen_shannon_divergence(
    p: np.ndarray | pd.Series,
    q: np.ndarray | pd.Series,
    bins: int = 20
) -> float
```

Computes Jensen-Shannon divergence between two distributions.

**Returns:** Divergence score between 0 and 1 (0 = identical, 1 = completely different)

---

## Type Definitions

### Common Types

```python
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import shap

# Model types
Model = Any  # sklearn, xgboost, lightgbm models

# Data types
DataFrame = pd.DataFrame
Array = np.ndarray
Series = pd.Series

# SHAP types
SHAPExplanation = shap.Explanation
SHAPValues = Union[Array, SHAPExplanation]

# Configuration types
Config = Dict[str, Any]
Metadata = Dict[str, Any]
```

---

## Constants

```python
# Default configuration values
DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_RANDOM_STATE = 42
DEFAULT_DRIFT_THRESHOLD = 0.2
DEFAULT_RECONSTRUCTION_TOLERANCE = 1e-3

# API configuration
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8080
DEFAULT_API_WORKERS = 4

# Logging levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
```

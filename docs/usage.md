# Usage Guide

This guide provides comprehensive examples of using SHAP Analytics for various tasks.

## Basic SHAP Computation

### Computing SHAP Values

```python
from shap_analytics import compute_shap_values
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train your model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute SHAP values
shap_values = compute_shap_values(
    model=model,
    X_train=X_train,
    X_test=X_test,
    background_size=100,  # Number of background samples
    random_state=42
)

# Access SHAP values for class 1 (for binary classification)
shap_values_class1 = shap_values[..., 1]
print(f"SHAP values shape: {shap_values_class1.shape}")
```

### Validating Background Sample

Ensure your background sample is statistically representative:

```python
from shap_analytics import validate_background_sample

is_valid = validate_background_sample(
    X_train=X_train,
    sample_size=100,
    threshold=0.1,  # Maximum allowed normalized difference
    random_state=42
)

if is_valid:
    print("✅ Background sample is valid")
else:
    print("⚠️ Background sample may not be representative")
```

## Drift Monitoring

### Detecting Feature Drift

Monitor feature drift between training and new data:

```python
from shap_analytics import monitor_feature_drift

drift_scores = monitor_feature_drift(
    X_train=X_train,
    X_new=X_production,
    threshold=0.2,  # Drift alert threshold
    bins=20  # Histogram bins for distribution estimation
)

# Identify high-drift features
high_drift_features = {
    feature: score
    for feature, score in drift_scores.items()
    if score > 0.2
}

print(f"High drift features: {high_drift_features}")
```

## Verification

### SHAP Reconstruction Verification

Verify that SHAP values correctly reconstruct model predictions:

```python
from shap_analytics import verify_shap_reconstruction

is_valid = verify_shap_reconstruction(
    shap_values=shap_values,
    X=X_test,
    model=model,
    sample_index=0,
    tolerance=1e-3
)

if is_valid:
    print("✅ SHAP reconstruction verified")
```

## Visualization

### Creating Interactive Dashboards

```python
from shap_analytics import create_interactive_dashboard

dashboard = create_interactive_dashboard(
    shap_values=shap_values,
    X=X_test,
    model=model,
    feature_names=X_test.columns.tolist()
)

# Save dashboard to HTML
dashboard.write_html("shap_dashboard.html")
```

## Model Registry

### Tracking SHAP Explanations

```python
from shap_analytics import SHAPRegistry

registry = SHAPRegistry(storage_path="./shap_registry")

# Register SHAP explanation
registry.register(
    model_name="breast_cancer_rf",
    model_version="v1.0",
    shap_values=shap_values,
    metadata={
        "background_size": 100,
        "n_features": X_train.shape[1],
        "timestamp": "2025-10-31T10:00:00Z"
    }
)

# Retrieve explanation
explanation = registry.get(
    model_name="breast_cancer_rf",
    model_version="v1.0"
)
```

## API Deployment

### Serving SHAP via REST API

```python
from shap_analytics import serve_shap_api
import uvicorn

# Create FastAPI app
app = serve_shap_api(
    model=model,
    X_train=X_train,
    background_size=100
)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:8080/health
```

**Compute SHAP Values**
```bash
curl -X POST http://localhost:8080/explain \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [[1.0, 2.0, 3.0, ...]],
    "background_size": 100
  }'
```

## Utilities

### Model Serialization

```python
from shap_analytics.utils import serialize_model, load_model

# Save model with checksum
serialize_model(model, "models/model.joblib", compute_checksum=True)

# Load model with verification
loaded_model = load_model("models/model.joblib", verify_checksum=True)
```

### JSON I/O

```python
from shap_analytics.utils import save_json, load_json

# Save configuration
config = {"background_size": 100, "threshold": 0.2}
save_json(config, "config/shap_config.json")

# Load configuration
config = load_json("config/shap_config.json")
```

### Structured Logging

```python
from shap_analytics.utils import setup_logger

logger = setup_logger(
    name=__name__,
    log_file="logs/shap_analytics.log",
    level=logging.INFO
)

logger.info("Computing SHAP values", extra={
    "model": "RandomForest",
    "n_samples": len(X_test)
})
```

## Advanced Usage

### Converting SHAP to DataFrame

```python
from shap_analytics.utils import shap_to_dataframe

shap_df = shap_to_dataframe(shap_values, X_test)
print(shap_df.head())

# Compute mean absolute SHAP for feature importance
feature_importance = shap_df.abs().mean().sort_values(ascending=False)
print(feature_importance)
```

### Computing Mean Absolute SHAP

```python
from shap_analytics.utils import compute_mean_abs_shap
import numpy as np

mean_abs_shap = compute_mean_abs_shap(shap_values, axis=0)
feature_importance = dict(zip(X_test.columns, mean_abs_shap))

# Sort by importance
sorted_features = sorted(
    feature_importance.items(),
    key=lambda x: x[1],
    reverse=True
)

print("Top 5 most important features:")
for feature, importance in sorted_features[:5]:
    print(f"  {feature}: {importance:.4f}")
```

## MLflow Integration

```python
import mlflow
from shap_analytics import compute_shap_values

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Compute and log SHAP values
    shap_values = compute_shap_values(model, X_train, X_test)

    # Log SHAP artifacts
    shap_df = shap_to_dataframe(shap_values, X_test)
    shap_df.to_csv("shap_values.csv")
    mlflow.log_artifact("shap_values.csv")
```

## Best Practices

1. **Background Sample Size**: Use 50-100 samples for TreeExplainer
2. **Drift Monitoring**: Run regularly on production data
3. **Verification**: Always verify SHAP reconstruction for critical applications
4. **Caching**: Use Redis for API deployment to cache explanations
5. **Logging**: Enable structured JSON logging for production systems
6. **Type Checking**: Run `mypy --strict` to catch type errors early

import os
import json
import time
import logging
import tempfile
import joblib
import warnings
import threading
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def retrain_model_periodically(model_class, X_train: pd.DataFrame, y_train: pd.Series,
                               monitor_func, drift_threshold: float = 0.2,
                               retrain_interval_hours: int = 24) -> None:
    """
    Automatically retrain model when drift exceeds threshold or at fixed intervals.

    Args:
        model_class: Scikit-learn compatible model class or callable.
        X_train: Training data.
        y_train: Target vector.
        monitor_func: Function returning drift score between 0 and 1.
        drift_threshold: Drift level to trigger retraining.
        retrain_interval_hours: Minimum time between retrains.
    """
    last_retrain = 0.0
    while True:
        drift_score = monitor_func(X_train, X_train.sample(frac=0.2, random_state=42))
        now = time.time()
        if drift_score > drift_threshold or (now - last_retrain) > retrain_interval_hours * 3600:
            print(f"🔁 Retraining triggered (drift={drift_score:.3f}).")
            model = model_class()
            model.fit(X_train, y_train)
            joblib.dump(model, "model_latest.joblib")
            last_retrain = now
        else:
            print(f"Drift={drift_score:.3f} < threshold={drift_threshold}, skipping retrain.")
        time.sleep(3600)  # check hourly


def cache_explanations_locally(shap_values: shap.Explanation,
                               cache_dir: str | Path = tempfile.gettempdir(),
                               ttl_hours: int = 24) -> Path:
    """
    Cache SHAP explanations locally for faster reuse.

    Args:
        shap_values: SHAP Explanation object.
        cache_dir: Directory for temporary cache.
        ttl_hours: Expiration time for cached files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / "shap_cache.pkl"

    # Save cache
    joblib.dump({"timestamp": time.time(), "values": shap_values}, cache_path)

    # Evict old cache
    for f in cache_dir.glob("*.pkl"):
        try:
            data = joblib.load(f)
            if (time.time() - data.get("timestamp", 0)) > ttl_hours * 3600:
                f.unlink()
        except Exception:
            continue

    print(f"✅ Cached SHAP explanations at {cache_path}")
    return cache_path


def enhance_visualizations(shap_values: shap.Explanation, X: pd.DataFrame,
                           output_path: str | Path = "interactive_summary.html") -> None:
    """
    Generate an interactive SHAP summary plot using Plotly and color-blind-safe palette.
    """
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    fig = px.bar(mean_abs, x=mean_abs.values, y=mean_abs.index,
                 orientation="h", title="Feature Importance (Mean |SHAP|)",
                 color=mean_abs.values, color_continuous_scale="Viridis")
    fig.update_layout(yaxis_title="Features", xaxis_title="Mean |SHAP|")
    fig.write_html(output_path)
    print(f"✅ Interactive SHAP summary saved to {output_path}")


def integrate_with_ci_cd(workflow_dir: str = ".github/workflows") -> None:
    """
    Create a GitHub Action workflow to validate SHAP drift after model updates.
    """
    Path(workflow_dir).mkdir(parents=True, exist_ok=True)
    yaml_content = """name: Validate SHAP Drift
on:
  push:
    branches: [ main ]
jobs:
  validate-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install shap pandas numpy scipy matplotlib
      - name: Validate drift
        run: python -m src.validate_drift
"""
    (Path(workflow_dir) / "validate_drift.yml").write_text(yaml_content)
    print(f"✅ CI/CD workflow created at {workflow_dir}/validate_drift.yml")


def improve_error_handling(log_file: str = "logs/shap_app.log") -> logging.Logger:
    """
    Configure structured JSON logging with retries for transient API errors.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shap_logger")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Structured logging initialized.")

    def safe_api_call(func, *args, retries: int = 3, delay: float = 2.0, **kwargs):
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed ({i+1}/{retries}): {e}")
                time.sleep(delay)
        logger.error("API call failed after retries.")
        return None

    logger.safe_api_call = safe_api_call
    print(f"✅ JSON logger configured at {log_file}")
    return logger


def extend_feature_analysis(shap_values: shap.Explanation, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute feature correlation heatmap and SHAP interaction metrics.
    """
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    corr = shap_df.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation")
    plt.title("Feature Correlation Heatmap (SHAP Values)")
    plt.savefig("reports/shap_correlation.png", bbox_inches="tight")
    plt.close()

    # Approximate pairwise SHAP interaction strengths
    distances = pairwise_distances(shap_df.T)
    interactions = pd.DataFrame(1 / (1 + distances), index=X.columns, columns=X.columns)
    interactions.to_csv("reports/shap_interactions.csv")
    print("✅ Feature correlation and interaction metrics computed.")
    return {"corr_path": "reports/shap_correlation.png", "interactions": interactions}


def add_unit_tests(test_dir: str = "tests") -> None:
    """
    Create pytest boilerplate for SHAP helper functions.
    """
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    test_file = Path(test_dir) / "test_shap_helpers.py"
    test_file.write_text(
        """import pytest
import pandas as pd
from src.shap_explain import enhance_visualizations

def test_visualization_runs(tmp_path):
    X = pd.DataFrame([[1,2],[3,4]], columns=["a","b"])
    class Dummy: values = X.values
    enhance_visualizations(Dummy(), X, tmp_path/'viz.html')
    assert (tmp_path/'viz.html').exists()
"""
    )
    print(f"✅ Unit test template created at {test_file}")


def optimize_performance(X: pd.DataFrame, shap_func) -> float:
    """
    Benchmark SHAP computation performance using multiprocessing backend.
    """
    from joblib import Parallel, delayed
    n_samples = min(len(X), 100)
    start = time.time()
    Parallel(n_jobs=4)(delayed(shap_func)(X.iloc[[i]]) for i in range(n_samples))
    elapsed = time.time() - start
    print(f"✅ SHAP performance benchmark: {elapsed:.2f}s for {n_samples} samples")
    return elapsed


def document_configuration_schema(output_path: str = "docs/config_schema.md") -> None:
    """
    Auto-generate Markdown documentation for configuration schema.
    """
    config_schema = {
        "MODEL_PATH": "Path to serialized model",
        "DRIFT_THRESHOLD": "Drift value to trigger retraining",
        "CACHE_TTL_HOURS": "Expiration time for cached SHAP values",
        "LOG_LEVEL": "Logging verbosity (INFO, DEBUG, WARNING)",
    }

    lines = ["# Configuration Schema\n"]
    for key, desc in config_schema.items():
        lines.append(f"- **{key}** — {desc}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    print(f"✅ Configuration schema documented in {output_path}")


def prepare_release_automation() -> None:
    """
    Prepare semantic-release configuration and changelog template.
    """
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)

    semantic_yml = """name: Semantic Release
on:
  push:
    branches: [ main ]
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Semantic Release
        uses: cycjimmy/semantic-release-action@v4
        with:
          branches: main
"""
    (workflow_dir / "semantic_release.yml").write_text(semantic_yml)
    Path("CHANGELOG.md").write_text("# Change Log\n\nAll notable changes will be documented here.\n")
    print("✅ Semantic-release workflow and changelog prepared.")

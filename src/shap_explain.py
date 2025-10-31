"""
SHAP Classification Example
Python >= 3.10
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit


def main() -> None:
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.3f}")

    # Background sample for baseline
    background = shap.utils.sample(X_train, 100, random_state=42)
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = explainer(X_test)

    print("Expected value (base prediction):", explainer.expected_value)
    mean_prob = model.predict_proba(X_train)[:, 1].mean()
    print("Mean predicted probability for malignant:", mean_prob)

    # Explain single prediction
    i = 0
    shap_values_class1 = shap_values[..., 1]
    proba = model.predict_proba(X_test.iloc[i:i+1])[0, 1]
    print(f"Predicted probability for malignant: {proba:.3f}")

    shap.plots.waterfall(shap_values_class1[i])

    # Global summary plot
    shap.summary_plot(shap_values_class1, X_test, feature_names=X_test.columns)

    # Dependence plot
    shap.dependence_plot("mean radius", shap_values_class1.values, X_test)

    # Verify probability reconstruction
    log_odds = explainer.expected_value[1] + shap_values_class1[i].values.sum()
    prob_from_shap = expit(log_odds)
    print("Reconstructed probability:", prob_from_shap)
    
    
def export_shap_dataframe(shap_values: shap.Explanation, X: pd.DataFrame, output_path: str) -> None:
    """
    Export SHAP values with corresponding feature data to a CSV file.

    Args:
        shap_values (shap.Explanation): SHAP explanation object from TreeExplainer.
        X (pd.DataFrame): DataFrame containing features corresponding to shap_values.
        output_path (str): File path where the CSV file will be saved.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not hasattr(shap_values, "values"):
        raise ValueError("Invalid SHAP object provided — must contain 'values' attribute.")

    # Create a DataFrame with one row per (sample, feature)
    df = pd.DataFrame(shap_values.values, columns=X.columns)
    df.insert(0, "sample_index", np.arange(len(X)))

    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ SHAP values exported to {output_path}")


def generate_html_report(shap_values: shap.Explanation, X: pd.DataFrame, output_path: str) -> None:
    """
    Generate a static HTML report summarizing SHAP results.

    Args:
        shap_values (shap.Explanation): Computed SHAP values.
        X (pd.DataFrame): Feature matrix used for SHAP computation.
        output_path (str): Output HTML file path.
    """
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Create global summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    summary_png = output.with_suffix(".png")
    plt.savefig(summary_png)
    plt.close()

    html_content = f"""
    <html>
    <head><title>SHAP Summary Report</title></head>
    <body>
        <h1>SHAP Feature Importance Summary</h1>
        <p>This report shows the global importance of features derived from SHAP values.</p>
        <img src="{summary_png.name}" alt="SHAP Summary Plot">
    </body>
    </html>
    """
    output.write_text(html_content, encoding="utf-8")
    print(f"✅ HTML report created at {output_path}")


def validate_background_sample(X_train: pd.DataFrame, sample_size: int = 100) -> None:
    """
    Validate that the background sample is statistically representative of the training data.

    Args:
        X_train (pd.DataFrame): Training feature set.
        sample_size (int): Number of samples to use for validation.
    """
    background = X_train.sample(min(sample_size, len(X_train)), random_state=42)
    diffs = (X_train.mean() - background.mean()).abs() / (X_train.std() + 1e-9)
    exceeded = diffs[diffs > 0.1]

    if not exceeded.empty:
        warnings.warn(f"⚠️ Background sample deviates significantly for features: {', '.join(exceeded.index)}")
    else:
        print("✅ Background sample validated successfully — no significant drift detected.")


def log_shap_summary_to_cloud(shap_summary: Dict[str, Any], service: str = "s3") -> None:
    """
    Upload SHAP summary statistics to a cloud storage service.

    Args:
        shap_summary (dict): SHAP summary data to upload.
        service (str): Cloud backend, 's3' or 'gcs'.
    """
    if service == "s3":
        import boto3

        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise EnvironmentError("S3_BUCKET_NAME environment variable not set")

        s3 = boto3.client("s3")
        key = f"shap_reports/summary_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(shap_summary))
        print(f"✅ Uploaded SHAP summary to s3://{bucket}/{key}")

    elif service == "gcs":
        from google.cloud import storage

        client = storage.Client()
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise EnvironmentError("GCS_BUCKET_NAME environment variable not set")

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"shap_reports/summary_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}.json")
        blob.upload_from_string(json.dumps(shap_summary))
        print(f"✅ Uploaded SHAP summary to gs://{bucket_name}/{blob.name}")

    else:
        raise ValueError("Unsupported service: choose either 's3' or 'gcs'")


def monitor_feature_drift(X_train: pd.DataFrame, X_new: pd.DataFrame) -> Dict[str, float]:
    """
    Compare feature distributions between training and new data using Jensen–Shannon divergence.

    Args:
        X_train (pd.DataFrame): Training feature data.
        X_new (pd.DataFrame): New feature data.

    Returns:
        Dict[str, float]: Per-feature drift score (0–1).
    """
    from scipy.spatial.distance import jensenshannon

    drift_scores = {}
    for col in X_train.columns:
        p, _ = np.histogram(X_train[col], bins=20, density=True)
        q, _ = np.histogram(X_new[col], bins=20, density=True)
        p = np.clip(p, 1e-12, None)
        q = np.clip(q, 1e-12, None)
        drift_scores[col] = float(jensenshannon(p, q))

    print("✅ Feature drift monitoring complete.")
    return drift_scores


def schedule_weekly_explanation_update() -> None:
    """
    Schedule weekly recomputation of SHAP explanations.
    This is a placeholder for integration with CI/CD or cron-based automation.
    """
    import subprocess

    try:
        subprocess.run(["echo", "Recomputing SHAP explanations..."], check=True)
        print("✅ Weekly SHAP recomputation simulated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to schedule weekly update: {e}")


# ============================================================
# FUTURE ENHANCEMENTS AND DEVELOPMENT TASKS
# These tags will be detected by the Create issues from TODOs Action
# ============================================================


def retrain_model_periodically() -> None:
    """Schedule retraining of the model when new data is available."""
    # TODO: Implement automatic retraining pipeline integrated with SHAP refresh
    # TODO: Add threshold-based trigger when drift exceeds 0.2 in monitor_feature_drift()
    # FIXME: Handle large dataset retraining without exhausting Lambda memory
    pass


def cache_explanations_locally() -> None:
    """Introduce caching layer for SHAP explanations."""
    # TODO: Cache SHAP explanations in /tmp or Redis for faster debugging
    # NOTE: Evaluate trade-offs between cache size and recomputation speed
    # TODO: Integrate TTL eviction policy for cached explanations
    pass


def enhance_visualizations() -> None:
    """Improve visual reports for SHAP explanations."""
    # TODO: Add interactive Plotly version of summary plot
    # TODO: Support color-blind-safe palettes and theme toggles
    # NOTE: Consider embedding these visualizations in Streamlit dashboards
    pass


def integrate_with_ci_cd() -> None:
    """Hook SHAP validation into CI/CD pipeline."""
    # TODO: Add GitHub Action to validate SHAP drift on each model update
    # HACK: Temporary skip for unstable model versions (< 1.0.0)
    # TODO: Upload plots as workflow artifacts for team review
    pass


def improve_error_handling() -> None:
    """Add robust exception handling and structured logging."""
    # TODO: Integrate Python's logging config with json-based formatter
    # FIXME: Some cloud logging backends still truncate SHAP payloads over 1 MB
    # TODO: Add retry logic for transient API errors in log_shap_summary_to_cloud()
    pass


def extend_feature_analysis() -> None:
    """Add more advanced statistical diagnostics."""
    # TODO: Implement feature correlation heatmap among top SHAP features
    # TODO: Compute pairwise SHAP interaction values
    # BUG: Current version mislabels correlated categorical variables
    # NOTE: Compare with partial dependence plots for consistency
    pass


def add_unit_tests() -> None:
    """Create comprehensive unit and integration tests."""
    # TODO: Write pytest suite for all SHAP helper functions
    # TODO: Mock AWS/GCS uploads in CI environment
    # NOTE: Add coverage badge in README after tests are stable
    pass


def optimize_performance() -> None:
    """Profile and optimize performance bottlenecks."""
    # TODO: Profile SHAP computation time vs. sample size
    # TODO: Introduce multiprocessing or joblib parallel backend
    # FIXME: Some SHAP plots block matplotlib backend on macOS CI
    pass


def document_configuration_schema() -> None:
    """Document configuration files and environment variables."""
    # TODO: Create config schema (pydantic or dataclasses) for runtime params
    # TODO: Auto-generate Markdown docs from schema fields
    # NOTE: Sync default config with pyproject.toml for consistency
    pass


def prepare_release_automation() -> None:
    """Set up semantic-release for automated versioning."""
    # TODO: Add semantic-release workflow and conventional commits enforcement
    # TODO: Generate CHANGELOG.md automatically
    # NOTE: Coordinate with TODO Action to include closed items in changelog
    pass

if __name__ == "__main__":
    main()

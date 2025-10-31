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

if __name__ == "__main__":
    main()

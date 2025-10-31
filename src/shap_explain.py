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
    
    
# ============================================================
# FUTURE DEVELOPMENT PLACEHOLDERS – AUTO-DETECTED BY GITHUB ACTION
# ============================================================

def export_shap_dataframe(shap_values, X, output_path: str) -> None:
    """
    Export SHAP values with corresponding features to a CSV file.
    Useful for auditing or offline inspection.

    Args:
        shap_values: SHAP values array or Explanation object.
        X: Input features used for computing SHAP values.
        output_path: File path where the CSV will be saved.
    """
    # TODO: Implement this export function (convert shap_values to a DataFrame)
    # Expected columns: ["feature", "shap_value", "sample_index"]
    # Add CLI flag in the future to enable automatic export
    pass


def generate_html_report(shap_values, X, output_path: str) -> None:
    """
    Generate a simple HTML report summarizing SHAP insights.
    """
    # TODO: Build a static HTML summary with SHAP plots and key statistics
    # Possibly use Jinja2 templates for styling
    # Add markdown export for ReadTheDocs compatibility
    pass


def validate_background_sample(X_train, sample_size: int = 100) -> None:
    """
    Validate that the background sample is representative enough.
    """
    # NOTE: Implement statistical checks (mean/std/quantiles) vs. full training data
    # If the difference exceeds tolerance, raise a warning or log
    pass


def log_shap_summary_to_cloud(shap_summary: dict, service: str = "s3") -> None:
    """
    Upload SHAP summary metrics to cloud storage for monitoring drift.
    """
    # FIXME: Currently unimplemented. Add AWS S3 and GCS integrations.
    # Should serialize shap_summary as JSON and upload with timestamp.
    # Consider using boto3 (AWS) or google-cloud-storage clients.
    pass


def monitor_feature_drift(X_train, X_new) -> None:
    """
    Compare feature distributions between training data and new incoming data.

    Returns:
        dict with drift statistics per feature.
    """
    # TODO: Add feature drift computation (e.g., Jensen-Shannon divergence)
    # Integration point with SHAP drift explanation module
    pass


def schedule_weekly_explanation_update() -> None:
    """
    Placeholder for scheduling weekly SHAP recomputation.
    """
    # HACK: Use this only after adding persistent model storage
    # Should trigger re-explanations automatically once pipelines are ready
    pass


if __name__ == "__main__":
    main()

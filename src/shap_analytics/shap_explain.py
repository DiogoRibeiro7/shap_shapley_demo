"""
SHAP Computation and Core Explanation Logic.

This module handles the core SHAP computation, model explanation generation,
and validation of SHAP values. It focuses purely on the computation logic
without visualization or deployment concerns.

Python >= 3.10
"""

import warnings

from typing import Any

import numpy as np
import pandas as pd
import shap

from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .utils.common import (
    compute_jensen_shannon_divergence,
    setup_logger,
)

__all__ = [
    "compute_shap_values",
    "main",
    "monitor_feature_drift",
    "validate_background_sample",
    "verify_shap_reconstruction",
]


# Module-level logger
logger = setup_logger(__name__)


def main() -> None:
    """
    Run SHAP classification example on breast cancer dataset.

    This demonstrates:
    - Loading and splitting data
    - Training a RandomForest classifier
    - Computing SHAP values using TreeExplainer
    - Validating probability reconstruction
    - Generating basic SHAP visualizations
    """
    logger.info("Starting SHAP classification example")

    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    # Compute SHAP values
    shap_values = compute_shap_values(model, X_train, X_test)
    logger.info("SHAP values computed successfully")

    # Validate background sample
    validate_background_sample(X_train, sample_size=100)

    # Explain single prediction
    i = 0
    shap_values_class1 = shap_values[..., 1]
    proba = model.predict_proba(X_test.iloc[i:i+1])[0, 1]
    logger.info(f"Sample {i} predicted probability: {proba:.3f}")
    print(f"Predicted probability for malignant: {proba:.3f}")

    # Verify probability reconstruction
    verify_shap_reconstruction(shap_values, X_test, model, sample_index=i)

    # Generate SHAP plots
    logger.info("Generating SHAP visualizations")
    shap.plots.waterfall(shap_values_class1[i])
    shap.summary_plot(shap_values_class1, X_test, feature_names=X_test.columns)
    shap.dependence_plot("mean radius", shap_values_class1.values, X_test)

    logger.info("SHAP analysis completed successfully")


def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    background_size: int = 100,
    random_state: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer with background data sampling.

    Args:
        model: Trained tree-based model (RandomForest, XGBoost, etc.).
        X_train: Training features used to sample background data.
        X_test: Test features for which to compute SHAP values.
        background_size: Number of samples to use for background distribution.
        random_state: Random seed for reproducibility.

    Returns:
        SHAP Explanation object containing computed values.

    Raises:
        ValueError: If model is not compatible with TreeExplainer.
    """
    logger.info(f"Computing SHAP values with background size={background_size}")

    # Background sample for baseline
    background = shap.utils.sample(X_train, background_size, random_state=random_state)
    logger.debug(f"Background sample shape: {background.shape}")

    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
        shap_values = explainer(X_test)
    except Exception as e:
        logger.error(f"Failed to compute SHAP values: {e}")
        raise ValueError(f"SHAP computation failed: {e}") from e

    # Log expected value
    expected_value = explainer.expected_value
    logger.info(f"Expected value (base prediction): {expected_value}")
    print("Expected value (base prediction):", expected_value)

    # Log mean predicted probability
    if hasattr(model, 'predict_proba'):
        mean_prob = model.predict_proba(X_train)[:, 1].mean()
        logger.info(f"Mean predicted probability (class 1): {mean_prob:.3f}")
        print("Mean predicted probability for malignant:", mean_prob)

    return shap_values


def validate_background_sample(
    X_train: pd.DataFrame,
    sample_size: int = 100,
    threshold: float = 0.1,
    random_state: int = 42,
) -> bool:
    """
    Validate that background sample is statistically representative.

    Checks if the background sample has similar mean and standard deviation
    to the full training data for all features.

    Args:
        X_train: Training feature set.
        sample_size: Number of samples to use for validation.
        threshold: Maximum allowed normalized difference in means.
        random_state: Random seed for reproducibility.

    Returns:
        True if validation passes, False otherwise.

    Warns:
        UserWarning: If background sample deviates significantly.
    """
    logger.info(f"Validating background sample (size={sample_size}, threshold={threshold})")

    sample_size = min(sample_size, len(X_train))
    background = X_train.sample(sample_size, random_state=random_state)

    # Compute normalized difference in means
    diffs = (X_train.mean() - background.mean()).abs() / (X_train.std() + 1e-9)
    exceeded = diffs[diffs > threshold]

    if not exceeded.empty:
        warning_msg = (
            f"Background sample deviates significantly for {len(exceeded)} features: "
            f"{', '.join(exceeded.index)}"
        )
        logger.warning(warning_msg)
        warnings.warn(f"⚠️ {warning_msg}", stacklevel=2)
        return False
    else:
        logger.info("Background sample validated successfully")
        print("✅ Background sample validated successfully — no significant drift detected.")
        return True


def monitor_feature_drift(
    X_train: pd.DataFrame,
    X_new: pd.DataFrame,
    threshold: float = 0.2,
    bins: int = 20,
) -> dict[str, float]:
    """
    Monitor feature drift between training and new data using Jensen-Shannon divergence.

    This function compares feature distributions to detect data drift that might
    affect SHAP value reliability.

    Args:
        X_train: Training feature data (reference distribution).
        X_new: New feature data (current distribution).
        threshold: Drift threshold for alerting (0-1 scale).
        bins: Number of histogram bins for distribution estimation.

    Returns:
        Dictionary mapping feature names to drift scores (0-1).

    Example:
        >>> drift_scores = monitor_feature_drift(X_train, X_new, threshold=0.2)
        >>> high_drift = {k: v for k, v in drift_scores.items() if v > 0.2}
    """
    logger.info(f"Monitoring feature drift (threshold={threshold}, bins={bins})")

    drift_scores = {}
    for col in X_train.columns:
        drift_score = compute_jensen_shannon_divergence(
            X_train[col],
            X_new[col],
            bins=bins
        )
        drift_scores[col] = drift_score

        if drift_score > threshold:
            logger.warning(f"High drift detected in feature '{col}': {drift_score:.3f}")

    # Summary statistics
    high_drift_count = sum(1 for v in drift_scores.values() if v > threshold)
    avg_drift = np.mean(list(drift_scores.values()))

    logger.info(
        f"Drift monitoring complete: {high_drift_count}/{len(drift_scores)} features "
        f"above threshold, avg drift={avg_drift:.3f}"
    )
    print("✅ Feature drift monitoring complete.")

    return drift_scores


def verify_shap_reconstruction(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    model: Any,
    sample_index: int = 0,
    tolerance: float = 1e-3,
) -> bool:
    """
    Verify SHAP values correctly reconstruct model predictions.

    For tree models with log-odds output, verifies that:
    base_value + sum(shap_values) = model_prediction

    Args:
        shap_values: SHAP Explanation object.
        X: Feature data.
        model: Trained model.
        sample_index: Index of sample to verify.
        tolerance: Maximum allowed reconstruction error.

    Returns:
        True if reconstruction is within tolerance, False otherwise.

    Raises:
        ValueError: If SHAP values are incompatible with the model.
    """
    logger.info(f"Verifying SHAP reconstruction for sample {sample_index}")

    try:
        # Get model prediction
        if hasattr(model, 'predict_proba'):
            model_proba = model.predict_proba(X.iloc[sample_index:sample_index+1])[0, 1]
        else:
            raise ValueError("Model does not support predict_proba")

        # Get SHAP prediction
        shap_values_class1 = shap_values[..., 1]
        base_value = shap_values.base_values[sample_index][1]
        shap_sum = shap_values_class1[sample_index].values.sum()
        log_odds = base_value + shap_sum
        shap_proba = expit(log_odds)

        # Compute error
        error = abs(model_proba - shap_proba)

        logger.info(f"Model probability: {model_proba:.6f}")
        logger.info(f"SHAP reconstructed probability: {shap_proba:.6f}")
        logger.info(f"Reconstruction error: {error:.6e}")

        print(f"Reconstructed probability: {shap_proba:.6f}")

        if error > tolerance:
            logger.warning(
                f"SHAP reconstruction error {error:.6e} exceeds tolerance {tolerance}"
            )
            return False
        else:
            logger.info("SHAP reconstruction verified successfully")
            return True

    except Exception as e:
        logger.error(f"Failed to verify SHAP reconstruction: {e}")
        raise ValueError(f"SHAP verification failed: {e}") from e


if __name__ == "__main__":
    main()

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


if __name__ == "__main__":
    main()

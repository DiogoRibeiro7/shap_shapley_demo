"""
SHAP Analytics - Professional SHAP Value Computation and Analysis.

A comprehensive library for computing, analyzing, visualizing, and deploying
SHAP (SHapley Additive exPlanations) values in production ML systems.

Main modules:
- shap_explain: Core SHAP computation and validation
- shap_expansion: Advanced utilities (visualization, drift monitoring, API)
- shap_future: Experimental features and upcoming functionality
- shap_backlog: Feature roadmap and planned enhancements

Example:
    >>> from shap_analytics import compute_shap_values
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import pandas as pd
    >>>
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> shap_values = compute_shap_values(model, X_train, X_test)

Minimum Python version: 3.10
"""

__version__ = "0.1.0"
__author__ = "SHAP Analytics Contributors"
__license__ = "MIT"

from .shap_explain import (
    compute_shap_values,
    monitor_feature_drift,
    validate_background_sample,
    verify_shap_reconstruction,
)
from .utils.common import (
    compute_mean_abs_shap,
    ensure_directory,
    get_timestamp,
    load_json,
    load_model,
    save_json,
    serialize_model,
    setup_logger,
    shap_to_dataframe,
)

__all__ = [
    "__author__",
    "__license__",
    "__version__",
    "compute_mean_abs_shap",
    "compute_shap_values",
    "ensure_directory",
    "get_timestamp",
    "load_json",
    "load_model",
    "monitor_feature_drift",
    "save_json",
    "serialize_model",
    "setup_logger",
    "shap_to_dataframe",
    "validate_background_sample",
    "verify_shap_reconstruction",
]

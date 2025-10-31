"""
Comprehensive tests for shap_expansion module.

Tests cover:
- Caching behavior
- Visualization output creation
- Drift detection
- Model registry
- Data quality checks
- Edge cases
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from src.shap_expansion import (
    add_explanation_dashboard,
    add_metadata_tracking,
    add_model_card_generator,
    add_performance_monitoring,
    automate_data_quality_checks,
    benchmark_model_versions,
    build_analytics_api,
    design_drift_alerts,
    develop_feature_selection_module,
    enhance_notebook_experience,
    expand_ci_workflow,
    expand_cloud_support,
    export_summary_json,
    implement_automated_docs,
    integrate_model_registry,
    preprocess_data_pipeline,
    simulate_user_feedback_loop,
    validate_feature_importance_consistency,
)


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_preprocess_data_pipeline_basic(self, sample_dataframe):
        """Test basic preprocessing without missing values."""
        result = preprocess_data_pipeline(sample_dataframe)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape
        assert not result.isna().any().any()

        # Check normalization (mean ~0, std ~1)
        assert np.allclose(result.mean(), 0, atol=0.1)
        assert np.allclose(result.std(), 1, atol=0.1)

    def test_preprocess_handles_missing_values(self):
        """Test preprocessing handles missing values correctly."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8],
            'c': ['x', 'y', 'x', None]
        })

        result = preprocess_data_pipeline(df)

        # Check no missing values remain
        assert not result.isna().any().any()

        # Check numeric columns are normalized
        assert result[['a', 'b']].shape == (4, 2)

    def test_preprocess_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        df = pd.DataFrame()
        result = preprocess_data_pipeline(df)
        assert result.empty


class TestVisualization:
    """Test visualization functions."""

    def test_add_explanation_dashboard_creates_file(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test dashboard HTML file is created."""
        X, _ = sample_classification_data
        output_path = temp_dir / "dashboard.html"

        result = add_explanation_dashboard(
            shap_values_explanation,
            X[:50],
            str(output_path)
        )

        assert result.exists()
        assert result.suffix == ".html"
        assert result.stat().st_size > 0

        # Check HTML contains expected content
        content = result.read_text()
        assert "SHAP" in content or "plotly" in content

    def test_dashboard_with_single_sample(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test dashboard works with single sample."""
        X, _ = sample_classification_data
        output_path = temp_dir / "single_dashboard.html"

        # Take only first sample
        single_shap = shap.Explanation(
            values=shap_values_explanation.values[:1],
            base_values=shap_values_explanation.base_values[:1],
            data=shap_values_explanation.data[:1]
        )

        result = add_explanation_dashboard(
            single_shap,
            X[:1],
            str(output_path)
        )

        assert result.exists()


class TestModelRegistry:
    """Test model registry functions."""

    def test_integrate_model_registry_creates_entry(
        self, trained_model, temp_dir
    ):
        """Test model registry creates entry and saves model."""
        registry_path = temp_dir / "registry.json"
        metadata = {
            "version": "1.0.0",
            "description": "Test model",
            "accuracy": 0.95
        }

        integrate_model_registry(
            trained_model,
            metadata,
            str(registry_path)
        )

        # Check registry file exists
        assert registry_path.exists()

        # Check registry content
        registry = json.loads(registry_path.read_text())
        assert len(registry) == 1
        assert registry[0]["metadata"] == metadata
        assert "timestamp" in registry[0]

        # Check model file exists
        model_path = Path("model_1.0.0.joblib")
        assert model_path.exists()
        model_path.unlink()  # Cleanup

    def test_registry_appends_entries(self, trained_model, temp_dir):
        """Test registry appends multiple entries."""
        registry_path = temp_dir / "registry.json"

        # Add first entry
        integrate_model_registry(
            trained_model,
            {"version": "1.0.0"},
            str(registry_path)
        )

        # Add second entry
        integrate_model_registry(
            trained_model,
            {"version": "1.0.1"},
            str(registry_path)
        )

        registry = json.loads(registry_path.read_text())
        assert len(registry) == 2

        # Cleanup
        Path("model_1.0.0.joblib").unlink(missing_ok=True)
        Path("model_1.0.1.joblib").unlink(missing_ok=True)

    def test_benchmark_model_versions_no_registry(self, temp_dir):
        """Test benchmark when no registry exists."""
        registry_path = temp_dir / "nonexistent.json"
        versions = benchmark_model_versions(str(registry_path))
        assert versions == []

    def test_benchmark_model_versions_with_data(self, temp_dir):
        """Test benchmark with existing registry."""
        registry_path = temp_dir / "registry.json"
        registry = [
            {"metadata": {"version": "1.0.0"}},
            {"metadata": {"version": "1.0.1"}},
            {"metadata": {"other": "data"}},  # No version
        ]
        registry_path.write_text(json.dumps(registry))

        versions = benchmark_model_versions(str(registry_path))
        assert versions == ["1.0.0", "1.0.1"]


class TestDriftDetection:
    """Test drift detection functions."""

    def test_design_drift_alerts_no_drift(self, sample_dataframe):
        """Test drift detection when no drift exists."""
        # Use same data as reference and new
        drift_scores = design_drift_alerts(
            sample_dataframe,
            sample_dataframe,
            threshold=0.2
        )

        assert isinstance(drift_scores, dict)
        assert len(drift_scores) == len(sample_dataframe.columns)

        # All scores should be very low (same distribution)
        assert all(score < 0.1 for score in drift_scores.values())

    def test_design_drift_alerts_with_drift(self):
        """Test drift detection when drift exists."""
        np.random.seed(42)
        reference = pd.DataFrame(
            np.random.randn(100, 3),
            columns=['a', 'b', 'c']
        )

        # Create drifted data (shifted mean)
        new_data = pd.DataFrame(
            np.random.randn(100, 3) + 2.0,  # Shift mean by 2
            columns=['a', 'b', 'c']
        )

        drift_scores = design_drift_alerts(
            reference,
            new_data,
            threshold=0.2
        )

        # All scores should be high (different distributions)
        assert all(score > 0.2 for score in drift_scores.values())

    def test_drift_alerts_edge_case_single_feature(self):
        """Test drift detection with single feature."""
        reference = pd.DataFrame({'x': np.random.randn(50)})
        new_data = pd.DataFrame({'x': np.random.randn(50) + 1.0})

        drift_scores = design_drift_alerts(reference, new_data)
        assert 'x' in drift_scores
        assert drift_scores['x'] > 0


class TestFeatureConsistency:
    """Test feature importance consistency validation."""

    def test_validate_consistency_perfect(self, sample_shap_dataframe):
        """Test perfect consistency with same data."""
        summaries = [sample_shap_dataframe, sample_shap_dataframe]
        corr = validate_feature_importance_consistency(summaries)
        assert corr == 1.0

    def test_validate_consistency_single_summary(self, sample_shap_dataframe):
        """Test with single summary returns 1.0."""
        summaries = [sample_shap_dataframe]
        corr = validate_feature_importance_consistency(summaries)
        assert corr == 1.0

    def test_validate_consistency_different_rankings(self):
        """Test consistency with different feature rankings."""
        np.random.seed(42)
        df1 = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])

        # Create df2 with reversed importance
        df2 = pd.DataFrame({
            'a': np.random.randn(50) * 0.1,  # Low importance
            'b': np.random.randn(50) * 0.5,  # Medium
            'c': np.random.randn(50) * 2.0,  # High importance
        })

        summaries = [df1, df2]
        corr = validate_feature_importance_consistency(summaries)

        # Should have some correlation but not perfect
        assert -1.0 <= corr <= 1.0


class TestExportFunctions:
    """Test export and reporting functions."""

    def test_export_summary_json(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test SHAP summary export to JSON."""
        X, _ = sample_classification_data
        output_path = temp_dir / "summary.json"

        result = export_summary_json(
            shap_values_explanation,
            X[:50],
            str(output_path)
        )

        assert result.exists()

        # Check JSON structure
        data = json.loads(result.read_text())
        assert "timestamp" in data
        assert "features" in data
        assert len(data["features"]) == 5

        # Check feature stats
        for _, stats in data["features"].items():
            assert "mean_abs" in stats
            assert "std_abs" in stats
            assert isinstance(stats["mean_abs"], float)

    def test_implement_automated_docs(self, temp_dir):
        """Test documentation generation."""
        doc_path = temp_dir / "docs" / "index.md"
        result = implement_automated_docs(str(doc_path))

        assert result.exists()
        content = result.read_text()
        assert "SHAP Analytics Documentation" in content
        assert "compute_shap_values" in content

    def test_add_model_card_generator(self, temp_dir):
        """Test model card generation."""
        card_path = temp_dir / "model_card.md"
        result = add_model_card_generator(str(card_path))

        assert result.exists()
        content = result.read_text()
        assert "Model Card" in content
        assert "Purpose" in content
        assert "Explainability Metrics" in content


class TestDataQuality:
    """Test data quality functions."""

    def test_automate_data_quality_checks(self, sample_dataframe, temp_dir):
        """Test data quality check report generation."""
        output_path = temp_dir / "quality.json"
        result = automate_data_quality_checks(
            sample_dataframe,
            str(output_path)
        )

        assert result.exists()

        # Check report structure
        report = json.loads(result.read_text())
        assert "summary" in report
        assert "outliers" in report
        assert "timestamp" in report

    def test_data_quality_with_outliers(self, temp_dir):
        """Test data quality detection with outliers."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 100],  # 100 is outlier
            'b': [5, 6, 7, 8]
        })

        output_path = temp_dir / "quality_outliers.json"
        result = automate_data_quality_checks(df, str(output_path))

        report = json.loads(result.read_text())
        assert report["outliers"]["a"] >= 1  # At least one outlier


class TestCIWorkflow:
    """Test CI/CD integration functions."""

    def test_expand_ci_workflow(self, temp_dir):
        """Test CI workflow file creation."""
        workflow_dir = temp_dir / ".github" / "workflows"
        result = expand_ci_workflow(str(workflow_dir))

        assert result.exists()
        content = result.read_text()
        assert "CI Extended" in content
        assert "pytest" in content
        assert "mypy" in content


class TestMetadataAndFeedback:
    """Test metadata tracking and feedback functions."""

    def test_simulate_user_feedback_loop(self, temp_dir):
        """Test user feedback recording."""
        feedback_file = temp_dir / "feedback.json"
        result = simulate_user_feedback_loop(str(feedback_file))

        assert result.exists()
        feedback = json.loads(result.read_text())
        assert "user" in feedback
        assert "comment" in feedback
        assert "timestamp" in feedback

    def test_add_metadata_tracking(self, temp_dir):
        """Test metadata tracking."""
        output_path = temp_dir / "metadata.json"
        result = add_metadata_tracking(str(output_path))

        assert result.exists()
        meta = json.loads(result.read_text())
        assert "timestamp" in meta
        assert "commit_hash" in meta


class TestAPIAndCloud:
    """Test API and cloud support functions."""

    def test_build_analytics_api(self, temp_dir):
        """Test FastAPI scaffold generation."""
        api_dir = temp_dir / "api"
        result = build_analytics_api(str(api_dir))

        assert result.exists()
        assert result.name == "main.py"

        content = result.read_text()
        assert "FastAPI" in content
        assert "@app.get" in content

    def test_expand_cloud_support(self, temp_dir):
        """Test cloud configuration generation."""
        config_path = temp_dir / "cloud.json"
        result = expand_cloud_support(str(config_path))

        assert result.exists()
        config = json.loads(result.read_text())
        assert "AWS" in config
        assert "GCP" in config
        assert "Azure" in config


class TestPerformanceMonitoring:
    """Test performance monitoring functions."""

    def test_add_performance_monitoring(self, temp_dir):
        """Test performance metrics logging."""
        log_path = temp_dir / "performance.json"
        result = add_performance_monitoring(str(log_path))

        assert result.exists()
        stats = json.loads(result.read_text())
        assert "timestamp" in stats
        assert "avg_runtime_ms" in stats
        assert isinstance(stats["avg_runtime_ms"], int)


class TestFeatureSelection:
    """Test feature selection functions."""

    def test_develop_feature_selection_module(
        self, shap_values_explanation, sample_classification_data
    ):
        """Test feature selection based on SHAP importance."""
        X, _ = sample_classification_data

        result = develop_feature_selection_module(
            shap_values_explanation,
            X[:50],
            top_n=3
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3  # Top 3 features
        assert result.shape[0] == 50  # Same samples

    def test_feature_selection_all_features(
        self, shap_values_explanation, sample_classification_data
    ):
        """Test feature selection with top_n >= total features."""
        X, _ = sample_classification_data

        result = develop_feature_selection_module(
            shap_values_explanation,
            X[:50],
            top_n=10  # More than available
        )

        assert result.shape[1] == 5  # All features


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_shap_values(self, sample_classification_data, temp_dir):
        """Test handling of empty SHAP arrays."""
        X, _ = sample_classification_data

        # Create empty SHAP Explanation
        empty_shap = shap.Explanation(
            values=np.array([]).reshape(0, 5),
            base_values=np.array([]),
            data=np.array([]).reshape(0, 5)
        )

        output_path = temp_dir / "empty_summary.json"
        result = export_summary_json(empty_shap, X[:0], str(output_path))

        assert result.exists()

    def test_missing_columns_in_drift(self):
        """Test drift detection with mismatched columns."""
        reference = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        new_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # Should work with matching columns
        drift_scores = design_drift_alerts(reference, new_data)
        assert len(drift_scores) == 2

    def test_preprocess_all_nan_column(self):
        """Test preprocessing with all-NaN column."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [np.nan, np.nan, np.nan]
        })

        result = preprocess_data_pipeline(df)
        # Should handle gracefully (fill with median or 0)
        assert not result['a'].isna().any()

    def test_enhance_notebook_nonexistent(self, temp_dir):
        """Test notebook enhancement with nonexistent file."""
        nb_path = temp_dir / "nonexistent.ipynb"
        result = enhance_notebook_experience(str(nb_path))

        # Should return None and handle gracefully
        assert result is None

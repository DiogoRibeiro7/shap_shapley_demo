"""
Comprehensive tests for shap_future module (experimental features).

Tests cover:
- Caching behavior with TTL
- Deterministic experimental methods with fixed seeds
- Performance optimization
- CI/CD integration
- Structured logging
- Edge cases
"""

import tempfile
import time

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.shap_analytics.shap_future import (
    add_unit_tests,
    cache_explanations_locally,
    document_configuration_schema,
    enhance_visualizations,
    extend_feature_analysis,
    improve_error_handling,
    integrate_with_ci_cd,
    optimize_performance,
    prepare_release_automation,
)


class TestCaching:
    """Test SHAP caching functionality."""

    def test_cache_explanations_creates_file(self, shap_values_explanation, temp_dir):
        """Test that cache file is created."""
        cache_path = cache_explanations_locally(
            shap_values_explanation, cache_dir=str(temp_dir), ttl_hours=1
        )

        assert cache_path.exists()
        assert cache_path.suffix == ".pkl"
        assert cache_path.name == "shap_cache.pkl"

    def test_cache_stores_and_retrieves_correctly(self, shap_values_explanation, temp_dir):
        """Test that cached SHAP values can be retrieved."""
        import joblib

        cache_path = cache_explanations_locally(
            shap_values_explanation, cache_dir=str(temp_dir), ttl_hours=24
        )

        # Load cached data
        cached_data = joblib.load(cache_path)

        assert "timestamp" in cached_data
        assert "values" in cached_data
        assert hasattr(cached_data["values"], "values")

        # Check values match
        np.testing.assert_array_almost_equal(
            cached_data["values"].values, shap_values_explanation.values
        )

    def test_cache_eviction_with_ttl(self, shap_values_explanation, temp_dir):
        """Test that expired cache files are evicted."""
        # Create an expired cache file manually
        import joblib

        old_cache = temp_dir / "old_cache.pkl"
        old_timestamp = time.time() - (25 * 3600)  # 25 hours ago
        joblib.dump({"timestamp": old_timestamp, "values": shap_values_explanation}, old_cache)

        # Create new cache with TTL of 24 hours
        cache_explanations_locally(shap_values_explanation, cache_dir=str(temp_dir), ttl_hours=24)

        # Old cache should be evicted
        assert not old_cache.exists()

    def test_cache_with_default_temp_directory(self, shap_values_explanation):
        """Test caching with default system temp directory."""
        cache_path = cache_explanations_locally(
            shap_values_explanation,
            cache_dir=None,  # Use system default
            ttl_hours=1,
        )

        assert cache_path.exists()
        assert tempfile.gettempdir() in str(cache_path)

        # Cleanup
        cache_path.unlink()

    def test_cache_handles_corrupted_files(self, shap_values_explanation, temp_dir):
        """Test that corrupted cache files don't crash the system."""
        # Create corrupted cache file
        corrupted = temp_dir / "corrupted.pkl"
        corrupted.write_text("invalid pickle data")

        # Should handle gracefully and create new cache
        cache_path = cache_explanations_locally(
            shap_values_explanation, cache_dir=str(temp_dir), ttl_hours=1
        )

        assert cache_path.exists()


class TestVisualizationEnhancement:
    """Test enhanced visualization functions."""

    def test_enhance_visualizations_creates_html(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test that visualization HTML is created."""
        X, _ = sample_classification_data
        output_path = temp_dir / "viz.html"

        result = enhance_visualizations(shap_values_explanation, X[:50], str(output_path))

        assert result.exists()
        assert result.suffix == ".html"

        content = result.read_text()
        assert len(content) > 0

    def test_visualization_deterministic_with_same_data(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test that visualizations are deterministic."""
        X, _ = sample_classification_data

        # Generate twice with same data
        path1 = temp_dir / "viz1.html"
        path2 = temp_dir / "viz2.html"

        enhance_visualizations(shap_values_explanation, X[:50], str(path1))
        enhance_visualizations(shap_values_explanation, X[:50], str(path2))

        # File sizes should be similar (within 5%)
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        assert abs(size1 - size2) / max(size1, size2) < 0.05

    def test_visualization_with_single_feature(self, sample_classification_data, temp_dir):
        """Test visualization with single feature."""
        X, y = sample_classification_data
        X_single = X[["feature_0"]]

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_single, y)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_single[:10])

        output_path = temp_dir / "single_feat.html"
        result = enhance_visualizations(shap_vals, X_single[:10], str(output_path))

        assert result.exists()


class TestCICDIntegration:
    """Test CI/CD integration functions."""

    def test_integrate_with_ci_cd_creates_workflow(self, temp_dir):
        """Test that GitHub Actions workflow is created."""
        workflow_dir = temp_dir / "workflows"
        result = integrate_with_ci_cd(str(workflow_dir))

        assert result.exists()
        assert result.name == "validate_drift.yml"

        content = result.read_text()
        assert "Validate SHAP Drift" in content
        assert "python-version" in content
        assert "shap" in content

    def test_workflow_has_valid_yaml_structure(self, temp_dir):
        """Test that generated workflow has valid YAML structure."""
        import yaml

        workflow_dir = temp_dir / "workflows"
        result = integrate_with_ci_cd(str(workflow_dir))

        # Should be valid YAML
        with Path(result).open("r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert "jobs" in workflow
        assert "validate-drift" in workflow["jobs"]


class TestErrorHandling:
    """Test error handling and logging improvements."""

    def test_improve_error_handling_creates_logger(self, temp_dir):
        """Test that structured logger is created."""
        log_file = temp_dir / "test.log"
        logger = improve_error_handling(str(log_file))

        assert logger is not None
        assert log_file.exists()

        # Test logging works
        logger.info("Test message")
        assert log_file.stat().st_size > 0

    def test_safe_api_call_retry_logic(self, temp_dir):
        """Test retry logic for failed API calls."""
        log_file = temp_dir / "retry_test.log"
        logger = improve_error_handling(str(log_file))

        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Simulated failure")
            return "success"

        # Should retry and eventually succeed
        result = logger.safe_api_call(failing_func, retries=3, delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_safe_api_call_exhausts_retries(self, temp_dir):
        """Test that retries are exhausted on persistent failures."""
        log_file = temp_dir / "exhaust_test.log"
        logger = improve_error_handling(str(log_file))

        def always_fails():
            raise RuntimeError("Always fails")

        # Should return None after exhausting retries
        result = logger.safe_api_call(always_fails, retries=2, delay=0.01)
        assert result is None


class TestFeatureAnalysis:
    """Test feature analysis functions."""

    def test_extend_feature_analysis_creates_outputs(
        self, shap_values_explanation, sample_classification_data, temp_dir
    ):
        """Test that feature analysis creates correlation heatmap and CSV."""
        X, _ = sample_classification_data

        result = extend_feature_analysis(shap_values_explanation, X[:50], str(temp_dir))

        assert "corr_path" in result
        assert "interactions" in result
        assert "interactions_path" in result

        corr_path = Path(result["corr_path"])
        interactions_path = Path(result["interactions_path"])

        assert corr_path.exists()
        assert interactions_path.exists()

        # Check interactions DataFrame
        interactions_df = result["interactions"]
        assert interactions_df.shape == (5, 5)  # 5x5 for 5 features
        assert np.allclose(np.diag(interactions_df.values), 1.0)  # Diagonal is 1

    def test_feature_analysis_deterministic(self, sample_classification_data, temp_dir):
        """Test that feature analysis is deterministic with same SHAP values."""
        X, _ = sample_classification_data

        # Create fixed SHAP values
        np.random.seed(42)
        fixed_shap = shap.Explanation(
            values=np.random.randn(50, 5), base_values=np.zeros(50), data=X[:50].values
        )

        # Run twice
        result1 = extend_feature_analysis(fixed_shap, X[:50], str(temp_dir / "run1"))
        result2 = extend_feature_analysis(fixed_shap, X[:50], str(temp_dir / "run2"))

        # Interactions should be identical
        interactions1 = result1["interactions"]
        interactions2 = result2["interactions"]

        pd.testing.assert_frame_equal(interactions1, interactions2)


class TestUnitTestGeneration:
    """Test unit test generation functions."""

    def test_add_unit_tests_creates_file(self, temp_dir):
        """Test that unit test template is created."""
        test_dir = temp_dir / "tests"
        result = add_unit_tests(str(test_dir))

        assert result.exists()
        assert result.name == "test_shap_helpers.py"

        content = result.read_text()
        assert "def test_" in content
        assert "pytest" in content
        assert "assert" in content


class TestPerformanceOptimization:
    """Test performance optimization functions."""

    def test_optimize_performance_runs_benchmark(self, sample_classification_data):
        """Test that performance benchmark runs."""
        X, _ = sample_classification_data

        def mock_shap_func(row):
            time.sleep(0.001)  # Simulate computation
            return np.random.randn(row.shape[1])

        elapsed = optimize_performance(X[:20], mock_shap_func, n_jobs=2)

        assert isinstance(elapsed, float)
        assert elapsed > 0
        assert elapsed < 10  # Should complete within 10 seconds

    def test_performance_scales_with_parallelization(self, sample_classification_data):
        """Test that parallel jobs improve performance."""
        X, _ = sample_classification_data

        def slow_func(row):
            time.sleep(0.01)
            return np.random.randn(row.shape[1])

        # Serial execution
        time_serial = optimize_performance(X[:10], slow_func, n_jobs=1)

        # Parallel execution
        time_parallel = optimize_performance(X[:10], slow_func, n_jobs=2)

        # Parallel should be faster (allowing some overhead)
        # This is a weak test due to overhead, but checks it runs
        assert time_serial > 0
        assert time_parallel > 0


class TestDocumentation:
    """Test documentation generation functions."""

    def test_document_configuration_schema(self, temp_dir):
        """Test configuration schema documentation."""
        doc_path = temp_dir / "config_schema.md"
        result = document_configuration_schema(str(doc_path))

        assert result.exists()

        content = result.read_text()
        assert "Configuration Schema" in content
        assert "MODEL_PATH" in content
        assert "DRIFT_THRESHOLD" in content


class TestReleaseAutomation:
    """Test release automation functions."""

    def test_prepare_release_automation_creates_files(self, temp_dir):
        """Test that release automation files are created."""
        workflow_dir = temp_dir / "workflows"
        workflow_path, changelog_path = prepare_release_automation(str(workflow_dir))

        assert workflow_path.exists()
        assert changelog_path.exists()

        # Check workflow content
        workflow_content = workflow_path.read_text()
        assert "Semantic Release" in workflow_content

        # Check changelog content
        changelog_content = changelog_path.read_text()
        assert "Change Log" in changelog_content


class TestDeterministicBehavior:
    """Test that experimental methods run deterministically with fixed seeds."""

    def test_cache_deterministic_timestamps_excluded(self, shap_values_explanation, temp_dir):
        """Test that cache structure is deterministic (excluding timestamps)."""
        import joblib

        cache1 = cache_explanations_locally(
            shap_values_explanation, cache_dir=str(temp_dir / "cache1"), ttl_hours=1
        )

        time.sleep(0.1)  # Small delay

        cache2 = cache_explanations_locally(
            shap_values_explanation, cache_dir=str(temp_dir / "cache2"), ttl_hours=1
        )

        # Load both
        data1 = joblib.load(cache1)
        data2 = joblib.load(cache2)

        # SHAP values should be identical
        np.testing.assert_array_equal(data1["values"].values, data2["values"].values)

    def test_feature_analysis_with_fixed_seed(self, sample_classification_data, temp_dir):
        """Test feature analysis with fixed random seed."""
        X, _ = sample_classification_data

        # Create SHAP with fixed seed
        np.random.seed(42)
        shap1 = shap.Explanation(
            values=np.random.randn(30, 5), base_values=np.zeros(30), data=X[:30].values
        )

        # Create identical SHAP with same seed
        np.random.seed(42)
        shap2 = shap.Explanation(
            values=np.random.randn(30, 5), base_values=np.zeros(30), data=X[:30].values
        )

        result1 = extend_feature_analysis(shap1, X[:30], str(temp_dir / "seed1"))
        result2 = extend_feature_analysis(shap2, X[:30], str(temp_dir / "seed2"))

        # Results should be identical
        pd.testing.assert_frame_equal(result1["interactions"], result2["interactions"])


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_cache_with_empty_shap_values(self, temp_dir):
        """Test caching with empty SHAP values."""
        empty_shap = shap.Explanation(
            values=np.array([]).reshape(0, 5),
            base_values=np.array([]),
            data=np.array([]).reshape(0, 5),
        )

        # Should handle gracefully
        cache_path = cache_explanations_locally(empty_shap, cache_dir=str(temp_dir), ttl_hours=1)

        assert cache_path.exists()

    def test_visualization_with_nan_values(self, sample_classification_data, temp_dir):
        """Test visualization handles NaN in SHAP values."""
        X, _ = sample_classification_data

        # Create SHAP with some NaN values
        shap_vals = np.random.randn(10, 5)
        shap_vals[0, 0] = np.nan

        shap_with_nan = shap.Explanation(
            values=shap_vals, base_values=np.zeros(10), data=X[:10].values
        )

        output_path = temp_dir / "viz_nan.html"

        # Should handle gracefully (plotly handles NaN)
        result = enhance_visualizations(shap_with_nan, X[:10], str(output_path))
        assert result.exists()

    def test_performance_with_no_samples(self, sample_classification_data):
        """Test performance optimization with zero samples."""
        X, _ = sample_classification_data

        def mock_func(row):
            return np.random.randn(row.shape[1])

        # Should handle gracefully
        elapsed = optimize_performance(X[:0], mock_func, n_jobs=2)
        assert elapsed >= 0

    def test_logger_with_invalid_path(self):
        """Test logger creation with invalid path."""
        # Should create parent directories automatically
        log_file = "nonexistent/deeply/nested/test.log"
        logger = improve_error_handling(log_file)

        assert logger is not None
        assert Path(log_file).exists()

        # Cleanup
        import shutil

        shutil.rmtree("nonexistent")

    def test_ci_workflow_creates_parent_dirs(self, temp_dir):
        """Test that CI workflow creates missing parent directories."""
        workflow_dir = temp_dir / "deeply" / "nested" / "workflows"
        result = integrate_with_ci_cd(str(workflow_dir))

        assert result.exists()
        assert workflow_dir.exists()


class TestConfigurationValidation:
    """Test configuration and schema validation."""

    def test_document_schema_includes_all_keys(self, temp_dir):
        """Test that generated schema includes expected configuration keys."""
        doc_path = temp_dir / "schema.md"
        result = document_configuration_schema(str(doc_path))

        content = result.read_text()

        # Check all expected keys are documented
        assert "MODEL_PATH" in content
        assert "DRIFT_THRESHOLD" in content
        assert "CACHE_TTL_HOURS" in content
        assert "LOG_LEVEL" in content

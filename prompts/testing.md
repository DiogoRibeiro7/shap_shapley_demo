# Testing Prompts

## Test Infrastructure Setup

### Prompt for conftest.py

Generate pytest fixtures for a SHAP analytics library:

1. **Session-scoped Fixtures**:
   - `random_seed`: Fixed seed (42) for reproducibility
   - `breast_cancer_data`: Load sklearn breast cancer dataset
   - `train_test_split_data`: Split data 80/20
   - `trained_rf_model`: Trained RandomForest with small n_estimators for speed

2. **Function-scoped Fixtures**:
   - `small_dataset`: Generate synthetic classification data (50 samples, 5 features)
   - `small_trained_model`: Quick RF model for fast tests
   - `temp_dir`: Temporary directory for test outputs
   - `sample_config`: Dictionary with default config values
   - `sample_metadata`: Dictionary with model metadata
   - `mock_shap_values`: Random SHAP values array for testing

3. **Auto-use Fixtures**:
   - `reset_random_seed`: Reset np.random.seed before each test

4. **Custom Markers**:
   - @pytest.mark.unit
   - @pytest.mark.integration
   - @pytest.mark.slow

---

## Core Module Tests

### Prompt for test_shap_explain.py

Generate comprehensive tests for shap_explain module:

1. **TestComputeSHAPValues**:
   - Test basic computation returns Explanation object
   - Test output shape matches input
   - Test expected_value attribute exists
   - Test error handling for invalid model

2. **TestValidateBackgroundSample**:
   - Test validation passes for representative sample
   - Test with small sample size
   - Test when sample size exceeds data size

3. **TestMonitorFeatureDrift**:
   - Test no drift with identical distributions
   - Test drift detected with shifted distributions
   - Test return type is dict with float values
   - Test all scores are between 0 and 1

4. **TestVerifySHAPReconstruction**:
   - Test reconstruction verification passes
   - Test multiple sample indices
   - Test error when model lacks predict_proba

5. **Integration Tests**:
   - Test complete workflow: validate → compute → verify → monitor

Test Standards:
- Use descriptive test names: `test_<function>_<scenario>`
- Assert specific conditions, not just "doesn't crash"
- Test edge cases: empty data, single sample, NaN values
- Use pytest.raises for error testing
- Mark slow tests with @pytest.mark.slow

---

## Utility Module Tests

### Prompt for test_utils.py

Generate tests for utility modules:

1. **TestDirectoryOperations**:
   - Test directory creation
   - Test with existing directory
   - Test nested directory creation
   - Test from file path (creates parent)

2. **TestJSONOperations**:
   - Test save and load round-trip
   - Test creates parent directories
   - Test error on nonexistent file
   - Test with complex nested data

3. **TestModelSerialization**:
   - Test serialize and load round-trip
   - Test with checksum validation
   - Test checksum mismatch detection
   - Test error on nonexistent model

4. **TestJensenShannonDivergence**:
   - Test identical distributions (score ≈ 0)
   - Test different distributions (score > threshold)
   - Test with pandas Series input

5. **TestLogging**:
   - Test logger setup with console handler
   - Test logger with file output
   - Test multiple handlers (console + file)
   - Test log messages are written correctly

---

## Coverage Requirements

### Coverage Configuration (.coveragerc)

```ini
[run]
source = src
branch = True
omit =
    */tests/*
    */test_*.py
    */__pycache__/*

[report]
precision = 2
show_missing = True
skip_covered = False
fail_under = 80
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
```

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -ra
    --strict-markers
    --strict-config
    --cov=src/shap_analytics
    --cov-report=term-missing:skip-covered
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

---

## Test Execution Commands

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src/shap_analytics --cov-report=html

# Run specific test file
pytest tests/test_shap_explain.py -v

# Run specific test
pytest tests/test_shap_explain.py::TestComputeSHAPValues::test_basic -v

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto

# Run with verbose output and show locals on failure
pytest -vv --showlocals

# Generate coverage badge
coverage-badge -o coverage.svg -f
```

---

## Mocking Guidelines

When mocking in tests:

1. **Use pytest-mock for mocking**:
   ```python
   def test_with_mock(mocker):
       mock_func = mocker.patch('module.function')
       mock_func.return_value = "mocked"
   ```

2. **Mock external dependencies**:
   - HTTP requests (use responses or httpx)
   - File system (use tmp_path fixture)
   - Time (use freezegun)
   - Random (use fixed seeds)

3. **Don't mock what you're testing**:
   - Only mock external dependencies
   - Test the actual implementation

4. **Use fixtures for reusable mocks**:
   ```python
   @pytest.fixture
   def mock_model(mocker):
       model = mocker.Mock()
       model.predict_proba.return_value = [[0.3, 0.7]]
       return model
   ```

---

## Performance Testing

For performance benchmarking:

```python
import time
import pytest

@pytest.mark.slow
def test_shap_performance():
    start = time.perf_counter()
    shap_values = compute_shap_values(model, X_train, X_test)
    duration = time.perf_counter() - start

    assert duration < 1.0, f"SHAP computation took {duration:.2f}s, expected <1s"
```

---

## Test Data Management

Best practices for test data:

1. **Use fixtures for data**:
   - Session-scoped for expensive setup
   - Function-scoped for mutable data

2. **Generate synthetic data**:
   - Use sklearn.datasets.make_classification
   - Use numpy random with fixed seed

3. **Small datasets for tests**:
   - 50-100 samples is sufficient
   - 5-10 features is sufficient

4. **Avoid committed test data files**:
   - Generate data in fixtures
   - Use temporary files for outputs

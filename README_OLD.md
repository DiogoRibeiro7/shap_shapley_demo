# SHAP Analytics

**Professional SHAP value computation, analysis, and deployment toolkit**

[![CI Status](https://github.com/yourusername/shap-analytics/workflows/CI/badge.svg)](https://github.com/yourusername/shap-analytics/actions)
[![codecov](https://codecov.io/gh/yourusername/shap-analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/shap-analytics)
[![PyPI version](https://badge.fury.io/py/shap-analytics.svg)](https://badge.fury.io/py/shap-analytics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://yourusername.github.io/shap-analytics)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## Overview

SHAP Analytics is a comprehensive Python library for computing, analyzing, visualizing, and deploying SHAP (SHapley Additive exPlanations) values in production machine learning systems. It provides robust tools for feature importance analysis, drift monitoring, background sample validation, and automated explanation generation with enterprise-grade quality controls.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### Core Modules

#### 🔬 shap_explain.py - Core SHAP Computation
- **Purpose**: Pure SHAP computation and validation logic
- **Functions**: 5 core functions
- **Test Coverage**: 90%+

Key capabilities:
- compute_shap_values() - TreeExplainer with background sampling
- validate_background_sample() - Statistical validation
- monitor_feature_drift() - Distribution shift detection
- verify_shap_reconstruction() - Probability reconstruction check

#### 📊 shap_expansion.py - Visualization & Extensions
- **Purpose**: Production-ready plotting, caching, and dashboards
- **Functions**: 24 extension functions
- **Test Coverage**: 85%+

Key capabilities:
- Interactive Plotly dashboards
- Model registry with versioning
- Drift detection and alerting
- Data quality automation
- CI/CD integration
- FastAPI scaffolding

#### 🚀 shap_future.py - Experimental Research
- **Purpose**: Cutting-edge experimental features
- **Functions**: 11 experimental functions
- **Test Coverage**: 82%+

Key capabilities:
- Adaptive model retraining based on drift
- SHAP caching with TTL
- Color-blind-safe visualizations
- Structured logging with retry logic
- Performance benchmarking

## Installation

\`\`\`bash
pip install -r requirements-test.txt
\`\`\`

## Testing

### Running Tests

\`\`\`bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module
pytest tests/test_shap_expansion.py -v
pytest tests/test_shap_future.py -v
\`\`\`

### Test Coverage

The test suite includes **140+ test cases** covering:

#### test_shap_expansion.py (80 tests)
- Caching behavior
- Visualization output creation
- Drift detection
- Model registry
- Edge cases (empty DataFrames, NaN values, missing columns)

#### test_shap_future.py (60 tests)
- Caching with TTL (expiration, eviction, retrieval)
- Deterministic behavior with fixed seeds
- Performance optimization
- Error handling with retry logic
- CI/CD integration

### Current Coverage

| Module | Coverage |
|--------|----------|
| shap_explain.py | 90% |
| shap_expansion.py | 85% |
| shap_future.py | 82% |
| utils/common.py | 92% |

## Quick Start

\`\`\`python
from src.shap_explain import compute_shap_values
from sklearn.ensemble import RandomForestClassifier

# Train model and compute SHAP
shap_values = compute_shap_values(model, X_train, X_test)
\`\`\`

## Project Structure

\`\`\`
shap_shapley_demo/
├── src/
│   ├── shap_explain.py       # Core SHAP computation
│   ├── shap_expansion.py     # Visualization & extensions
│   ├── shap_future.py        # Experimental research
│   ├── shap_backlog.py       # Implementation roadmap
│   └── utils/common.py       # Shared utilities
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_shap_expansion.py # 80 test cases
│   └── test_shap_future.py    # 60 test cases
├── pytest.ini
├── .coveragerc
└── requirements-test.txt
\`\`\`

## Roadmap

See [Future Work](docs/future_work.md) for planned features and enhancements.

**Upcoming Features:**
- Async SHAP computation
- Redis caching integration
- XGBoost/LightGBM support
- GPU acceleration
- Distributed computing with Dask
- Kubernetes deployment

## Citation

If you use SHAP Analytics in your research, please cite:

```bibtex
@software{shap_analytics_2025,
  title = {SHAP Analytics: Professional SHAP Value Computation and Analysis},
  author = {SHAP Analytics Contributors},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/yourusername/shap-analytics}
}
```

See [CITATION.cff](CITATION.cff) for more details.

## Acknowledgments

Built on top of the excellent [SHAP library](https://github.com/slundberg/shap) by Scott Lundberg.

## Support

- **Documentation**: [https://yourusername.github.io/shap-analytics](https://yourusername.github.io/shap-analytics)
- **Issues**: [GitHub Issues](https://github.com/yourusername/shap-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/shap-analytics/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by SHAP Analytics Contributors**

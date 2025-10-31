# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository structure and package setup
- Core SHAP computation module with TreeExplainer support
- Background sample validation utilities
- Feature drift monitoring with Jensen-Shannon divergence
- SHAP value reconstruction verification
- Interactive dashboard creation utilities
- SHAP registry for metadata tracking
- FastAPI-based REST API for SHAP serving
- Comprehensive test suite with pytest
- Type checking with mypy (strict mode)
- Code quality enforcement with Ruff
- Pre-commit hooks for automated checks
- GitHub Actions CI/CD workflows
- Documentation with MkDocs Material
- Docker containerization support
- Poetry-based dependency management
- Structured JSON logging with LambdaJsonFormatter
- MLflow integration for experiment tracking

### Changed
- Reorganized source code into professional package structure
- Updated import paths for new package layout
- Enhanced documentation with architecture diagrams

### Fixed
- Type hints for all public functions and classes
- Import resolution for utility modules

## [0.1.0] - 2025-10-31

### Added
- Initial release of SHAP Analytics
- Core explainability computation engine
- Utility modules for common operations
- Basic visualization support
- Model serialization and checksum validation
- Logging infrastructure

[Unreleased]: https://github.com/yourusername/shap-analytics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/shap-analytics/releases/tag/v0.1.0

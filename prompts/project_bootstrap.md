# Project Bootstrap Prompt

## Initial Setup Request

Create a complete Python package named `shap_analytics`, minimum Python version 3.10, using Poetry for dependency management and project metadata.

## Directory Layout

```
shap_analytics/
├── pyproject.toml
├── poetry.lock
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── CITATION.cff
├── docs/
│   ├── index.md
│   ├── usage.md
│   ├── architecture.md
│   ├── api_reference.md
│   └── future_work.md
├── src/shap_analytics/
│   ├── __init__.py
│   ├── shap_explain.py
│   ├── shap_expansion.py
│   ├── shap_future.py
│   ├── shap_backlog.py
│   └── utils/
│       ├── __init__.py
│       ├── common.py
│       ├── io_utils.py
│       └── logging_utils.py
├── tests/
│   ├── test_shap_explain.py
│   ├── test_shap_expansion.py
│   ├── test_shap_future.py
│   ├── test_utils.py
│   └── conftest.py
├── .github/workflows/
│   ├── ci.yml
│   ├── todo.yml
│   └── docs.yml
├── .pre-commit-config.yaml
├── .mypy.ini
├── .gitignore
├── .env.example
├── Dockerfile
├── compose.yaml
└── setup.sh
```

## Core Requirements

### Dependencies
- numpy, pandas, matplotlib, scikit-learn, shap, scipy
- click, fastapi, uvicorn, pydantic, pydantic-settings
- pytest, mypy, ruff, mkdocs-material, mlflow
- plotly, joblib, redis (optional)

### Code Quality
- Enable strict typing: `mypy --strict`
- Lint with Ruff: `ruff check .`
- Run unit tests: `pytest -q --cov=src/shap_analytics`
- Pre-commit hooks: ruff, mypy, pytest

### GitHub Workflows
1. **ci.yml**: Lint, type check, test on Python 3.10, 3.11, 3.12
2. **todo.yml**: Detect TODO comments, sync with GitHub issues using juulsn/todo-issue
3. **docs.yml**: Build and deploy MkDocs to GitHub Pages on pushes to main

### Documentation
- Build with mkdocs-material
- Pages: Overview, architecture, module APIs, backlog roadmap, contributing guide
- Automated API reference generation with mkdocstrings

### Docker
Multi-stage build:
1. Stage 1: Install Poetry + dependencies
2. Stage 2: Copy minimal runtime + FastAPI entrypoint
3. Expose port 8080
4. Non-root user for security

### Logging
- Implement LambdaJsonFormatter for structured JSON logging
- Support console and file output
- Include context and performance metrics

### Testing
- Create pytest fixtures in conftest.py
- Each module gets at least one test file
- Coverage threshold: 80%
- Mark tests as unit/integration/slow

### Metadata
- CITATION.cff with author info and ORCID
- LICENSE (MIT)
- README.md badges for CI, Codecov, PyPI
- CODE_OF_CONDUCT.md using Contributor Covenant
- CHANGELOG.md following Keep a Changelog format

### Security
- Include .env.example with placeholder secrets
- Config validation via pydantic.BaseSettings
- Model checksum validation (SHA256)
- Input validation with pydantic

### Setup Script
```bash
#!/usr/bin/env bash
poetry install
pre-commit install
poetry run pytest -q
```

## Verification Commands

After generation, verify with:

```bash
mypy --strict src/
ruff check src/
pytest -q
```

## Success Criteria

1. All files generated with correct content
2. Type checking passes with `mypy --strict`
3. Linting passes with `ruff check`
4. Tests run successfully with >80% coverage
5. Documentation builds without errors
6. Docker image builds successfully
7. Pre-commit hooks install correctly

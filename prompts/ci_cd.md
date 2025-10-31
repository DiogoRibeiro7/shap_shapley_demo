# CI/CD Workflow Prompts

## GitHub Actions CI Workflow

### Prompt for ci.yml

Generate a comprehensive CI workflow for a Python package using Poetry that:

1. **Linting Job**:
   - Run Ruff linter: `ruff check src/ tests/`
   - Run Ruff formatter check: `ruff format --check src/ tests/`
   - Use Python 3.10
   - Cache Poetry dependencies

2. **Type Checking Job**:
   - Run MyPy in strict mode: `mypy --strict src/shap_analytics`
   - Use Python 3.10
   - Install all dependencies including dev

3. **Test Job**:
   - Test on Python 3.10, 3.11, 3.12
   - Run pytest with coverage: `pytest --cov=src/shap_analytics --cov-report=xml`
   - Upload coverage to Codecov for Python 3.10 only
   - Use matrix strategy for multiple Python versions

4. **Build Job**:
   - Build package with Poetry: `poetry build`
   - Upload build artifacts (dist/)
   - Run after lint, typecheck, and test pass

5. **Security Scan Job**:
   - Run Trivy vulnerability scanner
   - Upload results to GitHub Security tab
   - Scan filesystem for vulnerabilities

### Trigger Conditions

- Push to `main` and `develop` branches
- Pull requests to `main` and `develop`
- Manual workflow dispatch

### Caching Strategy

- Cache Poetry virtual environment based on `poetry.lock` hash
- Cache key: `venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}`

---

## TODO to Issue Workflow

### Prompt for todo.yml

Generate a workflow that automatically creates GitHub issues from TODO comments in code using `juulsn/todo-issue`:

1. **Trigger**:
   - Push to main branch
   - Only run on specific file changes (*.py, *.ts, *.go, etc.)
   - Exclude docs and CI files
   - Provide workflow_dispatch for manual runs

2. **Token Selection**:
   - Support both GitHub token and PAT
   - Check for `secrets.PAT_GITHUB` or fall back to `github.token`
   - Verify token validity before use

3. **Keywords to Detect**:
   - TODO, FIXME, BUG, HACK, NOTE
   - Support ASSIGNEE:, LABELS:, PRIORITY: in comment body

4. **Exclusion Pattern**:
   - Exclude: .git/, .github/, .venv/, __pycache__/, node_modules/
   - Exclude: build/, dist/, site/, htmlcov/
   - Exclude: .pytest_cache/, .mypy_cache/, .ruff_cache/

5. **Configuration**:
   - Auto-label with "todo"
   - Title similarity threshold: 0.7
   - Don't reopen closed issues
   - Case insensitive matching

---

## Documentation Deployment Workflow

### Prompt for docs.yml

Generate a workflow for building and deploying MkDocs documentation:

1. **Build Job**:
   - Install Poetry and dependencies with docs group
   - Build docs: `mkdocs build --strict`
   - Upload docs artifact (site/)
   - Cache Poetry dependencies

2. **Deploy Job**:
   - Only run on push to main branch
   - Deploy to GitHub Pages: `mkdocs gh-deploy --force`
   - Notify deployment URL in summary
   - Require build job to pass first

3. **Permissions**:
   - contents: write (for GitHub Pages deployment)

### MkDocs Configuration

Ensure mkdocs.yml includes:
- theme: material
- plugins: search, mkdocstrings[python]
- markdown_extensions: pymdownx.superfences, pymdownx.tabbed, etc.

---

## Pre-commit Hook Configuration

### Prompt for .pre-commit-config.yaml

Generate pre-commit configuration with:

1. **Standard Hooks**:
   - trailing-whitespace
   - end-of-file-fixer
   - check-yaml, check-toml, check-json
   - check-added-large-files (max 1000KB)
   - check-merge-conflict
   - check-ast, debug-statements

2. **Ruff Hooks**:
   - ruff linter with --fix and --exit-non-zero-on-fix
   - ruff-format for code formatting

3. **MyPy Hook**:
   - Run mypy --strict --ignore-missing-imports
   - Only on src/ files
   - Include additional type stubs

4. **Pytest Hook** (local):
   - Run pytest --maxfail=1 --disable-warnings -q
   - fail fast on first error
   - Run on commit stage

---

## Secrets Configuration

Document required secrets for CI/CD:

```yaml
# Required Secrets
PAT_GITHUB: Personal Access Token for issue creation (optional)
CODECOV_TOKEN: Token for Codecov uploads (optional)

# Optional Secrets for future use
PYPI_TOKEN: For publishing to PyPI
AWS_ACCESS_KEY_ID: For S3 deployments
AWS_SECRET_ACCESS_KEY: For S3 deployments
DOCKER_USERNAME: For Docker Hub publishing
DOCKER_PASSWORD: For Docker Hub publishing
```

---

## Semantic Release Configuration

### Future Enhancement

For automated version bumps and changelog generation:

```yaml
# .releaserc.json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/github",
    ["@semantic-release/exec", {
      "prepareCmd": "poetry version ${nextRelease.version}"
    }]
  ]
}
```

Commit format:
- `feat: description` - Minor version bump
- `fix: description` - Patch version bump
- `BREAKING CHANGE:` - Major version bump

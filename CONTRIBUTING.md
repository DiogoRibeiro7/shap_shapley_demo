# Contributing to SHAP Shapley Demo

First off, thank you for considering contributing to SHAP Shapley Demo! It's people like you that make this project a great tool for the ML explainability community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug reports**: File issues for bugs you find
- ✨ **Feature requests**: Suggest new features or enhancements
- 📝 **Documentation**: Improve or add to our documentation
- 🧪 **Tests**: Add tests to increase coverage
- 💻 **Code**: Submit PRs to fix bugs or add features
- 🎨 **Design**: Improve visualizations or UI elements
- 📊 **Examples**: Add examples and tutorials

### Before You Start

- Check if there's already an [open issue](https://github.com/yourusername/shap_shapley_demo/issues) for your concern
- For major changes, please open an issue first to discuss your approach
- Make sure you have read and understood our [README.md](README.md)

## How to Contribute

### Reporting Bugs

**Before submitting a bug report:**

1. Check the [issue tracker](https://github.com/yourusername/shap_shapley_demo/issues) to avoid duplicates
2. Gather information about the bug (Python version, OS, error messages, etc.)
3. Create a minimal reproducible example

**When filing a bug report, include:**

- **Title**: Clear, descriptive title
- **Description**: Detailed description of the issue
- **Steps to reproduce**: List of steps to reproduce the behavior
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, package versions
- **Error messages**: Full error traceback
- **Screenshots**: If applicable

**Template:**

```markdown
## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. [First Step]
2. [Second Step]
3. [And so on...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- SHAP: [e.g., 0.42.0]
- Package Version: [e.g., 1.0.0]

## Error Message
```
[Paste full error traceback here]
```

## Additional Context
[Any other context, screenshots, or information]
```

### Suggesting Features

**Before submitting a feature request:**

1. Check if the feature already exists
2. Check if there's an open issue for it
3. Consider if it fits the project scope

**When suggesting a feature, include:**

- **Title**: Clear, descriptive title
- **Problem**: What problem does this solve?
- **Proposed solution**: How would you implement it?
- **Alternatives**: Other solutions you've considered
- **Additional context**: Examples, mockups, or references

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Clarifying confusing sections
- Adding examples
- Improving API documentation
- Creating tutorials

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/shap_shapley_demo.git
   cd shap_shapley_demo
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/shap_shapley_demo.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements-test.txt
   pip install -e .
   ```

6. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

7. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_shap_expansion.py -v

# Run specific test
pytest tests/test_shap_expansion.py::TestCaching::test_cache_stores_and_retrieves_correctly -v
```

### Running Linters

```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix

# Format code
black src/ tests/

# Type checking
mypy src/ --check-untyped-defs
```

## Pull Request Process

### Before Submitting

1. ✅ Update your branch with latest upstream changes
2. ✅ Run all tests and ensure they pass
3. ✅ Run linters and fix any issues
4. ✅ Add tests for new functionality
5. ✅ Update documentation as needed
6. ✅ Update CHANGELOG.md with your changes

### PR Guidelines

1. **Branch naming**:
   - `feature/description` for new features
   - `fix/description` for bug fixes
   - `docs/description` for documentation
   - `refactor/description` for refactoring
   - `test/description` for test additions

2. **Commit messages**:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit first line to 72 characters
   - Reference issues and PRs liberally

   Example:
   ```
   Add caching mechanism for SHAP values

   - Implement TTL-based cache with configurable expiration
   - Add cache eviction logic for expired entries
   - Include comprehensive tests for cache behavior

   Closes #123
   ```

3. **PR description**:
   - Use the PR template
   - Explain what changes you made and why
   - Link related issues
   - Add screenshots/GIFs for UI changes
   - List breaking changes (if any)

4. **PR size**:
   - Keep PRs focused and reasonably sized
   - Split large changes into multiple PRs
   - Aim for < 400 lines changed per PR

### Review Process

1. At least one maintainer must approve the PR
2. All CI checks must pass
3. Coverage should not decrease (maintain ≥80%)
4. All conversations must be resolved
5. Maintainer will merge when ready

### After Merge

- Delete your branch (both locally and on GitHub)
- Pull latest changes from upstream
- Celebrate! 🎉

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with `isort`

### Type Hints

- All functions must have type hints
- Use `from __future__ import annotations` for forward references
- Prefer built-in types over `typing` module when possible (Python 3.10+)

Example:
```python
def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    background_size: int = 100,
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer."""
    ...
```

### Docstrings

We use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed. Explain what the function does,
    any important details, edge cases, etc.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When validation fails.
        TypeError: When wrong type is provided.

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    ...
```

### Code Organization

**Module structure:**
```python
"""Module docstring."""

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .utils.common import helper_function

# Constants
DEFAULT_VALUE = 42

# Public API
__all__ = [
    "public_function",
    "PublicClass",
]

# Implementation
...
```

### Naming Conventions

- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_`
- **Very private**: Prefix with `__`

## Testing Guidelines

### Test Requirements

- All new code must have tests
- Maintain or increase coverage (target: ≥80%)
- Tests should be fast (< 1s per test)
- Use descriptive test names

### Test Structure

```python
class TestFeatureName:
    """Test suite for feature X."""

    def test_function_behavior_expected_result(self, fixture):
        """Test that function does X when given Y."""
        # Arrange
        input_data = ...
        expected = ...

        # Act
        result = function(input_data)

        # Assert
        assert result == expected
```

### Test Categories

Use pytest markers:

```python
@pytest.mark.slow
def test_expensive_operation():
    """Test that takes > 1s."""
    ...

@pytest.mark.integration
def test_full_workflow():
    """Integration test."""
    ...
```

### Fixtures

- Use fixtures for common test data
- Keep fixtures in `conftest.py`
- Document fixtures with docstrings

## Documentation

### Documentation Types

1. **Code documentation**: Docstrings in code
2. **API documentation**: Generated from docstrings
3. **User guide**: README.md and tutorials
4. **Developer guide**: This file (CONTRIBUTING.md)

### Building Documentation

```bash
# Generate API docs (if using Sphinx)
cd docs
make html
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up-to-date with code

## Community

### Getting Help

- 📖 Read the [documentation](README.md)
- 💬 Ask questions in [Discussions](https://github.com/yourusername/shap_shapley_demo/discussions)
- 🐛 File [issues](https://github.com/yourusername/shap_shapley_demo/issues)

### Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- CHANGELOG.md

### License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Quick Checklist

Before submitting your PR, make sure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

---

**Thank you for contributing! 🎉**

Questions? Feel free to ask in the [Discussions](https://github.com/yourusername/shap_shapley_demo/discussions)!

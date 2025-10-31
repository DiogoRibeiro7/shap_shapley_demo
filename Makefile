POETRY_RUN=poetry run

.PHONY: install setup precommit format lint typecheck test coverage security docs clean update-deps onboard

install:
	poetry install --no-interaction

setup: install precommit

precommit:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg

format:
	$(POETRY_RUN) black src tests
	$(POETRY_RUN) isort src tests
	$(POETRY_RUN) ruff format src tests

lint:
	$(POETRY_RUN) ruff check src tests
	$(POETRY_RUN) bandit -r src/shap_analytics -lll

typecheck:
	$(POETRY_RUN) mypy --strict src/shap_analytics

test:
	$(POETRY_RUN) pytest

coverage:
	$(POETRY_RUN) pytest --cov=src/shap_analytics --cov-report=term-missing --cov-report=xml
	$(POETRY_RUN) coverage report --fail-under=80

security:
	poetry export --format=requirements.txt --without-hashes --output /tmp/requirements.txt
	$(POETRY_RUN) safety check --full-report --requirements /tmp/requirements.txt
	$(POETRY_RUN) bandit -r src/shap_analytics -lll

docs:
	poetry run mkdocs build

clean:
	rm -rf .venv dist build __pycache__ .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov

update-deps:
	poetry update

onboard:
	$(POETRY_RUN) python scripts/onboard.py

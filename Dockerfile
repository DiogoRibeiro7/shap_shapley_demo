# Multi-stage Dockerfile for SHAP Analytics
# Stage 1: Builder - Install dependencies
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --without dev --no-interaction --no-ansi && \
    rm -rf $POETRY_CACHE_DIR

# Stage 2: Runtime - Minimal image with application
FROM python:3.10-slim as runtime

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Create non-root user
RUN useradd -m -u 1000 shap && \
    chown -R shap:shap /app

USER shap

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command: Run FastAPI server
CMD ["python", "-m", "uvicorn", "shap_analytics.shap_expansion:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

# Alternative entrypoint for CLI
# ENTRYPOINT ["python", "-m", "shap_analytics.shap_explain"]

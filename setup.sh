#!/usr/bin/env bash
#
# Setup script for SHAP Analytics development environment
#
# This script:
# - Checks for required dependencies (Python, Poetry)
# - Installs project dependencies
# - Sets up pre-commit hooks
# - Runs initial tests
# - Generates initial documentation
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    log_info "Starting SHAP Analytics setup..."

    # Check Python version
    log_info "Checking Python version..."
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2)
    log_success "Python $python_version found"

    # Check Poetry
    log_info "Checking for Poetry..."
    if ! command_exists poetry; then
        log_warning "Poetry not found. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"

        if ! command_exists poetry; then
            log_error "Failed to install Poetry. Please install manually: https://python-poetry.org/docs/"
            exit 1
        fi
    fi

    poetry_version=$(poetry --version | cut -d' ' -f3)
    log_success "Poetry $poetry_version found"

    # Install dependencies
    log_info "Installing project dependencies..."
    poetry install --no-interaction

    if [ $? -eq 0 ]; then
        log_success "Dependencies installed successfully"
    else
        log_error "Failed to install dependencies"
        exit 1
    fi

    # Install pre-commit hooks
    log_info "Installing pre-commit hooks..."
    if command_exists pre-commit; then
        poetry run pre-commit install
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found. Skipping hook installation."
        log_info "Run 'poetry run pre-commit install' manually later."
    fi

    # Create necessary directories
    log_info "Creating project directories..."
    mkdir -p logs models data reports
    log_success "Project directories created"

    # Copy .env.example to .env if not exists
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_success ".env file created. Please update with your configuration."
    else
        log_info ".env file already exists. Skipping."
    fi

    # Run initial tests
    log_info "Running initial test suite..."
    if poetry run pytest -q --maxfail=1 --disable-warnings; then
        log_success "All tests passed!"
    else
        log_warning "Some tests failed. Please review and fix."
    fi

    # Run type checking
    log_info "Running type checker..."
    if poetry run mypy --strict src/shap_analytics 2>/dev/null; then
        log_success "Type checking passed!"
    else
        log_warning "Type checking found issues. Run 'poetry run mypy --strict src/' to see details."
    fi

    # Run linter
    log_info "Running linter..."
    if poetry run ruff check src/ 2>/dev/null; then
        log_success "Linting passed!"
    else
        log_warning "Linting found issues. Run 'poetry run ruff check src/' to see details."
    fi

    # Build documentation
    log_info "Building documentation..."
    if poetry run mkdocs build --quiet 2>/dev/null; then
        log_success "Documentation built successfully. Open site/index.html to view."
    else
        log_warning "Documentation build failed. Run 'poetry run mkdocs build' to see details."
    fi

    # Print summary
    echo ""
    echo "========================================="
    log_success "Setup completed successfully!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Activate the virtual environment:"
    echo "     poetry shell"
    echo ""
    echo "  2. Run the example:"
    echo "     poetry run python -m shap_analytics.shap_explain"
    echo ""
    echo "  3. Run tests:"
    echo "     poetry run pytest -v"
    echo ""
    echo "  4. Start development server (API):"
    echo "     poetry run uvicorn shap_analytics.shap_expansion:app --reload"
    echo ""
    echo "  5. View documentation:"
    echo "     poetry run mkdocs serve"
    echo ""
    echo "  6. Run all quality checks:"
    echo "     poetry run pre-commit run --all-files"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main function
main "$@"

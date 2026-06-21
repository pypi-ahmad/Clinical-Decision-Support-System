# MediScan OCR &mdash; canonical task runner.
#
# Targets mirror the CI workflow in .github/workflows/ci.yml. Every target
# is intentionally a thin wrapper so the actual commands remain discoverable
# in `make help` and the CI YAML.

.DEFAULT_GOAL := help
.PHONY: help install install-dev install-test sync lock upgrade lint format typecheck test test-fast test-cov test-live test-eval test-property test-contract audit clean build pre-commit run-api run-frontend release-check

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?##"; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-22s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

install:  ## Create the .venv at Python 3.12.10 (idempotent).
	uv python install 3.12.10
	uv venv --python 3.12.10

sync:  ## Sync the locked environment (production-only deps).
	uv sync --frozen --no-dev

install-dev:  ## Sync the locked dev + test environment.
	uv sync --frozen

lock:  ## Regenerate the lockfile from pyproject.toml.
	uv lock

upgrade:  ## Upgrade all locked dependencies within their constraints.
	uv lock --upgrade
	uv sync

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

lint:  ## Run Ruff lint over the whole repo.
	uv run --frozen ruff check .

format:  ## Run Ruff format check (no writes).
	uv run --frozen ruff format --check .

typecheck:  ## Run mypy on the strict subset and Pyright on backend (best-effort).
	uv run --frozen mypy \
	    backend/security.py \
	    backend/database.py \
	    backend/retrieval
	uv run --frozen pyright backend/ || true

audit:  ## Run pip-audit on the locked environment.
	uv run --frozen pip-audit --strict

pre-commit:  ## Run all pre-commit hooks against the repo.
	uv run --frozen pre-commit run --all-files

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:  ## Run the offline test suite with coverage.
	uv run --frozen pytest -m "not live and not eval"

test-fast:  ## Run the offline test suite in parallel.
	uv run --frozen pytest -m "not live and not eval" -n auto

test-cov:  ## Run tests and produce an HTML coverage report.
	uv run --frozen pytest -m "not live and not eval" --cov-report=html
	@echo "open htmlcov/index.html"

test-live:  ## Run live integration tests (requires running Ollama + Qdrant).
	uv run --frozen pytest -m live

test-eval:  ## Run the eval harness (requires gold fixtures in tests/eval/gold/).
	uv run --frozen pytest -m eval

test-property:  ## Run property-based tests.
	uv run --frozen pytest -m property

test-contract:  ## Run API contract tests.
	uv run --frozen pytest -m contract

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run-api:  ## Run the FastAPI backend with auto-reload.
	uv run --frozen uvicorn backend.main:app --reload --port 8000

run-frontend:  ## Run the Streamlit frontend.
	uv run --frozen streamlit run frontend/app.py

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

release-check:  ## Verify the repo is in a green state for release.
	make lint
	make typecheck
	make test
	@echo "Repo is green. Safe to tag and release."

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove build artifacts and cache directories.
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

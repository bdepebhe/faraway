.PHONY: install install-dev test lint format typecheck clean cov

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest

test-v:
	uv run pytest -v

cov:
	uv run pytest --cov=faraway --cov-report=term-missing

badge:
	uv run pytest --cov=faraway --cov-report=xml
	@# Truncate line-rate to 2 decimal places (displays as whole %) to avoid flaky badge diffs
	sed -i '' 's/line-rate="0\.\([0-9][0-9]\)[0-9]*"/line-rate="0.\1"/g' coverage.xml
	uv run genbadge coverage -i coverage.xml -o assets/coverage-badge.svg

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy faraway

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


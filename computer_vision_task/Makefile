.PHONY: install lint format typecheck test test-all run ci clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m 'not slow'

test-all:
	uv run pytest

run:
	uv run python -m pipeline.cli run

clean:
	rm -rf output/ .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

ci: lint typecheck test-all

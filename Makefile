.PHONY: sync
sync:
	uv sync --extra dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run mypy unilab

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-fast
test-fast:
	uv run pytest -m "not slow"

.PHONY: test-all
test-all: check test

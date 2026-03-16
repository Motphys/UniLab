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
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest -m "not slow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow" --cov=unilab --cov-report=term-missing

.PHONY: test-fast
test-fast:
	uv run pytest -m "not slow"

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow"

.PHONY: test-all
test-all: check test-cov

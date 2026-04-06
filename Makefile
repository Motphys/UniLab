.PHONY: sync
sync:
	uv sync --extra dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run mypy src/unilab
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest -m "not slow and not veryslow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow and not veryslow" --cov=unilab --cov-report=term-missing

.PHONY: test-fast
test-fast:
	uv run pytest -m "not slow and not veryslow"

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow and not veryslow" -v

.PHONY: test-veryslow
test-veryslow:
	uv run pytest -m "veryslow" -v

.PHONY: test-all
test-all: check test-cov

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete

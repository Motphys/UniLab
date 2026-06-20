.PHONY: sync
sync:
	uv sync --extra mujoco --extra motrix

.PHONY: setup
setup:
	uv sync --extra mujoco --extra motrix
	uv run --no-sync unilab-complete install

.PHONY: setup-mujoco
setup-mujoco:
	uv sync --extra mujoco
	uv run --no-sync unilab-complete install

.PHONY: setup-motrix
setup-motrix:
	uv sync --extra motrix
	uv run --no-sync unilab-complete install

.PHONY: install-completion
install-completion:
	uv run --no-sync unilab-complete install

.PHONY: sync-rocm
sync-rocm:
	@cp pyproject.rocm.toml pyproject.toml
	@if [ -f uv.rocm.lock ]; then cp uv.rocm.lock uv.lock; fi
	uv sync --extra mujoco --extra motrix
	cp uv.lock uv.rocm.lock

.PHONY: sync-xpu
sync-xpu:
	uv sync --extra mujoco --extra motrix --no-install-package torch
	uv pip install torch==2.7.0 --torch-backend xpu

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
	uv run pytest -m "not slow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow" -v

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
	rm -f train_appo.log train_offpolicy.log train_rsl_rl.log
	find src/unilab/assets/.cache -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/caches -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/checkpoints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/scenes -type f ! -name '.gitkeep' -delete 2>/dev/null || true

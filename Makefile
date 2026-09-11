.PHONY: sync
sync:
	uv sync --extra mujoco --extra motrix

.PHONY: setup
setup:
	uv sync --extra mujoco --extra motrix
	uv run --no-sync unilab-complete install

# Installs the Python extra and builds DrakeUni's native extension. By default
# the host-compatible official tarball is downloaded; use DRAKE_HOME=<prefix>
# to build against an existing installation.
.PHONY: setup-drake
setup-drake:
	@ if [ -n "$(DRAKE_HOME)" ]; then \
		bash scripts/tools/setup_drake_env.sh --drake-home "$(DRAKE_HOME)"; \
	else \
		bash scripts/tools/setup_drake_env.sh --download-drake; \
	fi

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
check: format type check-tests

.PHONY: check-tests
check-tests:
	uv run ruff check tests --select F401,F821,F811,F841 --output-format concise

.PHONY: test
test:
	uv run pytest -m "not slow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow" -v

.PHONY: test-benchmark-smoke
test-benchmark-smoke:
	uv run python scripts/benchmark/smoke_test.py

.PHONY: test-all
test-all: check test-cov test-benchmark-smoke

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
	rm -f train_appo.log train_td3.log train_sac.log train_flashsac.log train_rsl_rl.log MUJOCO_LOG.TXT
	find src/unilab/assets/.cache -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/caches -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/checkpoints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/scenes -type f ! -name '.gitkeep' -delete 2>/dev/null || true

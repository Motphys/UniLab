# UniLab Claude Code Guidelines

**Always use `uv run`, not `python`.**

UniLab is a **high-performance, modular, contract-driven** RL infrastructure repo.
Read [RL Infrastructure Development Standard](docs/zh_CN/00-development-architecture.md) for full architecture context.

## Project Overview

CPU simulation + shared-memory data path + GPU training:

```
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
```

## Common Commands

```bash
make format         # ruff format + ruff check --fix
make type           # mypy src/unilab + pyright
make check          # format + type (run before commits)
make test           # non-slow tests
make test-slow      # integration tests (requires MuJoCo)
```

## Code Style

- Formatter & linter: **ruff** (`line-length = 100`, `target-version = "py310"`)
- Type checking: **mypy** + **pyright**
- Run `make format` before committing; `make check` for full validation
- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)

## Layered Architecture

Dependencies flow strictly downward. **Fix problems at the layer where they originate.**

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| L0 Backend | `base/backend/` | `SimBackend` physics abstraction |
| L1 Env | `envs/`, `base/np_env.py` | MDP semantics, obs, reward, reset |
| L2 Config | `config/`, `base/registry.py`, `conf/` | Schema, task/reward composition, registration |
| L3 Algo & IPC | `algos/`, `ipc/` | Learner, runner, collector, shared-memory |
| L4 Scripts | `scripts/` | Assembly only, no core business rules |

## Key Contracts (Do Not Break)

- `NpEnvState.obs` must be `dict`
- `reset()` returns `(obs_dict, info_dict)`
- `obs_groups_spec` drives wrapper and learner dimensions
- Reward injected via Hydra config, not hardcoded
- `training.sim_backend` and `motrix_legacy` must respect explicit overrides
- Backend-specific logic stays in backend/env adaptation layer, never leaks into training scripts

## High-Signal Files

- Env contract: `src/unilab/base/np_env.py`
- Backend contract: `src/unilab/base/backend/base.py`
- Registry: `src/unilab/base/registry.py`
- Config schema: `src/unilab/config/structured_configs.py`
- Backend factory: `src/unilab/base/backend/__init__.py`
- Async runner: `src/unilab/ipc/async_runner.py`

## Training Entrypoints

| Algorithm | Script |
|-----------|--------|
| PPO (torch) | `scripts/train_rsl_rl.py` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` |
| APPO | `scripts/train_appo.py` |
| SAC / TD3 | `scripts/train_offpolicy.py` |

## Configuration

Hydra + dataclass. Schema in `structured_configs.py`, runtime config in `conf/{ppo,appo,offpolicy}/`.

- Add a new task: create YAML under `conf/{algo}/task/` with `# @package _global_`
- Backend-specific overrides go in YAML config (e.g., `motrix_legacy.env_cfg_override`), not in MJCF files
- Generic env parameters (like `iterations`) belong in `EnvCfg` base class and are passed through `create_backend`

## Backend Notes

- **MuJoCo**: `BatchEnvPool` with multithreaded stepping, `implicitfast` solver (non-iterative)
- **Motrix**: `model.step_n()` for multi-substep, iterative solver (sensitive to `max_iterations`)
- Both backends accept `iterations` parameter via `EnvCfg` base class
- Backend differences are isolated in `base/backend/` and env adaptation layers

## Testing

- Regular tests: no marker, no MuJoCo required, `make test`
- `@pytest.mark.slow`: requires MuJoCo, run with `make test-slow`
- `@pytest.mark.veryslow`: full training smoke tests, run with `make test-veryslow`

## Validation Requirements

| Change | Minimum validation |
|--------|--------------------|
| Hydra / task / reward | `make test` |
| Env contract / obs | `make test` |
| Runner / IPC | `make test`, possibly `make test-slow` |
| Training path | Related tests + 1-iteration smoke run |
| Docs-only | Check commands, paths, config names, CI |

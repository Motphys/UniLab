# UniLab Agent Principles

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。
先看 [RL Infrastructure Development Standard](docs/zh_CN/00-development-architecture.md)。AGENTS 只保留 agent 必须记住的理念。

## Core Principles

1. Contract first: 不为了一次通过绕过 env / backend / runner contract。
2. Fix at owner layer: `scripts/` 只组装流程，不承载长期业务规则。
3. Config first: task / reward / backend 优先通过 Hydra + registry 表达。
4. Backend isolation: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. Evidence only: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. Validate near risk: 在最接近风险的边界补验证，不只跑顶层命令。

## Read Order

1. `AGENTS.md`
2. `docs/zh_CN/00-development-architecture.md`
3. `CONTRIBUTING.md`
4. 当前任务相关代码与测试

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；`training.sim_backend` 和 `motrix_legacy` 必须尊重显式 override。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |

## Validation

- Hydra / task / reward：`make test`
- env / obs / reset：`make test`
- runner / IPC：`make test`，必要时 `make test-slow`
- training path：相关测试 + 1-iteration smoke run
- docs-only：核对命令、路径、配置名、CI 和 support claim

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- config schema: `src/unilab/config/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`

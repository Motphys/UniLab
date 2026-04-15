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
7. Cold-path asset access only: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## Read Order

1. `AGENTS.md`
2. `docs/zh_CN/00-development-architecture.md`
3. `CONTRIBUTING.md`
4. 当前任务相关代码与测试

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |

## Validation

- Hydra / task / reward：`make test`
- env / obs / reset：`make test`
- runner / IPC：`make test`，必要时 `make test-slow`
- training path：相关测试 + 1-iteration smoke run
- docs-only：核对命令、路径、配置名、CI 和 support claim

## GitHub CLI (gh) 使用指南

### Issue 查看

```bash
# 查看特定 issue
gh issue view <number>

# 示例: 查看 issue #174
gh issue view 174

# 如果提示缺少 scope，使用 API 方式
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'

# 示例
gh api repos/unilabsim/UniLab/issues/174 --jq '.body'
```

### PR 创建与管理

```bash
# 创建 PR（当前分支推送到远程后）
gh pr create --title "标题" --body "内容" --base main

# 查看 PR 列表
gh pr list

# 查看当前分支的 PR 状态
gh pr view
```

### CI 工作流查看

```bash
# 查看工作流运行状态
gh run list

# 查看特定工作流的最新运行
gh run list --workflow=<workflow-name>

# 查看运行日志
gh run view <run-id>

# 查看失败的运行
gh run list --status=failure
```

### 常用组合

```bash
# 完整的工作流程示例
# 1. 查看 issue
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'

# 2. 创建分支并修改代码后，推送并创建 PR
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- config schema: `src/unilab/config/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`

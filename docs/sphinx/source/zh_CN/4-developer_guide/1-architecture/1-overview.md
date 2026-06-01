# 架构概览

UniLab 是一个 contract 驱动的机器人学习基础设施仓库。其核心规则是：在拥有该
契约的 owner 层修复行为。

## 运行时模型

异步路径采用从 CPU 仿真到加速器 learner 的流水线：

```text
CPU physics backend -> collector / IPC -> learner
MuJoCo or Motrix      shared memory       torch or mlx
```

PPO 与 MLX PPO 是同步的单进程路径。APPO 与 off-policy 算法则使用异步 runner、
共享缓冲区，以及位于 `src/unilab/ipc/` 与 `src/unilab/algos/` 下的权重同步原语。

## 分层边界

| 层 | 路径 | 拥有 |
| --- | --- | --- |
| Backend | `src/unilab/base/backend/` | `SimBackend`、物理状态、可选能力 |
| Env | `src/unilab/envs/`、`src/unilab/base/np_env.py` | MDP 语义、观测、奖励、reset |
| Config 与 registry | `conf/`、`src/unilab/base/registry.py`、`src/unilab/structured_configs.py` | Schema、owner YAML、env/backend 注册 |
| 算法与 IPC | `src/unilab/algos/`、`src/unilab/ipc/` | Learner、runner、buffer、权重同步 |
| Scripts | `scripts/`、`src/unilab/cli.py` | 轻量装配与 CLI 路由 |

## 设计规则

- 将 backend 差异保留在 backend 实现、env 适配层与 owner YAML 中。
- 使用 `uv run train --algo <algo> --task <task> --sim <backend>` 或
  `uv run eval ...` 来选择对外的算法/任务/backend 路由。这些 flag 会 compose
  出匹配的 owner YAML；`training.sim_backend` 只是一个身份字段。
- 优先用 config 表达，而不是分支。任何扩展的优先级顺序是：config schema ->
  registry -> env/backend 适配层 -> 最后才是脚本分支。
- 不要在 `step`、`reset` 或 interval domain randomization 等热路径中解析 XML 或
  资源。
- 如果共享的 env 代码需要某个 backend 操作，先将其加入 `SimBackend`，再使用。
- 使用证据等级的表述。例如 `Registered`、`Configured`、`Tested`、`Benchmarked`
  或 `Recommended`；没有仓库内证据就不要声称稳定支持。
- 上浮可复用原语。通用逻辑应放在 `src/unilab/base/` 或 `src/unilab/utils/`，
  不要在多个 workflow 中复制粘贴。
- 在最接近风险的边界处进行验证：Hydra 改动用 config 测试，观测/reset 改动用
  env 测试，runner 改动用 IPC 测试。

## 验证

在贴近风险处验证。顶层 smoke run 只能补充、不能替代对改动真正触及边界的验证。

| 改动类型 | 最少验证 |
| --- | --- |
| 仅文档 | `uv run pytest tests/scripts/test_check_docs.py -q`，并逐条对照仓库手动核对 support claim |
| Hydra / task / reward 配置 | `make test`（`tests/config/`、`tests/scripts/`） |
| Env contract / 观测 | `make test`（`tests/base/test_np_env.py` 与 env 测试）加 1 iteration smoke run |
| Runner / IPC | `make test`；必要时补 `make test-slow` |
| Backend 路径 | 对应 backend 的 smoke run，必要时补 slow test |
| 训练入口 | 相关测试加 1 iteration smoke run |

快速路径用 `make test`；提 PR 前用 `make test-all`（`make check` 加
`make test-cov`）。

## 评审清单

1. 这次改动触及了哪个 contract？
2. 这个问题是否应该在更低层解决？
3. backend 或 task 行为是通过 config 表达的，还是被脚本特判掩盖了？
4. support 声明是否有 registry / config / test / benchmark 证据？
5. 验证是否发生在最接近风险的边界？

## 关键文件

- `scripts/train_rsl_rl.py`
- `scripts/train_mlx_ppo.py`
- `scripts/train_appo.py`
- `scripts/train_offpolicy.py`
- `src/unilab/base/np_env.py`
- `src/unilab/base/backend/base.py`
- `src/unilab/base/registry.py`
- `src/unilab/ipc/async_runner.py`
- `src/unilab/training/run.py`

## 相关 ADR

- {doc}`ADR 索引 </adr/ADR-0000-index>`
- {doc}`运行时模型与分层边界 </adr/ADR-0001-runtime-model-and-layer-boundaries>`
- {doc}`Backend 能力边界 </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`
- {doc}`任务 owner 与 config compose 契约 </adr/ADR-0003-task-owner-and-config-compose-contract>`
- {doc}`Registry bootstrap 契约 </adr/ADR-0004-registry-bootstrap-contract>`
- {doc}`统一 obs/critic env 与 IPC 契约 </adr/ADR-0005-unified-obs-critic-env-and-ipc-contract>`

# RL Infrastructure 开发标准

语言: 简体中文

UniLab 是一个**高性能、模块化、contract 驱动**的 RL infrastructure 仓库。这个标准只回答一个问题: **什么样的改动是对的**。

工程属性: 高性能、结构化、系统性、模块化、可复用、可观测。

---

## 1. Runtime Model

三段式零拷贝管线:

```text
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
                                  ▲                   │
                                  └── SharedWeightSync ┘
```

- 后端切换通过 **contract + registry + config** 完成，而不是脚本分支
- Env 层保持 numpy / vectorized；GPU 由 learner 独占
- Collector 和 learner 通过 IPC + shared memory 解耦，并共享统一 lifecycle

---

## 2. Layered Architecture

依赖方向必须严格单向。**问题在哪一层产生，就在哪一层解决**。

| Layer | 目录 | 职责 | 不应承担 |
|-------|------|------|----------|
| L0 Backend | `base/backend/` | `SimBackend` 物理后端抽象 | 训练逻辑、reward |
| L1 Env | `envs/`, `base/np_env.py` | MDP 语义、observation、reward、reset | 调度、日志策略 |
| L2 Config & Registry | `config/`, `base/registry.py`, `conf/` | schema、task / reward 组合、注册 | 零散业务默认值 |
| L3 Algo & IPC | `algos/`, `ipc/` | learner、runner、collector、shared-memory 通路 | env / backend 细节 |
| L4 Scripts | `scripts/` | 只做装配 | 核心业务规则 |

---

## 3. Design Principles

1. **Contract first**: 先保护 contract，再做局部修补。承重墙包括 `registry.make`、`NpEnvState.obs: dict`、`reset -> (obs, info)`、`obs_groups_spec`、`SimBackend`，以及 collector / learner 共享内存协议。
2. **Own your layer**: scripts 不修 env bug，env 不修 backend bug。
3. **Config over branching**: 扩展优先级是 config schema -> registry -> env / backend 适配层 -> 最后才是脚本分支。
4. **Backend isolation**: MuJoCo / Motrix 差异收敛在 backend 实现、env 适配层和 backend-specific profile 中；能力缺口必须显式写出来。
5. **Evidence-graded claims**: 使用 `Registered`、`Configured`、`Benchmarked`、`Recommended` 这类表述；没有证据就不要写稳定支持。
6. **Validate near risk**: 顶层 smoke run 只能补充，不能替代贴近风险边界的验证。
7. **Reusable primitives**: 通用逻辑上浮到 `base/` 或 `utils/`，不要在多个 workflow 中复制粘贴。

---

## 4. Training Entrypoints

| 路径 | 入口 | 主链路 |
|------|------|--------|
| PPO (torch) | `scripts/train_rsl_rl.py` | `registry.make` -> `RslRlVecEnvWrapper` -> `rsl_rl.OnPolicyRunner` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | `registry.make` -> MLX `RolloutBuffer` -> `PPOTrainer` |
| APPO | `scripts/train_appo.py` | `APPORunner` -> collector -> `SharedOnPolicyStorage` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `OffPolicyRunner` -> collector -> `ReplayBuffer` |

动手前，先定位自己正在修改哪条链路。

---

## 5. Configuration

UniLab 使用 dataclass + Hydra。schema 位于 `src/unilab/config/structured_configs.py`，运行时配置位于 `conf/{ppo,appo,offpolicy}/`。

合成顺序: `{algo}/config*.yaml` -> `task=...` -> CLI override。

- `task` 是唯一 owner 配置入口：同一个 task + backend（offpolicy 再加 algo）对应一个 YAML，里面直接放这个组合的 `training.task_name` / `training.sim_backend` / `reward` / `env` / task-specific `algo`
- PPO / APPO 入口形如 `conf/{ppo,appo}/task/<task>/<backend>.yaml`；offpolicy 入口形如 `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`
- 这里的 `task` 不是旧设计里“只表达任务、再去别处拼 backend/reward/algo”的 group；它本身就是最终 owner 配置入口
- `training.sim_backend` 是 task owner YAML 的身份字段，不是独立的 backend switch；切换后端必须改 `task=.../<backend>`，不能只 override `training.sim_backend`
- reward 必须显式注入
- 如果 backend 选择会影响 task 或 reward 行为，就必须通过 config 表达
- 动态 override 必须尊重 CLI，但不能破坏 task owner 的 backend identity

---

## 6. Env

扩展流程:

1. 用 `@registry.envcfg("EnvName")` 注册 config dataclass
2. 用 `@registry.env("EnvName", sim_backend=...)` 注册实现类
3. 通过 `registry.make(...)` 构造

Env **负责** MDP 语义、observation 结构、reward、reset，以及 backend 数据到训练语义的映射。Env **不负责** 训练编排、多进程调度或顶层日志。

---

## 7. Backend

`SimBackend` (`src/unilab/base/backend/base.py`) 必须提供 base pose / velocity、DOF state、body pose / velocity（world 与 baselink 坐标系）以及 named sensor。

已知 backend-specific 分支包括: `backend_type == "motrix"` 会触发 `_process_rigid_body_props`；部分 play / debug / video / symmetry 路径仍然是 MuJoCo-first。

---

## 8. Async And Runner

所有异步算法共享 `src/unilab/ipc/async_runner.py` 中的 `AsyncRunner`: 统一的 spawn 模型、统一的 collector lifecycle、统一的 shared-resource cleanup。

- **APPO**: collector 写入 `SharedOnPolicyStorage`；learner 使用 V-trace；actor 权重通过 `SharedWeightSync` 回传
- **Off-policy**: collector 写入 `ReplayBuffer`；learner 从中采样；`SharedWeightSync` 同步权重；同时支持同步和异步采集

不要在 shared runner 之外复制并行协议、绕过 shared-resource lifecycle，或引入隐式耦合。

---

## 9. Validation

| 改动 | 最少验证 |
|------|----------|
| Hydra / task / reward | `make test`（`tests/config/`, `tests/scripts/`） |
| env contract / observation | `make test`（`tests/base/test_np_env.py` 等） |
| runner / IPC | `make test`；必要时补 `make test-slow` |
| 训练主链路 | 相关测试 + 1 iteration smoke run |
| backend 路径 | 对应 backend smoke run，必要时补 slow test |
| docs-only | 手动核对命令、路径、配置名、CI 和 support claim |

---

## 10. Review Checklist

1. 这次改动影响了哪个 contract？
2. 这个问题是否应该在更低层解决？
3. backend 或 task 行为是通过 config 表达的，还是被脚本特判掩盖了？
4. support 声明是否有 registry / config / test / benchmark 证据？
5. 验证是否发生在最接近风险的边界？

---

## 11. High-Signal Files

- `scripts/train_{rsl_rl,mlx_ppo,appo,offpolicy}.py`
- `src/unilab/base/{registry,np_env}.py`
- `src/unilab/base/backend/base.py`
- `src/unilab/config/structured_configs.py`
- `src/unilab/utils/{reward_utils,obs_utils}.py`
- `src/unilab/ipc/async_runner.py`

---

## Navigation

- Previous: [README](README.md)
- Next: [Getting Started](01-getting-started.md)

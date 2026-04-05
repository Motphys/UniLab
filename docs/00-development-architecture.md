# RL Infrastructure Development Standard

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。
本标准只回答一个问题：**什么样的改动是对的。**

工程属性：高性能 · 结构化 · 系统性 · 模块化 · 可复用 · 可观测。

---

## 1. Runtime Model

三段式零拷贝管线：

```
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
                                  ▲                   │
                                  └── SharedWeightSync ┘
```

- Backend 通过 **contract + registry + 配置** 切换，不通过脚本分支。
- Env 层是 numpy / vectorized；GPU 归 learner 独占。
- Collector 与 learner 通过 IPC + shared memory 解耦，lifecycle 统一。

---

## 2. Layered Architecture

依赖方向严格单向，**问题在哪一层产生，就在哪一层解决**。

| Layer | 目录 | 职责 | 禁止承担 |
|-------|------|------|---------|
| L0 Backend | `base/backend/` | `SimBackend` 物理后端抽象 | 训练逻辑、reward |
| L1 Env | `envs/`, `base/np_env.py` | MDP 语义、obs、reward、reset | 调度、日志策略 |
| L2 Config & Registry | `config/`, `base/registry.py`, `conf/` | schema、task / reward 组合、注册 | 散落业务默认值 |
| L3 Algo & IPC | `algos/`, `ipc/` | learner、runner、collector、shm 通路 | env/backend 细节 |
| L4 Scripts | `scripts/` | **只做装配** | 核心业务规则 |

---

## 3. Design Principles

1. **Contract-First** — 先保护 contract，再谈局部修补。承重墙：
   `registry.make`、`NpEnvState.obs: dict`、`reset → (obs, info)`、
   `obs_groups_spec`、`SimBackend`、collector/learner shm 协议。
2. **Own your layer** — scripts 不修 env bug，env 不修 backend bug。
3. **Config over branching** — 扩展优先级：
   config schema → registry → env/backend 适配层 →（最后）脚本分支。
4. **Backend isolation** — MuJoCo / Motrix 差异收敛在 backend 实现、
   env 适配层、backend-specific profile；capability gap 必须显式写出。
5. **Evidence-graded claims** — `Registered` / `Configured` /
   `Benchmarked` / `Recommended`，无证据不写"稳定支持"。
6. **Validate near risk** — 顶层 smoke 是补充，不是替代。
7. **Reusable primitives** — 通用逻辑上浮到 `base/` / `utils/`，不复制粘贴。

---

## 4. Training Entrypoints

| 路径 | 入口 | 主链路 |
|------|------|--------|
| PPO (torch) | `scripts/train_rsl_rl.py` | `registry.make` → `RslRlVecEnvWrapper` → `rsl_rl.OnPolicyRunner` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | `registry.make` → MLX `RolloutBuffer` → `PPOTrainer` |
| APPO | `scripts/train_appo.py` | `APPORunner` → collector → `SharedOnPolicyStorage` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `OffPolicyRunner` → collector → `ReplayBuffer` |

动手前先定位自己在哪条链路上。

---

## 5. Configuration

dataclass + Hydra。schema 在 `src/unilab/config/structured_configs.py`，
运行时配置在 `conf/{ppo,appo,offpolicy}/`。

合成顺序：`{algo}/config*.yaml` → `task=...` → `reward[_{backend}]` →
CLI override →（必要时）`motrix_legacy`。

- reward 必须显式注入。
- backend 影响 reward/task 时必须通过配置表达。
- 动态 override 必须尊重 CLI。

---

## 6. Env

扩展流程：

1. `@registry.envcfg("EnvName")` 注册 config dataclass
2. `@registry.env("EnvName", sim_backend=...)` 注册实现类
3. `registry.make(...)` 构造

Env **承担**：MDP 语义、obs 结构、reward、reset、backend→训练语义映射。
Env **不承担**：训练 orchestration、多进程调度、顶层日志。

---

## 7. Backend

`SimBackend` (`src/unilab/base/backend/base.py`) 必须提供：
base pose/vel、DOF state、body pose/vel（world & baselink）、named sensor。

已知 backend 分支：`backend_type == "motrix"` 触发 `_process_rigid_body_props`；
play/debug/video/symmetry 部分路径仍 MuJoCo-first。

---

## 8. Async & Runner

所有异步算法共享 `src/unilab/ipc/async_runner.py` 的 `AsyncRunner`：
统一 `spawn`、统一 collector lifecycle、统一 shared resource cleanup。

- **APPO**：collector 写 `SharedOnPolicyStorage`，learner V-trace，
  actor 权重经 `SharedWeightSync` 回传。
- **Off-policy**：collector 写 `ReplayBuffer`，learner 采样，
  `SharedWeightSync` 同步，支持 sync / async collection。

禁止：外部复制并行协议、绕过 shared resource lifecycle、引入隐式耦合。

---

## 9. Validation

| 改动 | 最少验证 |
|------|---------|
| Hydra / task / reward | `make test`（`tests/config/`, `tests/scripts/`） |
| env contract / obs | `make test`（`tests/base/test_np_env.py` 等） |
| runner / IPC | `make test`，必要时 `make test-slow` |
| 训练主链路 | 相关测试 + 1-iteration smoke run |
| backend path | 对应 backend smoke run，必要时 slow test |
| docs-only | 手动核对命令、路径、配置名、CI、support claim |

---

## 10. Review Checklist

1. 这次改动影响了哪个 contract？
2. 是否应该在**更低层**解决？
3. backend / task 行为通过**配置**表达还是被**脚本特判**掩盖？
4. support 声明是否有 registry / config / test / benchmark 证据？
5. 验证是否发生在**最接近风险**的边界？

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

- Previous: [README](../README.md)
- Next: [Getting Started](01-getting-started.md)

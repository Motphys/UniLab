# RL Infrastructure Development Standard

Languages: English | [简体中文](../zh_CN/00-development-architecture.md) | [日本語](../ja/00-development-architecture.md) | [한국어](../ko/00-development-architecture.md)

UniLab is a **high-performance, modular, contract-driven** RL infrastructure repository. This standard answers one question only: **what kind of change is correct**.

Engineering attributes: high performance, structured, systematic, modular, reusable, observable.

---

## 1. Runtime Model

Three-stage zero-copy pipeline:

```text
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
                                  ▲                   │
                                  └── SharedWeightSync ┘
```

- Switch backends through **contracts + registry + config**, not script branches
- Keep envs numpy-based and vectorized; reserve GPU ownership for the learner
- Decouple collectors and learners through IPC + shared memory with a unified lifecycle

---

## 2. Layered Architecture

Dependencies are strictly one-way. **Fix a problem in the layer that owns it**.

| Layer | Directory | Responsibility | Must Not Own |
|-------|-----------|----------------|--------------|
| L0 Backend | `base/backend/` | `SimBackend` abstraction over physics backends | training logic, rewards |
| L1 Env | `envs/`, `base/np_env.py` | MDP semantics, observations, rewards, resets | scheduling, logging policy |
| L2 Config & Registry | `config/`, `base/registry.py`, `conf/` | schema, task / reward composition, registration | scattered business defaults |
| L3 Algo & IPC | `algos/`, `ipc/` | learners, runners, collectors, shared-memory paths | env / backend details |
| L4 Scripts | `scripts/` | assembly only | core business rules |

---

## 3. Design Principles

1. **Contract first**: protect the contract before local patching. Load-bearing walls include `registry.make`, `NpEnvState.obs: dict`, `reset -> (obs, info)`, `obs_groups_spec`, `SimBackend`, and the collector / learner shared-memory protocol.
2. **Own your layer**: scripts do not fix env bugs, and envs do not fix backend bugs.
3. **Config over branching**: extend in this order: config schema -> registry -> env / backend adapter layer -> script branch only as a last resort.
4. **Backend isolation**: keep MuJoCo / Motrix differences inside backend implementations, env adapter layers, and backend-specific profiles. Capability gaps must be explicit.
5. **Evidence-graded claims**: use `Registered`, `Configured`, `Benchmarked`, and `Recommended`. Do not claim stable support without evidence.
6. **Validate near risk**: top-level smoke runs are complements, not substitutes.
7. **Reusable primitives**: lift generic logic into `base/` or `utils/`; do not copy and paste it across workflows.

---

## 4. Training Entrypoints

| Path | Entrypoint | Main Chain |
|------|------------|------------|
| PPO (torch) | `scripts/train_rsl_rl.py` | `registry.make` -> `RslRlVecEnvWrapper` -> `rsl_rl.OnPolicyRunner` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | `registry.make` -> MLX `RolloutBuffer` -> `PPOTrainer` |
| APPO | `scripts/train_appo.py` | `APPORunner` -> collector -> `SharedOnPolicyStorage` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `OffPolicyRunner` -> collector -> `ReplayBuffer` |

Locate the chain you are changing before you start editing.

---

## 5. Configuration

UniLab uses dataclasses plus Hydra. The schema lives in `src/unilab/config/structured_configs.py`; runtime configuration lives in `conf/{ppo,appo,offpolicy}/`.

Composition order: `{algo}/config*.yaml` -> `task=...` -> `reward[_{backend}]` -> CLI override -> `motrix_legacy` when needed.

- Rewards must be injected explicitly
- If backend selection changes task or reward behavior, express it through config
- Dynamic overrides must respect the CLI

---

## 6. Env

Extension flow:

1. Register the config dataclass with `@registry.envcfg("EnvName")`
2. Register implementation classes with `@registry.env("EnvName", sim_backend=...)`
3. Construct via `registry.make(...)`

An env **owns** MDP semantics, observation structure, rewards, resets, and the mapping from backend data to training semantics. An env **does not own** training orchestration, multiprocessing, or top-level logging.

---

## 7. Backend

`SimBackend` (`src/unilab/base/backend/base.py`) must provide base pose / velocity, DOF state, body pose / velocity in world and baselink frames, and named sensors.

Known backend-specific branches: `backend_type == "motrix"` triggers `_process_rigid_body_props`; some play / debug / video / symmetry paths are still MuJoCo-first.

---

## 8. Async And Runner

All async algorithms share `AsyncRunner` in `src/unilab/ipc/async_runner.py`: one spawn model, one collector lifecycle, and one shared-resource cleanup path.

- **APPO**: collectors write to `SharedOnPolicyStorage`; the learner uses V-trace; actor weights flow back through `SharedWeightSync`
- **Off-policy**: collectors write to `ReplayBuffer`; the learner samples it; `SharedWeightSync` synchronizes weights; both sync and async collection modes are supported

Do not duplicate the parallel protocol outside the shared runner, bypass the shared-resource lifecycle, or introduce hidden coupling.

---

## 9. Validation

| Change | Minimum Validation |
|--------|--------------------|
| Hydra / task / reward | `make test` (`tests/config/`, `tests/scripts/`) |
| env contract / observations | `make test` (`tests/base/test_np_env.py`, etc.) |
| runner / IPC | `make test`; add `make test-slow` when needed |
| training main path | relevant tests + one-iteration smoke run |
| backend path | backend-specific smoke run, and slow tests when needed |
| docs-only | manually verify commands, paths, config names, CI, and support claims |

---

## 10. Review Checklist

1. Which contract does this change affect?
2. Should the fix live in a lower layer?
3. Is backend or task behavior expressed through config, or hidden behind script special cases?
4. Does every support claim have registry / config / test / benchmark evidence?
5. Did validation happen at the boundary closest to the risk?

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

- Previous: [README](../../README.md)
- Next: [Getting Started](01-getting-started.md)

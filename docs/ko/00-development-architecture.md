# RL Infrastructure 개발 표준

언어: [English](../en/00-development-architecture.md) | [简体中文](../zh_CN/00-development-architecture.md) | [日本語](../ja/00-development-architecture.md) | 한국어

UniLab은 **고성능, 모듈형, contract-driven** RL infrastructure 저장소입니다. 이 표준은 한 가지 질문에만 답합니다. **어떤 변경이 올바른가**입니다.

엔지니어링 속성: 고성능, 구조화, 체계성, 모듈성, 재사용성, 관측 가능성.

---

## 1. Runtime Model

3단계 zero-copy pipeline:

```text
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
                                  ▲                   │
                                  └── SharedWeightSync ┘
```

- backend 전환은 script 분기가 아니라 **contract + registry + config**로 처리합니다
- Env 계층은 numpy / vectorized를 유지하고 GPU는 learner가 독점합니다
- collector와 learner는 IPC + shared memory로 분리하되 lifecycle은 통일합니다

---

## 2. Layered Architecture

의존 방향은 엄격하게 단방향이어야 합니다. **문제가 생긴 계층에서 문제를 해결해야 합니다**.

| Layer | 디렉터리 | 책임 | 가지면 안 되는 책임 |
|-------|----------|------|---------------------|
| L0 Backend | `base/backend/` | 물리 backend 추상 `SimBackend` | 학습 로직, reward |
| L1 Env | `envs/`, `base/np_env.py` | MDP 의미론, observation, reward, reset | scheduling, logging policy |
| L2 Config & Registry | `config/`, `base/registry.py`, `conf/` | schema, task / reward 조합, 등록 | 흩어진 비즈니스 기본값 |
| L3 Algo & IPC | `algos/`, `ipc/` | learner, runner, collector, shared-memory 경로 | env / backend 세부사항 |
| L4 Scripts | `scripts/` | 조립만 담당 | 핵심 비즈니스 규칙 |

---

## 3. Design Principles

1. **Contract first**: 지역 패치보다 contract 보호가 우선입니다. 핵심 축은 `registry.make`, `NpEnvState.obs: dict`, `reset -> (obs, info)`, `obs_groups_spec`, `SimBackend`, collector / learner shared-memory protocol입니다.
2. **Own your layer**: scripts는 env bug를 고치지 않고 env는 backend bug를 고치지 않습니다.
3. **Config over branching**: 확장 순서는 config schema -> registry -> env / backend adapter 계층 -> 마지막 수단으로 script 분기입니다.
4. **Backend isolation**: MuJoCo / Motrix 차이는 backend 구현, env adapter 계층, backend-specific profile 안에 가둬야 합니다. capability gap은 명시적으로 적어야 합니다.
5. **Evidence-graded claims**: `Registered`, `Configured`, `Benchmarked`, `Recommended`를 사용하세요. 근거 없이 stable support를 주장하지 마세요.
6. **Validate near risk**: 최상위 smoke run은 보완일 뿐, 위험 경계 근처 검증의 대체물이 아닙니다.
7. **Reusable primitives**: 범용 로직은 `base/`나 `utils/`로 끌어올리고 workflow마다 복붙하지 마세요.

---

## 4. Training Entrypoints

| 경로 | 엔트리포인트 | 메인 체인 |
|------|--------------|-----------|
| PPO (torch) | `scripts/train_rsl_rl.py` | `registry.make` -> `RslRlVecEnvWrapper` -> `rsl_rl.OnPolicyRunner` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | `registry.make` -> MLX `RolloutBuffer` -> `PPOTrainer` |
| APPO | `scripts/train_appo.py` | `APPORunner` -> collector -> `SharedOnPolicyStorage` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `OffPolicyRunner` -> collector -> `ReplayBuffer` |

수정하기 전에 자신이 어느 체인을 건드리는지 먼저 파악하세요.

---

## 5. Configuration

UniLab은 dataclass + Hydra를 사용합니다. schema는 `src/unilab/config/structured_configs.py`에 있고 runtime config는 `conf/{ppo,appo,offpolicy}/`에 있습니다.

합성 순서: `{algo}/config*.yaml` -> `task=...` -> `reward[_{backend}]` -> CLI override -> 필요할 때 `motrix_legacy`.

- reward는 명시적으로 주입해야 합니다
- backend 선택이 task 또는 reward 동작을 바꾼다면 config로 표현해야 합니다
- 동적 override는 CLI를 존중해야 합니다

---

## 6. Env

확장 절차:

1. `@registry.envcfg("EnvName")`로 config dataclass를 등록합니다
2. `@registry.env("EnvName", sim_backend=...)`로 구현 클래스를 등록합니다
3. `registry.make(...)`로 생성합니다

Env가 **가지는 책임**은 MDP 의미론, observation 구조, reward, reset, backend 데이터에서 학습 의미론으로의 매핑입니다. Env는 학습 orchestration, 멀티프로세스 제어, 상위 logging을 가지지 않습니다.

---

## 7. Backend

`SimBackend` (`src/unilab/base/backend/base.py`)는 base pose / velocity, DOF state, body pose / velocity (world와 baselink 좌표계), named sensor를 제공해야 합니다.

알려진 backend-specific 분기에는 `backend_type == "motrix"`에 따른 `_process_rigid_body_props`가 있으며, 일부 play / debug / video / symmetry 경로는 아직 MuJoCo-first입니다.

---

## 8. Async And Runner

모든 async algorithm은 `src/unilab/ipc/async_runner.py`의 `AsyncRunner`를 공유합니다. spawn 모델, collector lifecycle, shared-resource cleanup을 통일합니다.

- **APPO**: collector는 `SharedOnPolicyStorage`에 쓰고 learner는 V-trace를 사용하며 actor weight는 `SharedWeightSync`로 되돌립니다
- **Off-policy**: collector는 `ReplayBuffer`에 쓰고 learner가 이를 sample하며 `SharedWeightSync`로 weight를 동기화하고 sync / async collection을 모두 지원합니다

shared runner 바깥에서 병렬 protocol을 복제하거나 shared-resource lifecycle을 우회하거나 암묵적 결합을 만들지 마세요.

---

## 9. Validation

| 변경 | 최소 검증 |
|------|-----------|
| Hydra / task / reward | `make test` (`tests/config/`, `tests/scripts/`) |
| env contract / observation | `make test` (`tests/base/test_np_env.py` 등) |
| runner / IPC | `make test`, 필요하면 `make test-slow` |
| 학습 메인 경로 | 관련 테스트 + 1-iteration smoke run |
| backend 경로 | backend별 smoke run, 필요하면 slow test |
| docs-only | 명령, 경로, config 이름, CI, support claim 수동 확인 |

---

## 10. Review Checklist

1. 이번 변경은 어떤 contract에 영향을 주는가?
2. 더 낮은 계층에서 해결해야 하는 문제는 아닌가?
3. backend / task 동작이 config로 표현되어 있는가, 아니면 script 특수 처리 뒤에 숨어 있는가?
4. 모든 support claim에 registry / config / test / benchmark 근거가 있는가?
5. 검증이 가장 위험에 가까운 경계에서 수행되었는가?

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

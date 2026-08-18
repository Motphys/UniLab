---
orphan: true
---

# ADR-0006 Community Manager API On NumPy Runtime

语言: 简体中文

- Status: Accepted
- Date: 2026-08-17
- Owners: Env / Config / Backend maintainers
- Supersedes: None
- Superseded by: None

## Context

UniLab 的 `NpEnv`、Hydra owner YAML、registry、`SimBackend` 与 heterogeneous
training runtime 已形成稳定 contract，但 task 的 observation、action、reward、
termination、event、command 与 curriculum 仍主要由各 env 的私有方法组装。用户迁移
Isaac Lab 或 mjlab task 时，需要重写 manager term、配置和 lifecycle。

本决策采用 mjlab v1.6.0 的 manager package 作为可逐文件审查的迁移基线：

- repository: `mujocolab/mjlab`
- commit: `0fb8a681136be94ffc636a3dd423cabb97d91f10`
- source: `src/mjlab/managers/` 的 12 个 Python 文件
- license: Apache-2.0；上游 `LICENSE` 声明
  `Copyright 2025, The mjlab Developers`

该基线只定义 manager-facing API 与语义。它不把 mjlab 的 Torch、Warp、scene
composer、viewer、simulation 或 training runtime 带入 UniLab，也不恢复或参考 UniLab
历史上的 Manager-Based API 实现。

## Decision

### 1. Source-aligned public surface

`src/unilab/managers/` 按 pinned mjlab package 的模块职责和 exports 直接迁移。以下名称
是 canonical public surface；Torch 类型替换为 NumPy 类型不构成改名：

| Module | Canonical exports |
| --- | --- |
| `manager_base` | `ManagerBase`, `ManagerTermBase`, `ManagerTermBaseCfg` |
| `action_manager` | `ActionManager`, `ActionTerm`, `ActionTermCfg` |
| `observation_manager` | `ObservationManager`, `ObservationGroupCfg`, `ObservationTermCfg` |
| `reward_manager` | `RewardManager`, `RewardTermCfg` |
| `termination_manager` | `TerminationManager`, `TerminationTermCfg` |
| `event_manager` | `EventManager`, `EventMode`, `EventTermCfg` |
| `command_manager` | `CommandManager`, `CommandTerm`, `CommandTermCfg`, `NullCommandManager` |
| `curriculum_manager` | `CurriculumManager`, `CurriculumTermCfg`, `NullCurriculumManager` |
| `metrics_manager` | `MetricsManager`, `MetricsTermCfg`, `NullMetricsManager` |
| `recorder_manager` | `RecorderManager`, `RecorderTerm`, `RecorderTermCfg`, `NullRecorderManager` |
| `scene_entity_config` | `SceneEntityCfg` |

Manager cfg 使用 plain dataclass instance；term 集合使用保持插入顺序的 typed `dict`。
`func + params`、function/class term、class term 的 `(cfg, env)` 构造和局部
`reset(env_ids)` 语义保持不变。显式空配置或 term 值为 `None` 表示用户选择禁用，允许
使用 upstream Null manager/no-op 语义。

未来 env lifecycle 的 canonical 名称沿用迁移源的 `ManagerBasedRlEnv` 与
`ManagerBasedRlEnvCfg`。如果为 Isaac Lab 拼写提供 `ManagerBasedRLEnv` /
`ManagerBasedRLEnvCfg`，它们必须是同一对象的无分支 alias，不能形成第二套实现。
其他别名必须由实际 migration fixture 证明有价值，不能预先扩张 API。

### 2. NumPy runtime boundary

Manager-facing tensor、buffer、term return、env IDs 和 entity view 使用
`np.ndarray` 或 `slice`。Torch 的 `device`、`.to()`、`.cpu()` 与 Tensor-only API 不属于
UniLab manager contract；manager package 不能 import Torch、runner、learner 或 IPC。

数值转换保持下列语义：

- shape、dtype 和更新时序与上游一致；action history 与 observation history 不改变顺序；
- buffer 在 manager 构造或 reset owner 边界分配，step 热路径复用；
- 随机采样使用由 env 拥有并可复现的 NumPy generator，不依赖进程全局 RNG；
- shape 不匹配以及非有限 term 输出在最近 manager/term 边界直接报错；reward 不使用
  `nan_to_num` 把非法值静默变成零；
- observation 明确配置的 noise/delay/history/NaN policy 可以保留，但默认不能掩盖非法
  输出。

### 3. UniLab env、config 与 IPC boundary

Managers 只依赖一个 typed env context。P0 context 包含 `num_envs`、physics/control dt、
episode counters、NumPy RNG、各 manager 属性，以及正式 scene/entity facade；不能要求
`device` 或 backend 私有对象。

Manager 内可以使用社区常见的 `policy` / `actor` / `critic` observation group。env owner
必须显式把 actor-facing group 映射为 `NpEnvState.obs["obs"]`，并把可选 critic group
映射为 `NpEnvState.obs["critic"]`。runner、learner 与 IPC 不推断、不拼接 group。
`reset() -> (obs_dict, info_dict)`、final observation 与 `obs_groups_spec` 保持现有 contract。

Production task 的唯一配置 source of truth 是 Hydra owner YAML。它完整声明 scene/backend
tuning、manager/group/term 的顺序与启停、具体 cfg 类型、callable、params、weight 和
observation group mapping；compose 后按以下冷路径进入现有 registry：

`owner YAML -> DictConfig -> typed config materialization -> Registry factory -> ManagerBasedRlEnv`

具体 cfg 类型使用 Hydra `_target_`，term callable 使用完整 dotted reference。通用
materializer 将其解析为 plain dataclass instance，并在未知字段、target/callable 解析失败、
抽象或错误 term cfg 类型及缺少必填字段时 fail-closed。Python 只拥有 term 实现、公共 cfg
类型和通用 factory，不保存第二份 task-specific term 清单或默认值；直接构造 typed cfg
只用于底层单测。DictConfig 与解析逻辑不能进入 reset/step 热路径，scripts 不解释 term
业务规则。

### 4. Scene/entity owner boundary

`SceneEntityCfg` 和 term 所需的最小 NumPy entity facade 属于 `src/unilab/base/` 公共
contract；backend 负责通过 `SimBackend` materialize 名称、ID 和 state/control view，env
负责把 facade 组合进 manager context。该决策解决 #586 的 owner 问题，但不引入完整
scene composer 或通用 asset hierarchy。

- entity name 以及 joint/body/geom/site/actuator selector 在 init/materialization/cache
  冷路径解析一次；热路径只持有已解析 `list[int]`、`np.ndarray` 或 `slice`；
- selector 保留上游 name/regex、`preserve_order`、names/IDs consistency check 和全选压缩为
  `slice(None)` 的语义；
- root/joint/body/site/geom/control 能力只能来自 `SimBackend` 已声明方法；不暴露 backend
  model/data 私有对象；
- tendon/camera/light/material/texture/pair 等迁移表面可以存在，但 backend 未声明能力时在
  resolve/materialization 直接 `NotImplementedError`，不能返回空 ID 或跳过；
- 新 backend 能力必须作为独立 child 扩展 `SimBackend` 并补 conformance tests，不能在
  manager 或 env 中用 `getattr` / `hasattr` 探测私有实现。

Named sensor 使用 `SimBackend.bind_sensor_data(names)` 在 materialization 冷路径校验名称、
每个 sensor 的展平宽度、batch shape 与 finite 值，并返回 immutable
`BackendSensorView`。term 热路径只调用 `view.read()`；MuJoCo、Drake 和 MJWarp adapter
分别保留已解析的 host-cache slice 或数值 slot。MotrixSim 当前公开接口只提供 named
sensor accessor、没有数值 sensor ID，因此 Motrix adapter 在 scene materialization 时缓存
可用名称，并把原生批量 accessor 与 immutable 名称 tuple 封装为 backend-owned opaque
reader；term 不接触名称解析、XML 或 model metadata，未知名称在进入原生调用前 fail-closed。

这是 pinned mjlab sensor-facing 语义的 intentional NumPy/backend adaptation：社区侧的
tensor/device view 在 UniLab 表达为按请求名称顺序拼接的二维 NumPy batch
`(num_envs, sum(sensor_widths))`。名称顺序、单 sensor 宽度和当前值可见，Torch device、
backend model/data 与原生 handle 不属于 manager contract。

### 5. Fail-closed capability rule

用户显式禁用与实现缺失是两种不同状态。前者允许 Null manager；后者必须失败：

| Failure | Required behavior |
| --- | --- |
| cfg/term 类型错误、签名或 shape 不匹配 | `TypeError` / `ValueError`，包含 manager 与 term |
| term 输出 NaN/Inf | `ValueError`，包含 manager、group/term 与非法值类别 |
| backend/entity capability 未实现 | `NotImplementedError`，包含 manager、term、capability 与 backend |
| selector name/ID 不存在或不一致 | `KeyError` / `ValueError`，包含 entity 与 selector |

不得 warning 后 skip、返回零/旧值、自动换 backend、禁用 feature 或回退到旧 env。当前
不新增公共 exception hierarchy；只有 consumer 证明需要 machine-readable 分类时再单独
决策。

### 6. Performance and deletion policy

优先级固定为：社区 Manager-Based API 语义与结构一致性，优先于改变公共设计的局部性能
优化。在此约束下，生产级 NumPy 热路径不能引入明显可避免的重复解析、逐环境 Python
循环、数组复制或临时分配。优化必须由同配置、同硬件 benchmark 证明有足够收益，并优先
保持在内部预解析、预分配和批量 NumPy 实现；低收益但增加专用 fast path、缓存协议或长期
复杂度的方案不采用。

Production task 迁移后必须在同一 task-family child 删除被替代的旧 dispatch、重复
reward/config helper 和 bridge。Umbrella 完成时只保留一套 manager lifecycle，不保留
fallback 到旧单体 env 的永久兼容路径。

## Stable Contracts

### Compatibility matrix

状态只表示本 ADR 固定的迁移目标；实际 support claim 仍需要代码、注册、配置和测试证据。

| Surface | Target | Notes |
| --- | --- | --- |
| manager modules、class/config names、dict order | Compatible | 直接保留 pinned mjlab 1.6.0 表面 |
| function/class term、`params`、local reset | Compatible | class term 在冷路径实例化 |
| action split/apply/history、reward dt scaling、termination timeout split | Compatible | NumPy 实现保持时序 |
| observation groups、clip/scale/noise/delay/history | Adapted | 数值为 NumPy；group 在 env boundary 显式映射 |
| manager buffers、env IDs、RNG | Adapted | Torch→NumPy；无 device API |
| `ManagerBasedRlEnv` return | Adapted | 保留 `NpEnvState` 与 UniLab reset/final-observation contract |
| config container | Adapted | Hydra owner YAML 唯一持有 task 配置，冷路径物化为 plain typed instances |
| `SceneEntityCfg` selectors | Adapted | 语义保留；只解析 `SimBackend` 已声明能力 |
| named sensor view | Adapted | 冷路径 bind；有序展平 NumPy batch；reader 由 backend 拥有 |
| event/domain randomization | Adapted | 调度语义保留；mutation 走 backend DR/capability contract |
| Metrics/Recorder | Adapted | lifecycle hook 存在时启用；缺失时显式失败或显式空配置 |
| Torch device、Warp mutation、viewer glue | Unsupported | 不进入 manager core，不提供静默替代 |
| Omniverse/USD/mjlab Scene/Simulation | Unsupported | 不属于 UniLab runtime |

### Mechanical migration example

迁移前的 mjlab term：

```python
import torch
from mjlab.managers import RewardTermCfg

def joint_error(env) -> torch.Tensor:
    return torch.square(env.joint_pos - env.target_joint_pos).sum(dim=1)

term = RewardTermCfg(func=joint_error, weight=-1.0)
```

迁移后的 UniLab term 只改 import、数值类型和对应 NumPy 运算：

```python
import numpy as np
from unilab.managers import RewardTermCfg

def joint_error(env) -> np.ndarray:
    return np.square(env.joint_pos - env.target_joint_pos).sum(axis=1)

term = RewardTermCfg(func=joint_error, weight=-1.0)
```

如果迁移还要求重写 term 结构、增加 backend 分支或改 runner/IPC，说明 adapter boundary
不够薄，必须停止并拆出 owner child。

### Provenance and change accounting

每个 source-derived Python 文件必须注明上游 repository、tag/commit、原始路径、
Apache-2.0 和 UniLab 的修改类别。实现 PR 分别报告：

1. source-derived：保留的上游结构/语义；
2. mechanical：import、typing、Torch→NumPy 和格式转换；
3. UniLab-specific glue：新 facade、contract adapter 或行为；
4. deleted：删除的上游不适用代码和 UniLab 旧实现。

不得通过重新分类隐藏 glue 超预算；不建立长期 upstream mirror 或自动 sync tooling。

## Alternatives Considered

- 重新设计一套更适合 UniLab 的 managers，再提供兼容 facade。拒绝：会形成
  UniLab-only 方言和两套行为，增加用户迁移与长期维护成本。
- 在 manager 热路径保留 Torch。拒绝：破坏 NumPy runtime、backend isolation 与
  heterogeneous CPU physics → accelerator learner 数据面。
- 一次迁移完整 mjlab scene/simulation/entity runtime。拒绝：复制第二套 backend/scene
  abstraction，并引入 Warp/MuJoCo/Viewer 假设。
- 先设计 compiler、fused term protocol 或专用 fast path。拒绝：在 benchmark 证明瓶颈前
  增加结构复杂度，并可能牺牲社区 term 语义。
- 缺失能力 warning + skip 或回退旧 env。拒绝：配置表面与真实执行不一致，不能用于生产。
- task-owned Python factory 声明 callable/term，Hydra 只做字段 overlay。拒绝：会让同一 task
  在 Python 和 YAML 中拥有两份配置，增加迁移 friction 和语义漂移。

## Consequences

- Manager port 的审查基线是 pinned upstream diff，而不是重新解释每个 manager 的职责。
- NumPy、UniLab env/config contract 和显式 unsupported 是允许的偏离；其他偏离必须在
  compatibility matrix 中先记录。
- Scene/entity 采用最小 base facade，#586 不再阻塞 manager port；真实 backend 能力仍按
  独立 child 和 conformance evidence 接入。
- Config/Registry 永久维护一个通用 Hydra `_target_` / dotted callable 到 typed manager cfg
  的冷路径 materializer；production task 不维护 Python config mirror。
- 迁移初期允许 production 旧 task 与未接入的 manager package 同时存在，但 task 一旦迁移
  就必须删除对应旧实现；umbrella 结束时不能保留双 lifecycle。
- 性能 gate 关注明显低效与实测瓶颈，不以复杂度换取未经证明的小收益。

## Evidence In Repo

- Env contract: `src/unilab/base/np_env.py`
- Backend contract: `src/unilab/base/backend/base.py`
- Scene config owner: `src/unilab/base/scene.py`
- Config schema and registry: `src/unilab/structured_configs.py`,
  `src/unilab/base/config_materialization.py`, `src/unilab/base/registry.py`, `conf/`
- Observation/IPC contract: `docs/sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md`
- Layer boundary: `docs/sphinx/source/adr/ADR-0001-runtime-model-and-layer-boundaries.md`
- Upstream checkout used for the decision:
  `/home/user/ws/simulator/mjlab/src/mjlab/managers/` at `0fb8a681`

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`Manager-Based API contract </zh_CN/4-developer_guide/1-architecture/6-manager_based_api>`
- {doc}`RL Infrastructure 开发标准 </zh_CN/4-developer_guide/0-index>`
- [Roadmap #1042](https://github.com/unilabsim/UniLab/issues/1042)
- [Implementation issue #1043](https://github.com/unilabsim/UniLab/issues/1043)
- [Entity abstraction decision #586](https://github.com/unilabsim/UniLab/issues/586)

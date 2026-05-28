# Domain Randomization Contract


本页记录 domain randomization 的开发者 contract、MuJoCo `BatchEnvPool` 随机场接口快照，以及新任务接入最低标准。用户侧配置入口见 {doc}`Domain Randomization </en/user_guide/domain_randomization/index>`。

## Contract Boundary

Domain randomization 必须遵守三类生命周期边界：

- **init-lifecycle**：改变模型身份或模型几何的项，只能在 env/backend 初始化、materialization 或 cache build 等冷路径处理。
- **reset-lifecycle**：不改变模型身份、只改变同一个模型内参数或 reset 状态的项，通过 `ResetPlan.randomization` 下发。
- **interval-lifecycle**：step 间扰动，通过 `IntervalRandomizationPlan` 下发。

热路径不得读取 XML、解析 asset，或用 `getattr` / `hasattr` 探测 backend 私有能力决定 DR 行为。需要新增 backend 能力时，先把能力升格为显式 contract / capability。

## MuJoCo `BatchEnvPool` 随机场字段接口现状

这部分只描述 `python/mujoco/batch_env.py` / `python/mujoco/batch_env.cc` 当前已经暴露的接口，不推断未来设计。

### 当前支持的字段

当前 `SUPPORTED_FIELDS` 为：

- `body_mass`
- `body_ipos`
- `body_iquat`
- `body_inertia`
- `dof_armature`
- `gravity`
- `geom_friction`
- `kp`
- `kd`

注意：

- `geom_size` 不在 reset randomization 的 `SUPPORTED_FIELDS` 里；它在 UniLab 中通过 init-lifecycle model materialization 表达。
- 如果当前环境仍安装 `mujoco-uni==3.6.0.post6`，则 `gravity` 也不在发布包的 `SUPPORTED_FIELDS` 里；需要使用包含 gravity reset randomization 的本地构建或后续发布版本。

### 当前整块替换方式

当前整块替换入口仍然是：

- `BatchEnvPool.reset(env_ids, initial_state, randomization=...)`

其中：

- `env_ids` 指定这次 reset 要落到哪些 env
- `randomization` 是 `dict[str, ndarray]`
- key 必须属于上面的 `SUPPORTED_FIELDS`
- value 的首维必须等于 `len(env_ids)`
- value 的其余元素个数必须等于该字段在单个 `mjModel` 里的整块大小

当前字段大小按底层实现是：

- `body_mass`: `nbody`
- `body_ipos`: `3 * nbody`
- `body_iquat`: `4 * nbody`
- `body_inertia`: `3 * nbody`
- `dof_armature`: `nv`
- `gravity`: `3`
- `geom_friction`: `3 * ngeom`
- `kp`: `nu`
- `kd`: `nu`

对应到传参形状，可以理解为：

- `body_mass`: `(len(env_ids), nbody)`
- `body_ipos`: `(len(env_ids), 3 * nbody)`
- `body_iquat`: `(len(env_ids), 4 * nbody)`
- `body_inertia`: `(len(env_ids), 3 * nbody)`
- `dof_armature`: `(len(env_ids), nv)`
- `gravity`: `(len(env_ids), 3)`
- `geom_friction`: `(len(env_ids), 3 * ngeom)`
- `kp`: `(len(env_ids), nu)`
- `kd`: `(len(env_ids), nu)`

整块接口的语义仍然是“按 env 子集、按字段整块替换”。

### 当前读取方式

当前读取入口分成两类：

- 整块读取：`pool.get_field(env_id, name) -> np.ndarray`
- 索引读取：`pool.get_field_indexed(env_id, name, indices)`

其中索引读取当前支持：

- `indices` 为单个 `int`
- `indices` 为 `Sequence[int]`

返回语义当前是稳定的：

- `body_ipos` / `body_inertia` / `gravity` / `geom_friction`
  - 单索引返回 `(3,)`
  - 多索引返回 `(k, 3)`
- `body_iquat`
  - 单索引返回 `(4,)`
  - 多索引返回 `(k, 4)`
- `body_mass` / `dof_armature` / `kp` / `kd`
  - 单索引返回标量
  - 多索引返回 `(k,)`

`gravity` 只有一个逻辑索引 `0`，因此索引读取通常只用于读取整条 gravity 向量。

### 当前局部写入方式

当前局部写入入口是：

- `pool.set_field_indexed(env_id, name, indices, value)`

其中：

- `env_id` 只作用于一个目标 env
- `name` 必须属于 `SUPPORTED_FIELDS`
- `indices` 支持单个 `int` 或 `Sequence[int]`
- `value` 的 shape 必须和字段分量语义匹配

当前 setter 语义是：

- `body_ipos` / `body_inertia` / `gravity` / `geom_friction`
  - 单索引写入要求 `value.shape == (3,)`
  - 多索引写入要求 `value.shape == (k, 3)`
- `body_iquat`
  - 单索引写入要求 `value.shape == (4,)`
  - 多索引写入要求 `value.shape == (k, 4)`
- `body_mass` / `dof_armature` / `kp` / `kd`
  - 单索引写入要求 `value` 为标量
  - 多索引写入要求 `value.shape == (k,)`

`gravity` 的索引写入同样只应使用逻辑索引 `0`；常规 reset DR 更推荐通过整块 payload `gravity: (len(env_ids), 3)` 下发。

因此现在有两种使用方式：

- 如果要一次替换一个字段在多个 env 上的整块值，继续用 `reset(..., randomization=...)`
- 如果只想改单个 env 内的某几个 body / geom / dof / actuator 条目，直接用 `get_field_indexed` / `set_field_indexed`

这意味着像“只改某个 geom 的 friction”这类场景，已经不需要“先整块读出 flat payload、手工切片、再整块回写”的上层样板代码。

另外，当前底层对 refresh 的处理也已经固定：

- `body_mass`、`body_ipos`、`body_iquat`、`body_inertia`、`dof_armature` 会触发 `mj_setConst` refresh
- `gravity`、`geom_friction`、`kp`、`kd` 不触发 refresh

因此，当前 MuJoCo `BatchEnvPool` 的 reset-lifecycle randomization 接口可以概括为：

- 支持字段是固定白名单
- 读取方式同时支持整块读取和索引读取
- 写入方式同时支持整块替换和单 env 内的索引级局部写入
- 当前索引级接口按字段分量宽度返回稳定 shape，并且在需要时自动做 `mj_setConst` refresh

## 新任务接入最低标准

如果要保持和当前代码风格一致，新任务至少应满足：

1. 在任务文件里定义自己的 DR config dataclass
2. 在任务文件里定义 `DomainRandomizationProvider`
3. reset 通过 `ResetPlan` 返回 `qpos`、`qvel`、`info_updates`、`randomization`
4. 如需 interval 扰动，通过 `IntervalRandomizationPlan`
5. 在 env 构造函数里调用 `self._init_domain_randomization(...)`

如果某个随机项要做成“统一 DR 项”，还需要同时满足三层一致：

1. `ResetRandomizationPayload` in `src/unilab/dr/types.py` 里有明确字段
2. backend capability 明确声明支持，并在 backend 内真正落地
3. 任务 config / provider 真正采样并下发该字段

缺任何一层，都只能算“底层有能力”或“任务里自己做了随机”，还不能算仓库层面的统一 DR 项。

## 非标准 actuator/control 扩展

不是所有机器人都适合走 backend 原生 position actuator。新增 motor actuator 任务时，应把策略输出到物理控制量的转换放在 env owner 层，通过 `SimBackend.set_pre_step_control(...)` 注册 pre-step control callback。backend 只负责在每个物理 substep 前调用 callback，并在每步后刷新传感器；position actuator 任务不注册 callback，继续走直接控制路径。

Go2W 是当前 all-motor actuator 示例：

- `apply_action()` 输出 owner-level control：腿部 12 维是期望关节位置，轮子 4 维是 torque。
- pre-step callback 接收 `(backend, ctrl)`，从 backend named sensor 读取 joint position / velocity，按 `kp * (target - pos) - kd * vel` 计算腿部 motor torque，再拼接轮子 torque。
- `randomize_kp` / `randomize_kd` 的配置路径与 Go2 保持一致，但 Go2W 的 kp/kd 只写入 env owner 缓存，不进入 `ResetRandomizationPayload.kp/kd`，避免 MuJoCo batch-env 里的 position-actuator kp/kd 机制泄漏到 motor actuator。
- 传感器名、actuator 数量、`ctrl_range` 等 contract 在 env 初始化冷路径校验；pre-step 热路径只读 named sensor 并做数组运算，不解析 XML / asset，也不探测 backend 私有能力。

## Related Documents

- {doc}`RL Infrastructure 开发标准 </en/developer_guide/architecture/development_standard>`
- {doc}`ADR-0001 Runtime Model And Layer Boundaries </adr/ADR-0001-runtime-model-and-layer-boundaries>`
- {doc}`ADR-0002 Backend Capability Boundary For Play And Snapshot </adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot>`

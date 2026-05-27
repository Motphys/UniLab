# SceneCfg 与场景组合设计

语言: 简体中文

本文描述 UniLab 面向程序化地形和场景组合的目标设计。它覆盖 #197 当前 rough terrain scene profile / materialization 需求，并为 #270 的通用 cold-path scene composition primitives 留出扩展边界。

本文不是用户使用指南。当前可运行的 rough terrain 用法见 [Go2 Rough Terrain](../../users/zh_CN/D-tasks/05-go2-rough-terrain.md)。

## 1. 背景

旧的 UniLab 仿真入口主要通过 `model_file` 表达:

```text
task owner YAML / env cfg
  -> cfg.scene: SceneCfg
  -> create_backend(backend_type, scene, ...)
```

这对静态 XML 场景足够简单，但难以表达:

- 程序化 terrain profile 与 robot asset 的组合
- rough terrain / stairs / slope 等可配置 scene profile
- 多实体或组合场景
- 后端各自的 scene model 生成方式

#197 已经引入了 rough terrain generator 的主要能力: `TerrainGeneratorCfg` 生成 backend-agnostic heightfield，MuJoCo 路径通过 `SceneCfg` 在 cold path 生成 hfield PNG，并用 `MjSpec` 组装 terrain、robot 和 task sensor fragment。

目标设计应把这些职责收敛到明确的 scene contract 中。

## 2. 设计目标

1. `EnvCfg.scene` 是唯一 scene source；静态模型也必须用 `SceneCfg(model_file=...)` 表达。
2. task / env 通过 `SceneCfg` 描述需要什么场景，而不是描述后端如何生成场景。
3. terrain generation 保持 backend-agnostic，只产出通用 terrain 数据。
4. backend materialization 由 backend owner 负责，把 `SceneCfg` 转成该 backend 可加载的 scene model。
5. asset/XML/model metadata 只在 init / materialization / cache 等 cold path 处理。
6. `model_file` 只允许作为 `SceneCfg` 的静态模型字段，不再是 env cfg 字段或独立 source of truth。
7. materialized result 必须支持调试、回放和测试所需的可观测信息。

## 3. 非目标

- 不在本设计中定义训练 reward / observation / command 语义。G1 rough-terrain 训练语义属于独立 task owner work item。
- 不要求所有 backend 都输出 XML。MuJoCo 程序化组装路径直接编译为 `MjModel`；XML 只作为 robot / task fragment 的输入格式存在。
- 不要求第一阶段完成通用多实体编辑器。多实体 composition 可以在 `SceneCfg` contract 稳定后增量扩展。
- 不允许在 `step()` / `reset()` / domain randomization 中解析 XML、读取 asset 元数据或根据 asset 结构做运行时分支。

## 4. 核心抽象

### 4.1 SceneCfg

`EnvCfg.scene` 是 task / env 对场景的声明，类型固定为 `SceneCfg`。静态 XML 通过 `SceneCfg(model_file=...)` 表达；当前程序化组合只覆盖 robot XML、task fragment 与 terrain。

建议结构:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SceneCfg:
    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: "TerrainSceneCfg | None" = None
```

当前 contract 已收敛到单一路径: `scene` 必须是 `SceneCfg`。其中 `SceneCfg(model_file=...)` 表示完整静态 scene；`SceneCfg(model_file + terrain + fragment_files)` 表示通过 materializer 组合 scene。backend factory 只接收 `SceneCfg`，并把它原样分发给具体 backend 构造函数。

### 4.2 TerrainSceneCfg

程序化地形通过 terrain 子配置表达:

```python
@dataclass
class TerrainSceneCfg:
    kind: Literal["hfield"] = "hfield"
    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None
```

其中:

- `generator` 继续使用现有 `TerrainGeneratorCfg`
- `hfield_name` 是 materializer 查找 hfield slot 的稳定名字
- `geom_name` 可选；为空时 materializer 可找第一个引用该 hfield 的 geom

当前 Go2 rough terrain 可以表达为:

```yaml
scene:
  model_file: src/unilab/assets/robots/go2/go2.xml
  fragment_files:
    - src/unilab/assets/robots/go2/locomotion_task.xml
  terrain:
    kind: hfield
    hfield_name: terrain_hfield
    generator:
      seed: 42
      curriculum: false
      size: [8.0, 8.0]
      num_rows: 10
      num_cols: 20
      border_width: 20.0
```

这等价于当前的:

```text
cfg.scene.model_file
cfg.scene.fragment_files
cfg.scene.terrain.generator
```

语义集中在 scene contract 内: robot XML、task sensor fragment 和 terrain generator 都是 scene 的组成部分。

### 4.3 Backend Scene Context

`SceneCfg` 不再先包成额外的中间 scene wrapper；`create_backend(...)` 只做 backend 类型分发，`MuJoCoBackend` / `MotrixBackend` 构造函数直接接收 `SceneCfg`，并在各自 backend 模块内完成 cold-path materialization:

```python
backend = create_backend(backend_type, scene=cfg.scene, ...)
# 等价于 factory 内部分发:
backend = MuJoCoBackend(cfg.scene, ...)
```

backend 内部 materialization 结果由 backend owner 定义:

| Backend | 内部结果示例 |
|---------|---------------------|
| MuJoCo | `MjModel`；静态 `SceneCfg(model_file=...)` 可在 backend 内部继续走 XML path 加载 |
| Motrix | `motrixsim.SceneModel` |
| 其他 backend | 后端自己的 scene handle / asset bundle |

需要跨层传回 env 的 scene 上下文挂在 backend 实例上，例如 `terrain_origins`、`terrain_surface_sampler`、`scene_artifacts_dir`；临时 XML / 目录由 backend 的 scene cleanup hook 在 env `close()` 时释放。

静态 scene path 只存在于 `SceneCfg.model_file` 字段；程序化组装路径直接返回 `MjModel`，不把 XML path 写成通用 contract，也不生成最终 `scene.xml`。

## 5. 分层责任

### Config / Registry 层

负责:

- 在 env cfg 或 base cfg 中提供 `scene: SceneCfg | None`
- owner YAML 暴露可 override 的 scene / terrain 字段
- 将旧 `model_file` 入口迁移为 `scene`

不负责:

- 解析 XML
- 判断 backend 私有能力
- 在 Python 层解释算法超参数或训练语义

### Terrain 层

负责:

- `TerrainGeneratorCfg`
- terrain profile / preset 注册
- `TerrainGenerator.generate()` 生成 backend-agnostic `GeneratedTerrain`
- 产出 `terrain_origins`、height matrix、height range 等通用数据

不负责:

- 写 MuJoCo XML
- 调用 backend API
- 读取 robot asset

### Scene / Backend Materializer 层

负责:

- 根据 `backend_type` 选择 backend materializer
- backend 构造函数直接接收 `SceneCfg`，并把它编译或物化成 backend native model
- 管理 cold-path artifacts 生命周期
- 对 unsupported backend capability fail loudly

MuJoCo materializer 做:

```text
SceneCfg(model_file + fragment_files + hfield terrain)
  -> TerrainGenerator.generate()
  -> hfields/hfield.png
  -> MjSpec.add_hfield / worldbody.add_geom
  -> MjSpec.attach(robot_spec)
  -> merge task sensor/keyframe fragments
  -> MjSpec.compile()
  -> self._model = <MjModel>
```

Motrix materializer 应该:

- 若 Motrix 已支持等价 hfield scene model，生成 Motrix 自己的 scene artifact
- 若暂不支持，显式报 `UnsupportedSceneFeature`，而不是静默退回 flat scene 或在 env 中分支绕过

### Env 层

负责:

- 使用 `SceneCfg` 创建 backend
- 使用 `terrain_origins` 初始化 spawn / curriculum manager
- 维护 MDP 语义、obs、reward、reset

不负责:

- XML hfield 替换
- 临时 scene artifact 生成细节
- backend-specific scene composition

## 6. Backend 构造入口

目标入口:

```python
backend = create_backend_from_scene(
    cfg.scene,
    backend_type,
    num_envs,
    cfg.sim_dt,
    ...
)
```

当前入口:

```python
backend = create_backend(
    backend_type,
    cfg.scene,
    num_envs,
    cfg.sim_dt,
    ...
)
```

规则:

1. `scene` 是 `SceneCfg(model_file=...)` 且 `terrain is None`: 作为完整静态 scene 加载；若存在 `fragment_files`，先在 cold path 合并成临时 XML。
2. `scene` 是 `SceneCfg(model_file=..., terrain=...)`: backend 把 `model_file` 作为 robot model，在 cold path 组装 materialized scene，并合并 `fragment_files`。
3. `scene is None`: fail loudly。
4. backend 不支持某个 `SceneCfg` feature 时，必须报明确错误。

## 7. 当前 Go2 Rough 迁移方式

当前路径:

```text
Go2JoystickRoughCfg.scene
  -> create_backend(..., cfg.scene)
  -> materialize_mujoco_hfield_attached_scene(...)
  -> MjSpec.add_hfield + MjSpec.attach(robot_spec)
  -> create/load MuJoCo model
```

目标路径:

```text
Go2JoystickRoughCfg.scene
  -> create_backend(..., cfg.scene)
  -> materialize_mujoco_hfield_attached_scene(...)
  -> MuJoCoBackend._model = <MjModel>
```

`Go2JoystickRoughCfg.scene` 和 Go2 flat 的 `scene` 都是 `SceneCfg`。`Go2WalkTask` 只保留:

```python
terrain_origins = getattr(backend, "terrain_origins", None)
if terrain_origins is not None:
    self._spawn = TerrainSpawnManager(...)
```

env 不直接调用 MuJoCo / Motrix scene materializer，也不维护中间 scene wrapper。

## 8. 冷路径与热路径边界

允许在 cold path 做:

- resolve `scene`
- parse XML / scene descriptor
- generate heightfield PNG
- compile backend model
- cache body / geom / hfield ids
- compute `terrain_origins`

禁止在 hot path 做:

- 读取 XML / asset 文件
- 根据 XML 内容决定 reward / obs / reset 分支
- 通过 `getattr` / `hasattr` 探测 backend 私有 scene 能力
- 重新生成 terrain

热路径只能使用 cold path 明确产出的缓存数据，例如:

```text
terrain_origins
spawn level state
cached geom ids
backend public capability methods
```

## 9. Hydra 表达

第一阶段 owner YAML 应显式列出允许 override 的 scene / terrain 字段:

```yaml
env:
  scene:
    model_file: src/unilab/assets/robots/go2/go2.xml
    fragment_files:
      - src/unilab/assets/robots/go2/locomotion_task.xml
    terrain:
      kind: hfield
      hfield_name: terrain_hfield
      generator:
        seed: 42
        curriculum: false
        size: [8.0, 8.0]
        num_rows: 10
        num_cols: 20
        border_width: 20.0
```

`sub_terrains` 仍然建议由 Python preset / task cfg default factory 提供，不在命令行直接重建抽象子类型。若后续需要配置化 sub-terrain 组合，应新增 typed profile schema，而不是让 Hydra 直接实例化任意 `SubTerrainCfg`。

## 10. 验证计划

最小验证应贴近风险边界:

1. Config compose: `scene.model_file`、`scene.terrain.generator` CLI override 能被 Hydra 接受，并保留 Python default preset。
2. Materializer unit test: MuJoCo hfield materializer 生成 artifact，通过 `MjSpec` 组装 hfield、robot 和 task fragment。
3. Backend load test: MuJoCo 能从 materialized scene 创建 model。
4. Env contract test: rough terrain env 的 `reset()` / `step()` 输出仍满足 `NpEnvState.obs` dict contract。
5. Hot-path boundary test / review check: `step()`、`reset()`、DR provider 不解析 XML、不读取 asset 文件。
6. Docs test: 文档 claim 只写已注册、已配置、已测试的支持范围。

## 11. 分阶段落地

### Phase 1: Contract Skeleton

- 新增 `SceneCfg`、`TerrainSceneCfg`
- `EnvCfg` 增加 `scene: SceneCfg | None`
- 静态 scene 用 `SceneCfg(model_file=...)`
- 增加 compose / override 测试

### Phase 2: MuJoCo Terrain Materializer

- Go2 rough 使用 `materialize_mujoco_hfield_attached_scene`
- Go2 rough 从独立 `terrain_generator` 字段迁移到 `scene.terrain.generator`
- env 不再直接调用 XML materializer
- `terrain_origins` 通过 backend 场景属性传回 env

### Phase 3: Backend Entry Migration

- `create_backend` 只接受 `scene: SceneCfg`
- 静态 path 在 env cfg / owner YAML 中直接写入 `SceneCfg.model_file`
- 文档更新 `scene` 与 `SceneCfg` 的关系

### Phase 4: General Scene Composition

- 增加 typed robot / object / light / camera scene cfg
- 增加多实体或组合场景示例
- Motrix materializer 明确支持范围或 unsupported error

## 12. 已决策边界

1. `SceneCfg` materialization 生命周期由具体 backend 管理，env 在 `close()` 中调用 backend scene cleanup hook 清理 cold-path artifacts。
2. MuJoCo 程序化 materializer 返回预编译 `MjModel`，不返回最终 XML path。
3. `terrain_origins` 通过 backend 场景属性传回 env，用于 spawn / curriculum。
4. `SceneCfg` 第一阶段放在 `base/scene.py`，由 env cfg 引用。
5. Motrix rough terrain hfield 暂未支持时必须 fail loudly，不能静默退回 flat scene。

## Navigation

- Index: [Documentation](../../index.md)
- Related: [RL Infrastructure 开发标准](development-standard.md)
- Related: [Go2 Rough Terrain](../../users/zh_CN/D-tasks/05-go2-rough-terrain.md)

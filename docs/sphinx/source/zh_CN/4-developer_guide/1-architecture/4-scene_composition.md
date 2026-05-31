# 场景组合

场景组合是一个冷路径契约。env config 通过 `SceneCfg` 描述场景；backend
materializer 在初始化期间将该声明转换为 backend 的原生模型。

## 契约

`SceneCfg` 与 `TerrainSceneCfg` 位于 `src/unilab/base/scene.py`：

```python
@dataclass
class TerrainSceneCfg:
    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None

@dataclass
class SceneCfg:
    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
```

`EnvCfg.scene` 是唯一的 scene source。静态场景使用 `SceneCfg(model_file=...)`。
程序化地形场景则将机器人模型、任务 fragment 与地形配置组合在一起：

```yaml
env:
  scene:
    model_file: src/unilab/assets/robots/go2/go2.xml
    fragment_files:
      - src/unilab/assets/robots/go2/locomotion_task.xml
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
```

env 将场景交给 `create_backend(...)`；它不会直接调用 MuJoCo 或 Motrix 的
materializer。

## Backend 分发

`src/unilab/base/backend/__init__.py` 中的 `create_backend(...)` 按 backend 类型
路由，并对 `SceneCfg` 应用以下规则：

1. 静态 `SceneCfg(model_file=...)` 且无 terrain：加载完整静态场景（存在
   `fragment_files` 时在冷路径合并）。
2. `SceneCfg(model_file=..., terrain=...)`：把 `model_file` 当作机器人模型，在
   冷路径组装 materialized 场景，并合并 `fragment_files`。
3. `scene is None`：fail loudly（`create_backend` 抛出 `ValueError`）。
4. backend 不支持某个 `SceneCfg` feature 时，必须抛出明确错误。例如 Motrix 的
   崎岖地形必须 fail loudly，而不是静默退回 flat 场景。

## MuJoCo Materializer 流水线

对于程序化地形，MuJoCo backend 调用 `src/unilab/base/backend/mujoco/xml.py` 中的
`materialize_mujoco_hfield_attached_scene(...)`。冷路径步骤为：

1. `TerrainGenerator(terrain_cfg).write_png(...)` 生成与 backend 无关的高度场
   PNG。
2. `MjSpec.add_hfield(...)` 加上 `worldbody.add_geom(type=mjGEOM_HFIELD, ...)`
   放置地形。
3. `MjSpec.from_file(robot_path)` 加载机器人 spec，`spec.attach(...)` 将其挂到
   一个 frame 下。
4. 把每个 `fragment_files` 条目（任务 sensor、keyframe）合并进场景 XML。
5. `MjSpec.from_string(...).compile()` 返回编译好的 `MjModel`，连同
   `terrain_origins`（以及可选的 surface sampler）。

程序化路径返回预编译的 `MjModel`，不把最终 `scene.xml` path 作为契约暴露。诸如
`terrain_origins` 这类场景上下文通过 backend 属性传回 env。

## 分层所有权

| 层 | 拥有 |
| --- | --- |
| Config / registry | `SceneCfg` 字段与 owner YAML 的 override 入口 |
| Terrain | 与 backend 无关的高度矩阵、地形原点以及地形预设 |
| Backend materializer | XML/world 装配、原生模型编译、场景产物清理 |
| Env | MDP 语义、reset、reward、观测，以及对已缓存场景上下文的使用 |

## 冷路径边界

冷路径上允许：

- 读取 XML 与资源文件。
- 生成地形高度场。
- 编译 MuJoCo `MjModel` 或 Motrix 场景模型。
- 解析场景 ID、地形原点以及 scanner handle。

热路径上禁止：

- 在 `step`、`reset` 或 interval DR 期间解析 XML 或读取资源。
- 基于原始资源元数据对 reward 或观测逻辑做分支。
- 探测 backend 私有的场景方法，而不使用明确的契约。
- 在 env 构造完成后重新生成地形。

## Go2 崎岖地形证据

当前面向用户的程序化地形路径是 Go2 崎岖地形：

- Env owner：`src/unilab/envs/locomotion/go2/rough.py`
- 地形生成器：`src/unilab/terrains/terrain_generator.py`
- MuJoCo materializer：`src/unilab/base/backend/mujoco/xml.py`
- Motrix materializer：`src/unilab/base/backend/motrix/scene.py`
- Owner YAML：`conf/ppo/task/go2_joystick_rough/mujoco.yaml`、
  `conf/ppo/task/go2_joystick_rough/motrix.yaml`

用户使用说明见 {doc}`../../2-user_guide/6-terrain/1-procedural`。

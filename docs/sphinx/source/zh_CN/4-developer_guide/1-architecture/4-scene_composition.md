# 场景组合

语言: 简体中文

场景组合是一个冷路径契约。env config 通过 `SceneCfg` 描述场景；backend
materializer 在初始化期间将该声明转换为 backend 的原生模型。

## 契约

`SceneCfg` 位于 `src/unilab/base/scene.py`。静态场景使用
`SceneCfg(model_file=...)`。程序化地形场景则将机器人模型、任务 fragment 与地形
配置组合在一起：

```yaml
env:
  scene:
    model_file: src/unilab/assets/robots/go2/go2.xml
    fragment_files:
      - src/unilab/assets/robots/go2/locomotion_task.xml
    terrain:
      kind: hfield
      hfield_name: terrain_hfield
      geom_name: floor
```

env 将场景交给 `create_backend(...)`；它不会直接调用 MuJoCo 或 Motrix 的
materializer。

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

## Navigation

- Index: [文档](0-index.md)

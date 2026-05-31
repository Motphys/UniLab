# 程序化地形

本页只回答四个问题：

1. 如何运行当前仓库中已经存在的崎岖地形任务？
2. 哪些内容可以、哪些内容不能通过 Hydra 命令行修改？
3. 当我想修改子地形组成时，正确的入口是什么？
4. 当前已知的边界是什么——这些不是 bug，而是约束？

关于底层 contract（冷路径实例化、注册新的子地形、hfield 导出），参见 `src/unilab/base/backend/mujoco/xml.py`、`src/unilab/base/backend/motrix/scene.py` 和 `src/unilab/terrains/terrain_generator.py` 中的源码注释。

## 当前状态

当前仓库中只有一个任务注册并接入了程序化地形：

| 任务 | owner YAML | 后端 | 入口算法 | 代码 |
| --- | --- | --- | --- | --- |
| `Go2JoystickRough` | `mujoco.yaml`、`motrix.yaml` | MuJoCo / Motrix | PPO (`train_rsl_rl.py`) | `go2/rough.py` |

在 env 构建过程中：

1. `Go2JoystickRoughCfg` 声明了一个 `SceneCfg`，其 `model_file` 指向 `go2.xml`，`fragment_files` 从 `locomotion_task.xml` 引入 task 级别的接触传感器和 `home` keyframe，`scene.terrain` 声明了一个名为 `terrain_hfield` 的待生成 hfield。
2. 后端场景实例化器调用 `TerrainGenerator(...)` 生成一个与后端无关的合并高度矩阵以及 `terrain_origins`；地形生成器本身不依赖 MuJoCo 或 Motrix。
3. MuJoCo 实例化器使用 `MjSpec.add_hfield(...)` / `worldbody.add_geom(...)` 创建地形，然后用 `MjSpec.attach(...)` 把机器人 spec 附加到场景中，最后通过 `compile()` 生成 `MjModel`。
4. Motrix 实例化器使用 `motrixsim.msd.World` 创建地形世界，用 `World.attach(...)` 拼接机器人世界和 task fragment，最后通过 `msd.build(...)` 生成 `SceneModel`。
5. `go2.xml` 是机器人模型；`locomotion_task.xml` 是用于崎岖地形的 task fragment，包含与地形 `floor` 关联的接触传感器以及 task 级别的 `home` keyframe。
6. 后端实例持有冷路径场景产物，直到 env `close()`；`terrain_origins` 通过一个后端场景属性回传给 env，用于 spawn / curriculum。

`step()` / `reset()` / DR provider 永远不会读取 XML 或访问 asset 文件；所有与地形相关的事情都发生在冷路径上。

## 1. 直接训练

```bash
# 默认使用单 patch 的 random_rough，critic 额外接收一个 17×11 的高度扫描
uv run train --algo ppo --task go2_joystick_rough --sim mujoco
```

Motrix 后端使用相同的任务 owner：

```bash
uv run train --algo ppo --task go2_joystick_rough --sim motrix
```

## 2. 通过 Hydra 命令行覆盖地形参数

`Go2JoystickRough` 在 `conf/ppo/task/go2_joystick_rough/{mujoco,motrix}.yaml` 中显式列出了一组可覆盖字段；这些字段允许 Hydra struct 模式接受命令行覆盖。

| 字段 | 用途 | YAML 默认值 |
| --- | --- | --- |
| `env.scene.terrain.generator.seed` | 随机种子，`null` 表示每次重新随机化 | `42` |
| `env.scene.terrain.generator.curriculum` | `true`：每个子地形一列，难度沿行递增；`false`：按 `proportion` 随机采样 | `false` |
| `env.scene.terrain.generator.size` | 单个地形 patch 的 x/y 尺寸（米） | `[8.0, 8.0]` |
| `env.scene.terrain.generator.num_rows` | 网格行数（curriculum 模式下 = 难度等级数量） | `1` |
| `env.scene.terrain.generator.num_cols` | 网格列数（curriculum 模式下被忽略；列数 = `len(sub_terrains)`） | `1` |
| `env.scene.terrain.generator.border_width` | 网格周围平坦边界的宽度（米） | `1.0` |
| `env.scene.terrain.generator.difficulty_range` | 难度采样范围 `[min, max]`，∈ `[0, 1]` | `[0.0, 1.0]` |
| `env.terrain_scan.enabled` | 是否将后端原生高度扫描拼接到 critic obs | `true` |
| `env.terrain_scan.geom_name` | 高度扫描采样的 hfield geom 名称 | `floor` |

示例：本地小规模冒烟 + 固定种子 + curriculum 模式。

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.scene.terrain.generator.num_rows=4 \
    env.scene.terrain.generator.num_cols=6 \
    env.scene.terrain.generator.seed=42 \
    env.scene.terrain.generator.curriculum=true \
    algo.num_envs=64 algo.max_iterations=2 training.no_play=true
```

未在 YAML 中列出的字段（例如 `sub_terrains`）目前**无法**从命令行覆盖：

- `sub_terrains` 是 `dict[str, SubTerrainCfg]`，而 `SubTerrainCfg` 是一个抽象基类；从命令行重建子类类型并不安全。
- `terrain_scan.measured_points_x` / `terrain_scan.measured_points_y` 的默认网格由 `Go2JoystickRoughCfg` owner 定义；当需要修改扫描布局时，请在 owner cfg 中显式调整，并对照 critic obs 形状验证 `obs_groups_spec`。

## 3. 修改子地形

子地形在 `unilab.terrains.config` 的 `ALL_TERRAIN_PRESETS` 中注册。`Go2JoystickRough` 默认混合的 7 种子地形：

| 名称 | 实现 | 描述 |
| --- | --- | --- |
| `flat` | `HfFlatTerrainCfg` | 全零高度场，基线 patch |
| `pyramid_stairs` | `HfPyramidStairsTerrainCfg` | 金字塔形上升台阶（高度场中的同心方环） |
| `pyramid_stairs_inv` | `HfInvertedPyramidStairsTerrainCfg` | 倒金字塔形下降台阶 |
| `hf_pyramid_slope` | `HfPyramidSlopedTerrainCfg` | 高度场金字塔斜坡 |
| `hf_pyramid_slope_inv` | `HfPyramidSlopedTerrainCfg(inverted=True)` | 倒金字塔斜坡 |
| `random_rough` | `HfRandomUniformTerrainCfg` | 随机均匀噪声高度场 |
| `wave_terrain` | `HfWaveTerrainCfg` | 正弦波高度场 |

每种都有自己的难度参数（`step_height_range`、`slope_range`、`noise_range` 等）；完整的字段定义在 `heightfield_terrains.py` 中。所有子地形（包括 `flat` 和台阶）现在都通过 hfield 实现，分辨率统一由 `TerrainGeneratorCfg.horizontal_scale` / `vertical_scale` 控制。

内置组合定义在 `unilab.terrains.config` 中，`Go2JoystickRoughCfg` 在 `go2/rough.py` 中定义了自己的 owner 默认值：

- `Go2RoughTerrainCfg`：1 × 1，默认只采样 `random_rough`（proportion `0.2`，其余子地形作为可配置 profile 保留，但默认 proportion 为 `0.0`），随机模式。每个 env 实例获得自己独立的 cfg 对象。
- `ROUGH_TERRAINS_CFG`：10 × 20，按 proportion 混合 7 种子地形，随机模式。目前作为可复用 profile 保留；不是 `Go2JoystickRoughCfg` 的默认训练 profile。
- `STAIRS_TERRAINS_CFG`：10 × 4，curriculum 模式，难度从 flat → easy → moderate → challenging 递增。目前没有任何任务引用它；可以在自定义任务配置中使用。

## 4. 高度扫描观测

`Go2JoystickRoughEnv` 只把高度扫描拼接到 `critic` group；actor obs 遵循 45 维崎岖任务 contract。默认扫描点在 x 方向 17 个、y 方向 11 个，合计 187 维，因此 `obs_groups_spec` 为：

| obs group | 维度 | 内容 |
| --- | ---: | --- |
| `obs` | `45` | actor policy 输入 |
| `critic` | `235` | 崎岖 critic 48 维 + 高度扫描 187 维 |

高度扫描的 geom/body id 和采样偏移在 env init 期间缓存；热路径只使用由 `create_hfield_scanner(...)` 创建的后端持有的 scanner，并消费缓存的 id / 偏移。`step()` / `reset()` 中不解析 XML、不读取 asset 元数据。

## 5. 在新任务中启用程序化地形

新任务通过 `SceneCfg` 启用程序化地形。`SceneCfg` 位于 `src/unilab/base/scene.py`，`scene.terrain.generator` 使用 `TerrainGeneratorCfg`。

```yaml
env:
  scene:
    model_file: .../robot.xml
    fragment_files:
      - .../locomotion_task.xml
    terrain:
      kind: hfield
      hfield_name: terrain_hfield
      geom_name: floor
      generator:
        seed: 42
        size: [8.0, 8.0]
        num_rows: 10
        num_cols: 20
        border_width: 20.0
```

env 的 `__init__` 不需要直接调用 XML 实例化器；把 `scene` 交给后端构造函数即可：

```python
from unilab.base.backend import create_backend

backend = create_backend(..., cfg.scene)
terrain_origins = getattr(backend, "terrain_origins", None)
```

注意：`TerrainGenerator.__init__` 会原地修改传入的 cfg（把值写入每个 `sub_cfg.size`）。如果同一个 `TerrainGeneratorCfg` 实例被多个 env 共享，它们会相互污染；你必须使用 `default_factory` 或 `copy.deepcopy` 来确保每个实例获得自己的 cfg。`Go2JoystickRoughCfg` 通过 `scene.terrain.generator=Go2RoughTerrainCfg()` 处理这一点。

## 6. 可视化与离线回放

要在不启动训练的情况下预览已实例化的场景：

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --num_envs 4
```

## 7. 验证

```bash
# 程序化地形 + hfield PNG 实例化器单元/集成测试
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q

# Hydra compose + Go2JoystickRoughCfg 任务 owner 测试
uv run pytest tests/config/test_locomotion_params.py -k rough -q

# Go2 崎岖地形 spawn + 高度扫描 contract 测试
uv run pytest tests/envs/locomotion/test_go2_terrain_spawn.py tests/envs/locomotion/test_go2_rough_height_scan.py -q

# Hydra 命令行覆盖 + registry 深度合并环路
uv run pytest tests/config/test_locomotion_params.py \
    -k "apply_cfg_overrides or hydra_terrain_override" -q

# 端到端冒烟：Hydra 命令行覆盖网格尺寸 + 种子，2 次迭代 PPO
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.scene.terrain.generator.num_rows=4 env.scene.terrain.generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64

uv run train --algo ppo --task go2_joystick_rough --sim motrix \
    env.scene.terrain.generator.num_rows=4 env.scene.terrain.generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64
```

## 已知约束

- **MuJoCo 和 Motrix 实例化器都有自动化冒烟覆盖**：MuJoCo 路径返回 `MjModel`，Motrix 路径返回 `SceneModel`。生产训练性能与收敛质量仍需由独立的 benchmark 记录；冒烟测试不对其作出保证。
- **MuJoCo 组装路径依赖 `MjSpec.attach`**：机器人 XML、地形和 task 传感器 fragment 在实例化阶段组装，并直接编译为 `MjModel`。
- **Motrix 组装路径依赖 `motrixsim.msd.World.attach`**：`go2.xml` 提供机器人模型，`locomotion_task.xml` 作为携带接触传感器和 task 级别 keyframe 的 task fragment 被接入。
- **高度扫描支持通过 `create_hfield_scanner(...)`**：崎岖 env 在初始化期间缓存 scanner id 和偏移，然后在观测/奖励代码中消费 scanner 输出，热路径上不解析 XML。
- **`scene.terrain.generator` 是冷路径配置**：在 env 构建之后修改 generator 不会影响已经实例化的场景。要更换地形，必须重建 env（即重新运行训练命令）。
- **`import unilab.terrains` 不依赖 mujoco**：`TerrainGenerator.generate()` / `write_png()` 是纯 numpy + imageio 路径。

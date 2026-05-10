# 程序化地形 (Procedural Terrain)

语言: 简体中文

本页只回答四个问题:

1. 怎么把当前仓库已有的 rough terrain 任务跑起来？
2. Hydra 命令行能改什么、不能改什么？
3. 想改子地形组合时，正确的入口是什么？
4. 哪些是当前已知的边界，不是 bug 而是约束？

底层 contract（cold-path materialization、注册新 sub-terrain、hfield PNG 导出）见 [`base/backend/xml.py`](../../../src/unilab/base/backend/xml.py) 与 [`terrains/terrain_generator.py`](../../../src/unilab/terrains/terrain_generator.py) 的源码注释。

## 现状

当前仓库注册并接入程序化地形的任务只有一个：

| 任务 | owner YAML | 后端 | 入口算法 | 代码 |
| --- | --- | --- | --- | --- |
| `Go2JoystickRough` | [`conf/ppo/task/go2_joystick_rough/mujoco.yaml`](../../../conf/ppo/task/go2_joystick_rough/mujoco.yaml) | MuJoCo | PPO (`train_rsl_rl.py`) | [`go2/rough.py`](../../../src/unilab/envs/locomotion/go2/rough.py) |

env 构造期会执行：

1. 加载 [`scene_rough.xml`](../../../src/unilab/assets/robots/go2/scene_rough.xml) 模板（包含一个名为 `terrain_hfield` 的 hfield asset 和一个名为 `floor` 的 hfield geom）。
2. `TerrainGenerator(cfg.terrain_generator)` 只生成 backend-agnostic 的合并 height matrix，不依赖 MuJoCo。
3. backend XML materializer 把合并后的 height matrix 写成 per-instance 的 `hfields/hfield.png`（uint16 PNG），并在临时 `scene.xml` 中替换 hfield 的 `file` / `size` / geom `pos`。
4. 同步复制模板依赖（如 `go2.xml` 与 mesh assets）到 `tempfile.TemporaryDirectory`，backend 用 `model_file=<materialized_xml>` 编译 `MjModel`。
5. tempdir 一直持有到 env `close()`；`Go2JoystickRoughEnv.get_playback_model()` 返回该路径，离线回放视频复用同一个 materialized scene。

`step()` / `reset()` / DR provider 不读 XML、不访问 asset 文件；地形相关全部发生在冷路径。

## 1. 直接训练

```bash
# 默认 single-patch random_rough，critic 额外接入 17×11 height scan
uv run train --algo ppo --task go2_joystick_rough --sim mujoco
```

## 2. Hydra 命令行覆盖地形参数

`Go2JoystickRough` 在 [conf/ppo/task/go2_joystick_rough/mujoco.yaml](../../../conf/ppo/task/go2_joystick_rough/mujoco.yaml) 里显式列出了一组可覆盖字段；这些字段允许 Hydra struct 模式接受命令行覆盖。

| 字段 | 作用 | yaml 默认值 |
| --- | --- | --- |
| `env.terrain_generator.seed` | 随机种子，`null` 表示每次随机 | `42` |
| `env.terrain_generator.curriculum` | `true`：每种 sub-terrain 一列、难度沿行递增；`false`：按 `proportion` 随机采样 | `false` |
| `env.terrain_generator.size` | 单个 terrain patch 的 x/y 尺寸（米） | `[8.0, 8.0]` |
| `env.terrain_generator.num_rows` | grid 行数（curriculum 模式 = 难度等级数） | `1` |
| `env.terrain_generator.num_cols` | grid 列数（curriculum 模式被忽略，列数 = `len(sub_terrains)`） | `1` |
| `env.terrain_generator.border_width` | grid 外圈 flat border 宽度（米） | `1.0` |
| `env.terrain_generator.difficulty_range` | 难度采样区间 `[min, max]`，∈ `[0, 1]` | `[0.0, 1.0]` |
| `env.terrain_scan.enabled` | 是否向 critic obs 拼接 backend-native height scan | `true` |
| `env.terrain_scan.geom_name` | height scan 采样的 hfield geom 名称 | `floor` |

示例：本地小规模 smoke + 固定种子 + curriculum 模式。

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.terrain_generator.num_rows=4 \
    env.terrain_generator.num_cols=6 \
    env.terrain_generator.seed=42 \
    env.terrain_generator.curriculum=true \
    algo.num_envs=64 algo.max_iterations=2 training.no_play=true
```

未列在 yaml 里的字段（如 `sub_terrains`）当前**不能**通过命令行覆盖：

- `sub_terrains` 是 `dict[str, SubTerrainCfg]`，`SubTerrainCfg` 是抽象基类，从命令行重建子类型不安全。
- `terrain_scan.measured_points_x` / `terrain_scan.measured_points_y` 的默认网格由 `Go2JoystickRoughCfg` owner 定义；需要改 scan layout 时应在 owner cfg 里显式调整，并同步验证 `obs_groups_spec` 与 critic obs shape。

## 3. 修改 sub-terrain

注册在 [`unilab.terrains.config`](../../../src/unilab/terrains/config.py) 的 `ALL_TERRAIN_PRESETS`。`Go2JoystickRough` 默认混合的 7 种：

| 名称 | 实现 | 描述 |
| --- | --- | --- |
| `flat` | `HfFlatTerrainCfg` | 全零 heightfield，作为 baseline patch |
| `pyramid_stairs` | `HfPyramidStairsTerrainCfg` | 金字塔形上行台阶（heightfield 同心方环） |
| `pyramid_stairs_inv` | `HfInvertedPyramidStairsTerrainCfg` | 倒金字塔形下行台阶 |
| `hf_pyramid_slope` | `HfPyramidSlopedTerrainCfg` | heightfield 金字塔斜坡 |
| `hf_pyramid_slope_inv` | `HfPyramidSlopedTerrainCfg(inverted=True)` | 倒置金字塔斜坡 |
| `random_rough` | `HfRandomUniformTerrainCfg` | 随机均匀噪声 heightfield |
| `wave_terrain` | `HfWaveTerrainCfg` | 正弦波 heightfield |

每种都有自己的难度参数（`step_height_range`、`slope_range`、`noise_range` 等），完整字段定义见 [`heightfield_terrains.py`](../../../src/unilab/terrains/heightfield_terrains.py)。所有子地形（含 `flat` 与楼梯）现在都通过 hfield 实现，分辨率由 `TerrainGeneratorCfg.horizontal_scale` / `vertical_scale` 统一控制。

内置组合定义在 [`unilab.terrains.config`](../../../src/unilab/terrains/config.py)，`Go2JoystickRoughCfg` 在 [`go2/rough.py`](../../../src/unilab/envs/locomotion/go2/rough.py) 中定义自己的 owner 默认值：

- `Go2RoughTerrainCfg`：1 × 1，默认只采样 `random_rough`（比例 `0.2`，其余 sub-terrain 保留为可配置 profile，但默认比例为 `0.0`），random 模式。每个 env 实例都会拿到独立的 cfg 对象。
- `ROUGH_TERRAINS_CFG`：10 × 20，7 种 sub-terrain 按比例混合，random 模式。当前作为可复用 profile 保留，不是 `Go2JoystickRoughCfg` 的默认训练 profile。
- `STAIRS_TERRAINS_CFG`：10 × 4，curriculum 模式，难度从 flat → easy → moderate → challenging。当前没有任务直接引用，可在自定义 task config 里使用。

## 4. Height Scan Observation

`Go2JoystickRoughEnv` 只把 height scan 拼到 `critic` 组，actor obs 仍沿用 flat Go2 joystick 的 49 维 contract。默认 scan points 为 x 方向 17 个、y 方向 11 个，共 187 维，因此 `obs_groups_spec` 为：

| obs group | 维度 | 内容 |
| --- | ---: | --- |
| `obs` | `49` | actor policy 输入 |
| `critic` | `239` | flat critic 52 维 + height scan 187 维 |

height scan 的 geom/body id 与采样 offsets 在 env init 阶段缓存，热路径只调用 backend contract `sample_hfield_height(...)` 并消费缓存后的 id / offsets；不在 `step()` / `reset()` 解析 XML 或读取 asset 元数据。

## 5. 在新任务里启用程序化地形

`terrain_generator` 是基类 [`EnvCfg`](../../../src/unilab/base/base.py) 上的字段，默认 `None`。在自己的 task cfg 里赋值即可启用：

```python
import copy
from dataclasses import dataclass, field
from unilab.terrains import ROUGH_TERRAINS_CFG, TerrainGeneratorCfg

@dataclass
class MyTaskCfg(...):
    # 模板 XML 里需要有 name="terrain_hfield" 的 hfield asset，以及一个使用
    # 该 hfield 的 geom（默认命名为 floor，方便 contact sensor 复用）。
    model_file: str = ".../scene_rough.xml"
    terrain_generator: TerrainGeneratorCfg = field(default_factory=lambda: copy.deepcopy(ROUGH_TERRAINS_CFG))
```

env 的 `__init__` 沿用 rough task owner 的物化模板：

```python
import tempfile
from unilab.base.backend.xml import materialize_terrain_hfield_scene

self._materialized_dir = tempfile.TemporaryDirectory(prefix="unilab_terrain_")
model_file, terrain_origins = materialize_terrain_hfield_scene(cfg.model_file, terrain_cfg=cfg.terrain_generator, output_dir=self._materialized_dir.name)
backend = create_backend(..., model_file=model_file, ...)
```

注意：`TerrainGenerator.__init__` 会原地修改传入的 cfg（向每个 `sub_cfg.size` 写值）。如果在多个 env 之间共享同一个 `TerrainGeneratorCfg` 实例会互相污染，必须用 `default_factory` 或 `copy.deepcopy` 保证每个实例拿到独立 cfg；`Go2JoystickRoughCfg` 通过 `default_factory=Go2RoughTerrainCfg` 已处理。

## 6. 可视化与离线回放

不开训练直接看一下 materialized 场景：

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --num_envs 4
```

## 7. 验证

```bash
# 程序化地形 + hfield PNG materializer 单元/集成测试
uv run pytest tests/terrains tests/utils/test_xml_utils.py -q

# Hydra compose + Go2JoystickRoughCfg 的 task owner 测试
uv run pytest tests/config/test_locomotion_params.py -k rough -q

# Go2 rough terrain spawn + height scan contract 测试
uv run pytest tests/envs/locomotion/test_go2_terrain_spawn.py tests/envs/locomotion/test_go2_rough_height_scan.py -q

# Hydra 命令行覆盖 + registry deep-merge 闭环
uv run pytest tests/config/test_locomotion_params.py \
    -k "apply_cfg_overrides or hydra_terrain_override" -q

# 端到端 smoke：Hydra 命令行覆盖 grid 大小 + 种子，2 iter PPO
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.terrain_generator.num_rows=4 env.terrain_generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64
```

## 已知约束

- **当前只在 MuJoCo 后端验证过**：terrain 生成本身不依赖 MuJoCo，但运行时仍需要 MuJoCo backend 编译 hfield scene。Motrix 没有 owner YAML，也没有 hfield 程序化场景冒烟测试。生产训练请只用 `--sim mujoco`。
- **模板 XML 必须包含可替换 hfield**：当前 materializer 约定查找名为 `terrain_hfield` 的 hfield asset，并更新第一个引用它的 geom。Go2 模板把该 geom 命名为 `floor`，所以现有 contact sensor 不需要重定向。
- **height scan 依赖 backend contract**：Go2 rough 当前注册为 MuJoCo owner；其他 backend 必须先对齐 `sample_hfield_height(...)` contract，不能在训练脚本中探测私有能力或降级分支。
- **`terrain_generator` 是 cold-path 配置**：env 构造完成后再修改 `cfg.terrain_generator` 不会影响已 materialize 的场景。要换地形必须重新构造 env（即重新跑训练命令）。
- **`import unilab.terrains` 不依赖 mujoco**：`TerrainGenerator.generate()` / `write_png()` 是纯 numpy + imageio 路径。

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Dexterous In-Hand Manipulation](07-dexterous-inhand-manipulation.md)
- Next: [Simulation Backends](02-simulation-backends.md)

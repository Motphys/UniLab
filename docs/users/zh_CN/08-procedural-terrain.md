# 程序化地形 (Procedural Terrain)

语言: 简体中文

本页只回答四个问题:

1. 怎么把当前仓库已有的 rough terrain 任务跑起来？
2. Hydra 命令行能改什么、不能改什么？
3. 想改子地形组合时，正确的入口是什么？
4. 哪些是当前已知的边界，不是 bug 而是约束？

底层 contract（cold-path materialization、注册新 sub-terrain、`MjSpec` 序列化）见 [`scene/composer.py`](../../../src/unilab/scene/composer.py) 与 [`terrains/terrain_generator.py`](../../../src/unilab/terrains/terrain_generator.py) 的源码注释。

## 现状

当前仓库注册并接入程序化地形的任务只有一个：

| 任务 | owner YAML | 后端 | 入口算法 | 代码 |
| --- | --- | --- | --- | --- |
| `Go2JoystickRough` | [`conf/ppo/task/go2_joystick_rough/mujoco.yaml`](../../../conf/ppo/task/go2_joystick_rough/mujoco.yaml) | MuJoCo | PPO (`train_rsl_rl.py`) | [`go2/joystick.py`](../../../src/unilab/envs/locomotion/go2/joystick.py) |

env 构造期会执行：

1. 加载 [`scene_flat.xml`](../../../src/unilab/assets/robots/go2/scene_flat.xml)（机器人基础 XML，包含一个名为 `floor` 的占位 geom）。
2. `TerrainGenerator(cfg.terrain_generator)` 在 `MjSpec` 上写入 `terrain` body、地形 geoms / heightfield / 灯光。
3. 把所有引用 `floor` 的 contact sensor 重定向到 `terrain` 子树，再把 `floor` 占位 geom 删除。
4. 写出 `scene.xml` + `assets/` 到 per-instance 的 `tempfile.TemporaryDirectory`，backend 用 `model_file=<materialized_xml>` 编译 `MjModel`。
5. tempdir 一直持有到 env `close()`；`Go2WalkTask.get_playback_model()` 返回该路径，离线回放视频复用同一个 materialized scene。

`step()` / `reset()` / DR provider 不读 XML、不访问 asset 文件；地形相关全部发生在冷路径。

## 1. 直接训练

```bash
# 默认 grid 10×20、7 种 sub-terrain 混合
uv run train --algo ppo --task go2_joystick_rough --sim mujoco
```

PPO + MuJoCo 之外的组合（APPO、offpolicy、Motrix）目前没有 owner YAML。Motrix 后端虽然有 `@registry.env(..., sim_backend="motrix")` 注册，但当前 materializer 只验证过 MuJoCo 编译路径，没有 owner 配置。

训练结束后默认会自动回放并导出 `play_video.mp4`，渲染场景就是上述 materialized scene。

## 2. Hydra 命令行覆盖地形参数

`Go2JoystickRough` 在 [conf/ppo/task/go2_joystick_rough/mujoco.yaml](../../../conf/ppo/task/go2_joystick_rough/mujoco.yaml) 里显式列出了一组可覆盖字段；这些字段允许 Hydra struct 模式接受命令行覆盖。

| 字段 | 作用 | yaml 默认值 |
| --- | --- | --- |
| `env.terrain_generator.seed` | 随机种子，`null` 表示每次随机 | `null` |
| `env.terrain_generator.curriculum` | `true`：每种 sub-terrain 一列、难度沿行递增；`false`：按 `proportion` 随机采样 | `false` |
| `env.terrain_generator.num_rows` | grid 行数（curriculum 模式 = 难度等级数） | `10` |
| `env.terrain_generator.num_cols` | grid 列数（curriculum 模式被忽略，列数 = `len(sub_terrains)`） | `20` |
| `env.terrain_generator.border_width` | grid 外圈 flat border 宽度（米） | `20.0` |
| `env.terrain_generator.difficulty_range` | 难度采样区间 `[min, max]`，∈ `[0, 1]` | `[0.0, 1.0]` |
| `env.terrain_generator.add_lights` | 是否在 grid 上方加方向光 | `true` |

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

内置组合定义在 [`unilab.terrains.config`](../../../src/unilab/terrains/config.py)：

- `ROUGH_TERRAINS_CFG`：10 × 20，7 种 sub-terrain 按比例混合，random 模式。`Go2JoystickRoughCfg` 默认值与之一致（通过 `Go2RoughTerrainCfg` 子类锁定，每个 env 实例都会拿到独立的 cfg 对象）。
- `STAIRS_TERRAINS_CFG`：10 × 4，curriculum 模式，难度从 flat → easy → moderate → challenging。当前没有任务直接引用，可在自定义 task config 里使用。

## 4. 在新任务里启用程序化地形

`terrain_generator` 是基类 [`EnvCfg`](../../../src/unilab/base/base.py) 上的字段，默认 `None`。在自己的 task cfg 里赋值即可启用：

```python
import copy
from dataclasses import dataclass, field
from unilab.terrains import ROUGH_TERRAINS_CFG, TerrainGeneratorCfg

@dataclass
class MyTaskCfg(...):
    # 不能含 floor geom 之外其他冲突的 worldbody；至少保留一个名字为 `terrain_floor_geom`
    # 的 placeholder geom 让 contact sensor 在 load 阶段先校验通过。
    model_file: str = ".../scene_flat.xml"
    terrain_generator: TerrainGeneratorCfg = field(default_factory=lambda: copy.deepcopy(ROUGH_TERRAINS_CFG))
    # 可选：base XML 里 placeholder geom 的名字。默认 "floor"。
    terrain_floor_geom: str = "floor"
```

env 的 `__init__` 沿用 `Go2WalkTask.__init__` 的物化模板：

```python
import tempfile
from pathlib import Path
from unilab.scene.composer import compose_and_materialize

self._materialized_dir = tempfile.TemporaryDirectory(prefix="unilab_terrain_")
scene = compose_and_materialize(base_xml=Path(cfg.model_file), terrain_cfg=cfg.terrain_generator, output_dir=Path(self._materialized_dir.name), floor_geom=cfg.terrain_floor_geom)
backend = create_backend(..., model_file=str(scene.scene_xml), ...)
```

注意：`TerrainGenerator.__init__` 会原地修改传入的 cfg（向每个 `sub_cfg.size` 写值）。如果在多个 env 之间共享同一个 `TerrainGeneratorCfg` 实例会互相污染，必须用 `default_factory` 或 `copy.deepcopy` 保证每个实例拿到独立 cfg；`Go2JoystickRoughCfg` 通过 `default_factory=Go2RoughTerrainCfg` 已处理。

## 5. 可视化与离线回放

不开训练直接看一下 materialized 场景：

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --num_envs 4
```

## 6. 验证

```bash
# 程序化地形 + materializer 单元/集成测试
uv run pytest tests/terrains tests/scene -q

# Hydra compose + Go2JoystickRoughCfg 的 task owner 测试
uv run pytest tests/config/test_locomotion_params.py -k rough -q

# Hydra 命令行覆盖 + registry deep-merge 闭环
uv run pytest tests/config/test_locomotion_params.py \
    -k "apply_cfg_overrides or hydra_terrain_override" -q

# 端到端 smoke：Hydra 命令行覆盖 grid 大小 + 种子，2 iter PPO
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
    env.terrain_generator.num_rows=4 env.terrain_generator.seed=42 \
    algo.max_iterations=2 algo.num_envs=64
```

## 已知约束

- **当前只在 MuJoCo 后端验证过**：materializer 通过 `mujoco.MjSpec` 编辑 spec 并 `MjSpec.to_xml()` 序列化。Motrix 走相同的 `model_file` 入口理论可行，但仓库里没有 Motrix owner YAML，也没有 Motrix 路径下 hfield / 程序化场景的冒烟测试。生产训练请只用 `--sim mujoco`。
- **Heightfield 高度着色会丢失**：`HfPyramidSlopedTerrainCfg` 等通过 `color_by_height` 写入 in-memory buffer texture，`MjSpec.to_xml()` 不能序列化这种 texture，因此 materialized 场景里 heightfield 会失去高度色，但物理一致，不影响训练。
- **base XML 必须包含名为 `floor` 的 placeholder geom**：被 contact sensor 引用的 geom 必须在 load 阶段就存在，composer 会在 retarget 后再把它 strip 掉。如果不叫 `floor`，可以通过 `cfg.terrain_floor_geom` 覆盖。
- **`terrain_generator` 是 cold-path 配置**：env 构造完成后再修改 `cfg.terrain_generator` 不会影响已 materialize 的场景。要换地形必须重新构造 env（即重新跑训练命令）。
- **`import unilab.terrains` 不依赖 mujoco**：模块对 `import mujoco` 用 `try/except ImportError` 包装，仅 `compose_and_materialize` 与 `TerrainGenerator.compile` 真正调用 mujoco；这是为了保证 [`tests/envs/test_env_configs.py::test_registry_bootstrap_and_config_imports_do_not_require_mujoco`](../../../tests/envs/test_env_configs.py) 这条 contract 不被破坏。

## 5. Spawn 与 terrain-level 课程

启用 `terrain_generator` 后，每个 env 在初始化时被分配一个固定的 `(level, type_col)`，reset 时 `qpos[xyz] = init_qpos + terrain_origins[level, type_col] + [0, 0, spawn_height_margin]`。这样机器人脚下的实际地表高度（台阶顶面、坑底平台、斜坡顶等）会被自动加到出生 z 上，避免卡进高地形或从低地形悬空摔下。Flat 变体（无 `terrain_generator`）所有 env 在世界原点附近 ±0.5m 内出生（reset 自带的 xy 抖动），无需额外分散。

实现见 [`src/unilab/envs/locomotion/common/terrain_spawn.py`](../../../src/unilab/envs/locomotion/common/terrain_spawn.py)。`Go2JoystickRoughCfg` 已经持有一个 `terrain_curriculum` 字段（默认 `TerrainCurriculumCfg()`），其内部字段如下：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `enabled` | `False` | 启用动态升降；关闭时仅做 cell-aware spawn 分配 |
| `promote_frac` | `0.5` | 走过 > `promote_frac` × `cell_size` 升级一档 |
| `demote_frac` | `0.25` | 走过 < `demote_frac` × `cell_size` 降级一档 |
| `cycle_top_frac` | `0.5` | 顶行 cycle 下界 = `num_rows` × `cycle_top_frac` |
| `spawn_height_margin` | `0.05` | 出生 z 上抬量（米），用于吸收 random/wave 表面近似 |
| `seed` | `None` | RNG 种子（type_col 抽样、初始 level 随机分布、cycle 重采） |

通过 Hydra override 启用例如：`task.terrain_curriculum.enabled=true task.terrain_curriculum.seed=0`。

**两种模式**：

- **`enabled=False`（默认）**：每个 env 的 `level` 在初始化时从 `[0, num_rows)` 均匀随机抽取后固定，全程不变。训练数据天然覆盖所有难度行；`type_col` 同样随机固定。这是接通 `terrain_origins` 之后、不引入动态课程时的默认行为。
- **`enabled=True`**：所有 env 从 `level=0` 开始；episode 结束（terminated 或 truncated）时按水平走过的距离对 `(promote_frac × cell_size, demote_frac × cell_size)` 阈值升降。`level` 越过 `num_rows-1` 时随机回到 `[num_rows × cycle_top_frac, num_rows-1]`，避免全部 env 集中在最难行造成难度分布退化。`level` 在 0 处 clamp 不会进入负值。

**`spawn_height_margin` 的意义**：`HfPyramidStairs` / `HfInvertedPyramidStairs` / `HfPyramidSloped` / `HfFlat` 的 `terrain_origins[2]` 是 cell 中心实际表面（精确）；但 `HfRandomUniform` 是 noise 中线、`HfWaveTerrain` 是 0（中线），都是近似 —— cell 中心实际表面可能略高于该值。默认 5cm margin 让机器人脚悬空一点点，物理 1-2 帧就稳。如果你把 `noise_range` / `amplitude_range` 调到默认两倍以上，建议同步把 margin 调大（比如 0.1）。

**W&B 日志**（episode 结束时写入 `state.info["log"]`）：

| Key | 含义 |
| --- | --- |
| `terrain_curriculum/mean_level` | 当前所有 env `level` 的平均值，启用 curriculum 时随训练上升 |
| `terrain_curriculum/max_level` | 当前最高 `level` |
| `terrain_curriculum/mean_walked` | 本批次完成的 episode 平均水平距离 |
| `terrain_curriculum/num_promoted` | 本批次升级的 env 数量（`enabled=False` 时恒为 0） |
| `terrain_curriculum/num_demoted` | 本批次降级的 env 数量（`enabled=False` 时恒为 0） |
| `terrain_curriculum/num_skipped` | 因 `_has_started=False`（首次 done 之前）跳过统计的 env 数 |

**奖励权重的迁移**：现有 `Go2JoystickRoughCfg` 的 reward 是基于"机器人脚下永远是 z=0 平地"调出的；切到 cell-aware spawn 后，inverted pit 中的机器人 base z 会是 `init_z - 0.6m` 量级，pyramid 顶上则是 `init_z + 0.6m` 量级。`base_height` reward 的目标值需要相应调整或改用相对脚的高度差。建议在启用 curriculum 重训前先用 `enabled=False` 跑一个 baseline 确认 reward 分布稳定。

**已知限制**：`HfRandomUniform` / `HfWaveTerrain` 的 spawn z 是中线近似而非精确表面；后续可加可选项"精确 hfield surface 查询"读 `userdata`，本节描述的版本不做。

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Dexterous In-Hand Manipulation](07-dexterous-inhand-manipulation.md)
- Next: [Simulation Backends](02-simulation-backends.md)

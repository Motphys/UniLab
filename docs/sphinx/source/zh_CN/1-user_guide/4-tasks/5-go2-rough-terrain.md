# 程序化地形

语言: 简体中文

## 任务

- `go2_joystick_rough`

## 默认命令

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco
uv run train --algo ppo --task go2_joystick_rough --sim motrix
```

## 这页关心什么

- rough terrain 的训练入口
- terrain generator 的常见覆盖项
- 任务当前已知边界

## 当前任务范围

- 当前用户侧 rough terrain 入口就是 `go2_joystick_rough`
- MuJoCo 和 Motrix 都有 PPO owner
- 默认 profile 是单 patch `random_rough`

## 常见覆盖

```bash
uv run train --algo ppo --task go2_joystick_rough --sim mujoco \
  env.scene.terrain.generator.num_rows=4 \
  env.scene.terrain.generator.num_cols=6 \
  env.scene.terrain.generator.curriculum=true \
  env.scene.terrain.generator.seed=42
```

常见可覆盖字段：

| 字段 | 默认值 | 作用 |
|------|--------|------|
| `env.scene.terrain.generator.seed` | `42` | 固定 terrain 随机种子 |
| `env.scene.terrain.generator.curriculum` | `false` | 开启后按难度列组织 terrain |
| `env.scene.terrain.generator.size` | `[8.0, 8.0]` | 单 patch 尺寸 |
| `env.scene.terrain.generator.num_rows` | `1` | grid 行数 |
| `env.scene.terrain.generator.num_cols` | `1` | grid 列数 |
| `env.scene.terrain.generator.border_width` | `1.0` | 外圈 flat border 宽度 |
| `env.scene.terrain.generator.difficulty_range` | `[0.0, 1.0]` | 难度采样区间 |
| `env.terrain_scan.enabled` | `true` | critic 是否拼接 height scan |
| `env.terrain_scan.geom_name` | `floor` | 被采样的地形 geom |

## 地形组合与边界

- 当前 rough env 的 actor obs 仍沿用 flat Go2 主体；critic 可以额外接入 height scan。
- 默认 scan 网格是 x 方向 17 点、y 方向 11 点；actor 维度 49，critic 维度 239。
- 当前默认会用到的子地形包括 `flat`、`pyramid_stairs`、`pyramid_stairs_inv`、`hf_pyramid_slope`、`hf_pyramid_slope_inv`、`random_rough`、`wave_terrain`。
- `Go2RoughTerrainCfg` 是默认训练 profile：1x1，默认只采样 `random_rough`。
- `ROUGH_TERRAINS_CFG` 是多子地形混合 profile：10x20，随机混合 7 类子地形。
- `STAIRS_TERRAINS_CFG` 是 curriculum 楼梯 profile：10x4，难度沿列递增。
- `sub_terrains` 不是适合直接从命令行重建的覆盖项。
- `scene.terrain.generator` 是 cold-path 配置；env 创建后再改不会影响已 materialize 的场景。
- height scan 当前主要是 MuJoCo rough env 路径；Motrix rough 不要求和 MuJoCo 完全等价。

如果你要在新任务里接程序化地形，最小入口仍是 `SceneCfg` + `scene.terrain.generator`，不是在热路径里临时改 XML：

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
```

## 关联入口

- 后端规则：看 [02 仿真后端](../2-simulation-backends.md)
- 任务总索引：看 [D 任务索引](1-task-index.md)

## 可视化与验证

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --num_envs 4
uv run pytest tests/config/test_locomotion_params.py -k rough -q
uv run pytest tests/envs/locomotion/test_go2_terrain_spawn.py tests/envs/locomotion/test_go2_rough_height_scan.py -q
```

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [Sharpa Inhand](4-sharpa-inhand.md)
- Next: [Go2 Arm Manip Loco](6-go2-arm-manip-loco.md)

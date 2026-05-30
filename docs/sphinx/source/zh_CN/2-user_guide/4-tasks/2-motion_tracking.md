# 动作追踪

语言: 简体中文

G1 动作追踪任务位于 `src/unilab/envs/motion_tracking/` 下，并通过
`conf/ppo/`、`conf/appo/` 以及选定的 off-policy 路径中的 task owner YAML 选择。

## Task Owners

| CLI Task | Registered Env | Owner Evidence |
| --- | --- | --- |
| `g1_motion_tracking` | `G1MotionTracking` | `conf/ppo/task/g1_motion_tracking/`, `conf/appo/task/g1_motion_tracking/` |
| `g1_flip_tracking` | `G1FlipTracking` | `conf/ppo/task/g1_flip_tracking/`, `conf/appo/task/g1_flip_tracking/` |
| `g1_wall_flip_tracking` | `G1WallFlipTracking` | `conf/ppo/task/g1_wall_flip_tracking/`, `conf/appo/task/g1_wall_flip_tracking/` |
| `g1_climb_tracking` | G1 climb tracking env | `conf/ppo/task/g1_climb_tracking/`, `conf/appo/task/g1_climb_tracking/` |
| `g1_box_tracking` | G1 box tracking env | `conf/ppo/task/g1_box_tracking/` |
| `g1_wbt_obs` | `G1MotionTrackingSAC` | `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` |

## PPO 与 APPO

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_motion_tracking --sim motrix training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
```

## SAC WBT 路径

```bash
uv run train --algo sac --task g1_wbt_obs --sim mujoco training.use_amp=true
uv run train --algo sac --task g1_wbt_obs --sim mujoco \
  training.use_amp=true
```

`g1_wbt_obs` owner 是与部署对齐的 off-policy 观测配置。

## 动作文件

动作 NPZ 文件通过 `env.motion_file` 读取。预期的训练载荷包括 `fps`、关节
位置/速度、body 位姿和 body 速度数组。转换与检查辅助工具在 `scripts/motion/` 中：

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop
```

有关更详细的动作转换说明，请参阅 `scripts/motion/README.md`。

## Navigation

- Index: [文档](0-index.md)

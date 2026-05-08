# G1 运动跟踪指南

语言: 简体中文

本页是 G1 全身运动跟踪任务的操作指南，覆盖从 motion 文件准备、训练、回放到交互式调试的完整流程。常规训练和回放优先使用统一 CLI：

```bash
uv run train --algo <algo> --task <task> --sim <backend>
uv run eval --algo <algo> --task <task> --sim <backend> --load-run <run>
```

## 适用任务

| 使用场景                           | CLI task               | 注册环境名              | 主要入口             |
| ---------------------------------- | ---------------------- | ----------------------- | -------------------- |
| 通用 G1 whole-body motion tracking | `g1_motion_tracking` | `G1MotionTracking`    | PPO、MLX PPO、APPO   |
| flip clip 专用 profile             | `g1_flip_tracking`   | `G1FlipTracking`      | PPO、MLX PPO、APPO   |
| wall-assisted flip 专用 profile    | `g1_wall_flip_tracking` | `G1WallFlipTracking` | PPO、MLX PPO、APPO   |
| holosoma-aligned FastSAC WBT       | `g1_sac_wbt`         | `G1MotionTrackingSAC` | FastSAC，MuJoCo only |

当前已提交的 owner YAML 位于：

- PPO / MLX PPO：`conf/ppo/task/g1_motion_tracking/`、`conf/ppo/task/g1_flip_tracking/`、`conf/ppo/task/g1_wall_flip_tracking/`
- APPO：`conf/appo/task/g1_motion_tracking/`、`conf/appo/task/g1_flip_tracking/`、`conf/appo/task/g1_wall_flip_tracking/`
- FastSAC WBT：`conf/offpolicy/task/sac/g1_sac_wbt/mujoco.yaml`

默认 motion：

- `g1_motion_tracking`：`src/unilab/assets/motions/g1/dance1_subject2_part.npz`
- `g1_flip_tracking`：`src/unilab/assets/motions/g1/flip_360_001__A304.npz`
- `g1_wall_flip_tracking`：`src/unilab/assets/motions/g1/flip_from_wall_104__A304.npz`
- `g1_sac_wbt`：与 `g1_motion_tracking` 相同

## 推荐流程

1. 选择 task：普通动作先用 `g1_motion_tracking`，flat flip 数据先用 `g1_flip_tracking`，带墙起跳/接触数据先用 `g1_wall_flip_tracking`。
2. 准备或选择 `.npz` motion 文件。
3. 用 MuJoCo replay 检查 `.npz` 的姿态、速度和 body layout。
4. 用较小训练预算做 smoke run。
5. 放大 `algo.num_envs` 和 `algo.max_iterations` 做正式训练。
6. 用 `uv run eval` 回放 checkpoint；需要逐帧看 target 或 reward 时再用交互式调试脚本。

## 快速开始

PPO + MuJoCo 是最直接的本地路径：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
```

训练结束后回放最新 run：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1
```

Motrix 路径使用同一个 task 名，只切换 backend：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1
```

macOS / MacBook 上，只要回放会打开 MotrixSim 原生 renderer，统一 CLI 会自动路由到 `mxpython`。如果训练时不需要自动回放，追加：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim motrix training.no_play=true
```

## Motion NPZ 格式

训练环境读取预处理后的 `.npz` 文件。标准文件包含 7 个 key：

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

`MotionLoader` 支持单个路径，也支持路径列表。多 clip 列表会在 loader 内拼接，但 episode 不会跨 clip 边界继续播放；到达当前 clip 末尾后，该 env 会进入 reset。

```yaml
env:
  motion_file:
    - src/unilab/assets/motions/g1/dance1_subject2_part.npz
    - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

## 准备 Motion

### 使用仓库内置 motion

直接覆盖 `env.motion_file` 即可：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  env.motion_file=src/unilab/assets/motions/g1/gangnam_style.npz
```

### Unitree CSV 转 NPZ

Unitree CSV 可以通过 `scripts/motion/csv_to_npz.py` 转成训练可读的 NPZ。输出 FPS 通常设为 50 Hz，以匹配当前训练控制频率。

```bash
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50
```

只导出一个时间片段：

```bash
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

### BONES-SEED CSV 转 NPZ

BONES-SEED G1 CSV 使用固定 36 列布局，可以先 dry-run 检查解析：

```bash
uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
```

转换一个本地 CSV：

```bash
uv run scripts/motion/bones_seed_csv_to_npz.py \
  --input path/to/flip_090_001__A304.csv \
  --output temp/flip_090_001__A304.npz
```

更多 BONES-SEED 输入列约定见 [motion scripts README](../../../scripts/motion/README.md)。

### Holosoma NPZ remap

holosoma 等管线导出的 NPZ 可能包含 root free-joint、碰撞球体或额外手指 body。训练前先 remap 到 UniLab 标准训练模型布局：

```bash
uv run scripts/motion/remap_fullbody_npz.py \
  --input path/to/holosoma_motion.npz \
  --output src/unilab/assets/motions/g1/motion_remapped.npz
```

如果只想验证 layout 而不写文件：

```bash
uv run scripts/motion/remap_fullbody_npz.py \
  --input path/to/holosoma_motion.npz \
  --output temp/motion_remapped.npz \
  --dry-run
```

## 检查 Motion

生成或 remap 完 NPZ 后，先用 MuJoCo viewer 回放：

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop
```

慢放检查：

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

如果回放里 body 姿态明显错位，先检查：

- NPZ 是否包含标准 7 个 key
- `fps` 是否符合预期
- body layout 是否需要 `scripts/motion/remap_fullbody_npz.py`
- joint 顺序是否匹配当前 G1 训练模型

## 训练

### PPO

PPO owner YAML 默认训练预算：

- `g1_motion_tracking`：`algo.max_iterations=15000`
- `g1_flip_tracking`：`algo.max_iterations=30000`
- `g1_wall_flip_tracking`：`algo.max_iterations=30000`

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_wall_flip_tracking --sim mujoco
```

`g1_wall_flip_tracking` 是独立的 wall-assisted flip profile，使用带墙场景和 `flip_from_wall_104__A304.npz`，不改变 `g1_flip_tracking` 的 flat flip 默认配置。

Motrix：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run train --algo ppo --task g1_flip_tracking --sim motrix
uv run train --algo ppo --task g1_wall_flip_tracking --sim motrix
```

smoke run 可以降低训练预算：

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  algo.num_envs=128 algo.max_iterations=5 training.no_play=true
```

### APPO

APPO 使用异步 collector/learner 路径，适合吞吐优先的训练：

```bash
uv run train --algo appo --task g1_motion_tracking --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix
```

smoke run：

```bash
uv run train --algo appo --task g1_motion_tracking --sim mujoco \
  algo.num_envs=128 algo.max_iterations=5 training.no_play=true
```

## 回放 Checkpoint

统一回放入口是 `uv run eval`。`--load-run -1` 表示加载该 task 日志目录下的最新 run。

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1
uv run eval --algo appo --task g1_motion_tracking --sim mujoco --load-run -1
uv run eval --algo sac --task g1_sac_wbt --sim mujoco --load-run -1
```

指定某个 run：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco \
  --load-run 2026-03-16_01-35-12_mujoco
```

MuJoCo PPO 回放支持跟随镜头：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1 \
  training.cam_tracking=true \
  training.cam_tracking_env_idx=0 \
  training.cam_tracking_extra_envs=2
```

当前 play/eval 会复用 task 的 env 配置。如果只想临时减少回放随机性，可以在 eval 命令里显式覆盖相关 env 字段，例如：

```bash
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1 \
  env.sampling_mode=start \
  env.joint_position_range=[0.0,0.0]
```

## 交互式调试

`scripts/play_interactive.py` 基于 MuJoCo viewer，可直接显示 target body、reward reference 和速度信息。暂不支持 Motrix 原生 renderer。

可视化 motion target：

```bash
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_target_bodies=true \
  interactive.target_show_axes=true
```

只看部分 body：

```bash
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_target_bodies=true \
  interactive.target_body_names=torso_link,left_wrist_yaw_link,right_wrist_yaw_link
```

查看 reward debug 信息：

```bash
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_reward_debug=true \
  interactive.reward_debug_show_velocity=true \
  interactive.reward_debug_show_connectors=true \
  interactive.target_max_bodies=4
```

如果要加载策略，传入 run 和 checkpoint：

```bash
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.action_mode=policy \
  algo.load_run=2026-03-16_01-35-12_mujoco \
  algo.checkpoint=model_5000.pt
```

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Algorithms](04-algorithms.md)
- Next: [Domain Randomization](06-domain-randomization.md)

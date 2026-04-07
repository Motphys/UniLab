# G1 Motion Tracking

UniLab 当前提供一个 G1 的 whole-body motion tracking 任务。

- Hydra task：`g1_motion_tracking`
- 注册环境名：`G1MotionTracking`
- 后端注册：`mujoco` 和 `motrix`
- 已提交的 Motrix 特化配置：PPO 和 APPO 的 motion-tracking reward
- 默认 motion 文件：`src/unilab/assets/motions/g1/dance1_subject2_part.npz`

## Environment Entrypoints

```bash
# PPO (RSL-RL, MuJoCo)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking

# PPO (RSL-RL, Motrix)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix

# APPO (MuJoCo)
uv run python scripts/train_appo.py task=g1_motion_tracking

# APPO (Motrix)
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix

# 回放最新 checkpoint
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.play_only=true

# Motrix PPO 回放会打开原生 renderer
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true

# APPO MuJoCo play
uv run python scripts/train_appo.py task=g1_motion_tracking training.play_only=true

# APPO Motrix play 会打开原生 renderer
uv run python scripts/train_appo.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true
```

G1 motion tracking 的 Motrix 训练 / 回放主链路优先走 `scripts/train_rsl_rl.py` 和
`scripts/train_appo.py`。调试脚本 `scripts/play_interactive.py` 仍按 MuJoCo viewer 路径使用。

## Interactive Debugging

`scripts/play_interactive.py` 可以直接查看 target body，也可以查看 reward 使用的参考位姿和速度。
当前该脚本按 MuJoCo viewer 实现，不支持 Motrix 原生 renderer。

```bash
# 可视化 motion target
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_show_axes

# 只查看部分 body
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_body_names torso_link,left_wrist_yaw_link,right_wrist_yaw_link

# 查看 reward debug 信息
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_reward_debug \
  --reward_debug_show_velocity \
  --reward_debug_show_connectors \
  --target_max_bodies 4
```

如果需要加载指定 run 或 checkpoint，可额外传入 `--load_run` 和 `--checkpoint`。

## Motion Preprocess

训练环境读取的是预处理后的 `.npz` 文件。使用 `scripts/motion/csv_to_npz.py` 可以把 Unitree 格式的 CSV 转成训练可用的 NPZ：

```bash
# 全量转换
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50

# 只导出一个时间片段
uv run python scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

## Replay NPZ

生成好 NPZ 之后，可以用 `scripts/motion/replay_npz.py` 在 MuJoCo viewer 中直接检查动作：

```bash
# 循环播放
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop

# 0.5x 慢放
uv run python scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

## Config Note

`task=g1_motion_tracking` 默认读取环境配置里的 `motion_file`。如果要切换到自定义 motion，先生成 `.npz`，再更新环境配置里的默认 `motion_file`。

`motion_file` 现在同时支持单个字符串路径和字符串列表。列表模式下，训练会在多个 motion clip 之间采样，并且每个 episode 都会保持在当前 clip 内，不会跨文件串帧。例如：

```yaml
motion_file:
  - src/unilab/assets/motions/g1/dance1_subject2_part.npz
  - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

如果要验证 Motrix 路径，优先使用训练脚本内置 play mode，而不是 MuJoCo-only 的调试脚本：

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)

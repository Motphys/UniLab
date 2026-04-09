# G1 Motion Tracking

语言: [English](../en/05-g1-motion-tracking.md) | 简体中文 | [日本語](../ja/05-g1-motion-tracking.md) | [한국어](../ko/05-g1-motion-tracking.md)

UniLab 当前提供两个 G1 whole-body motion tracking task：

- Hydra task：`g1_motion_tracking`（兼容历史默认）
- 注册环境名：`G1MotionTracking`
- Hydra task：`g1_flip_tracking`（flip 专用 profile）
- 注册环境名：`G1FlipTracking`
- 后端注册：`mujoco` 和 `motrix`
- 已提交的 Motrix 特化配置：PPO 和 APPO 的 motion-tracking reward
- `g1_motion_tracking` 默认 motion：`src/unilab/assets/motions/g1/dance1_subject2_part.npz`
- `g1_flip_tracking` 默认 motion：`src/unilab/assets/motions/g1/flip_360_001__A304.npz`

## Environment Entrypoints

```bash
# PPO (RSL-RL, MuJoCo)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking

# PPO (RSL-RL, MuJoCo, flip profile)
uv run python scripts/train_rsl_rl.py task=g1_flip_tracking

# PPO (RSL-RL, Motrix)
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix

# PPO (RSL-RL, Motrix, flip profile)
uv run python scripts/train_rsl_rl.py task=g1_flip_tracking training.sim_backend=motrix

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

# APPO MuJoCo 回放
uv run python scripts/train_appo.py task=g1_motion_tracking training.play_only=true

# APPO Motrix 回放会打开原生 renderer
uv run python scripts/train_appo.py task=g1_motion_tracking \
  training.sim_backend=motrix \
  training.play_only=true
```

对于 G1 motion tracking，Motrix 的训练和回放主路径应优先走 `scripts/train_rsl_rl.py` 和 `scripts/train_appo.py`。调试脚本 `scripts/play_interactive.py` 仍沿用 MuJoCo viewer 路径。

## Interactive Debugging

`scripts/play_interactive.py` 可以直接可视化 target body，也可以显示 reward 使用的参考位姿与速度。该脚本基于 MuJoCo viewer 实现，不支持 Motrix 原生 renderer。

```bash
# 可视化 motion target
uv run python scripts/play_interactive.py \
  --task G1MotionTracking \
  --show_target_bodies \
  --target_show_axes

# 只看部分 body
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

如果需要指定 run 或 checkpoint，还可以额外传入 `--load_run` 和 `--checkpoint`。

## Motion Preprocessing

训练环境读取预处理后的 `.npz` 文件。使用 `scripts/motion/csv_to_npz.py` 可以把 Unitree 格式的 CSV 转成训练环境可直接加载的 NPZ:

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

生成好 NPZ 后，可以用 `scripts/motion/replay_npz.py` 在 MuJoCo viewer 中直接检查动作:

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

## Configuration Note

`task=g1_motion_tracking` 默认读取环境配置里的单个 `motion_file`（历史默认是 `dance1_subject2_part.npz`）。`task=g1_flip_tracking` 提供 flip 专用默认 profile（更保守的 reset 随机化与 termination）。

PPO 默认训练预算也做了分流：`g1_motion_tracking` 保持历史默认 `max_iterations=15000`，`g1_flip_tracking` 使用更长的 `max_iterations=30000`。

`motion_file` 现在同时支持单个字符串路径和字符串列表。列表模式下，训练会在多个 motion clip 之间采样，并且每个 episode 都会保持在当前 clip 内，不会跨文件串帧。例如：

```yaml
motion_file:
  - src/unilab/assets/motions/g1/dance1_subject2_part.npz
  - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

`sampling_mode` 的语义现在显式区分：

- `start`：保持历史行为，总是从全局第 0 帧开始
- `clip_start`：从随机 clip 的首帧开始，适合多 clip 列表
- `uniform` / `adaptive`：在拼接后的全局帧空间采样，但 episode 会在当前 clip 边界处截断

验证 Motrix 路径时，优先使用训练脚本自带的 play mode，而不是只支持 MuJoCo 的调试脚本:

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)

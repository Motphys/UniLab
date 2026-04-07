# G1 Motion Tracking

语言: [English](../en/05-g1-motion-tracking.md) | 简体中文 | [日本語](../ja/05-g1-motion-tracking.md) | [한국어](../ko/05-g1-motion-tracking.md)

UniLab 当前提供一个 G1 的全身 motion tracking 任务。

- Hydra task: `g1_motion_tracking`
- 注册的 env 名称: `G1MotionTracking`
- 已注册后端: `mujoco` 和 `motrix`
- 已落地的 Motrix 专用配置: PPO 和 APPO 的 motion-tracking reward
- 默认 motion 文件: `src/unilab/assets/motions/g1/dance1_subject2_part.npz`

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

`task=g1_motion_tracking` 默认读取 env config 中声明的 `motion_file`。如果要切换到自定义 motion，先生成 `.npz`，再更新 env config 中默认的 `motion_file`。

验证 Motrix 路径时，优先使用训练脚本自带的 play mode，而不是只支持 MuJoCo 的调试脚本:

```bash
uv run python scripts/train_rsl_rl.py task=g1_motion_tracking training.sim_backend=motrix
uv run python scripts/train_appo.py task=g1_motion_tracking training.sim_backend=motrix
```

## Navigation

- Previous: [Algorithms](04-algorithms.md)
- Next: [Collaboration Workflow](06-collaboration.md)

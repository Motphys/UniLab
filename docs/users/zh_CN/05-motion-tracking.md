# G1 Motion Tracking

语言: 简体中文

UniLab 当前提供两个 G1 whole-body motion tracking task family 和一个 FastSAC off-policy 入口：

- task family：`g1_motion_tracking`
- 注册环境名：`G1MotionTracking`
- task family：`g1_flip_tracking`（flip 专用 profile）
- 注册环境名：`G1FlipTracking`
- task family：`g1_sac_wbt`（FastSAC off-policy，从 holosoma 迁移）
- 注册环境名：`G1MotionTrackingSAC`
- 后端注册：`mujoco` 和 `motrix`（`g1_sac_wbt` 目前仅 `mujoco`）
- 已提交的 Motrix 特化配置：PPO 和 APPO 的 motion-tracking reward
- `g1_motion_tracking` 默认 motion：`src/unilab/assets/motions/g1/dance1_subject2_part.npz`
- `g1_flip_tracking` 默认 motion：`src/unilab/assets/motions/g1/flip_360_001__A304.npz`
- `g1_sac_wbt` 默认 motion：与 `g1_motion_tracking` 相同
- 常规训练入口统一使用 `uv run train --algo <algo> --task <family> --sim <backend>`

## Environment Entrypoints

```bash
# PPO (RSL-RL, MuJoCo)
uv run train --algo ppo --task g1_motion_tracking --sim mujoco

# PPO (RSL-RL, MuJoCo, flip profile)
uv run train --algo ppo --task g1_flip_tracking --sim mujoco

# PPO (RSL-RL, Motrix)
uv run train --algo ppo --task g1_motion_tracking --sim motrix

# PPO (RSL-RL, Motrix, flip profile)
uv run train --algo ppo --task g1_flip_tracking --sim motrix

# APPO (MuJoCo)
uv run train --algo appo --task g1_motion_tracking --sim mujoco

# APPO (Motrix)
uv run train --algo appo --task g1_motion_tracking --sim motrix

# FastSAC (MuJoCo, holosoma-aligned WBT)
uv run train --algo sac --task g1_sac_wbt --sim mujoco

# FastSAC with AMP (recommended for CUDA)
uv run train --algo sac --task g1_sac_wbt --sim mujoco training.use_amp=true

# 回放最新 checkpoint
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1

# Motrix PPO 回放会打开原生 renderer；统一 CLI 会在 macOS / MacBook 自动路由到 mxpython
uv run eval --algo ppo --task g1_motion_tracking --sim motrix --load-run -1

# APPO MuJoCo 回放
uv run eval --algo appo --task g1_motion_tracking --sim mujoco --load-run -1

# APPO Motrix 回放会打开原生 renderer；统一 CLI 会在 macOS / MacBook 自动路由到 mxpython
uv run eval --algo appo --task g1_motion_tracking --sim motrix --load-run -1
```

对于 G1 motion tracking，Motrix 的训练和回放主路径应优先走 `uv run train` 和 `uv run eval`。macOS / MacBook 上只要会打开 MotrixSim 原生 renderer，统一 CLI 会自动路由到 `mxpython`；不需要可视化的训练可追加 `training.no_play=true`。调试脚本 `scripts/play_interactive.py` 仍沿用 MuJoCo viewer 路径。

## Interactive Debugging

`scripts/play_interactive.py` 可以直接可视化 target body，也可以显示 reward 使用的参考位姿与速度。该脚本基于 MuJoCo viewer 实现，不支持 Motrix 原生 renderer。

```bash
# 可视化 motion target
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_target_bodies=true \
  interactive.target_show_axes=true

# 只看部分 body
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_target_bodies=true \
  interactive.target_body_names=torso_link,left_wrist_yaw_link,right_wrist_yaw_link

# 查看 reward debug 信息
uv run scripts/play_interactive.py \
  task=g1_motion_tracking/mujoco \
  interactive.show_reward_debug=true \
  interactive.reward_debug_show_velocity=true \
  interactive.reward_debug_show_connectors=true \
  interactive.target_max_bodies=4
```

如果需要指定 run 或 checkpoint，可以传入 `algo.load_run` 和 `algo.checkpoint`。

## Motion Preprocessing

训练环境读取预处理后的 `.npz` 文件。使用 `scripts/motion/csv_to_npz.py` 可以把 Unitree 格式的 CSV 转成训练环境可直接加载的 NPZ:

```bash
# 全量转换
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 \
  --output_fps 50

# 只导出一个时间片段
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 \
  --output_fps 50 \
  --start_time 4.0 \
  --end_time 9.0
```

## Remap Full-body NPZ (Holosoma → UniLab)

部分动捕管线（如 holosoma）导出的 NPZ 基于更详细的 MuJoCo 模型（例如包含碰撞球体、手指关节等 51 bodies），并且在 `joint_pos` / `joint_vel` 中包含了 root free-joint。UniLab 训练环境期望 NPZ 与训练模型布局（如 `scene_flat.xml` 的 31 bodies）对齐，且只包含被驱动关节的自由度。

`scripts/motion/remap_fullbody_npz.py` 负责完成这一转换：

```bash
# 基本用法：将 holosoma 导出的 NPZ 转换为 UniLab 训练格式
uv run scripts/motion/remap_fullbody_npz.py \
  --input src/unilab/assets/motions/g1/holosoma_dance.npz \
  --output src/unilab/assets/motions/g1/dance_remapped.npz
```

转换完成后，可以使用 `scripts/motion/replay_npz.py` 在 MuJoCo viewer 中回放验证，或直接用于 motion tracking 训练。

## Replay NPZ

生成好 NPZ 后，可以用 `scripts/motion/replay_npz.py` 在 MuJoCo viewer 中直接检查动作:

```bash
# 循环播放
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --loop

# 0.5x 慢放
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz \
  --speed 0.5
```

## FastSAC WBT (holosoma Migration)

`task=sac/g1_sac_wbt/mujoco` 提供从 holosoma 迁移的 G1 whole-body tracking FastSAC 训练。超参数对齐 holosoma `exp:g1-29dof-wbt-fast-sac`（`gamma=0.99, tau=0.05, num_atoms=501, target_entropy_ratio=0.5`）。环境 `G1MotionTrackingSAC` 在 PPO 版 `G1MotionTracking` 基础上为 critic 增加了 `base_lin_vel`（asymmetric actor-critic）。

```bash
# 默认训练
uv run train --algo sac --task g1_sac_wbt --sim mujoco

# 推荐：开启 AMP 加速
uv run train --algo sac --task g1_sac_wbt --sim mujoco training.use_amp=true

# 自定义并行环境数和迭代次数
uv run train --algo sac --task g1_sac_wbt --sim mujoco \
    algo.num_envs=4096 algo.max_iterations=10000 training.use_amp=true

# 使用 wandb 记录
uv run train --algo sac --task g1_sac_wbt --sim mujoco \
    training.use_amp=true training.logger=wandb

# 指定 motion 文件
uv run train --algo sac --task g1_sac_wbt --sim mujoco \
    env.motion_file=src/unilab/assets/motions/g1/dance1_subject2_part.npz
```

## Configuration Note

`task=g1_motion_tracking/mujoco` 默认读取环境配置里的单个 `motion_file`（历史默认是 `dance1_subject2_part.npz`）。`task=g1_flip_tracking/mujoco` 提供 flip 专用默认 profile（更保守的 reset 随机化与 termination）。

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

验证 Motrix 路径时，优先使用统一训练入口的 play mode，而不是只支持 MuJoCo 的调试脚本:

```bash
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run train --algo appo --task g1_motion_tracking --sim motrix
```

## PPO Motion Tracking：Play Tracking + Viser

`task=g1_motion_tracking/mujoco` 的 PPO 路径支持两项可视化能力：

- Play 回放支持跟随镜头：`training.cam_tracking=true`
- 浏览器可视化脚本：`scripts/play_viser.py`（仅 MuJoCo）

新增配置（`training`）：

- `cam_tracking`（默认 `false`）
- `cam_tracking_env_idx`（默认 `0`）
- `cam_tracking_extra_envs`（默认 `2`）

仅 Viser 功能需要额外依赖：

```bash
uv sync --extra viser
```

必要命令运行示例：

```bash
# 1) PPO Play 回放（启用 tracking，亦可选择默认配置，即非跟随镜头）
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run xxx training.cam_tracking=true algo.checkpoint=xxx
```

```bash
# 2) Viser 可视化（零动作）
uv run scripts/play_viser.py task=g1_motion_tracking/mujoco interactive.action_mode=zero viser.max_envs=8 viser.port=8080
```

```bash
# 3) Viser 可视化（策略）
uv run scripts/play_viser.py task=g1_motion_tracking/mujoco interactive.action_mode=policy algo.load_run=xxx algo.checkpoint=xxx viser.max_envs=4 viser.port=8080
```


## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Algorithms](04-algorithms.md)
- Next: [Domain Randomization](06-domain-randomization.md)

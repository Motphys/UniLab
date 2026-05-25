# G1 运动跟踪指南

语言: 简体中文

> **Motion 资产已迁移到 Hugging Face**：`.npz` 文件不再随仓库分发，首次使用时由 `MotionLoader` 自动从
> [unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)
> 下载。`uv sync` 已自动安装所需依赖。详见
> [迁移指南](../../../developers/zh_CN/motion-asset-migration.md)。

## 任务范围

| 场景 | task | 常用算法 | 默认 motion |
|------|------|----------|-------------|
| 通用全身 tracking | `g1_motion_tracking` | PPO、MLX PPO、APPO | `dance1_subject2_part.npz` |
| flat flip clip | `g1_flip_tracking` | PPO、MLX PPO、APPO | `flip_360_001__A304.npz` |
| wall-assisted flip | `g1_wall_flip_tracking` | PPO、MLX PPO、APPO | `flip_from_wall_104__A304.npz` |
| holosoma-aligned WBT | `g1_sac_wbt` | SAC | 与 `g1_motion_tracking` 共用 |
| crawl-slope WBT (SAC) | `g1_sac_wbt` + 自定义场景 | SAC | `motion_crawl_slope_uni.npz` |
| sim2real-aligned WBT (deploy chain) | `g1_wbt_obs` | SAC | 与 `g1_motion_tracking` 共用 |

## 配置入口

- PPO / MLX PPO：`conf/ppo/task/g1_motion_tracking/`、`conf/ppo/task/g1_flip_tracking/`、`conf/ppo/task/g1_wall_flip_tracking/`
- APPO：`conf/appo/task/g1_motion_tracking/`、`conf/appo/task/g1_flip_tracking/`、`conf/appo/task/g1_wall_flip_tracking/`
- SAC WBT：`conf/offpolicy/task/sac/g1_sac_wbt/{mujoco,mujoco_deploy,motrix}.yaml`、`conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml`

## 推荐流程

1. 先选 task，再确认 motion 来源。
2. 先用 MuJoCo replay 检查 `.npz` 的姿态和 body layout。
3. 先做 smoke run，再放大 `algo.num_envs` 和 `algo.max_iterations`。
4. 常规回放用 `uv run eval`；需要看 target / reward 时再用交互脚本。

## Motion 资产与检查

标准 `.npz` 需要包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。`env.motion_file` 也支持路径列表：

```yaml
env:
  motion_file:
    - src/unilab/assets/motions/g1/dance1_subject2_part.npz
    - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

```bash
uv run scripts/motion/csv_to_npz.py --input_file src/unilab/assets/motions/g1/dance1_subject2.csv --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz --input_fps 30 --output_fps 50
uv run scripts/motion/csv_to_npz.py --input_file src/unilab/assets/motions/g1/dance1_subject2.csv --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz --input_fps 30 --output_fps 50 --start_time 4.0 --end_time 9.0
uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
uv run scripts/motion/bones_seed_csv_to_npz.py --input path/to/flip_090_001__A304.csv --output temp/flip_090_001__A304.npz
uv run scripts/motion/remap_fullbody_npz.py --input path/to/holosoma_motion.npz --output src/unilab/assets/motions/g1/motion_remapped.npz
uv run scripts/motion/remap_fullbody_npz.py --input path/to/holosoma_motion.npz --output temp/motion_remapped.npz --dry-run
uv run scripts/motion/replay_npz.py --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --loop
uv run scripts/motion/replay_npz.py --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --speed 0.5
```

如果 MuJoCo replay 里 body 姿态明显错位，优先检查：NPZ 是否包含标准 7 个 key、`fps` 是否匹配控制频率、body layout 是否需要 remap、joint 顺序是否匹配当前 G1 训练模型。

## 训练与回放

PPO owner 默认预算：`g1_motion_tracking` 为 `algo.max_iterations=15000`，`g1_flip_tracking` 和 `g1_wall_flip_tracking` 为 `30000`。

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_wall_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run train --algo ppo --task g1_flip_tracking --sim motrix
uv run train --algo ppo --task g1_wall_flip_tracking --sim motrix
uv run train --algo ppo --task g1_motion_tracking --sim mujoco algo.num_envs=128 algo.max_iterations=5 training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix
uv run train --algo sac --task g1_sac_wbt --sim mujoco training.use_amp=true
uv run train --algo sac --task g1_sac_wbt --sim mujoco --profile deploy training.use_amp=true
uv run train --algo sac --task g1_wbt_obs --sim mujoco training.use_amp=true
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1 training.cam_tracking=true training.cam_tracking_env_idx=0
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_sac_wbt/motrix training.play_only=true algo.load_run=/abs/path/to/logs/fast_sac/G1MotionTrackingSAC/2026-04-23_14-06-57_mujoco
```

`--profile deploy` 会切到面向部署的独立 SAC owner；`g1_wbt_obs` 是更彻底的 sim2real 部署 task（pelvis IMU + 5 步 per-term 历史 actor obs，与 deploy `ObservationManager` byte-对齐；额外 encoder-bias / foot-friction DR、`joint_acc_l2` / `joint_torque_l2` 奖励；部署工具在 `scripts/deploy/`，obs 对齐由 `tests/scripts/test_obs_alignment_g1_wbt.py` 三路 cross-check）。Motrix sim2sim 回放时用绝对路径透传 `algo.load_run`，不要塞进 `--load-run`。

## SAC WBT on crawl-slope 场景

在斜坡地形上跑 `g1_sac_wbt`，需要同时切换 motion 片段和 MuJoCo 场景文件，并固定 episode 长度、关闭 reset 随机化：

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/train_offpolicy.py \
  algo=sac task=sac/g1_sac_wbt/mujoco training.use_amp=true algo.seed=1 \
  +env.motion_file=src/unilab/assets/motions/g1/motion_crawl_slope_uni.npz \
  +env.scene.model_file=src/unilab/assets/robots/g1/scene_crawl_slope.xml \
  +env.sampling_mode=start \
  env.truncate_on_clip_end=true \
  +env.max_episode_seconds=20.0 \
  '+env.pose_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.velocity_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.joint_position_range=[0,0]'
```

关键覆写：`env.motion_file` 切爬坡动作；`env.scene.model_file` 切三段凸壳 mesh 场景；`sampling_mode=start` + `truncate_on_clip_end=true` 从 clip 起点出发并截断；randomization 全置零复用 motion 精确初始状态。

## 交互式调试

```bash
uv run scripts/play_interactive.py task=g1_motion_tracking/mujoco interactive.show_target_bodies=true interactive.target_show_axes=true
uv run scripts/play_interactive.py task=g1_motion_tracking/mujoco interactive.show_reward_debug=true interactive.reward_debug_show_velocity=true
```

## 关联入口

- 通用训练规则：看 [03 训练指南](../03-training.md)
- 算法选择：看 [04 算法说明](../04-algorithms.md)
- 任务总索引：看 [D 任务索引](01-task-index.md)

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [任务索引](01-task-index.md)
- Next: [Allegro Inhand](03-allegro-inhand.md)

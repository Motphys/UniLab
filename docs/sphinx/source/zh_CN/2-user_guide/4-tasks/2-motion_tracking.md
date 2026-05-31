# 动作追踪

G1 动作追踪任务位于 `src/unilab/envs/motion_tracking/` 下，并通过
`conf/ppo/`、`conf/appo/` 以及选定的 off-policy 路径中的 task owner YAML 选择。

> **Motion 资产已迁移到 Hugging Face。** `.npz` 片段不再随仓库分发，首次使用时由
> `MotionLoader`（`src/unilab/envs/motion_tracking/g1/motion_loader.py`）按需从
> [unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)
> 下载，下载逻辑在 `src/unilab/assets/hub.py`（`_HF_MOTIONS_REPO_ID`）。`uv sync`
> 已自动安装所需的 `huggingface_hub` 依赖。

## Task Owners

每个 task 在 env 配置 dataclass 中定义了默认 motion 片段：

| CLI Task | Registered Env | 默认 motion | Owner Evidence |
| --- | --- | --- | --- |
| `g1_motion_tracking` | `G1MotionTracking` | `dance1_subject2_part.npz` | `conf/ppo/task/g1_motion_tracking/`, `conf/appo/task/g1_motion_tracking/` |
| `g1_flip_tracking` | `G1FlipTracking` | `flip_360_001__A304.npz` | `conf/ppo/task/g1_flip_tracking/`, `conf/appo/task/g1_flip_tracking/` |
| `g1_wall_flip_tracking` | `G1WallFlipTracking` | `flip_from_wall_104__A304.npz` | `conf/ppo/task/g1_wall_flip_tracking/`, `conf/appo/task/g1_wall_flip_tracking/` |
| `g1_climb_tracking` | G1 climb tracking env | 由 env 配置给出 | `conf/ppo/task/g1_climb_tracking/`, `conf/appo/task/g1_climb_tracking/` |
| `g1_box_tracking` | G1 box tracking env | 由 env 配置给出 | `conf/ppo/task/g1_box_tracking/` |
| `g1_wbt_obs` | `G1MotionTrackingSAC` | 与 `g1_motion_tracking` 共用 | `conf/offpolicy/task/sac/g1_wbt_obs/mujoco.yaml` |

默认值在代码中设定：`dance1_subject2_part.npz`（`g1/tracking.py`），
`flip_360_001__A304.npz` 与 `flip_from_wall_104__A304.npz`（`g1/flip_tracking.py`）。

## PPO 与 APPO

PPO owner 迭代预算（`--sim mujoco` owner YAML）：`g1_motion_tracking` 为
`algo.max_iterations=15000`；`g1_flip_tracking` 和 `g1_wall_flip_tracking` 为
`20000`。（`g1_flip_tracking` 的 Motrix owner YAML 将其提到 `30000`。）

```bash
uv run train --algo ppo --task g1_motion_tracking --sim mujoco
uv run train --algo ppo --task g1_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_wall_flip_tracking --sim mujoco
uv run train --algo ppo --task g1_motion_tracking --sim motrix
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo ppo --task g1_motion_tracking --sim mujoco \
  algo.num_envs=128 algo.max_iterations=5 training.no_play=true
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1
uv run eval --algo ppo --task g1_motion_tracking --sim mujoco --load-run -1 \
  training.cam_tracking=true training.cam_tracking_env_idx=0
```

## SAC WBT 路径

```bash
uv run train --algo sac --task g1_motion_tracking --sim mujoco training.use_amp=true
uv run train --algo sac --task g1_wbt_obs --sim mujoco training.use_amp=true
```

`g1_wbt_obs` owner 是与部署对齐的 off-policy 观测配置：pelvis IMU 状态
（`pelvis_local_linvel` / `pelvis_gyro` / `pelvis_upvector`）加上 per-term 历史观测
（`noise_config.obs_history_length: 5`），与部署侧的 `ObservationManager` 按字节对齐。
部署工具在 `scripts/deploy/`，观测对齐由 `tests/scripts/test_obs_alignment_g1_wbt.py`
交叉校验。当 Motrix sim2sim 回放需要引用其他日志根目录下的 checkpoint 时，用
`uv run eval` 透传绝对路径：

```bash
uv run eval --algo sac --task g1_motion_tracking --sim motrix \
  algo.load_run=/abs/path/to/logs/fast_sac/G1MotionTrackingSAC/2026-04-23_14-06-57_mujoco
```

## 动作文件

动作 NPZ 文件通过 `env.motion_file` 读取，也支持路径列表。标准片段必须包含七个 key：
`fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、
`body_ang_vel_w`（在 `g1/motion_loader.py` 中校验）：

```yaml
env:
  motion_file:
    - src/unilab/assets/motions/g1/dance1_subject2_part.npz
    - src/unilab/assets/motions/g1/walk1_subject5_from_csv.npz
```

转换与检查辅助工具在 `scripts/motion/` 中：

```bash
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_from_csv.npz \
  --input_fps 30 --output_fps 50
uv run scripts/motion/csv_to_npz.py \
  --input_file src/unilab/assets/motions/g1/dance1_subject2.csv \
  --output_file src/unilab/assets/motions/g1/dance1_subject2_clip.npz \
  --input_fps 30 --output_fps 50 --start_time 4.0 --end_time 9.0
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --loop
uv run scripts/motion/replay_npz.py \
  --npz_file src/unilab/assets/motions/g1/dance1_subject2_part.npz --speed 0.5
```

如果 MuJoCo replay 里 body 姿态明显错位，优先检查：NPZ 是否包含标准 7 个 key、
`fps` 是否匹配控制频率、body layout 是否需要 remap、joint 顺序是否匹配当前 G1 模型。
更详细的动作转换说明见 `scripts/motion/README.md`。

## SAC WBT on crawl-slope 场景

在斜坡地形上跑 `g1_motion_tracking`，需要同时切换 motion 片段和 MuJoCo 场景文件，
并固定 episode 长度、关闭 reset 随机化，以复用 clip 精确初始状态：

```bash
CUDA_VISIBLE_DEVICES=1 uv run train --algo sac --task g1_motion_tracking --sim mujoco \
  training.use_amp=true algo.seed=1 \
  +env.motion_file=src/unilab/assets/motions/g1/motion_crawl_slope_uni.npz \
  +env.scene.model_file=src/unilab/assets/robots/g1/scene_crawl_slope.xml \
  +env.sampling_mode=start \
  env.truncate_on_clip_end=true \
  +env.max_episode_seconds=20.0 \
  '+env.pose_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.velocity_randomization={x:[0,0],y:[0,0],z:[0,0],roll:[0,0],pitch:[0,0],yaw:[0,0]}' \
  '+env.joint_position_range=[0,0]'
```

关键覆写：`env.motion_file` 切爬坡动作；`env.scene.model_file` 切斜坡场景
（`scene_crawl_slope.xml` 在 `src/unilab/assets/robots/g1/` 下）；`sampling_mode=start`
加 `truncate_on_clip_end=true` 从 clip 起点出发并在结尾截断；randomization 范围全置零
复用 motion 精确初始状态。

## 交互式调试

常规 checkpoint 回放用 `uv run eval`。需要 target body 或 reward debug overlay 时，
`scripts/play_interactive.py` 是 MuJoCo 专用的低层调试入口，当前没有暴露为统一
`uv run eval` 参数。

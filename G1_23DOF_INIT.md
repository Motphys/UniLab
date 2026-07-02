# G1 23-DoF 支持初始化文档

## 背景

Unitree G1 人形机器人有两个版本：
- **29-DoF（官方标准版）**：包含 6 个腿部关节 × 2 + 3 个腰部关节 + 7 个手臂关节 × 2 = 29 自由度
- **23-DoF（精简版）**：去掉双手腕部共 6 个关节（各 3 个），保留 23 个自由度

本分支从上游仓库 `UniLab/`（commit `b4924d44`）移植了完整的 23-DoF 支持，包括配置、资产、motion 数据和 Python 环境代码。

---

## 文件清单

### 官方原始配置（29-DoF，请勿修改）

以下为仓库原有文件，属于 29-DoF 调优版本，**不应修改**：

| 路径 | 说明 |
|------|------|
| `conf/appo/task/g1_*` | APPO 算法 G1 任务配置（不含 `23dof`） |
| `conf/ppo/task/g1_*` | PPO 算法 G1 任务配置（不含 `23dof`） |
| `conf/offpolicy/task/*/g1_*` | Offpolicy 算法 G1 任务配置（不含 `23dof`） |
| `src/unilab/assets/robots/g1/g1.xml` | 29-DoF 机器人模型 |
| `src/unilab/assets/robots/g1/g1_sphere_hand.xml` | 29-DoF 球形手模型 |
| `src/unilab/assets/robots/g1/scene_*.xml`（不含 `23dof`） | 29-DoF 场景文件 |
| `src/unilab/envs/locomotion/g1/joystick.py` | 原始 29-DoF 行走环境类 |
| `src/unilab/envs/motion_tracking/g1/*.py` | 原始 29-DoF 动作追踪环境类 |

### 新增 23-DoF 配置（从父目录复制，测试中）

以下文件从 `UniLab/` 复制，**处于测试阶段，训练参数已按 29-DoF 调优版对齐**：

| 路径 | 说明 |
|------|------|
| `conf/appo/task/g1_23dof_*/` | APPO 23-DoF 任务配置（5 个任务 × 后端） |
| `conf/ppo/task/g1_23dof_*/` | PPO 23-DoF 任务配置（7 个任务 × 后端） |
| `conf/offpolicy/task/sac/g1_23dof_*/` | SAC 23-DoF 任务配置（6 个任务 × 后端） |
| `conf/offpolicy/task/flashsac/g1_23dof_walk_flat/` | FlashSAC 23-DoF walk flat |
| `conf/offpolicy/task/td3/g1_23dof_walk_flat/` | TD3 23-DoF walk flat |

### 新增 23-DoF 资产

| 文件 | 说明 |
|------|------|
| `src/unilab/assets/robots/g1/g1_23dof.xml` | 23-DoF 机器人模型（无手腕关节） |
| `src/unilab/assets/robots/g1/g1_23dof_sphere_hand.xml` | 23-DoF 球形手模型 |
| `src/unilab/assets/robots/g1/scene_flat_23dof.xml` | 23-DoF 平地场景 |
| `src/unilab/assets/robots/g1/scene_rough_23dof.xml` | 23-DoF 崎岖地形场景 |
| `src/unilab/assets/robots/g1/scene_flat_23dof_with_largebox.xml` | 23-DoF 大箱子场景 |
| `src/unilab/assets/robots/g1/scene_flat_23dof_with_wall.xml` | 23-DoF 带墙场景 |
| `src/unilab/assets/robots/g1/scene_climb_20_z_scale_1_23dof.xml` | 23-DoF 攀爬场景 |
| `src/unilab/assets/robots/g1/assets/torso_link_23dof_rev_1_0.STL` | 23-DoF 躯干 STL |

### 新增 23-DoF Motion 数据

| 文件 | 说明 |
|------|------|
| `src/unilab/assets/motions/g1/dance1_subject2_part_23dof.npz` | 舞蹈 motion（23-DoF） |
| `src/unilab/assets/motions/g1/sub3_largebox_003_boxconverted_23dof.npz` | 搬箱子 motion（23-DoF） |
| `src/unilab/assets/motions/g1/flip_360_001__A304_23dof.npz` | 前空翻 motion（23-DoF） |
| `src/unilab/assets/motions/g1/flip_from_wall_104__A304_23dof.npz` | 墙壁空翻 motion（23-DoF） |
| `src/unilab/assets/motions/g1/climb_20_z_scale_1.0_23dof.npz` | 攀爬 motion（23-DoF） |
| `src/unilab/assets/motions/g1/23dof_amp/` | AMP 风格 23-DoF motion 数据 |

### 新增 23-DoF Python 环境代码（与 29-DoF 共存于同一文件）

以下文件中新增了 `class *23Dof*(...)` 类、config 类和 registry 注册。与 29-DoF 类共存，互不干扰：

| 文件 | 新增 23-DoF 内容 |
|------|-----------------|
| `src/unilab/envs/locomotion/g1/joystick.py` | `G1Walk23DofEnv`, `G1Walk23DofFlatCfg`, `G1Walk23DofRoughCfg` |
| `src/unilab/envs/motion_tracking/g1/tracking.py` | `G1MotionTracking23DofCfg`, `G1MotionTracking23DofEnvCfg`, `G1MotionTracking23DofDeployEnvCfg` |
| `src/unilab/envs/motion_tracking/g1/flip_tracking.py` | `G1FlipTracking23DofCfg/EnvCfg`, `G1WallFlipTracking23DofCfg/EnvCfg`, `G1ClimbTracking23DofCfg/EnvCfg` |
| `src/unilab/envs/motion_tracking/g1/box_tracking.py` | `G1BoxTracking23DofCfg/EnvCfg` |
| `src/unilab/envs/motion_tracking/g1/tracking_obs.py` | `G1WBTObs23DofCfg` |
| `src/unilab/envs/motion_tracking/g1/tracking_sac.py` | `G1MotionTrackingSAC23DofCfg/Env` |
| `src/unilab/envs/motion_tracking/g1/flip_tracking_sac.py` | `G1FlipTrackingSAC23DofCfg/Env`, `G1WallFlipTrackingSAC23DofCfg/Env` |

---

## 训练参数对齐

23-DoF 配置复制后，逐项比对了与 29-DoF 对应配置的训练参数差异。

### 对齐原则
- **DoF-specific 差异**（`task_name` 含 `23Dof`、`action_scale` 数组长度 23、模型文件引用 `*_23dof.xml`）：**保留**
- **训练参数差异**（`num_learning_epochs`、`entropy_coef`、`max_iterations`、`reward.scales` 等）：**已对齐至 29-DoF 调优值**

### 主要对齐项

| 参数 | 复制时旧值 | 对齐后值 |
|------|:---------:|:--------:|
| `algo.max_iterations`（部分） | 5000 | 3500（匹配 29-DoF） |
| `algo.algorithm.num_learning_epochs`（部分motrix） | 5 | 10 |
| `algo.algorithm.num_mini_batches`（部分motrix） | 4 | 8 |
| `algo.algorithm.entropy_coef`（部分motrix） | 0.01 | 0.005 |
| `env.*` block（部分motrix） | 缺失 | 补齐至与 29-DoF 一致 |
| `reward.motion_body_pos`（部分motrix） | 1.0 | 2.0 |
| `reward.motion_body_ori`（部分motrix） | 1.0 | 1.5 |
| `reward.action_rate_l2`（部分motrix） | -0.1 | -0.005 |

---

## 使用方式

```bash
# APPO 训练
uv run train --algo appo --task g1_23dof_flip_tracking --sim motrix
uv run train --algo appo --task g1_23dof_flip_tracking --sim mujoco

# PPO 训练
uv run train --algo ppo --task g1_23dof_walk_flat --sim mujoco
uv run train --algo ppo --task g1_23dof_flip_tracking --sim motrix

# SAC 训练
uv run train --algo sac --task g1_23dof_walk_flat --sim mujoco
```

## 验证状态

- [ ] 23-DoF APPO flip tracking 训练曲线与 29-DoF 基线对比
- [ ] 23-DoF PPO walk flat 训练收敛
- [ ] 23-DoF SAC walk flat 训练收敛
- [ ] Sim2Sim 跨后端 play 验证
- [ ] 所有非 slow 测试通过 (`uv run pytest -m "not slow"`)

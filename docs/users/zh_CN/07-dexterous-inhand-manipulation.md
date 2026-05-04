# Dexterous In-Hand Manipulation 训练

语言: 简体中文

本页只说明如何运行当前仓库已有的 dexterous inhand manipulation 流程。后端选择必须通过 `task=<task>/<backend>` 完成，不要单独 override `training.sim_backend` 来切后端。

## Allegro Inhand Rotation

Allegro 的环境注册名是 `AllegroInhandRotation`，常规训练 task owner 是 `allegro_inhand`。完整流程是先生成 grasp cache，再训练 rotation policy。

`allegro_inhand` 是一个 in-hand manipulation 的最小训练示例。策略观测包含 privileged information，默认不启用 domain randomization。

### 配置文件

- `scripts/train_rsl_rl.py` 主配置：`conf/ppo/config.yaml`
- `task=allegro_inhand/mujoco`：`conf/ppo/task/allegro_inhand/mujoco.yaml`
- `task=allegro_inhand/motrix`：`conf/ppo/task/allegro_inhand/motrix.yaml`
- `task=allegro_inhand_grasp/mujoco`：`conf/ppo/task/allegro_inhand_grasp/mujoco.yaml`，并继承 `conf/ppo/task/allegro_inhand/mujoco.yaml`
- `task=allegro_inhand_grasp/motrix`：`conf/ppo/task/allegro_inhand_grasp/motrix.yaml`，并继承 `conf/ppo/task/allegro_inhand/motrix.yaml`
- `scripts/train_appo.py` 主配置：`conf/appo/config.yaml`
- `task=allegro_inhand/mujoco`：`conf/appo/task/allegro_inhand/mujoco.yaml`
- `task=allegro_inhand/motrix`：`conf/appo/task/allegro_inhand/motrix.yaml`

### 1. 生成 Grasp Cache

grasp cache 生成任务是 `allegro_inhand_grasp`：

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand_grasp/mujoco training.no_play=true
```

Motrix owner 也存在：

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand_grasp/motrix training.no_play=true
```

默认 rotation 配置读取：

```text
cache/allegro_grasp_50k.npy
```

如果使用自定义 cache，在训练时指定：

```bash
uv run scripts/train_rsl_rl.py \
  task=allegro_inhand/mujoco \
  env.grasp_cache_path=cache/my_allegro_grasp.npy
```

### 2. 训练 Policy

PPO:

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco
uv run scripts/train_rsl_rl.py task=allegro_inhand/motrix
```

APPO:

```bash
uv run scripts/train_appo.py task=allegro_inhand/mujoco
uv run scripts/train_appo.py task=allegro_inhand/motrix
```

### 3. 回放

MuJoCo 会导出 `play_video.mp4`：

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/mujoco training.play_only=true
uv run scripts/train_appo.py task=allegro_inhand/mujoco training.play_only=true
```

Motrix 当前使用原生交互式 renderer：

```bash
uv run scripts/train_rsl_rl.py task=allegro_inhand/motrix training.play_only=true
uv run scripts/train_appo.py task=allegro_inhand/motrix training.play_only=true
```

macOS / MacBook 上如果会打开 MotrixSim 原生 renderer，使用 `mxpython`：

```bash
uv run mxpython scripts/train_rsl_rl.py task=allegro_inhand/motrix training.play_only=true
```

## Sharpa Inhand Rotation

Sharpa 的环境注册名是 `SharpaInhandRotation`，常规训练 task owner 是 `sharpa_inhand`。当前训练 pipeline 是生成 grasp cache、训练 teacher policy、再训练 student policy。

`sharpa_inhand` 是一个完整的 [HORA](https://github.com/HaozhiQi/hora) 风格训练示例，训练流程包含完整的 domain randomization。

`sharpa_inhand` 当前只支持 MuJoCo backend，不支持 Motrix，因为 Motrix 目前还不支持完整的 domain randomization。

### 配置文件

- `scripts/train_rsl_rl.py` 主配置：`conf/ppo/config.yaml`
- `task=sharpa_inhand_grasp/mujoco`：`conf/ppo/task/sharpa_inhand_grasp/mujoco.yaml`
- `task=sharpa_inhand/mujoco_hora`：`conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`，并继承 `conf/ppo/task/sharpa_inhand/mujoco.yaml`
- `scripts/train_appo.py` 主配置：`conf/appo/config.yaml`
- `task=sharpa_inhand/mujoco_hora`：`conf/appo/task/sharpa_inhand/mujoco_hora.yaml`，并继承 `conf/appo/task/sharpa_inhand/mujoco.yaml`
- `scripts/train_hora_distill.py` 主配置：`conf/hora_distill/config.yaml`
- student `task=sharpa_inhand/mujoco`：`conf/hora_distill/task/sharpa_inhand/mujoco.yaml`
- student 默认 `teacher.algo_family=ppo`，`teacher.task=sharpa_inhand/mujoco_hora` 指向 `conf/ppo/task/sharpa_inhand/mujoco_hora.yaml`
- 如果蒸馏 APPO teacher，保持 `teacher.task=sharpa_inhand/mujoco_hora`，并将 `teacher.algo_family=appo`，此时会解析到 `conf/appo/task/sharpa_inhand/mujoco_hora.yaml`

### 1. 生成 Grasp Cache

grasp cache 生成任务是 `sharpa_inhand_grasp`. 按 object scale 分别采集：

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[0.8]' training.no_play=true
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[1.0]' training.no_play=true
uv run scripts/train_rsl_rl.py task=sharpa_inhand_grasp/mujoco 'env.domain_rand.scale_list=[1.2]' training.no_play=true
```

也可以用批量脚本一次生成多组 scale 的 grasp cache：

```bash
bash scripts/sharpa_collect_grasps.sh 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6
```

常规 Sharpa rotation 配置读取：

```text
cache/sharpa_grasp_linspace
```

如果使用自定义 cache，在后续训练命令中指定：

```bash
uv run scripts/train_rsl_rl.py \
  task=sharpa_inhand/mujoco \
  env.grasp_cache_path=cache/my_sharpa_grasp_cache
```

### 2. 训练 Teacher Policy

PPO teacher:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco_hora
```

APPO teacher:

```bash
uv run scripts/train_appo.py task=sharpa_inhand/mujoco_hora
```

### 3. 训练 Student Policy

从 PPO teacher 蒸馏 student：

```bash
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco
```

从 APPO teacher 蒸馏 student：

```bash
uv run scripts/train_hora_distill.py \
  task=sharpa_inhand/mujoco \
  teacher.algo_family=appo \
  teacher.task=sharpa_inhand/mujoco_hora
```

指定 teacher run：

```bash
uv run scripts/train_hora_distill.py \
  task=sharpa_inhand/mujoco \
  algo.load_run="2026-04-28_12-00-00_mujoco"
```

指定 APPO teacher run：

```bash
uv run scripts/train_hora_distill.py \
  task=sharpa_inhand/mujoco \
  teacher.algo_family=appo \
  teacher.task=sharpa_inhand/mujoco_hora \
  algo.load_run="2026-04-28_12-00-00_mujoco"
```

### 4. 回放

回放 teacher:

```bash
uv run scripts/train_rsl_rl.py task=sharpa_inhand/mujoco_hora training.play_only=true
uv run scripts/train_appo.py task=sharpa_inhand/mujoco_hora training.play_only=true
```

回放 student:

```bash
uv run scripts/train_hora_distill.py task=sharpa_inhand/mujoco training.play_only=true
```

## 常用命令

日志目录按 `algo.algo_log_name` 和环境名分组，常见路径包括：

- `logs/rsl_rl_ppo/AllegroInhandRotation/<run>/`
- `logs/appo/AllegroInhandRotation/<run>/`
- `logs/hora_ppo/SharpaInhandRotation/<run>/`
- `logs/hora_appo/SharpaInhandRotation/<run>/`
- `logs/hora_distill/SharpaInhandRotation/<run>/`

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Domain Randomization](06-domain-randomization.md)
- Next: [Simulation Backends](02-simulation-backends.md)

# Hydra 覆盖规则

语言: 简体中文

UniLab 的统一 CLI 和底层训练脚本都由 Hydra 驱动。

## 常见覆盖

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.max_iterations=10
uv run train --algo ppo --task go2_joystick_flat --sim mujoco algo.num_envs=1024
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.no_play=true
```

## backend 选择规则

- 统一 CLI：`--sim <backend>`
- 底层脚本：`task=<task>/<backend>`
- offpolicy：`task=<algo>/<task>/<backend>`

## 常见 config group

```text
task=go1_joystick_flat/mujoco
algo=sac
```

## 常见训练字段

```text
training.play_only=true
training.no_play=true
training.play_render_mode=record
algo.load_run=-1
training.logger=wandb
algo.max_iterations=1000
algo.num_envs=2048
```

## 不要这样做

不要单独 override：

```text
training.sim_backend=...
```

这不是独立 backend switch。

## 何时切到底层脚本

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco algo.max_iterations=10
```

查看完整 compose 结果：

```bash
uv run scripts/train_offpolicy.py --cfg job
```

## Navigation

- Index: [Documentation](../../../index.md)
- Previous: [评估、回放与恢复训练](02-playback-and-resume.md)
- Next: [日志、run 目录与 W&B](04-logging-and-wandb.md)

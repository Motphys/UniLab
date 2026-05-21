# APPO

语言: 简体中文

APPO 是 UniLab 的异步 PPO 路径，collector 和 learner 并行运行，适合吞吐优先场景。

## 默认入口

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
```

## 适用场景

- 需要异步 collector / learner
- 接受更复杂的运行链路

## 运行模型

- collector 负责 CPU 仿真，learner 负责 GPU 训练
- rollout 会先进入 replay queue，再由 learner 消费
- APPO 内部带 V-trace importance-sampling 修正，语义不同于同步 PPO
- 当前实现使用 4 槽 ring buffer 做 collector / learner 流水线

## 常见命令

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300

uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  training.replay_queue_size=2 \
  training.no_play=true
```

## 关联脚本

```bash
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco
```

## 关键字段

- `algo.steps_per_env`：单次 rollout 长度
- `training.replay_queue_size`：learner 侧缓存深度
- `training.collector_device`：collector 设备；默认跟随 learner
- `algo.save_interval`：checkpoint 保存间隔

回放：

```bash
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco training.play_only=true
uv run scripts/train_appo.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run="2026-03-16_01-35-12_mujoco"
```

## 和 PPO 的差异

- PPO：同步采集，路径更直观
- APPO：异步采集，CPU / GPU 更容易同时满载
- APPO：collector rollout 会先进入 replay queue，再由 learner 消费

## 日志根目录

```text
logs/appo/<task>/
```

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [MLX PPO](02-mlx-ppo.md)
- Next: [SAC](04-sac.md)

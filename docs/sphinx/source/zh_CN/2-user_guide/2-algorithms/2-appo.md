# APPO

语言: 简体中文

APPO 是 UniLab 的异步 PPO 路径。它使用 `scripts/train_appo.py`、
`conf/appo/config.yaml` 以及 `src/unilab/algos/torch/appo/` 下的运行时。该配置暴露
了 `algo.steps_per_env`、`training.collector_device` 和
`training.replay_queue_size`；算法配置中包含 V-trace 裁剪字段。

## 快速开始

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task g1_motion_tracking --sim motrix training.no_play=true
```

## 常用 Override

```bash
uv run train --algo appo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2048 \
  algo.max_iterations=300 \
  training.replay_queue_size=2
```

回放与检查点选择使用 `uv run eval`：

```bash
uv run eval --algo appo --task go2_joystick_flat --sim mujoco --load-run -1
```

默认日志族为 `2-appo`，来自 `conf/appo/config.yaml` 中的 `algo.algo_log_name`。

## Navigation

- Index: [文档](0-index.md)

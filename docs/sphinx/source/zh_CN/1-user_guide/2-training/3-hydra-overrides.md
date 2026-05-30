# Hydra 覆盖规则

语言: 简体中文

UniLab 的统一 CLI 和底层训练脚本都由 Hydra 驱动。

## 常见覆盖

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco algo.max_iterations=10
uv run train --algo ppo --task go2_joystick_flat --sim mujoco algo.num_envs=1024
uv run train --algo ppo --task go2_joystick_flat --sim mujoco training.no_play=true
```

## backend 选择规则

- 统一 CLI：`--sim <backend>`
- 算法选择：`--algo <algo>`
- 任务选择：`--task <task>`
- 普通超参 override 追加在命令末尾，route-defining 字段由统一 CLI 生成

## 不要把路由字段当 override

用户命令里不要透传 `task`、`algo` 或 `training.play_only` 这类路由字段；训练、评估和回放分别用 `train` / `eval` / `demo` 入口表达。

## 常见训练字段

```text
training.no_play=true
algo.load_run=-1
training.logger=wandb
algo.max_iterations=1000
algo.num_envs=2048
```

回放渲染模式优先通过 `uv run eval ... --render-mode record` 表达。

## 不要这样做

不要单独 override：

```text
training.sim_backend=...
```

这不是独立 backend switch。

## 查看完整 compose 结果

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

调试脚本级 Hydra compose 时再直接进入 `scripts/train_*.py`，用户训练和回放文档不把脚本级路由写成命令示例。

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [评估、回放与恢复训练](2-playback-and-resume.md)
- Next: [日志、run 目录与 W&B](4-logging-and-wandb.md)

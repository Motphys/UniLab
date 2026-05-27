# 训练指南

语言: 简体中文

本页汇总训练入口和训练相关主题。统一 CLI 是默认训练入口。

## 默认入口

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run demo
```

## 训练专题

- [B.01 统一 CLI](B-training/01-unified-cli.md)
- [B.02 评估、回放与恢复训练](B-training/02-playback-and-resume.md)
- [B.03 Hydra 覆盖规则](B-training/03-hydra-overrides.md)
- [B.04 日志、run 目录与 W&B](B-training/04-logging-and-wandb.md)
- [B.05 训练相关 Docker 用法](B-training/05-docker.md)

## 什么时候看底层脚本

只有在下面几种场景才建议直接进入 `scripts/train_*.py`：

- 需要调试特定训练栈
- 需要直接观察 Hydra compose 行为
- 需要对照脚本级日志目录或 adapter 行为

常规用户先用统一 CLI，不先记脚本路径。

## 关联入口

- 想选算法：看 [算法说明](04-algorithms.md)
- 想找任务命令：看 [任务索引](D-tasks/01-task-index.md)
- 想查 backend 行为差异：看 [仿真后端](02-simulation-backends.md)
- 想查精确支持状态：看 [后端支持矩阵](E-reference/01-backend-support-matrix.md)

## Navigation

- Index: [Documentation](../index.md)
- Previous: [Simulation Backends](02-simulation-backends.md)
- Next: [Unified CLI](B-training/01-unified-cli.md)

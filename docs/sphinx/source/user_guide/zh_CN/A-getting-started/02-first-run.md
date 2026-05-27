# 首次运行

语言: 简体中文

本页给出第一次验证环境最需要的命令。默认使用统一 CLI。

## 推荐第一跑

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

如果你更想先走默认、最完整的路径，也可以直接用 MuJoCo：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

## 评估和 demo

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
uv run demo
```

## 第一次最常见的运行差异

- macOS / MacBook 上，Motrix 交互回放会自动路由到 `mxpython`。
- Linux / server 上如果不想开窗口，优先用 `--render-mode record`。
- 如果训练阶段根本不想自动回放，直接加 `training.no_play=true`。

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.no_play=true
```

## 基本验证

需要做仓库级验证时：

```bash
make test-all
```

## 想直接打底层脚本时

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco
```

## 下一步

- 想系统了解训练命令：看 [训练指南](../03-training.md)
- 想查后端差异：看 [仿真后端](../02-simulation-backends.md)
- 想直达任务：看 [任务索引](../D-tasks/01-task-index.md)

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [安装与环境](01-install.md)
- Next: [Docker](03-docker.md)

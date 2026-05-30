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

# 预训练 checkpoint 回放（首次运行会从 Hugging Face 自动下载）
uv run demo dance
```

可用 demo 名：`teaser`、`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。

> 面向大陆用户：如果 `huggingface.co` 不可达，可在执行上面这些命令前切换镜像：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

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

## 想验证不同训练组合时

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo appo --task go1_joystick_flat --sim mujoco
```

## 下一步

- 想系统了解训练命令：看 [训练指南](../3-training.md)
- 想查后端差异：看 [仿真后端](../2-simulation-backends.md)
- 想直达任务：看 [任务索引](../4-tasks/1-task-index.md)

## Navigation

- Index: [Documentation](../../0-index.md)
- Previous: [安装与环境](1-install.md)
- Next: [Docker](3-docker.md)

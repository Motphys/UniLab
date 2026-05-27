# 训练相关 Docker 用法

语言: 简体中文

用户侧 Docker 基本规则与 [A.03 Docker](../A-getting-started/03-docker.md) 一致，这里只补训练命令视角。

## 容器内训练

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
```

## 容器内回放或评估

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

## 适用范围

- 根目录 `Dockerfile`：Linux NVIDIA / CUDA
- ROCm：使用 AMD 官方镜像，进入容器后先运行 `make sync-rocm`，命令走 `uv run ...`
- macOS Docker：不作为主路径
- 训练产物仍写回挂载后的仓库 `logs/` 目录
- 如果你要在容器里直接调脚本入口，命令形式与非容器环境完全一致

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [日志、run 目录与 W&B](04-logging-and-wandb.md)

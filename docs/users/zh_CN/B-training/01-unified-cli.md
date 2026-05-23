# 统一 CLI

语言: 简体中文

`uv run train`、`uv run eval` 和 `uv run demo` 是 UniLab 的默认训练接口。

## train

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go2_joystick_flat --sim mujoco
uv run train --algo sac --task g1_walk_flat --sim mujoco
uv run train --algo td3 --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo mlx_ppo --task go2_joystick_flat --sim mujoco
```

支持的算法：`ppo`、`mlx_ppo`、`appo`、`sac`、`td3`、`flashsac`

支持的后端：`mujoco`、`motrix`

## eval

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

## demo

```bash
uv run demo
uv run demo --preset go2_joystick_mujoco_ppo
uv run demo --refresh
uv run demo --device cpu
```

`demo` 会从本地预设 run 复制 checkpoint，再走回放命令；不是远端模型下载器。

## 规则

- `--algo`、`--task`、`--sim` 保持显式
- backend 选择走 `--sim`，不要单独切 `training.sim_backend`
- ROCm 环境在 `make sync-rocm` 后使用 `uv run ...`；Intel XPU 环境使用 `uv run --no-sync ...`
- `eval --load-run` 只接受 `-1` 或 run 目录名，不接受绝对路径

## render mode

- `auto`：MuJoCo 默认导出视频；Motrix 默认打开窗口
- `record`：只录视频
- `none`：不回放
- `interactive`：主要用于 Motrix 交互式回放

## Navigation

- Index: [Documentation](../../../README.md)
- Previous: [训练指南](../03-training.md)
- Next: [评估、回放与恢复训练](02-playback-and-resume.md)

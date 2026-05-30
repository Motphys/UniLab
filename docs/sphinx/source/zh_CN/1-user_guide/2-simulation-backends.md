# 仿真后端

语言: 简体中文

本页概览仿真后端的选择规则、适用场景和阅读路径。

## 当前后端

- **MuJoCo**：默认后端，能力最完整
- **Motrix**：可选后端，覆盖范围按 task owner、算法和测试证据逐步补齐

## 用户先记住的规则

1. 统一 CLI 里通过 `--sim mujoco|motrix` 选后端。
2. 训练和回放命令都保持 `--algo`、`--task`、`--sim` 显式。
3. 不要单独 override `training.sim_backend` 来切后端。

## 运行前提

- `uv sync --extra motrix` 会安装 Motrix 依赖。
- 任何 `--sim mujoco` 路径都仍然要求 MuJoCo runtime 可用。
- 某些 task / algo 组合只在矩阵中达到 `Registered` 或 `Configured`，不等于默认推荐路径。

## 什么时候选 MuJoCo

- 想走默认、最完整的训练路径
- 任务只提供 MuJoCo owner
- 需要 MuJoCo-only 调试工具或播放路径

## 什么时候选 Motrix

- 任务已经有 Motrix owner
- 需要 Motrix 的交互式 renderer
- 接受某些 task / algo 组合可能只到 `Registered` 或 `Configured` 级别

## playback 差异

- `mujoco`：`--render-mode auto` 默认导出 `play_video.mp4`
- `motrix`：`--render-mode auto` 默认打开交互式窗口，不受 `play_steps` 限制
- `--render-mode record`：两个后端都只录制视频
- `--render-mode none`：两个后端都不回放

macOS / MacBook 上，统一 CLI 会在 Motrix 交互回放时自动路由到 `mxpython`。

## owner config 规则

- 训练 owner YAML 是 backend 选择的真正入口，统一 CLI 会按 `--algo`、`--task`、`--sim` 路由到对应 owner。
- 用户命令不要手写 route-defining Hydra override；普通超参 override 仍然可以追加在命令末尾。
- `training.sim_backend` 只是 owner YAML 的身份字段，不是独立切换开关。

## 去哪里看细节

- [后端支持矩阵](5-reference/1-backend-support-matrix.md)
- [训练指南](3-training.md)
- [任务索引](4-tasks/1-task-index.md)

## Navigation

- Index: [Documentation](../0-index.md)
- Previous: [快速开始](1-getting-started.md)
- Next: [Backend Support Matrix](5-reference/1-backend-support-matrix.md)

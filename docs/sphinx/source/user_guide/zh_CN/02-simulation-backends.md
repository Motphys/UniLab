# 仿真后端

语言: 简体中文

本页概览仿真后端的选择规则、适用场景和阅读路径。

## 当前后端

- **MuJoCo**：默认后端，能力最完整
- **Motrix**：可选后端，覆盖范围按 task owner、算法和测试证据逐步补齐

## 用户先记住的规则

1. 统一 CLI 里通过 `--sim mujoco|motrix` 选后端。
2. 底层脚本里通过 `task=<task>/<backend>` 选后端。
3. 不要单独 override `training.sim_backend` 来切后端。

## 运行前提

- `uv sync --extra motrix` 会安装 Motrix 依赖。
- 任何 `task=.../mujoco` 路径都仍然要求 MuJoCo runtime 可用。
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

- `mujoco`：`training.play_render_mode=auto` 默认导出 `play_video.mp4`
- `motrix`：`training.play_render_mode=auto` 默认打开交互式窗口，不受 `play_steps` 限制
- `record`：两个后端都只录制视频
- `none`：两个后端都不回放

macOS / MacBook 上，统一 CLI 会在 Motrix 交互回放时自动路由到 `mxpython`。

## owner config 规则

- 训练 owner YAML 是 backend 选择的真正入口。
- PPO / APPO 通常用 `task=<task>/<backend>`。
- off-policy 通常用 `task=<algo>/<task>/<backend>`。
- `training.sim_backend` 只是 owner YAML 的身份字段，不是独立切换开关。

## 去哪里看细节

- [后端支持矩阵](E-reference/01-backend-support-matrix.md)
- [训练指南](03-training.md)
- [任务索引](D-tasks/01-task-index.md)

## Navigation

- Index: [Documentation](../../index.md)
- Previous: [Getting Started](01-getting-started.md)
- Next: [Backend Support Matrix](E-reference/01-backend-support-matrix.md)

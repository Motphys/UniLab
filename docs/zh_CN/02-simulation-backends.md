# 仿真后端

语言: 简体中文

UniLab 当前支持两个仿真后端:

- **MuJoCo**: 默认后端，能力最完整
- **Motrix**: 可选后端，任务和算法支持仍在持续补齐

## Runtime Prerequisites

- `uv sync --extra motrix` 会安装 Motrix 依赖。
- Motrix 路径的 registry bootstrap 和 Hydra 配置 compose 不再要求导入 MuJoCo。
- 任何 `task=.../mujoco` 的实际运行、MuJoCo playback、以及 MuJoCo-only 调试工具，仍然要求可用的 MuJoCo runtime。
- 某些任务目前仍然只有 MuJoCo owner 配置；例如 `AllegroInhandRotation` 只提供 `mujoco` task。

## Support Matrix

### MuJoCo

| 算法 | Go1 | Go2 | G1 |
|------|-----|-----|----|
| PPO (torch) | ✅ |  | ✅ |
| PPO (mlx) | ✅ |  | ✅ |
| SAC (torch) | ✅ | ⚠️ | ✅ |
| TD3 (torch) | ⚠️ | ⚠️ | ⚠️ |
| APPO (torch) | ✅ | ✅ | ✅ |

### Motrix

| 算法 | Go1 | Go2 | G1 |
|------|-----|-----|----|
| PPO (torch) | ⚠️ |  | ✅ |
| PPO (mlx) | ⚠️ |  |  |
| SAC (torch) | ⚠️ |  |  |
| TD3 (torch) |  |  |  |
| APPO (torch) |  |  | ✅ |

图例:

- `✅` 已支持
- `⚠️` 开发中

## Select A Backend

默认后端是 `mujoco`。通过 `task=<task>/<backend>` 切换到 `motrix`，不要用 `training.sim_backend=motrix` 单独切换后端。

实际要改参数时，不再去拆着找 `reward` / `backend preset` / `algo preset`。直接改对应的 `task` 文件：

- PPO / APPO: `conf/{ppo,appo}/task/<task>/<backend>.yaml`
- offpolicy: `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`

现在没有单独的 `reward/`、`backend preset`、`sim_backend/` 配置组。`task/` 是唯一 owner 入口，不再是旧的拆分式 task 配置。
`training.sim_backend` 由 owner YAML 设置，只用于标识最终选择的后端；不要把它当作独立 backend switch。

```bash
# 默认 MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick/mujoco

# 显式指定 Motrix
uv run python scripts/train_rsl_rl.py task=go1_joystick/motrix
```

## Playback Differences

- `mujoco`: 训练后的自动回放会导出 `play_video.mp4`
- `motrix`: 回放通常打开交互式 renderer 窗口，而不是导出视频

对 G1 motion tracking 来说，目前已验证的 Motrix 路径是 `PPO (torch) + motrix` 和 `APPO (torch) + motrix`。`scripts/play_interactive.py` 仍然沿用 MuJoCo 路径。

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick/mujoco training.play_only=true
```

## Notes

- backend 支持范围是阶段性的能力快照，不要把临时执行状态写成顶层 README 结论
- 具体推进应通过 GitHub milestone 和 issue 跟踪，而不是维护仓库内的临时状态列表

## Navigation

- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)

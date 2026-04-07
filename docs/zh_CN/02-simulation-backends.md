# 仿真后端

语言: [English](../en/02-simulation-backends.md) | 简体中文 | [日本語](../ja/02-simulation-backends.md) | [한국어](../ko/02-simulation-backends.md)

UniLab 当前支持两个仿真后端:

- **MuJoCo**: 默认后端，能力最完整
- **Motrix**: 可选后端，任务和算法支持仍在持续补齐

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

默认后端是 `mujoco`。通过 Hydra 参数 `training.sim_backend` 切换到 `motrix`。

```bash
# 默认 MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick

# 显式指定 Motrix
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix
```

## Playback Differences

- `mujoco`: 训练后的自动回放会导出 `play_video.mp4`
- `motrix`: 回放通常打开交互式 renderer 窗口，而不是导出视频

对 G1 motion tracking 来说，目前已验证的 Motrix 路径是 `PPO (torch) + motrix` 和 `APPO (torch) + motrix`。`scripts/play_interactive.py` 仍然沿用 MuJoCo 路径。

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## Notes

- backend 支持范围是阶段性的能力快照，不要把临时执行状态写成顶层 README 结论
- 具体推进应通过 GitHub milestone 和 issue 跟踪，而不是维护仓库内的临时状态列表

## Navigation

- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)

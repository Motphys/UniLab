# Simulation Backends

UniLab 当前支持两个仿真后端：

- **MuJoCo**：默认后端，功能最完整
- **Motrix**：可选后端，仍在持续补齐任务和算法支持

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
| PPO (torch) | ⚠️ |  |  |
| PPO (mlx) | ⚠️ |  |  |
| SAC (torch) | ⚠️ |  |  |
| TD3 (torch) |  |  |  |
| APPO (torch) |  |  |  |

说明：

- `✅` 已支持
- `⚠️` 开发中

## Select Backend

默认使用 `mujoco`。通过 Hydra 参数 `training.sim_backend` 切换到 `motrix`。

```bash
# 默认 MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick

# 指定 Motrix
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix
```

## Playback Differences

- `mujoco`：训练后自动回放时会生成 `play_video.mp4`
- `motrix`：回放通常打开交互式窗口，不导出视频

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## Notes

- 后端支持范围属于阶段性能力，不建议把执行状态写回 `README.md`
- 具体推进请看 GitHub milestone / issues，而不是仓库内临时列表

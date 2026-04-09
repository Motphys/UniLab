# Simulation Backends

Languages: English | [简体中文](../zh_CN/02-simulation-backends.md) | [日本語](../ja/02-simulation-backends.md) | [한국어](../ko/02-simulation-backends.md)

UniLab currently supports two simulation backends:

- **MuJoCo**: default backend, with the most complete feature coverage
- **Motrix**: optional backend, with task and algorithm support still being filled in

## Support Matrix

### MuJoCo

| Algorithm | Go1 | Go2 | G1 |
|-----------|-----|-----|----|
| PPO (torch) | ✅ |  | ✅ |
| PPO (mlx) | ✅ |  | ✅ |
| SAC (torch) | ✅ | ⚠️ | ✅ |
| TD3 (torch) | ⚠️ | ⚠️ | ⚠️ |
| APPO (torch) | ✅ | ✅ | ✅ |

### Motrix

| Algorithm | Go1 | Go2 | G1 |
|-----------|-----|-----|----|
| PPO (torch) | ⚠️ |  | ✅ |
| PPO (mlx) | ⚠️ |  |  |
| SAC (torch) | ⚠️ |  |  |
| TD3 (torch) |  |  |  |
| APPO (torch) |  |  | ✅ |

Legend:

- `✅` supported
- `⚠️` in progress

## Select A Backend

The default backend is `mujoco`. Switch to `motrix` with the Hydra parameter `training.sim_backend`.

```bash
# Default MuJoCo
uv run python scripts/train_rsl_rl.py task=go1_joystick

# Explicit Motrix
uv run python scripts/train_rsl_rl.py task=go1_joystick training.sim_backend=motrix
```

## Playback Differences

- `mujoco`: automatic playback after training exports `play_video.mp4`
- `motrix`: playback usually opens an interactive renderer window instead of exporting video

For G1 motion tracking, the validated Motrix paths are currently `PPO (torch) + motrix` and `APPO (torch) + motrix`. `scripts/play_interactive.py` still follows the MuJoCo path.

```bash
uv run python scripts/train_rsl_rl.py task=go1_joystick training.play_only=true
```

## Notes

- Backend support is a stage-specific capability snapshot; do not turn transient execution status into top-level README claims
- Track progress through GitHub milestones and issues instead of temporary in-repo status lists

## Navigation

- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)

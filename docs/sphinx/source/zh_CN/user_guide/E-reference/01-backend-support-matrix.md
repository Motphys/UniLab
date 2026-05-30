# 后端支持矩阵

语言: 简体中文

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Support Matrix

下面的矩阵由 registry、owner YAML 和测试清单自动汇总；不要手工编辑表格内容。需要刷新时运行：

```bash
uv run scripts/generate_support_matrix.py --write
```

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Evidence Grades

| 等级 | 仓库事实来源 |
|------|--------------|
| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |
| `Configured` | 存在对应的 owner YAML：`conf/{ppo,appo,offpolicy}/task/...`。 |
| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合。这里的 `Tested` 包含 config compose 与脚本/运行时测试，不等同于默认推荐路径。 |
| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |
| `Recommended` | 仓库中存在显式 recommendation 元数据。 |

`Tested` 只描述仓库中已有自动化覆盖，不代表该组合具备同名 MuJoCo owner 的全部 backend capability；例如 phase-1 Motrix owner 可能只覆盖训练 smoke 和明确启用的 DR 子集。

未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。
仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
|------------|------------|--------|--------|
| PPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | Tested |
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested |
| PPO (torch) | `go2_joystick_rough` (Go2 joystick rough) | Tested | Tested |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested |
| PPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | Tested |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested |
| PPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | Tested |
| PPO (torch) | `sharpa_inhand_grasp` (Sharpa in-hand grasp) | Tested | Tested |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | Tested |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | Tested |
| PPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | Tested |
| PPO (torch) | `g1_motion_tracking_deploy` (g1 motion tracking deploy) | Tested | Tested |
| PPO (torch) | `go1_joystick_rough` (go1 joystick rough) | Tested | Tested |
| PPO (torch) | `go2_arm_manip_loco` (go2 arm manip loco) | Tested | - |
| PPO (torch) | `go2_handstand` (go2 handstand) | Tested | Tested |
| PPO (torch) | `go2w_joystick_flat` (go2w joystick flat) | Tested | Tested |
| PPO (torch) | `go2w_joystick_rough` (go2w joystick rough) | Tested | Tested |
| PPO (mlx) | `go1_joystick_flat` (Go1 joystick) | Tested | Tested |
| PPO (mlx) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested |
| PPO (mlx) | `go2_joystick_rough` (Go2 joystick rough) | Configured | Configured |
| PPO (mlx) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| PPO (mlx) | `g1_motion_tracking` (G1 motion tracking) | Configured | Configured |
| PPO (mlx) | `g1_flip_tracking` (G1 flip tracking) | Configured | Configured |
| PPO (mlx) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Configured | Configured |
| PPO (mlx) | `allegro_inhand` (Allegro in-hand) | Configured | Configured |
| PPO (mlx) | `sharpa_inhand` (Sharpa in-hand) | Configured | Configured |
| PPO (mlx) | `sharpa_inhand_grasp` (Sharpa in-hand grasp) | Configured | Configured |
| PPO (mlx) | `allegro_inhand_grasp` (allegro inhand grasp) | Configured | Configured |
| PPO (mlx) | `g1_box_tracking` (g1 box tracking) | Configured | Configured |
| PPO (mlx) | `g1_climb_tracking` (g1 climb tracking) | Configured | Configured |
| PPO (mlx) | `g1_motion_tracking_deploy` (g1 motion tracking deploy) | Configured | Configured |
| PPO (mlx) | `go1_joystick_rough` (go1 joystick rough) | Configured | Configured |
| PPO (mlx) | `go2_arm_manip_loco` (go2 arm manip loco) | Configured | - |
| PPO (mlx) | `go2_handstand` (go2 handstand) | Configured | Configured |
| PPO (mlx) | `go2w_joystick_flat` (go2w joystick flat) | Configured | Configured |
| PPO (mlx) | `go2w_joystick_rough` (go2w joystick rough) | Configured | Configured |
| APPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | Registered |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested |
| APPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | Tested |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested |
| APPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | Tested |
| APPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | Tested |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| SAC (torch) | `g1_walk_rough` (G1 walk rough) | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | Registered |
| TD3 (torch) | `go1_joystick_flat` (Go1 joystick) | Registered | Tested |
| TD3 (torch) | `go2_joystick_flat` (Go2 joystick) | Registered | Tested |
| TD3 (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Registered |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`, `conf/offpolicy/task/**`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
- MLX-specific compose coverage only upgrades task owners listed in `tests/config/test_config_system.py::_PPO_MLX_TASKS`: `go1_joystick_flat`, `go2_joystick_flat`, `g1_walk_flat`.
- MLX runtime smoke: `tests/algos/test_mlx_ppo.py::test_mlx_ppo_one_iteration_real_env` currently exercises `go2_joystick_flat/mujoco`.
<!-- END GENERATED SUPPORT MATRIX -->

## Backend 选择规则

- 默认后端通常是 `mujoco`
- 切到 Motrix 用 `task=<task>/<backend>` 或统一 CLI 的 `--sim`
- 不要把 `training.sim_backend` 当独立 backend switch

## Playback Differences

- `mujoco`: `training.play_render_mode=auto` 会导出 `play_video.mp4`
- `motrix`: `training.play_render_mode=auto` 会打开交互式 renderer 窗口，不录制视频，不受 `play_steps` 限制
- `record`: 两个后端都只录制视频
- `none`: 不回放

## Navigation

- Index: [Documentation](../../index.md)
- Previous: [仿真后端](../02-simulation-backends.md)

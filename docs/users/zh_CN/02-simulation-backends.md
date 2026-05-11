# 仿真后端

语言: 简体中文

UniLab 当前支持两个仿真后端:

- **MuJoCo**: 默认后端，能力最完整
- **Motrix**: 可选后端，任务和算法支持仍在持续补齐

## Runtime Prerequisites

- `uv sync --extra motrix` 会安装 Motrix 依赖。
- Motrix 路径的 registry bootstrap 和 Hydra 配置 compose 不再要求导入 MuJoCo。
- 在 macOS / MacBook 上，只要命令会打开 MotrixSim 原生 renderer（训练后自动回放或 `training.play_only=true`），就需要用 `uv run mxpython` 启动；不需要可视化的训练仍可使用 `uv run scripts/... training.no_play=true`。
- 任何 `task=.../mujoco` 的实际运行、MuJoCo playback、以及 MuJoCo-only 调试工具，仍然要求可用的 MuJoCo runtime。
- 某些任务目前仍然只有 MuJoCo owner 配置；例如 APPO 路径下 `go1_joystick_flat` 对 Motrix 仅为 `Registered`。

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

未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。
仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
|------------|------------|--------|--------|
| PPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | Tested |
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested |
| PPO (torch) | `go2_joystick_rough` (Go2 joystick rough) | Tested | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested |
| PPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | Tested |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | Tested |
| PPO (torch) | `go2_handstand` (go2 handstand) | Tested | Tested |
| PPO (torch) | `go2w_joystick_flat` (go2w joystick flat) | Tested | - |
| PPO (torch) | `go2w_joystick_rough_tiles` (go2w joystick rough tiles) | Tested | - |
| PPO (torch) | `sharpa_inhand` (sharpa inhand) | Tested | - |
| PPO (torch) | `sharpa_inhand_grasp` (sharpa inhand grasp) | Tested | - |
| PPO (mlx) | `go1_joystick_flat` (Go1 joystick) | Tested | Tested |
| PPO (mlx) | `go2_joystick_flat` (Go2 joystick) | Tested | Tested |
| PPO (mlx) | `go2_joystick_rough` (Go2 joystick rough) | Configured | - |
| PPO (mlx) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| PPO (mlx) | `g1_motion_tracking` (G1 motion tracking) | Configured | Configured |
| PPO (mlx) | `g1_flip_tracking` (G1 flip tracking) | Configured | Configured |
| PPO (mlx) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Configured | Configured |
| PPO (mlx) | `allegro_inhand` (Allegro in-hand) | Configured | Configured |
| PPO (mlx) | `allegro_inhand_grasp` (allegro inhand grasp) | Configured | Configured |
| PPO (mlx) | `go2_handstand` (go2 handstand) | Configured | Configured |
| PPO (mlx) | `go2w_joystick_flat` (go2w joystick flat) | Configured | - |
| PPO (mlx) | `go2w_joystick_rough_tiles` (go2w joystick rough tiles) | Configured | - |
| PPO (mlx) | `sharpa_inhand` (sharpa inhand) | Configured | - |
| PPO (mlx) | `sharpa_inhand_grasp` (sharpa inhand grasp) | Configured | - |
| APPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | Registered |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | Registered |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Tested |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | Tested |
| APPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | Tested |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | Tested |
| APPO (torch) | `sharpa_inhand` (sharpa inhand) | Tested | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested |
| SAC (torch) | `g1_walk_rough` (G1 walk rough) | Tested | Registered |
| SAC (torch) | `g1_sac_wbt` (g1 sac wbt) | Tested | Tested |
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

## Select A Backend

训练 owner config 的默认后端通常是 `mujoco`。通过 `task=<task>/<backend>` 切换到 `motrix`，不要用 `training.sim_backend=motrix` 单独切换后端。底层 `registry.make(..., sim_backend=None)` 也会显式按 `mujoco`、`motrix` 的顺序解析默认 backend，而不是依赖 decorator 注册顺序。

实际要改参数时，不再去拆着找 `reward` / `backend preset` / `algo preset`。直接改对应的 `task` 文件：

- PPO / APPO: `conf/{ppo,appo}/task/<task>/<backend>.yaml`
- offpolicy: `conf/offpolicy/task/<algo>/<task>/<backend>.yaml`

现在没有单独的 `reward/`、`backend preset`、`sim_backend/` 配置组。`task/` 是唯一 owner 入口，不再是旧的拆分式 task 配置。
`training.sim_backend` 由 owner YAML 设置，只用于标识最终选择的后端；不要把它当作独立 backend switch。

```bash
# 默认 MuJoCo
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco

# 显式指定 Motrix
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/motrix
```

## Playback Differences

- `mujoco`: 训练后的自动回放会导出 `play_video.mp4`
- `motrix`: 回放通常打开交互式 renderer 窗口，而不是导出视频；macOS / MacBook 上需要用 `uv run mxpython` 启动

对 G1 motion tracking 来说，目前已验证的 Motrix 路径是 `PPO (torch) + motrix` 和 `APPO (torch) + motrix`。`scripts/play_interactive.py` 仍然沿用 MuJoCo 路径。

```bash
uv run scripts/train_rsl_rl.py task=go1_joystick_flat/mujoco training.play_only=true

# macOS / MacBook 上的 MotrixSim 原生 renderer
uv run mxpython scripts/train_rsl_rl.py task=go1_joystick_flat/motrix training.play_only=true
```

## Notes

- backend 支持范围是阶段性的能力快照，不要把临时执行状态写成顶层 README 结论
- 具体推进应通过 GitHub milestone 和 issue 跟踪，而不是维护仓库内的临时状态列表

## Architecture Decision References

- 总体架构标准: [RL Infrastructure 开发标准](../../developers/zh_CN/development-standard.md)
- 后端能力边界 ADR: [ADR-0002 Backend Capability Boundary For Play And Snapshot](../../developers/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot.md)
- task owner / compose contract ADR: [ADR-0003 Task Owner And Config Compose Contract](../../developers/adr/ADR-0003-task-owner-and-config-compose-contract.md)
- registry bootstrap ADR: [ADR-0004 Registry Bootstrap Contract](../../developers/adr/ADR-0004-registry-bootstrap-contract.md)
- ADR 索引: [ADR Index](../../developers/adr/README.md)

## Navigation

- Index: [Documentation](../../README.md)
- Previous: [Getting Started](01-getting-started.md)
- Next: [Training Guide](03-training.md)

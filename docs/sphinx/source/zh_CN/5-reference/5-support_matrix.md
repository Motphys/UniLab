# 后端支持矩阵

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Backend 选择规则

- 默认后端是 `mujoco`
- 切到 Motrix 用统一 CLI 的 `--sim motrix`
- `--sim mjwarp` 当前只对应 `g1_walk_flat` host adapter；PPO (torch) 与 SAC (torch) 为 Tested，其他入口按下方矩阵查证，使用前需安装 `mjwarp` extra
- `--algo`、`--task`、`--sim` 共同选择 owner YAML
- 不要把 `training.sim_backend` 当独立 backend switch

## Playback Differences

- `mujoco`: `--render-mode auto` 会导出 `play_video.mp4`
- `motrix`: `--render-mode auto` 会打开交互式 renderer 窗口，不录制视频，不受 `play_steps` 限制
- `mjwarp`: 仅支持显式、有限步数的 `record`，通过 task owner 的 MuJoCo visual model 离线录制；不支持 `auto`、interactive 或 native renderer
- `--render-mode record`: MuJoCo、mjwarp 和 Motrix 都只录制视频
- `--render-mode none`: 不回放

## Support Matrix

下面的矩阵由 registry、owner YAML 和测试/验证清单自动汇总；不要手工编辑表格内容。需要刷新时运行：

```bash
uv run scripts/generate_support_matrix.py --write
```

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Evidence Grades

| 等级 | 仓库事实来源 |
|------|--------------|
| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |
| `Configured` | 存在对应的 owner YAML：`conf/{ppo,appo,sac,td3,flashsac}/task/...`。 |
| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合，或存在显式 maintainer 完整训练验证并具备近风险自动化测试。这里的 `Tested` 不等同于默认推荐路径。 |
| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |
| `Recommended` | 仓库中存在显式 recommendation 元数据。 |

`Tested` 只描述仓库中已有自动化覆盖或显式 maintainer 训练验证，不代表该组合具备同名 MuJoCo owner 的全部 backend capability；例如 phase-1 Motrix owner 可能只覆盖训练 smoke 和明确启用的 DR 子集。

`mjwarp` 只支持 `g1_walk_flat` host adapter。PPO (torch) 与 SAC (torch) owner 已完成训练验证，并有 backend、contract 与 playback 自动化覆盖，因此标记为 `Tested`。mjwarp playback 仅支持显式、有限步数的 `record` 并复用 MuJoCo 离线 renderer，不支持 `auto`、interactive 或 native playback。其他 entrypoint 中出现的 `Registered` 只表示 env/backend registry identity，不代表对应算法、terrain、完整 DR 或 production training 支持。

未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。
仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | mjwarp | Motrix |
|------------|------------|--------|--------|--------|
| PPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | - | Tested |
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Tested |
| PPO (torch) | `go2_joystick_rough` (Go2 joystick rough) | Tested | - | Tested |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | Tested |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Tested |
| PPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Tested |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | - | Tested |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | Tested |
| PPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | - | Tested |
| PPO (torch) | `sharpa_inhand_grasp` (Sharpa in-hand grasp) | Tested | - | Tested |
| PPO (torch) | `a2_joystick_flat` (a2 joystick flat) | Tested | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_box_tracking` (g1 23dof box tracking) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_climb_tracking` (g1 23dof climb tracking) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_motion_tracking_deploy` (g1 23dof motion tracking deploy) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested |
| PPO (torch) | `g1_23dof_walk_rough` (g1 23dof walk rough) | Tested | - | Registered |
| PPO (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Tested |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | - | Tested |
| PPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | - | Tested |
| PPO (torch) | `g1_motion_tracking_deploy` (g1 motion tracking deploy) | Tested | - | Tested |
| PPO (torch) | `go1_joystick_rough` (go1 joystick rough) | Tested | - | Tested |
| PPO (torch) | `go2_arm_manip_loco` (go2 arm manip loco) | Tested | - | Tested |
| PPO (torch) | `go2_footstand` (go2 footstand) | Tested | - | Tested |
| PPO (torch) | `go2w_joystick_flat` (go2w joystick flat) | Tested | - | Tested |
| PPO (torch) | `go2w_joystick_rough` (go2w joystick rough) | Tested | - | Tested |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | - | Tested |
| PPO (torch) | `t800_walk_flat` (t800 walk flat) | Tested | - | - |
| APPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | - | Tested |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Tested |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | Tested |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Tested |
| APPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Tested |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | Tested |
| APPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | - | Tested |
| APPO (torch) | `g1_23dof_climb_tracking` (g1 23dof climb tracking) | Tested | - | Tested |
| APPO (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Tested |
| APPO (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested |
| APPO (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Registered |
| APPO (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Tested |
| APPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | - | Tested |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested |
| SAC (torch) | `g1_walk_rough` (G1 walk rough) | Tested | - | Tested |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | Tested |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Registered |
| SAC (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Registered |
| SAC (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Registered |
| SAC (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested |
| SAC (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested |
| SAC (torch) | `g1_23dof_walk_rough` (g1 23dof walk rough) | Tested | - | Tested |
| SAC (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Registered |
| SAC (torch) | `g1_23dof_wbt_obs` (g1 23dof wbt obs) | Tested | - | Registered |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | - | Registered |
| SAC (torch) | `t800_walk_flat` (t800 walk flat) | Tested | - | - |
| TD3 (torch) | `go1_joystick_flat` (Go1 joystick) | Registered | - | Tested |
| TD3 (torch) | `go2_joystick_flat` (Go2 joystick) | Registered | - | Tested |
| TD3 (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered |
| TD3 (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Registered |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Registered |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Configured | Tested |
| FlashSAC (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`, `conf/sac/task/**`, `conf/td3/task/**`, `conf/flashsac/task/**`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
- Validated mjwarp entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS`; near-risk coverage lives in `tests/base/test_mjwarp_backend.py`, `tests/base/test_backend_conformance.py`, `tests/base/test_mjwarp_differential.py`, and `tests/base/test_mjwarp_playback.py`.
<!-- END GENERATED SUPPORT MATRIX -->

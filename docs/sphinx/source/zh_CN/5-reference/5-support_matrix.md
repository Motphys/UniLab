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

`mjwarp` 完成训练验证的只有 `g1_walk_flat` host adapter：PPO (torch) 与 SAC (torch) owner 已完成训练验证，并有 backend、contract 与 playback 自动化覆盖，因此标记为 `Tested`。SAC `t800_walk_flat` 的 mjwarp owner 只有 owner YAML 与 compose 覆盖，标记为 `Configured`，不代表训练验证。mjwarp playback 仅支持显式、有限步数的 `record` 并复用 MuJoCo 离线 renderer，不支持 `auto`、interactive 或 native playback。其他 entrypoint 中出现的 `Registered` 只表示 env/backend registry identity，不代表对应算法、terrain、完整 DR 或 production training 支持。

`isaacgym` 是 Python 3.8 子进程后端，当前只接入 `g1_walk_flat`。SAC (torch) owner 已在真机（external Python 3.8 worker runtime，不在仓库 CI 覆盖）完成训练与 record playback 验证，标记为 `Tested`；其余 isaacgym cell 最高只到 `Configured`（registry + owner YAML + compose/contract 覆盖），不代表任何训练或 play 验证。playback 走 IsaacGym 原生渲染（viewer + camera sensor 离屏录制），有显示器时 `play_render_mode=auto` 打开交互 viewer，无显示器时自动降级为离屏录制。

`genesis` 是进程内后端（genesis-world==1.3.3，要求 torch>=2.8 与 CUDA；一进程只允许一次 `gs.init`），当前只接入 `g1_walk_flat` 的 PPO (torch) owner。该 cell 最高只到 `Configured`（registry + owner YAML + compose/contract 覆盖），不代表任何训练验证。真机 env smoke 慢车道测试已编码（`tests/envs/locomotion/g1/test_g1_owner_contract.py`），但因 adapter 侧缺口暂时 xfail：`ManagerBasedRlEnv` 构造期的 entity 校验在 env 的 `materialize()` 钩子之前读取状态 getter，而 genesis backend 在 materialize 前 fail-closed（且其 `materialize()` 非幂等；isaacgym 后端的惰性幂等 materialize 是既有先例）。Genesis 在 import 时丢弃 MJCF 全局 `<option>`，owner YAML 显式重声明 `genesis_integrator=implicitfast`。未支持边界：geom 名称契约、体帧运动学（`get_body_pos_b`/`get_body_quat_b`）、terrain spawn 与 height scanner、playback/render（`play_render_mode` 仅 `none` 安全进入，其余模式 fail-closed）、contact sensor 为 per-link net-force 阈值近似（非 geom 对 `data="found"`）、`get_geom_friction` 类绝对摩擦 DR fail-closed（geom 摩擦只有 per-env ratio API）。

未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。
仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | mjwarp | Motrix | IsaacGym | Genesis |
|------------|------------|--------|--------|--------|----------|---------|
| PPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | - | Tested | - | - |
| PPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Tested | - | - |
| PPO (torch) | `go2_joystick_rough` (Go2 joystick rough) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested | Configured | Configured |
| PPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `x2_wall_flip_tracking` (X2 wall flip tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | Tested | - | - |
| PPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | - | Tested | - | - |
| PPO (torch) | `sharpa_inhand_grasp` (Sharpa in-hand grasp) | Tested | - | Tested | - | - |
| PPO (torch) | `a2_joystick_flat` (a2 joystick flat) | Tested | - | - | - | - |
| PPO (torch) | `allegro_inhand_grasp` (allegro inhand grasp) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_box_tracking` (g1 23dof box tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_climb_tracking` (g1 23dof climb tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_motion_tracking_deploy` (g1 23dof motion tracking deploy) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_23dof_walk_rough` (g1 23dof walk rough) | Tested | - | Registered | - | - |
| PPO (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_box_tracking` (g1 box tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | - | Tested | - | - |
| PPO (torch) | `g1_motion_tracking_deploy` (g1 motion tracking deploy) | Tested | - | Tested | - | - |
| PPO (torch) | `go1_joystick_rough` (go1 joystick rough) | Tested | - | Tested | - | - |
| PPO (torch) | `go2_arm_manip_loco` (go2 arm manip loco) | Tested | - | Tested | - | - |
| PPO (torch) | `go2_footstand` (go2 footstand) | Tested | - | Tested | - | - |
| PPO (torch) | `go2w_joystick_flat` (go2w joystick flat) | Tested | - | Tested | - | - |
| PPO (torch) | `go2w_joystick_rough` (go2w joystick rough) | Tested | - | Tested | - | - |
| PPO (torch) | `stewart_balance` (stewart balance) | Tested | - | Tested | - | - |
| PPO (torch) | `t800_walk_flat` (t800 walk flat) | Tested | Registered | - | - | - |
| APPO (torch) | `go1_joystick_flat` (Go1 joystick) | Tested | - | Tested | - | - |
| APPO (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered | Registered | Registered |
| APPO (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `allegro_inhand` (Allegro in-hand) | Tested | - | Tested | - | - |
| APPO (torch) | `sharpa_inhand` (Sharpa in-hand) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_23dof_climb_tracking` (g1 23dof climb tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Registered | - | - |
| APPO (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Tested | - | - |
| APPO (torch) | `g1_climb_tracking` (g1 climb tracking) | Tested | - | Tested | - | - |
| SAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Tested | Tested | Tested | Registered |
| SAC (torch) | `g1_walk_rough` (G1 walk rough) | Tested | - | Tested | - | - |
| SAC (torch) | `g1_motion_tracking` (G1 motion tracking) | Tested | Configured | Tested | - | - |
| SAC (torch) | `g1_flip_tracking` (G1 flip tracking) | Tested | - | Registered | - | - |
| SAC (torch) | `g1_wall_flip_tracking` (G1 wall flip tracking) | Tested | - | Registered | - | - |
| SAC (torch) | `g1_23dof_flip_tracking` (g1 23dof flip tracking) | Tested | - | Registered | - | - |
| SAC (torch) | `g1_23dof_motion_tracking` (g1 23dof motion tracking) | Tested | - | Tested | - | - |
| SAC (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested | - | - |
| SAC (torch) | `g1_23dof_walk_rough` (g1 23dof walk rough) | Tested | - | Tested | - | - |
| SAC (torch) | `g1_23dof_wall_flip_tracking` (g1 23dof wall flip tracking) | Tested | - | Registered | - | - |
| SAC (torch) | `g1_23dof_wbt_obs` (g1 23dof wbt obs) | Tested | - | Registered | - | - |
| SAC (torch) | `g1_wbt_obs` (g1 wbt obs) | Tested | - | Registered | - | - |
| SAC (torch) | `t800_walk_flat` (t800 walk flat) | Tested | Configured | - | - | - |
| TD3 (torch) | `go1_joystick_flat` (Go1 joystick) | Registered | - | Tested | - | - |
| TD3 (torch) | `go2_joystick_flat` (Go2 joystick) | Registered | - | Tested | - | - |
| TD3 (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Registered | Registered | Registered | Registered |
| TD3 (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Registered | - | - |
| FlashSAC (torch) | `go2_joystick_flat` (Go2 joystick) | Tested | - | Registered | - | - |
| FlashSAC (torch) | `g1_walk_flat` (G1 walk flat) | Tested | Configured | Tested | Registered | Registered |
| FlashSAC (torch) | `g1_23dof_walk_flat` (g1 23dof walk flat) | Tested | - | Tested | - | - |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`, `conf/sac/task/**`, `conf/td3/task/**`, `conf/flashsac/task/**`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
- Validated mjwarp entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS`; near-risk coverage lives in `tests/base/test_mjwarp_backend.py`, `tests/base/test_backend_conformance.py`, `tests/base/test_mjwarp_differential.py`, and `tests/base/test_mjwarp_playback.py`.
- Validated isaacgym entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_ISAACGYM_ENTRYPOINT_TASKS` (real hardware via the external Python 3.8 worker runtime; not covered by repo CI).
- Validated genesis entrypoints are explicitly recorded in `_MAINTAINER_VALIDATED_GENESIS_ENTRYPOINT_TASKS` (currently empty: no maintainer training validation yet); near-risk coverage lives in `tests/base/test_genesis_backend.py` (fake runtime), `tests/base/test_genesis_runtime.py` (real-runtime slow lane), and the genesis env smoke in `tests/envs/locomotion/g1/test_g1_owner_contract.py`.
<!-- END GENERATED SUPPORT MATRIX -->

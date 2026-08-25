# Support Matrix

This matrix is generated conceptually from registry entries, owner YAMLs, and
tests. The generator implementation is `src/unilab/utils/support_matrix.py`; the
write target for the generated block is currently the Chinese reference page
`docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`. This English page
mirrors that generated content.

## Backend Selection Rules

- The default backend is `mujoco`.
- Switch to Motrix with `--sim motrix` on the unified CLI.
- `--sim mjwarp` currently maps only to the `g1_walk_flat` host adapter; PPO
  (torch) and SAC (torch) are Tested, other entrypoints follow the matrix
  below, and using it requires installing the `mjwarp` extra.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`.
- `motrix`: `--render-mode auto` opens an interactive renderer window; it does
  not record a video and is not bound by `play_steps`.
- `mjwarp`: only supports explicit, finite-step `record`, rendered offline
  through the task owner's MuJoCo visual model; `auto`, interactive, and
  native renderers are not supported.
- `--render-mode record`: MuJoCo, mjwarp, and Motrix all record a video only.
- `--render-mode none`: no playback.

## Evidence Grades

| Grade | Repository Evidence |
| --- | --- |
| `Registered` | The env/backend pair exists in `registry.list_registered_envs()` after `ensure_registries()`. |
| `Configured` | A matching owner YAML exists under `conf/{ppo,appo,sac,td3,flashsac}/task/...`. |
| `Tested` | Automated tests under `tests/` cover the entrypoint/task-owner/backend combination, or an explicit maintainer full-training validation with near-risk automated tests exists. `Tested` here does not mean the default recommended path. |
| `Benchmarked` | A checked-in benchmark manifest exists for the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repo. |

`Tested` only describes existing automated coverage or explicit maintainer
training validation; it does not imply the combination has all the backend
capabilities of the same-named MuJoCo owner. For example, a phase-1 Motrix
owner may only cover training smoke and an explicitly enabled DR subset.

`mjwarp` only supports the `g1_walk_flat` host adapter. The PPO (torch) and SAC
(torch) owners have completed training validation and have backend, contract,
and playback automated coverage, so they are marked `Tested`. mjwarp playback
only supports explicit, finite-step `record` and reuses the MuJoCo offline
renderer; it does not support `auto`, interactive, or native playback. A
`Registered` mark on other entrypoints only denotes env/backend registry
identity, not support for the corresponding algorithm, terrain, full DR, or
production training.

No checked-in benchmark manifest bound to these combinations has been detected,
so rows do not auto-promote to `Benchmarked`. There is also no separate
recommendation metadata in the repo, so rows do not auto-promote to
`Recommended`.

## Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | mjwarp | Motrix |
| --- | --- | --- | --- | --- |
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

## Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via
  `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`,
  `conf/sac/task/**`, `conf/td3/task/**`, `conf/flashsac/task/**`.
- Generic compose coverage:
  `tests/config/test_config_system.py::test_supported_task_composes`.
- Validated mjwarp entrypoints are explicitly recorded in
  `_MAINTAINER_VALIDATED_MJWARP_ENTRYPOINT_TASKS`; near-risk coverage lives in
  `tests/base/test_mjwarp_backend.py`,
  `tests/base/test_backend_conformance.py`,
  `tests/base/test_mjwarp_differential.py`, and
  `tests/base/test_mjwarp_playback.py`.

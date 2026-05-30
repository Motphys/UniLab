# Support Matrix

This matrix is generated conceptually from registry entries, owner YAMLs, and
tests. The generator implementation is `src/unilab/utils/support_matrix.py`; the
write target for the generated block is currently the Chinese reference page
`docs/sphinx/source/zh_CN/1-user_guide/5-reference/1-backend-support-matrix.md`.

## Evidence Grades

| Grade | Repository Evidence |
| --- | --- |
| `Registered` | The env/backend pair appears after `registry.ensure_registries()`. |
| `Configured` | A matching owner YAML exists under `conf/ppo/task`, `conf/appo/task`, or `conf/offpolicy/task`. |
| `Tested` | Automated tests cover the entrypoint/task-owner/backend combination through config compose or runtime smoke. |
| `Benchmarked` | A checked-in benchmark manifest exists for the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repo. |

The current generator reports no checked-in benchmark manifest and no separate
recommendation metadata, so rows do not auto-promote to `Benchmarked` or
`Recommended`.

## Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
| --- | --- | --- | --- |
| PPO (torch) | `go1_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2_joystick_rough` | Tested | Tested |
| PPO (torch) | `g1_walk_flat` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking` | Tested | Tested |
| PPO (torch) | `g1_flip_tracking` | Tested | Tested |
| PPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| PPO (torch) | `allegro_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand` | Tested | Tested |
| PPO (torch) | `sharpa_inhand_grasp` | Tested | Tested |
| PPO (torch) | `allegro_inhand_grasp` | Tested | Tested |
| PPO (torch) | `g1_box_tracking` | Tested | Tested |
| PPO (torch) | `g1_climb_tracking` | Tested | Tested |
| PPO (torch) | `g1_motion_tracking_deploy` | Tested | Tested |
| PPO (torch) | `go1_joystick_rough` | Tested | Tested |
| PPO (torch) | `go2_arm_manip_loco` | Tested | - |
| PPO (torch) | `go2_handstand` | Tested | Tested |
| PPO (torch) | `go2w_joystick_flat` | Tested | Tested |
| PPO (torch) | `go2w_joystick_rough` | Tested | Tested |
| PPO (mlx) | `go1_joystick_flat` | Tested | Tested |
| PPO (mlx) | `go2_joystick_flat` | Tested | Tested |
| PPO (mlx) | `go2_joystick_rough` | Configured | Configured |
| PPO (mlx) | `g1_walk_flat` | Tested | Tested |
| PPO (mlx) | `g1_motion_tracking` | Configured | Configured |
| PPO (mlx) | `g1_flip_tracking` | Configured | Configured |
| PPO (mlx) | `g1_wall_flip_tracking` | Configured | Configured |
| PPO (mlx) | `allegro_inhand` | Configured | Configured |
| PPO (mlx) | `sharpa_inhand` | Configured | Configured |
| PPO (mlx) | `sharpa_inhand_grasp` | Configured | Configured |
| PPO (mlx) | `allegro_inhand_grasp` | Configured | Configured |
| PPO (mlx) | `g1_box_tracking` | Configured | Configured |
| PPO (mlx) | `g1_climb_tracking` | Configured | Configured |
| PPO (mlx) | `g1_motion_tracking_deploy` | Configured | Configured |
| PPO (mlx) | `go1_joystick_rough` | Configured | Configured |
| PPO (mlx) | `go2_arm_manip_loco` | Configured | - |
| PPO (mlx) | `go2_handstand` | Configured | Configured |
| PPO (mlx) | `go2w_joystick_flat` | Configured | Configured |
| PPO (mlx) | `go2w_joystick_rough` | Configured | Configured |
| APPO (torch) | `go1_joystick_flat` | Tested | Registered |
| APPO (torch) | `go2_joystick_flat` | Tested | Registered |
| APPO (torch) | `g1_walk_flat` | Tested | Registered |
| APPO (torch) | `g1_motion_tracking` | Tested | Tested |
| APPO (torch) | `g1_flip_tracking` | Tested | Tested |
| APPO (torch) | `g1_wall_flip_tracking` | Tested | Tested |
| APPO (torch) | `allegro_inhand` | Tested | Tested |
| APPO (torch) | `sharpa_inhand` | Tested | Registered |
| APPO (torch) | `g1_climb_tracking` | Tested | Tested |
| SAC (torch) | `g1_walk_flat` | Tested | Tested |
| SAC (torch) | `g1_walk_rough` | Tested | Tested |
| SAC (torch) | `g1_motion_tracking` | Tested | Tested |
| SAC (torch) | `g1_wbt_obs` | Tested | Registered |
| TD3 (torch) | `go1_joystick_flat` | Registered | Tested |
| TD3 (torch) | `go2_joystick_flat` | Registered | Tested |
| TD3 (torch) | `g1_walk_flat` | Tested | Registered |
| FlashSAC (torch) | `go2_joystick_flat` | Tested | Registered |
| FlashSAC (torch) | `g1_walk_flat` | Tested | Registered |

## Source Index

- Registry bootstrap: `src/unilab/envs/**` registrations via
  `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/appo/task/**`,
  `conf/offpolicy/task/**`.
- Generic compose coverage:
  `tests/config/test_config_system.py::test_supported_task_composes`.
- MLX-specific compose coverage:
  `tests/config/test_config_system.py::_PPO_MLX_TASKS`.
- MLX runtime smoke:
  `tests/algos/test_mlx_ppo.py::test_mlx_ppo_one_iteration_real_env`.

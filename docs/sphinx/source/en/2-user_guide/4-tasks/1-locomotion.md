# Locomotion

Locomotion tasks are registered in `src/unilab/envs/locomotion/` and
`src/unilab/envs/motion_tracking/`. The available owner YAMLs under `conf/`
define which algorithm and backend combinations are runnable.

## Families

- Go1: `go1_joystick_flat`, `go1_joystick_rough`
- Go2: `go2_joystick_flat`, `go2_joystick_rough`, `go2_handstand`, `go2_footstand`
- Go2W: `go2w_joystick_flat`, `go2w_joystick_rough`
- G1 walking: `g1_walk_flat`, `g1_walk_rough`
- G1 motion tracking: `g1_motion_tracking`, `g1_flip_tracking`,
  `g1_wall_flip_tracking`, `g1_climb_tracking`, `g1_box_tracking`
- Go2 arm: `go2_arm_manip_loco`

## Examples

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_rough --sim motrix training.no_play=true
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run train --algo appo --task g1_motion_tracking --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Check the support matrix for evidence grade by entrypoint, task owner, and
backend: {doc}`../../5-reference/5-support_matrix`.

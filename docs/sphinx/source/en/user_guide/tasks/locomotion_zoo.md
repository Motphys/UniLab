# Locomotion Zoo

Locomotion tasks are registered in `src/unilab/envs/locomotion/` and
`src/unilab/envs/motion_tracking/`. The available owner YAMLs under `conf/`
define which algorithm and backend combinations are runnable.

## Families

- Go1: `go1_joystick_flat`, `go1_joystick_rough`
- Go2: `go2_joystick_flat`, `go2_joystick_rough`, `go2_handstand`
- Go2W: `go2w_joystick_flat`, `go2w_joystick_rough`
- G1 walking: `g1_walk_flat`, `g1_walk_rough`
- G1 motion tracking: `g1_motion_tracking`, `g1_flip_tracking`,
  `g1_wall_flip_tracking`, `g1_climb_tracking`, `g1_box_tracking`
- Go2 arm: `go2_arm_manip_loco`

## Examples

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_rough/motrix training.no_play=true
uv run scripts/train_appo.py task=g1_motion_tracking/mujoco training.no_play=true
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

Check the generated support matrix for evidence grade by entrypoint, task owner,
and backend: {doc}`/zh_CN/user_guide/E-reference/01-backend-support-matrix`.

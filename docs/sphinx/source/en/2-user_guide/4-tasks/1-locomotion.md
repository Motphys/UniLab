# Locomotion

Locomotion tasks are registered in `src/unilab/tasks/locomotion/` and
`src/unilab/tasks/motion_tracking/`. The available owner YAMLs under `src/unilab/conf/`
define which algorithm and backend combinations are runnable.

## Families

- Go1: `go1_joystick_flat`, `go1_joystick_rough`
- Go2: `go2_joystick_flat`, `go2_joystick_rough`, `go2_footstand`
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

## Go2 FootStand

`go2_footstand` is the Go2 front-feet-stand task. Its PPO owner YAMLs
register MuJoCo, Motrix, and Drake; the SAC owner currently targets Drake.

- Canonical PPO task config: `src/unilab/conf/ppo/task/go2_footstand/base.yaml`
- Backend owners: `src/unilab/conf/ppo/task/go2_footstand/{mujoco,motrix,drake}.yaml`
- Registered env: `Go2FootStand` (MuJoCo, Motrix, and Drake)
- Implementation: `src/unilab/tasks/locomotion/go2/footstand.py`
  (task-owned NumPy manager terms on the generic Manager-Based runtime)
- Go2 model XML: `src/unilab/assets/robots/go2/go2.xml`

```bash
uv run train --algo ppo --task go2_footstand --sim mujoco training.no_play=true
uv run eval --algo ppo --task go2_footstand --sim mujoco --load-run -1
```

### Teacher-Student Pipeline

The full FootStand recipe is a three-stage teacher-student pipeline; the shipped
`go2_footstand` config corresponds to stage 1 (the teacher PPO entry point):

1. **Teacher PPO (privileged obs).** The teacher uses privileged observations
   (e.g. base linear velocity) that are available in simulation but should not be
   relied on directly during real-robot deployment. A power curriculum first lets
   the policy learn the front-feet stand under a loose power budget (~400 W), then
   gradually tightens it toward ~200 W, so early exploration is not crushed by a
   low power limit and the final policy stays near a deployable energy envelope.
2. **Distillation to a deployable student.** The trained teacher is distilled into
   a student policy whose inputs keep only on-robot observations (no privileged
   information). The goal is to reproduce the teacher's behavior without privileged
   obs.
3. **Student RL fine-tune.** The distilled student is fine-tuned with a combined
   objective: a reward term similar to the teacher's, plus a teacher-regularization
   term that keeps the student from drifting away from the teacher too quickly.
   This preserves the stable motion while letting the student adapt to its own
   observation inputs and deployment constraints.

### Observation Layout

The `Go2FootStand` policy (actor) observation uses 15 history frames of 45 dims
each (`FRAME_OBS_DIM = 45`):

```text
linvel(3) + gyro(3) + gravity(3) + joint_position_delta(12) + joint_velocity(12) + last_action(12)
```

So the policy observation is `45 * 15 = 675`. The value (critic) observation
appends the current-step privileged tail (`PRIVILEGED_OBS_DIM = 49`)
after that history:

```text
gyro(3) + accelerometer(3) + linvel(3) + global_angvel(3) + dof_pos(12) + dof_vel(12) + torques(12) + height(1)
```

The value observation is therefore `675 + 49 = 724`.

### Rewards And Terminations

Defaults come from `src/unilab/conf/ppo/task/go2_footstand/base.yaml`; backend leaves only
override backend-specific terms and tuning. The reward scales
include stand `height`, `orientation`, `rear_feet_contact`, target front-leg angle
(`tar`), `action_rate`, `dof_pos_limits`, `front_leg_motion`, `rear_leg_symmetry`,
`knee_clearance`, `upright_stability`, `stay_still`, `pose`, plus `energy` and
`dof_acc` penalties; `termination` and `penalty_contact` drive the termination /
penalty paths (front-leg / front-body contact, low height, bad orientation, and a
high-energy cutoff in the `footstand` termination term).

### Tuning Keys

- `env.observations.policy.terms.frame.history_length`: policy history length
  (default `15`).
- `env.terminations.footstand.params.energy_threshold`: high-energy cutoff
  (default `200.0`).
- `env.events`: reset and domain-randomization terms. Backend owners explicitly
  set unsupported model-field terms to `null`.
- `reward.footstand.params.scales`: stand, contact, motion, and energy weights.

### Near-Risk Validation

```bash
uv run pytest tests/envs/locomotion/test_go2_footstand.py tests/config/test_locomotion_params.py -q
```

If the Go2 XML changed, at minimum confirm MuJoCo can load the model:

```bash
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/unilab/assets/robots/go2/go2.xml'); print(m.nq, m.nv, m.nu, m.nsensor)"
```

# Configuration

Domain randomization is configured inside the selected task owner YAML. Use
`--task` and `--sim` to select backend-specific behavior first, then override
fields inside that selected owner.

Two declaration paths exist today:

- Manager-Based (Compatible) tasks declare reset / interval randomization
  through Hydra `events:` manager terms in the owner YAML, for example
  `src/unilab/conf/ppo/task/go1_joystick_flat/base.yaml`.
- Only the Adapted families (sharpa / go2_arm and their hora / appo / ppo_him
  owners) still configure legacy provider fields under `env.domain_rand`.

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

Common lifecycle boundaries:

- Init-lifecycle items change model identity or geometry and must run during
  env/backend initialization.
- Reset-lifecycle items perturb state or model parameters at reset through a
  backend-supported payload.
- Interval-lifecycle items apply perturbations between steps.

The detailed task status and field semantics are in {doc}`0-index`.

Domain randomization is split by lifecycle: init, reset, and interval. The
legacy path's manager is `src/unilab/dr/manager.py`; task providers live near
the env owners, and backend capabilities are declared through
`src/unilab/base/backend/base.py`.

## Reset Gravity

Use `--sim mujoco` when enabling gravity reset randomization; Motrix does not
advertise the same gravity capability in the current backend. This item is only
available on the legacy provider path (Adapted-family owners).

```bash
uv run train --algo ppo --task sharpa_inhand_grasp --sim mujoco \
  env.domain_rand.randomize_gravity=true \
  'env.domain_rand.gravity_range=[[0.0,0.0,-10.5],[0.0,0.0,-8.5]]'
```

## Interval Push

Manager-Based tasks declare push through a `push_by_setting_velocity` interval
event term; `env.domain_rand.push_robots` is only available on the go2_arm
Adapted-family owners.

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim mujoco \
  env.domain_rand.push_robots=true \
  env.domain_rand.push_interval=500 \
  'env.domain_rand.max_force=[20.0,20.0,5.0]'
```

## Owner-Local Defaults

Keep ranges in the task owner YAML when they are part of the task contract. For
example, the rough quadruped family's base mass, center-of-mass, kp/kd, and
push randomization are declared as event terms in the shared base
`src/unilab/conf/ppo/task/quadruped_joystick_rough/base.yaml` (the `go2_joystick_rough`
backend owners compose it through Hydra defaults), while
`src/unilab/conf/ppo/task/sharpa_inhand/mujoco.yaml` configures object scale, friction, and
force disturbance for Sharpa.

For the full current inventory, see {doc}`0-index`.

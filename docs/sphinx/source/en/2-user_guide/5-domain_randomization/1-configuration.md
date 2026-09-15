# Configuration

Domain randomization is configured inside the selected Manager-Based task owner
YAML. Use `--task` and `--sim` to select backend-specific behavior first, then
override fields inside that selected owner.

The lifecycle boundaries are:

- Fixed model/tool identity is declared once on `env.fixed_model_variants` and
  realized during backend construction; it is not reset randomization.
- Reset-lifecycle event terms perturb state or curated model parameters through
  one `ResetStateTransaction` and one backend payload.
- Interval-lifecycle event terms apply perturbations between steps.

Backend support is declared through `unisim.backend.base`. A requested term that
the selected backend does not advertise fails closed.

## Reset Gravity

Use `--sim mujoco` when enabling gravity reset randomization; Motrix does not
advertise the gravity reset capability. Configure gravity through a reset event
term that calls `randomize_physics_scene_gravity`.

## Interval Push

Manager-Based tasks configure interval push through the `env.events.push_robot`
term. For example, the retained `g1_wbt_obs` owner uses
`push_by_setting_velocity` with an interval and per-axis velocity ranges.

```bash
uv run train --algo sac --task g1_wbt_obs --sim mujoco \
  'env.events.push_robot.interval_range_s=[10.0,10.0]'
```

## Owner-Local Defaults

Keep ranges in the task owner YAML when they are part of the task contract. For
example, a rough-task family can declare base mass, center-of-mass, kp/kd, and
push randomization as event terms in its shared owner base.

For the full current inventory, see {doc}`0-index`.

# Domain Randomization for Real-World Transfer

This page is the deployment checklist for domain randomization. For the
**contract** layer (what a DR provider must implement), see
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`.

## What to randomize, in priority order

```{list-table}
:header-rows: 1
:widths: 25 30 45

* - Category
  - Examples
  - Why it matters
* - Actuator dynamics
  - PD gains, action scale, one-step action delay when the task owner enables it
  - First-order driver of policy oscillation on hardware.
* - Mass / inertia
  - Trunk mass, link COM offsets, payload
  - Affects balance and tracking margins.
* - Friction
  - Foot ↔ ground μ, hand ↔ object μ
  - In-hand cube tasks fail without this.
* - Observation noise
  - IMU noise, joint encoder bias, deploy-side observation history
  - Keeps actor inputs close to deploy-side sensor behavior.
* - External forces
  - Pushes, gusts, tug on payload
  - Robustness to unmodeled disturbances.
* - Reset state
  - Initial pose, initial velocity
  - Reduces brittleness at episode boundary.
```

::::{admonition} Heuristic
:class: tip
If a parameter materially affects the closed-loop response and you do not have
a deploy-side measurement, keep the claim out of docs and encode a conservative
range in the task owner only after recording why that range is plausible.
::::

## How UniLab structures DR

Manager-Based tasks declare reset and interval randomization through
`env.events` in their owner YAML, executed by the manager lifecycle. See
`src/unilab/conf/ppo/task/quadruped_joystick_rough/base.yaml`.

Tasks may also attach a task-level provider (the
`DomainRandomizationProvider` interface in `src/unilab/dr/provider.py`) to
`src/unilab/dr/manager.py`; no in-repo task currently uses this path. The
capability boundary for both paths is described in
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`.

## Recipe: starting ranges

Use the selected owner YAML as the source of truth. Go2 rough owners compose
`src/unilab/conf/ppo/task/quadruped_joystick_rough/base.yaml`, which declares
base mass, COM, PD gains, and interval push. This excerpt shows its PD-gain
term; evaluate absolute gain ranges together with the robot's control settings.

```yaml
env:
  events:
    pd_gains:
      func: unilab.envs.mdp.pd_gains
      mode: reset
      params:
        kp_range: [17.5, 70.0]
        kd_range: [0.25, 1.0]
        operation: abs
```

## Curriculum: ramp DR with skill

DR that's too aggressive at step 0 stalls learning. UniLab curriculum
helpers are task-owned; keep their fields in the selected owner YAML and do not
add Python-side interpretation in training scripts.

## Validating DR coverage

After training, replay the checkpoint against the same backend owner YAML while
you sweep DR ranges in config:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1
```

Log reward components and task success metrics for each sweep point. A sharp
drop or a reward-component discontinuity is evidence that the DR range changed
the task contract rather than only widening deployment coverage.

## See also

- {doc}`../../2-user_guide/5-domain_randomization/0-index`
- {doc}`../../4-developer_guide/2-contracts/4-dr_contract`
- {doc}`../2-sim_to_sim/3-contact_and_friction_alignment`

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

Tasks that use DR attach a provider through the env initialization path:

```python
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider

class MyTaskEnv(NpEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_domain_randomization(LocomotionDRProvider(cfg.domain_rand))
```

The manager lives in `src/unilab/dr/manager.py`; providers live near their env
owners and conform to the contract in
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`.

## Recipe: starting ranges

Use the selected owner YAML as the source of truth. For example,
`conf/ppo/task/go2_joystick_rough/mujoco.yaml` enables base-mass, COM, kp/kd,
and push randomization; `conf/ppo/task/sharpa_inhand/mujoco.yaml` configures
PD-gain, friction, COM, mass, joint-noise, and contact-noise fields.

```yaml
# conf/ppo/task/go2_joystick_rough/mujoco.yaml
env:
  domain_rand:
    randomize_base_mass: true
    added_mass_range: [-1.0, 3.0]
    random_com: true
    randomize_kp: true
    kp_multiplier_range: [0.5, 2.0]
    randomize_kd: true
    kd_multiplier_range: [0.5, 2.0]
    push_robots: true
    push_interval: 625
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

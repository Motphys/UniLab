# Domain Randomization for Real-World Transfer

Sim-to-real ≈ DR done well. This page is the *recipe* layer; for the
**contract** layer (what your DR provider must implement), see
{doc}`../../developer_guide/contracts/domain_randomization`.

## What to randomize, in priority order

```{list-table}
:header-rows: 1
:widths: 25 30 45

* - Category
  - Examples
  - Why it matters
* - Actuator dynamics
  - PD gains, motor torque limits, response delay
  - First-order driver of policy oscillation on hardware.
* - Mass / inertia
  - Trunk mass, link COM offsets, payload
  - Bipeds especially: trunk mass off by 5% breaks balance.
* - Friction
  - Foot ↔ ground μ, hand ↔ object μ
  - In-hand cube tasks fail without this.
* - Observation noise
  - IMU bias / noise, joint encoder quantization
  - Cheap to randomize; pays back enormously.
* - External forces
  - Pushes, gusts, tug on payload
  - Robustness to unmodeled disturbances.
* - Reset state
  - Initial pose, initial velocity
  - Reduces brittleness at episode boundary.
```

::::{admonition} Heuristic
:class: tip
If a parameter materially affects the closed-loop response *and* you don't
have a precise measurement, randomize it across at least the 95%-confidence
interval of plausible real values.
::::

## How UniLab structures DR

Each task **must** ship a DR provider:

```python
from unilab.dr import DomainRandomizationManager
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider

class MyTaskEnv(NpEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dr = DomainRandomizationManager(
            provider=LocomotionDRProvider(cfg.dr),
            backend=self.backend,
        )

    def reset_idx(self, env_ids):
        self.dr.resample(env_ids)
        ...
```

The provider implementation lives at
{py:mod}`unilab.envs.locomotion.common.dr_provider` and conforms to the
contract in {doc}`../../developer_guide/contracts/domain_randomization`.

## Recipe: starting ranges

Use the following as a *first cut*. **Always** validate that your reward
varies smoothly across the range — flat reward means the policy is
ignoring it (good or bad depending on the parameter); discontinuous reward
means your DR range exits the basin of attraction.

```yaml
# conf/dr/locomotion_first_cut.yaml
mass:
  trunk_kg_factor:     [0.85, 1.15]
  link_com_offset_m:   [-0.02, 0.02]
friction:
  ground_static_mu:    [0.4, 1.2]
  ground_dynamic_mu:   [0.3, 1.0]
actuator:
  pd_kp_factor:        [0.85, 1.15]
  pd_kd_factor:        [0.85, 1.15]
  torque_delay_ms:     [0, 20]
  torque_noise_std:    [0.0, 0.05]
observation:
  imu_gyro_bias_dps:   [-1.0, 1.0]
  imu_gyro_noise_dps:  [0.0, 0.5]
  joint_pos_noise_rad: [0.0, 0.005]
push:
  prob_per_step:       0.005
  force_xy_N:          [-15, 15]
```

## Curriculum: ramp DR with skill

DR that's too aggressive at step 0 stalls learning. UniLab curriculum
helpers (`unilab.base.curriculum`) let you scale ranges as a function of a
**success metric**, not wall time:

```python
from unilab.base.curriculum import LinearCurriculum

cur = LinearCurriculum(
    metric="episode/avg_track_err",
    threshold_start=0.10,    # tighten when err < 0.10
    threshold_end=0.05,
    scale_start=0.3,         # start at 30% of nominal DR range
    scale_end=1.0,
)
```

## Validating DR coverage

After training, run a **DR sweep evaluation**:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --eval-dr-sweep
```

Expect the success rate to degrade gracefully as DR strength increases — a
cliff means the policy is overfit to a narrow regime, and hardware
performance will be unpredictable.

## See also

- {doc}`../../user_guide/domain_randomization/index`
- {doc}`../../developer_guide/contracts/domain_randomization`
- {doc}`../sim_to_sim/contact_and_friction_alignment`

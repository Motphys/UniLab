# Aligning Contact and Friction Between Backends

The single biggest source of MuJoCo-vs-Motrix drift is contact. This page
shows you how to identify and close the gap.

## Diagnostic: probe contact early

Drop the robot onto the floor from 1 m and measure peak vertical force.
Run the **same drop test** in both backends with the same robot and the
same friction declarations.

```python
from unilab.base.scene import Scene
from unilab.base.backend.mujoco import MuJoCoBackend
from unilab.base.backend.motrix import MotrixBackend

scene = Scene.from_owner_yaml("conf/diag/drop_test.yaml")
for B in [MuJoCoBackend, MotrixBackend]:
    b = B(scene)
    b.reset(initial_state)
    history = []
    for _ in range(500):
        b.step()
        history.append(b.foot_contact_force())
    plot(history, label=B.__name__)
```

You expect:

- Peak force magnitudes within ~20%.
- Settling time within ~10 ms.

Larger differences mean the friction / damping / restitution parameters
mean different things to the two backends.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Friction declared on geom in MuJoCo but on material in Motrix | One backend has μ=0 | Declare on both; verify in scene export |
| Solver iterations too low | MuJoCo sinks into floor | Bump to ≥ 30 |
| Contact-pair specific overrides | Inconsistent slip | Make pair-level overrides explicit in both backend YAMLs |
| Restitution mismatch | Bouncy vs sticky landing | Set explicitly; defaults differ |

## Aligning DR ranges

Once contact is aligned for the **nominal** parameters, audit DR:

- `friction_static_mu` and `friction_dynamic_mu` should be sampled
  identically in both backends.
- If a DR field is a no-op in one backend, log a warning at episode init.
  Silent ignorance leads to silent reward drift.

## See also

- {doc}`reward_parity_checks`
- {doc}`../sim_to_real/domain_randomization_for_real`
- {doc}`known_capability_gaps`

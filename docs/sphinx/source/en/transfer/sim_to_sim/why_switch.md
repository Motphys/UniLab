# Why Switch Backends?

UniLab supports two CPU physics backends: **MuJoCo** (via `mujoco-uni`) and
**Motrix** (via `motrixsim-core`). Both implement the same `SimBackend`
contract and the same env contract, but they differ in:

| Axis | MuJoCo | Motrix |
|---|---|---|
| Author / origin | DeepMind, fork at `unilabsim/mujoco_uni` | Motphys |
| Contact model | Soft-constraint solver | Penalty + Featherstone |
| Throughput on single CPU | Baseline | Often higher; multithread step |
| Snapshot / playback | Limited | Native, see Motrix capability docs |
| macOS video export | Requires extra tooling | Built-in |
| Determinism across versions | Tight | Improving |
| Asset support | XML / MJCF, MJX features | MJCF subset |

**The right reason to switch** is one of:

1. Your task uses a Motrix-only capability (e.g. video recording on macOS).
2. You need higher CPU throughput.
3. You want a robustness check: a policy that works on *both* backends is
   much more likely to transfer to real hardware than one that overfits to
   either.

**The wrong reason** is "I got curious." Each backend has its own quirks,
and supporting both costs time. Stick with one if your task list is
narrow.

## How to switch

UniLab does **not** support backend choice via runtime override. The
backend is part of the *task owner identity*:

```bash
# wrong — backend is not an override
uv run train --algo ppo --task go2_joystick_flat sim.backend=mujoco

# right — the owner YAML decides
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

The CLI resolves `--task <task>/--sim <backend>` to the owner YAML at
`conf/<algo>/<task>/<backend>.yaml`. If that file doesn't exist, the task
**does not support** the backend — see
{doc}`../../developer_guide/contracts/task_owner_config`.

## See also

- {doc}`owner_yaml_swap`
- {doc}`reward_parity_checks`
- {doc}`known_capability_gaps`

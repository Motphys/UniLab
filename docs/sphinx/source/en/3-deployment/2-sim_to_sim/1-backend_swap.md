# Backend Swap

UniLab supports two CPU physics backends: **MuJoCo** (via `mujoco-uni`) and
**Motrix** (via `motrixsim-core`). Both implement the same `SimBackend`
contract and the same env contract. Backend-specific behavior is exposed
through explicit methods and capability records.

| Axis | MuJoCo | Motrix |
|---|---|---|
| Backend class | `src/unilab/base/backend/mujoco/backend.py` | `src/unilab/base/backend/motrix/backend.py` |
| Playback capabilities | Physics-state playback in `get_play_capabilities()` | Native interactive renderer and native video capture in `get_play_capabilities()` |
| Height-field scan | Implements `create_hfield_scanner(...)` | Implements `create_hfield_scanner(...)` |
| DR capability reporting | `get_dr_capabilities()` | `get_dr_capabilities()` |

**The right reason to switch** is one of:

1. The target task has an owner YAML for the other backend.
2. The backend exposes the capability the workflow needs.
3. You want a sim-to-sim parity check before deployment or a backend change.

Switching is still a task-owner change, not an ad hoc runtime tweak.

## How to switch

UniLab does **not** support backend choice via passthrough Hydra overrides. The
backend is part of the *task owner identity* selected by `--task` and `--sim`:

```bash
# wrong — backend is not an override
uv run train --algo ppo --task go2_joystick_flat --sim motrix training.sim_backend=mujoco

# right — choose the backend with --sim
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
```

The CLI resolves `--algo`, `--task`, and `--sim` to an owner YAML such as
`conf/ppo/task/go2_joystick_flat/mujoco.yaml`. If that file doesn't exist, the
task **does not support** the backend — see
{doc}`../../4-developer_guide/2-contracts/3-task_owner`.

## See also

- {doc}`2-owner_yaml_swap`
- {doc}`4-reward_parity`
- {doc}`6-capability_gaps`

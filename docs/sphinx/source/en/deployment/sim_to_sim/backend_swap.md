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

UniLab does **not** support backend choice via runtime override. The
backend is part of the *task owner identity*:

```bash
# wrong — backend is not an override
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix training.sim_backend=mujoco

# right — the owner YAML decides
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
```

The `task=<task>/<backend>` override resolves to the owner YAML at
`conf/<algo>/<task>/<backend>.yaml`. If that file doesn't exist, the task
**does not support** the backend — see
{doc}`../../developer_guide/contracts/task_owner`.

## See also

- {doc}`owner_yaml_swap`
- {doc}`reward_parity`
- {doc}`capability_gaps`

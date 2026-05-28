# Choosing a Backend

UniLab selects the simulator through the task owner config. For direct script
usage, choose `task=<task>/<backend>` for PPO/APPO and
`task=<algo>/<task>/<backend>` for off-policy algorithms. Do not switch a run by
overriding `training.sim_backend` alone; that field is set by the owner YAML and
identifies the composed backend.

## Quick Choice

| Need | Prefer |
| --- | --- |
| Default path or broadest owner coverage | MuJoCo |
| Native interactive playback through the backend | Motrix |
| MuJoCo-only tools such as `scripts/play_viser.py` | MuJoCo |
| Task owner exists only under `conf/.../<task>/mujoco.yaml` | MuJoCo |
| Task owner exists under `conf/.../<task>/motrix.yaml` and the support matrix marks the combination as tested or configured | Motrix |

The support matrix is generated from registry, owner YAML, and tests; use it as
the current evidence source: {doc}`/zh_CN/user_guide/E-reference/01-backend-support-matrix`.

## Examples

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/motrix
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

`registry.make(..., sim_backend=None)` resolves the default backend in
`src/unilab/base/registry.py`; task-owner YAML remains the user-facing route.

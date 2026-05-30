# Choosing a Backend

UniLab selects the simulator through the task owner config. For normal usage,
choose the task and backend with `--task` and `--sim`; off-policy commands keep
the algorithm in `--algo`, not in `--task`. Do not switch a run by overriding
`training.sim_backend` alone; that field is set by the owner YAML and identifies
the composed backend.

## Quick Choice

| Need | Prefer |
| --- | --- |
| Default path or broadest owner coverage | MuJoCo |
| Native interactive playback through the backend | Motrix |
| MuJoCo-only tools such as `scripts/play_viser.py` | MuJoCo |
| Task owner exists only under `conf/.../<task>/mujoco.yaml` | MuJoCo |
| Task owner exists under `conf/.../<task>/motrix.yaml` and the support matrix marks the combination as tested or configured | Motrix |

The support matrix is generated from registry, owner YAML, and tests; use it as
the current evidence source: {doc}`/zh_CN/1-user_guide/5-reference/1-backend-support-matrix`.

## Examples

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo ppo --task go2_joystick_flat --sim motrix
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

`registry.make(..., sim_backend=None)` resolves the default backend in
`src/unilab/base/registry.py`; `--task` and `--sim` remain the user-facing
route.

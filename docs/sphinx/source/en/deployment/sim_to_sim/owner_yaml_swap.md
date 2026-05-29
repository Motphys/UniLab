# Adding a Backend YAML to an Existing Task

You have a task that runs on, say, Motrix. You want it on MuJoCo too. This
page is the recipe.

## What you DON'T do

- Add backend branching in Python. Backend-specific behaviour lives in the
  backend adapter and the owner YAML, never in env code. This is the
  ironclad rule of {doc}`../../developer_guide/contracts/backend_contract`.
- Set `training.sim_backend` as an override. That field is an **identity
  echo** of the owner YAML, not a switch.

## What you DO

1. Copy the existing owner YAML:
   ```bash
   cp conf/ppo/task/go2_joystick_flat/motrix.yaml \
      conf/ppo/task/go2_joystick_flat/mujoco.yaml
   ```
2. Inside the new file, set `training.sim_backend: mujoco`.
3. Adjust the **physical parameters** the new backend needs:
   - Contact friction / damping — MuJoCo uses solver parameters per
     contact pair; Motrix uses material props. Make sure your task asset
     file under `src/unilab/assets/robots/<robot>/` declares the values the
     backend needs.
   - Solver settings and timestep — keep them in the owner YAML or backend
     adapter that owns the behavior.
4. Re-resolve any backend-conditional DR ranges. Some randomizations are
   meaningful on one backend but no-ops on the other (e.g. friction
   damping coefficient).
5. Verify the env/backend pair is registered and importable through the
   registry bootstrap (see
   {doc}`../../developer_guide/architecture/registry`).

## Validation gate

Before claiming the backend is supported, you need (at minimum):

- One training run that reaches the same success threshold as the
  reference backend, *or* a documented reason it doesn't (e.g. capability
  gap).
- Reward parity check: see {doc}`reward_parity`.
- A test added under `tests/` that imports the env in the new backend and
  runs `reset` + `step(10)` without error.

::::{admonition} Evidence grades
:class: note
After validation, refresh the generated support data if the support status
changed:

```bash
uv run scripts/generate_support_matrix.py --write
```

See {doc}`/glossary` for the evidence-grade definitions.
::::

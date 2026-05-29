# Writing Providers

Task-level domain randomization providers live with the task env owner. They
sample task-specific state and return plans consumed by
`DomainRandomizationManager`.

## Provider Shape

Current provider examples define one or more of these plan methods:

- Build an init plan for model variants or geometry materialization.
- Return a reset plan with state updates and a reset randomization payload.
- Return an interval plan for push or body-force perturbations.

The shared types live in `src/unilab/dr/types.py`, and the manager lives in
`src/unilab/dr/manager.py`.

## Rules

- Keep XML, asset, and model metadata access on cold paths such as init,
  materialization, or cache creation.
- Do not probe backend private methods from env hot paths.
- Dispatch only fields that the backend declares through its DR capabilities.
- Put task-specific sampling in the task provider, not in training scripts.

## Evidence

Representative provider implementations are in:

- `src/unilab/envs/locomotion/go1/joystick.py`
- `src/unilab/envs/locomotion/g1/joystick.py`
- `src/unilab/envs/motion_tracking/g1/tracking.py`
- `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
- `src/unilab/envs/manipulation/sharpa_inhand/rotation.py`

Developer contract details are in
{doc}`../../developer_guide/contracts/dr_contract`.

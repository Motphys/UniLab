# Writing Providers

This page describes the legacy provider path: only the 3 Adapted families
(`sharpa_inhand` / `sharpa_inhand_grasp` / `go2_arm_manip_loco`) still declare
domain randomization through a task-level `DomainRandomizationProvider`.
Migrated Manager-Based tasks do not write providers; they declare randomization
through Hydra `events:` manager terms in the owner YAML (see {doc}`0-index`
and {doc}`1-configuration`).

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

Representative provider implementations are in (all on the Adapted-family
compatibility path):

- `src/unilab/tasks/locomotion/common/dr_provider.py` (`LocomotionDRProvider`,
  used by the go2_arm family)
- `src/unilab/tasks/locomotion/go2_arm/manip_loco.py`
- `src/unilab/tasks/manipulation/sharpa_inhand/rotation.py`

Developer contract details are in
{doc}`../../4-developer_guide/2-contracts/4-dr_contract`.

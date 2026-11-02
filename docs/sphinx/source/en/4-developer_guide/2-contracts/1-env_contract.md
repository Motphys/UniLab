# Env Contract

The environment contract is code-owned by `src/unilab/base/base.py` and
`src/unilab/base/torch_env.py`. Observation semantics are recorded in
{doc}`/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract` and superseded by
{doc}`/adr/ADR-0011-torch-only-manager-based-runtime`.

## Required Shape

- `TorchEnvState.obs` is `dict[str, torch.Tensor]`. It is not a flat tensor.
- The required actor observation key is `obs`.
- The only optional critic-only observation key is `critic`.
- `obs_groups_spec` maps each observation group name to its flat dimension.
  Wrappers and learners use this map to size actor and critic paths.
- `reset(env_indices)` accepts a one-dimensional Torch integer tensor (or
  `None`) and returns `(obs_dict, info_dict)` for the reset rows.
- `step(actions)` on `TorchEnv` accepts a contiguous float32 Torch tensor and
  returns `TorchEnvState`; external adapters may transfer tensors to a trainer
  device only at the adapter boundary.
- There is no legacy environment fallback. Unsupported devices, carriers, and
  backend tensor lifecycles fail closed.

## Owner Responsibilities

- Env code owns MDP semantics, observation construction, rewards, termination,
  truncation, reset behavior, and final-observation handling.
- When an async backend publishes a `CompletionToken`, `TorchEnv` uses that
  public receipt to order the current Torch stream before the next
  state-feedback Manager action/read/control write and before final Manager
  state update. Feed-forward stepping remains one batched backend call, and
  synchronous producers without a receipt perform no read gate. Task code must
  not override physics stepping or use backend-private streams/events for this
  ordering.
- Runners and learners must not invent critic observations by concatenating
  fields outside the env owner layer.
- If a third-party library still calls critic observations "privileged", keep
  that name translation inside the adapter. Inside UniLab, the key is `critic`.
- Manager terms declare tensor reads on the cold path. Ordinary
  `tensor_sensor_names` / `tensor_body_names` requests remain packed scalar/body
  reads. A term that consumes UniSim's ordered body-pair aggregation declares
  `tensor_body_pair_contact_names`; `EntityScene.compile_tensor_reads` cold-binds
  the ordered names on the requested Torch device and exposes the validated
  live batch through `SceneTensorReadPlan.body_pair_contact_view`. Access never
  binds or refreshes; reset, ordinary update, in-phase mutation repacking, and
  state-feedback substeps all call the plan's explicit refresh boundary.
  Event, reward, termination, and observation functions may declare this
  attribute. Declarations for one entity must use the identical ordered name
  tuple, which keeps slot semantics unambiguous without per-term rebinding.

## Evidence In Repo

- Env base contract: `src/unilab/base/base.py`
- Torch env lifecycle: `src/unilab/base/torch_env.py`
- RSL-RL adapter boundary: `src/unilab/rl/vec_env.py`
- Final observation helper: `src/unilab/base/final_observation.py`
- Tests: `tests/base/test_torch_env.py`,
  `tests/envs/test_manager_based_rl_env.py`,
  `tests/utils/test_final_observation.py`, `tests/ipc/`

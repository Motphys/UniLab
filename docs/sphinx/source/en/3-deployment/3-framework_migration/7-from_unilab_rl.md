# Migrating from unilab-rl (`uni_rl`)

This page is for users who installed the `uni_rl` extra
(`pip install unilab[uni_rl]`) or imported the `uni_rl` package directly.

The former `unilab-rl` distribution (import namespace `uni_rl`, repository
[unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)) is merged into
this repository under `unilab.rl` (issue #2372, decision in
{doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`). There is a single
`unilab` distribution: the APPO / SAC / FlashSAC runners, the async IPC
runtime, and the RL logging stack ship in the base install.

## Install changes

- Drop `--extra uni_rl` from `uv sync` commands and `unilab[uni_rl]` from
  pip installs; the RL stack needs no extra.
- The empty transitional `uni_rl` extra has been removed, so the old commands
  now fail with an unknown-extra error. Remove the extra from scripts and CI.
- Remove any standalone `unilab-rl` install from your environment
  (`pip uninstall unilab-rl`) so no stale copy of the old implementation
  lingers next to `unilab.rl`.
- CLI algorithm names, log paths, checkpoint fields, and metric schemas are
  unchanged.

## Import path mapping

| Old import (`uni_rl.*`) | New import |
|---|---|
| `uni_rl.env_contract` | `unilab.base.contracts.env` |
| `uni_rl.training_state` | no replacement (`unilab.training.experiment` owns checkpoint save/load) |
| `uni_rl.utils.observations` | `unilab.base.contracts.observations` |
| `uni_rl.algos.common.device` (probe) | `unilab.base.contracts.probe` |
| `uni_rl.algos.appo` / `.sac` / `.flash_sac` / `.common` | `unilab.rl.algos.*` (same subpackage names) |
| `uni_rl.algos.common.collector_timing` | `unilab.rl.runtime.collector_timing` |
| `uni_rl.ipc.async_runner` | `unilab.rl.runtime.async_runner` |
| `uni_rl.ipc.dp_launcher` | `unilab.rl.runtime.distributed.context` / `.launcher` / `.affinity`; the Hydra-side helper `apply_dp_rank_config` moved to `unilab.training.dp_config` |
| `uni_rl.ipc.dp_sync` | `unilab.rl.runtime.distributed.sync` |
| `uni_rl.ipc.*` (remaining modules) | `unilab.rl.runtime.ipc.*` |
| `uni_rl.offpolicy.*` | `unilab.rl.runtime.offpolicy.*` |
| `uni_rl.logging.*` | `unilab.rl.logging.*` |
| `uni_rl.utils.seed` | `unilab.rl.runtime.seed` (`apply_worker_seed`, `derive_worker_seed`; `TrainingSeedInfo`, `resolve_training_seed`, and parent-process `apply_training_seed` remain in `unilab.training.seed`) |
| `uni_rl.utils.nan_guard` | `unilab.training.tensor_diagnostics` (`NanGuard`); runner/collector code now receives an injected factory |
| `uni_rl.utils.tensor` | no replacement (the tensor-native collector no longer uses host conversion helpers) |
| `uni_rl.utils.device` | `unilab.rl.runtime.device` |
| `uni_rl.utils.tensor_runtime` | `unilab.rl.runtime.settings` / `unilab.rl.runtime.placement` (pure settings and placement); the Hydra resolver lives in `unilab.training.tensor_runtime` |

The `unilab.rl` package-level surface (`from unilab.rl import
RslRlVecEnvAdapter`, `UNILAB_DP_LOG_DIR`, `launch_torchrun_workers`, and the
other names in `src/unilab/rl/__init__.py`) is unchanged and resolved lazily.

Permanent `import uni_rl` support is **not** guaranteed; migrate to the
`unilab.rl.*` paths.

## Behavior changes

- The SAC / FlashSAC runner builders (`build_double_buffer_runner` in
  `unilab.rl.algos.sac.double_buffer` and `unilab.rl.algos.flash_sac.double_buffer`)
  are injection-based: the training
  assembly layer passes `runtime_settings`, `inference_placement`,
  `backend_device_resolver`, and `backend_device_configurer`; the old
  `backend_device_binder` parameter is removed. When omitted, runner-side
  defaults apply.
- Extension mechanisms are unchanged: `runtime_resolver`, `class_name`,
  `actor_adapter_modules`, and the actor registry keep working, and external
  dotted paths into your own packages remain valid. Only built-in dotted
  paths moved to the `unilab.rl.*` namespace.
- The boundary "uni_rl must not import UniLab" no longer applies; the RL
  stack now follows the internal layering rules of `unilab.rl` (the RL core
  does not depend on the env/training layers). See
  {doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl` for the layer DAG.

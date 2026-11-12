# Sync-Free Autoreset and Reset CUDA Graph

By default, `TorchEnv` resets done rows through row indices: it reads the done
mask on the host (`any()`), builds `env_ids` with `nonzero`, and launches reset
kernels whose shape depends on the number of reset rows. Each of these steps
synchronizes the host with the GPU.

An opt-in fixed-shape path removes these synchronizations, and on CUDA it can
replay the whole reset segment as one CUDA graph. Every switch defaults to
`False`, so tasks that do not opt in keep the selected-row behavior unchanged.

Implementation: `src/unilab/base/torch_env.py`,
`src/unilab/base/reset_selection.py`, `src/unilab/base/autoreset_graph.py`,
`src/unilab/base/host_transitions.py`. Tests:
`tests/envs/test_fixed_shape_autoreset.py`,
`tests/tasks/test_g1_motion_fixed_shape.py`.

## Requirements

- A Manager-based task whose every reset participant accepts a
  `ResetSelection` (see "Opting terms into `ResetSelection`" below).
  The G1 motion-tracking terms (`MotionCommand`, `JointPositionAction`, the
  motion body rewards and the anchor/body height terminations) already opt in.
- `training.defer_finite_validation=true`, the default in every shipped
  algorithm config.
- For the CUDA graph: a CUDA env device, and a backend that declares the
  `external_graph_capture` capability and implements
  `torch_graph_capture_scope`. MJWarp does both, from the UniSim release that
  ships `unisim.backend.graph_capture`.

## Configuration flags

All flags are fields of `ManagerBasedRlEnvCfg` (or of the base `EnvCfg` for the
backend flag). They live in the `env` section of the owner YAML. They are not
present in the shipped owner YAMLs, so set them with `++` from the command line:

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  ++env.masked_autoreset=true \
  ++env.persistent_step_buffers=true \
  ++env.autoreset_cuda_graph=true \
  ++env.mjwarp_defer_tensor_validation=true
```

| Flag | Default | Effect |
| --- | --- | --- |
| `masked_autoreset` | `False` | Runs the autoreset on every step as a full-width `reset(ResetSelection)` under the device done mask, with no `any()` gate and no `nonzero`. Requires `defer_finite_validation`. |
| `persistent_step_buffers` | `False` | Allocates the observation, reward, terminated and truncated carriers once and updates them with `copy_`. Consumers receive one of `published_buffer_depth` persistent publication slots. |
| `published_buffer_depth` | `2` | Size of the publication ring. A published state stays valid for `published_buffer_depth - 1` further steps. `0` publishes the working carriers themselves, which the next step overwrites. |
| `autoreset_cuda_graph` | `False` | Captures the masked autoreset segment, including the backend reset, into one `torch.cuda.graph` and replays it afterwards. Requires `masked_autoreset`, `persistent_step_buffers`, a CUDA device and a backend with external graph capture. |
| `autoreset_graph_warmup` | `2` | Eager executions before capture, so that lazily allocated buffers exist outside the graph. |
| `autoreset_graph_capture_mode` | `"thread_local"` | `torch.cuda.graph` capture error mode: `"global"`, `"thread_local"` or `"relaxed"`. |
| `mjwarp_defer_tensor_validation` | `False` | Latches MJWarp's fail-closed operand and reset-mask checks into device verdicts instead of reading each one on the host. The env publishes them at its deferred-validation poll. Inside a graph capture MJWarp always defers. The flag also removes the host reads from the eager steps outside the captured segment. UniLab forwards it only when it is `True`. |

Invalid combinations fail when the config is validated: for example,
`autoreset_cuda_graph` without `masked_autoreset` and `persistent_step_buffers`,
or `masked_autoreset` without `defer_finite_validation`.

## Behavior notes

**RNG draw order.** With `masked_autoreset` the reset lifecycle runs on every
step; an all-false mask leaves the state unchanged. Full-width samplers, such as
the `MotionCommand` resample or the reset-event interval times, therefore draw
from the Manager RNG on steps where no row is done. With the same seed,
trajectories differ from the selected-row path. Compare the two modes
statistically, not value by value; MJWarp is not bitwise deterministic in any
case.

**`final_observation`.** It is published on every step and is meaningful only
on done rows.

**Deferred validation polling.** Every fail-closed check in the reset segment is
latched on the device. This covers finite checks, `TorchEnv.require` verdicts,
reset-mask containment and MJWarp/packed-reset verdicts.
`poll_finite_validation` reads all of them together every
`training.finite_validation_interval` steps (32 in the shipped configs). A
failure is therefore raised at the next poll, up to `interval - 1` steps after
the faulty step. Packed-reset failures name the poll step
(`first detected at poll boundary, step N`). To localize a fault, rerun with
`training.finite_validation_interval=1`; the masked path still requires
`training.defer_finite_validation=true`.

**Stale diagnostic counters.** In masked mode the env has no host-side reset row
count or per-phase split. Consequences:

- The per-reset detail keys in `info["timing"]`
  (`RESET_DONE_DETAIL_TIMING_KEYS` in `src/unilab/base/backend_timing.py`, for
  example `reset_done_count`) are written as `0` on every step. Only
  `reset_done_reset_call_ms` is filled. A `0` there does not mean that no row
  reset.
- Episode logs (`Episode_Termination/*`, `Episode_Metrics/*` and command reset
  metrics) accumulate on the device. They are published into `info["log"]`
  only at the deferred-validation poll, and each value covers every episode
  that completed since the previous poll. Read them as interval aggregates,
  not per-step values.

**Host state under graph replay.** A CUDA graph replays device work only. Python
state that the reset segment changes, such as a pending-metrics flag or a host
lower bound, must be declared with
`unilab.base.host_transitions.record_host_transition`. The graph owner
re-applies these transitions after every replay. Undeclared host state keeps the
value it had at capture time.

## Opting terms into `ResetSelection`

A `ResetSelection` (`src/unilab/base/reset_selection.py`) is a persistent
full-width bool `mask` plus `row_ids = arange(num_envs)`. It is passed through
the existing `env_ids` reset arguments. A participant opts in by declaring
support and handling the selection as a masked full-width write.

- **Class-based terms** (commands, actions, class reward/termination/metric
  terms, class observation terms, stateful event terms) set
  `supports_reset_selection = True` and accept a `ResetSelection` in `reset`.
- **Function-style terms** (for example reset-mode event functions) use the
  `@reset_selection_capable` decorator, and their body must handle a
  `ResetSelection` argument.
- Write reset state with the helpers instead of indexing:
  - `fill_rows_(tensor, env_ids, value)` writes a constant;
  - `assign_rows_(tensor, env_ids, values)` writes a full-width tensor, or
    calls a factory `values(n)` that returns `n` rows (`num_envs` rows under a
    selection).

  Both accept row ids, `None`, a slice or a selection.
- Do not read the mask on the host (`bool(mask.any())`, `.item()`,
  `nonzero`) inside `reset`; that reintroduces the synchronization.

```python
from typing import ClassVar

import torch

from unilab.base.reset_selection import ResetSelection, fill_rows_
from unilab.envs.mdp.actions.actions import BaseAction


class MyAction(BaseAction):
    # Re-declare on every subclass that overrides reset().
    supports_reset_selection: ClassVar[bool] = True

    def reset(self, env_ids: torch.Tensor | ResetSelection | None = None) -> None:
        super().reset(env_ids)
        fill_rows_(self._my_state, env_ids, 0.0)
```

A declaration counts only at or below the class that defines the effective
`reset`. A subclass that overrides `reset` does not inherit its parent's opt-in
and is reported as a blocker until it declares support itself. A
`ResetSelection` is not a tensor, so unconverted code that receives one by
mistake fails at the first indexing operation instead of resetting wrong rows.

Some managers impose extra rules:

- `CurriculumManager` terms cannot run in a sync-free reset, because they
  publish host curriculum state.
- The reward, termination, observation and metrics managers check only their
  class-based terms. Stateless function terms need no declaration.

### Debugging `reset_selection_blockers`

With `masked_autoreset`, `init_state` calls
`ManagerBasedRlEnv.reset_selection_blockers()`. It collects one entry per
unsupported participant from every manager and raises:

```text
NotImplementedError: masked_autoreset needs every reset participant to accept a
ResetSelection; unsupported: EventManager: <term name> (<function name>); ...
```

Each entry has the form `<Manager>: <term name> (<function or class name>)`. A
manager that cannot take a selection for a structural reason reports a sentence
instead. One example is `TerminationManager: episode counts are read on the
host unless defer_finite_validation is enabled`.

To list blockers without starting a training run, construct the env with the
flag off and call the method:

```python
env.reset_selection_blockers()  # [] when the task can enable masked_autoreset
```

Then convert, replace or remove each named term. Reset-randomization events are
common blockers; the G1 motion-tracking FlashSAC owner
(`src/unilab/conf/flashsac/task/g1_motion_tracking/base.yaml`) is a DR-free
baseline for this reason.

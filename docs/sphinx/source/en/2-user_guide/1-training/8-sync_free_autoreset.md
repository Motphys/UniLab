# Sync-Free Autoreset and Reset CUDA Graph

By default, `TorchEnv` resets done rows through row indices: it reads the done
mask on the host (`any()`), builds `env_ids` with `nonzero`, and launches reset
kernels whose shape depends on the number of reset rows. Each of these steps
synchronizes the host with the GPU.

A fixed-shape path removes these synchronizations, and on CUDA it can
replay the whole reset segment as one CUDA graph. The path is controlled by a
single public switch, `env.autoreset_cuda_graph`, which defaults to `true`;
tasks that opt out with an explicit `false` keep the selected-row behavior.

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
- For the CUDA graph: a CUDA env device, and a backend that declares the
  `external_graph_capture` capability and implements
  `torch_graph_capture_scope`. MJWarp does both from `unisim-core` 2.7.2, the
  pinned version.

## Configuration flags

`env.autoreset_cuda_graph` is the single public switch and defaults to
`true`. It implies the whole fixed-shape chain; the intermediate switches
(`masked_autoreset`, `persistent_step_buffers`), the publication ring depth and
the graph warmup are internal — derived from this switch or owned as constants
by `TorchEnv` — and are no longer configuration keys. Supplying a deleted key
fails closed with a config error. The fixed-shape chain also requires the
deferred finite validation mode: `finite_validation_mode='deferred'` is the
default, and `autoreset_cuda_graph=true` with `finite_validation_mode='strict'`
fails closed at env construction. Set the switch to `false` explicitly to opt
out; that escape hatch is the fallback for debugging or when a graph-related
problem is suspected (the run then uses the selected-row eager path).

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp
```

| Flag | Default | Effect |
| --- | --- | --- |
| `autoreset_cuda_graph` | `True` | Runs the autoreset on every step as a full-width `reset(ResetSelection)` under the device done mask (no `any()` gate, no `nonzero`), keeps the observation, reward, terminated and truncated carriers persistent, defers every fail-closed finite check to the poll boundary, and — on CUDA — captures the masked autoreset segment, including the backend reset, into one `torch.cuda.graph` and replays it afterwards. Requires a CUDA device, a backend with external graph capture, and `finite_validation_mode='deferred'`. |
| `autoreset_graph_capture_mode` | `"thread_local"` | `torch.cuda.graph` capture error mode: `"global"`, `"thread_local"` or `"relaxed"`. |

The MJWarp backend options follow the env's single validation strictness
(`env.finite_validation_mode`, issue #2618): deferred mode latches MJWarp's
fail-closed operand and reset-mask checks into device verdicts (the removed
`mjwarp_defer_tensor_validation` switch) that the env publishes at its
deferred-validation poll, and also removes the host reads from the eager steps
outside the captured segment; strict mode keeps MJWarp's per-call operand check
immediate (`mjwarp_validate_tensor_finite`). Inside a graph capture MJWarp
always defers.

A published state stays valid for one further step (the ring holds two
persistent slots, which the off-policy double-buffer contract needs); the
capture happens after two eager warmup executions so that lazily allocated
buffers exist outside the graph.

Invalid combinations fail when the environment is constructed: for example,
`autoreset_cuda_graph` on a non-CUDA device, or a backend without the
`external_graph_capture` capability.

## Behavior notes

**RNG draw order.** On the fixed-shape path the reset lifecycle runs on every
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
`env.finite_validation_interval` steps (32 in the shipped configs). A
failure is therefore raised at the next poll, up to `interval - 1` steps after
the faulty step. Packed-reset failures name the poll step
(`first detected at poll boundary, step N`). To localize a fault, rerun with
`env.finite_validation_interval=1` so every step is a poll boundary.

**Stale diagnostic counters.** On the fixed-shape path the env has no
host-side reset row count or per-phase split. Consequences:

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

  Both accept row ids, `None`, or a selection.
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

With the fixed-shape path, `init_state` calls
`ManagerBasedRlEnv.reset_selection_blockers()`. It collects one entry per
unsupported participant from every manager and raises:

```text
NotImplementedError: masked_autoreset needs every reset participant to accept a
ResetSelection; unsupported: EventManager: <term name> (<function name>); ...
```

Each entry has the form `<Manager>: <term name> (<function or class name>)`. A
manager that cannot take a selection for a structural reason reports a sentence
instead. One example is `TerminationManager: episode counts are read on the
host unless finite_validation_mode='deferred'`.

To list blockers without starting a training run, construct the env with the
flag off and call the method:

```python
env.reset_selection_blockers()  # [] when the task can enable the fixed-shape path
```

Then convert, replace or remove each named term. Reset-randomization events are
common blockers; when a backend cannot negotiate them, the task owner must
disable those terms explicitly. The G1 motion-tracking owners keep the
canonical DR contract in
`src/unilab/conf/<algo>/task/g1_motion_tracking/base.yaml`, and backends without
reset-randomization support (for example the FlashSAC Newton owner) set the
unsupported event terms to `null` in their own leaf instead of forking the
contract.

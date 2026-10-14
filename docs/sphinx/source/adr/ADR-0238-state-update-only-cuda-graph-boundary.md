---
orphan: true
---

# ADR-0238 State-update-only CUDA graph boundary

语言: English

- Status: Proposed / issue-scoped candidate
- Date: 2026-10-14
- Owners: UniLab Manager runtime; UniSim backend contract owner
- Driving issue: #2338
- Scope: no-reset Manager state-update publication only; integration branch
  `dev/issue-2338-state-update-graph`
- Supersedes: None
- Superseded by: None

## Context

Issue #2311 closed no-go at its complete K=1 collector boundary. The evidence
does not show that Manager state-update compute itself is impossible to
capture; it shows that embedding MJWarp physics replay and reset publication
inside a caller-owned outer graph is not valid with the available public
boundaries. #2338 therefore defines a narrower candidate without changing
#2311's historical result or support claim.

The existing runtime already distinguishes the useful pieces:

- `ManagerBasedRlEnv.update_state()` publishes stable post-physics qpos, qvel,
  and sensor views through the public `SceneTensorReadPlan`;
- `SimBackend` already exposes public tensor lifecycle capabilities, step
  completion receipts, deferred validation, and graph lifecycle snapshots;
- Manager owners now retain persistent reward, observation-noise, command,
  motion-sampler, and validation carriers across eager steps;
- #2338 U1 extracts the fixed eager update body behind named owner methods and
  audits only that narrower boundary.

## Decision

Capture only the UniLab-owned, fixed-shape, no-reset Manager state-update
publication body after the backend step completion boundary. Policy inference,
action/control processing, physics, packed reset, post-reset observation,
semantic publication, replay ingress, and learner work stay outside.

### Exact boundary

Ordered outside the update graph:

1. learner policy inference and action transport;
2. action processing and control transform;
3. backend `step_tensor()` / MJWarp physics replay;
4. backend step completion receipt;
5. update graph replay;
6. update completion receipt;
7. external fixed-width reset orchestration and post-reset publication;
8. semantic logs, metrics, recorders, replay ingress, and learner updates.

Inside the first candidate:

- episode-step device carrier update;
- termination and timeout compute;
- terminated/truncated/reset-mask carrier publication;
- reward vector and reward-component carrier compute;
- metric compute that only writes persistent carriers;
- command dt, command update, and command post-compute;
- observation groups, history, noise, and mapping;
- persistent public observation/reward/done output carrier writes;
- persistent owner finite-verdict carrier writes.

The first candidate is the no-reset publication body. Reset preparation and
post-reset observations are not captured.

### Static operation inventory and classification

**Graph-eligible**

| Operation | Current owner | Disposition |
| --- | --- | --- |
| Episode-step carrier | `_advance_episode_steps` | Fixed-width device copy/add. |
| Termination and reset flags | `_compute_termination_and_reset_flags` | Fixed-width writes; no host mask read. |
| Reward and components | `_compute_rewards`, `RewardManager.compute` | Persistent full-width carriers. |
| Metric compute | `_compute_metrics` | Eligible only when owners write persistent carriers and publication is completion-owned. |
| Command update/post-compute | `_prepare_command_update`, `_compute_commands`, `_post_compute_commands` | Eligible only when no backend-state write/read repack occurs; otherwise preflight rejects. |
| Observations/history/noise | `_compute_observations`, `ObservationManager.compute` | Eligible only with pinned persistent term/output/noise carriers and recorded RNG order. |
| Public outputs | `_publish_step_outputs` | Deferred mode writes fixed carriers; eager clones remain outside graph mode. |

**Completion-owned / outside**

- backend step receipt;
- update completion token and counters;
- action/control/state/reset finite verdict publication;
- reward logs and component means;
- semantic log, metric publication, recorder, curriculum, replay ingress;
- Python timing dictionaries and update-state attribution;
- episode/global/Python control counters;
- `TorchEnvState` replacement and public transition construction;
- reset execution, terminal observations, and post-reset observations.

**Cold-path**

- mode selection and strict manifest construction;
- carrier allocation/layout validation;
- stable-view and stream preflight;
- capture identity allocation, warmup, capture, and teardown;
- RNG snapshot/version recording;
- invalidation/fallback bookkeeping.

**Rejected**

- any MJWarp physics replay or backend `step_tensor` call;
- packed reset execution or post-reset observation capture;
- policy/learner/replay-ingress work;
- eager output clones and eager invalid-row diagnostics;
- D2H transfer, `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `nonzero`, or
  `masked_select`;
- unclassified host bool reduction or data-dependent Python branch;
- capture-time allocation or allocator growth;
- backend-private graph/event/stream probing;
- observation layout recovery after drift;
- command/event paths that mutate backend state and require a mid-update read
  repack.

`tests/architecture/test_env_update_graph_static_audit.py` records the narrow
update-body audit. Its eager-only clone classifications are explicit, not
unclassified graph violations.

### Owner mode and manifests

#2338 must introduce a distinct cold-path owner mode:

```text
env_update_graph = disabled | eager | graph
```

It must not reuse or reinterpret `collector_graph`. `graph` is only a request.
A backend physics graph, physics READY lifecycle, or physics preflight verdict
never proves that the update graph is READY. Unsupported capabilities, malformed
manifests, owner rejection, or failed real capture fail closed. Explicit eager
fallback, if later allowed, is separately selected, counted, reason-bearing,
and cannot silently enable another mode.

The manifest needs its own contract version, capture identity, state, requested
and enabled flags, disable/invalidation reason, replay counters, backend step
receipt identity, update completion identity, RNG state/draw versions, and
carrier-layout evidence.

### Backend-first public preflight

Before UniLab consumes it, `SimBackend` must expose an update-specific public
preflight boundary, distinct from the existing generic graph-candidate preflight.
It must fail closed unless all of the following are true:

- public stable full-width qpos/qvel and requested sensor views exist;
- public step completion publication exists;
- selected Torch device is CUDA-resident and supported;
- declared stream/event ownership allows ordered execution after the backend
  step receipt;
- allocator readiness can be observed through the public boundary;
- backend-private probing is not required;
- deferred validation and persistent finite verdicts are supported.

The verdict must name the update boundary. It cannot be inferred from backend
physics graph enablement. A later `GraphCompletionToken` extension for an
`env_update` operation is also backend-first; the backend step receipt is an
ordering input and must never be relabeled as update completion evidence.

### Carrier contract

Before warmup and capture, every input/output address, shape, dtype, and device
is fixed:

- backend qpos/qvel and named sensor stable views;
- episode-step carrier;
- command state and motion frame/end state when command compute is captured;
- command dt;
- observation history and noise carriers;
- reward vector and reward components;
- terminated, truncated, reset mask, and public done carriers;
- observation-group public carriers;
- owner finite-verdict carriers;
- RNG evidence state and versions.

Carrier drift invalidates the graph. No graph-mode path may recover layout by
allocating replacement carriers.

### Stream and completion ordering

The only accepted lifecycle is:

```text
policy action (outside)
→ action/control transform (outside)
→ backend step graph replay
→ backend step completion receipt
→ ordered update-stream wait
→ update graph replay
→ update completion event/receipt
→ external reset/publication/ingress
```

The update graph publishes independent completion evidence with its own capture
identity and sequence. The backend receipt proves physics/state publication
ordering only. A stale backend receipt, mismatched capture identity, or update
replay failure rejects the transition and fails closed or executes the
pre-declared explicit fallback policy.

### RNG and warmup

If command/noise compute consumes RNG, the draw order is recorded before
capture. The complete pre-warmup semantic state is snapshotted; warmup may
advance device execution but cannot silently advance Python-side RNG accounting
or semantic state. Capture warmup restores the semantic state before the first
production replay unless the recorded draw is intentionally part of that
transition. State and draw versions enter every semantic comparison and run
manifest. Failed replay rolls back to the recorded boundary.

### Reset handling

The update graph may publish a reset mask, but reset remains external. The
external orchestrator consumes the fixed-width mask and must not branch on a
host bool mask read. Empty, partial, and full masks remain eager-equivalent and
exercise the same fixed-width reset boundary.

### Pre-registered Gate U

Semantic acceptance is new for #2338 and does not inherit #2311 Gate S.

It compares eager update execution with update-graph execution over:

- every observation group;
- observations containing history and configured noise;
- reward vector and all reward components;
- command state, motion frame/end state, and command metrics;
- terminated, truncated, and reset masks;
- every public validation verdict;
- episode/reset counters;
- RNG state and state/draw versions;
- capture warmup followed by replay;
- repeated deterministic replay;
- stale receipt/capture rejection;
- empty, partial, and full reset masks;
- fixed carrier addresses/shapes/dtypes/devices across warmup, capture, and
  replays.

Gate U additionally requires zero update-body static-audit violations and
runtime evidence that allocator statistics do not change across capture or
repeated replays. Throughput can never override this gate.

### Pre-registered Gate UP

Performance acceptance compares fresh-process eager versus update-graph paths
under both null and MPS execution. Each arm uses at least three paired,
interleaved repeats. Runs record both:

- update-stage attributable timing; and
- end-to-end `run_summary.json:tail_env_steps_per_sec`.

The default workload is the canonical FlashSAC / G1 motion tracking / MJWarp
owner configuration with `N=2048`, actor/critic observation groups, all reward
components, `algo.seed=1`, `env.seed=1`, `n_step=1`, and fixed null/MPS
orchestration. The final U5 decision must freeze exact workload shape, seed,
iterations, warmup/replay counts, report interval, host/driver/runtime versions,
and threshold before measurements are observed.

The initial threshold to beat is at least +10% median improvement in
update-stage attributable timing in every execution arm, with no end-to-end
regression and all safety observations valid. A performance no-go is valid and
must not become a support claim.

## Stable Contracts

- `env_update_graph` is distinct from `collector_graph`;
- backend update preflight is public and update-specific;
- update completion is independent from backend step completion;
- reset, replay ingress, and learner/inference work remain outside the first
  graph;
- graph mode fails closed until real capture succeeds;
- no backend-private probing or backend-name dispatch;
- RNG state/draw evidence and carrier layouts are manifest data;
- no UniLab release or graph support claim is authorized by this ADR.

## Alternatives Considered

- **Reuse `collector_graph`:** rejected as misleading because #2311 named and
  validated a complete collector boundary, not this no-reset update-only body.
- **Reuse generic backend graph preflight:** rejected because physics-graph
  readiness cannot prove update-boundary readiness.
- **Capture reset preparation too:** rejected as a separate future decision and
  contrary to the first #2338 candidate.
- **No update graph:** remains valid if U2/U3/U4/U5 find a structural barrier or
  performance no-go; independently valuable eager extraction and carrier work
  remains useful.

## Consequences

- U2 must add update-specific public backend preflight before UniLab consumes it.
- U3 must add an independent update capture identity and completion operation.
- Semantic/performance claims require the newly pre-registered gates.
- Eager semantics remain the baseline throughout U1.
- UniLab stays at version `1.3.5`; no UniLab tag, PyPI publication, GitHub
  release, or graph-enabled support claim is permitted while #2338 is open.

## Evidence In Repo

- Update-body extraction: `src/unilab/envs/manager_based_rl_env.py`
- Narrow static audit: `tests/architecture/test_env_update_graph_static_audit.py`
- Eager owner-order parity: `tests/envs/test_manager_based_rl_env.py`
- Stable read plan: `src/unilab/base/entity.py`
- Backend public contract: `../unisim/src/unisim/backend/base.py`
- MJWarp stable views and receipts:
  `../unisim/src/unisim/backend/mjwarp/backend.py`
- Driving issue: #2338
- Historical complete-collector no-go: #2311

## Related Documents

- {doc}`ADR Index </adr/README>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`

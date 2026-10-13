# ADR-0231 Integration-only K=1 MJWarp collector CUDA graph contract

语言: English

- Status: Proposed / integration-only
- Date: 2026-10-13; #2311 addendum 2026-10-14
- Owners: UniLab runtime, UniSim MJWarp backend, uni_rl off-policy collector
- Roadmaps: #2307 (historical contract), #2311 (M0 boundary addendum)
- Scope: integration-only; `dev/issue-2311-k1-collector-graph` until owners promote the result
- Supersedes: None
- Superseded by: None

## Context

The K=1 off-policy collector has a strict transition dependency:
`observation[t] -> learner action[t] -> env.step(action[t]) -> observation[t+1]`.
Phase A profiling on the canonical MJWarp / FlashSAC / G1 motion workload found addressable collector overhead of 19.54% in independent-context mode and 32.71% under MPS. The dominant blockers are manager update and episode reset dispatch, hot validation reductions, dynamic reset-row compaction, and reset-path temporary allocation. MJWarp already owns fixed-shape physics step/reset graphs, but that does not cover Manager-based observations, rewards, termination, events, commands, packed reset, or transition publication.

This is not a production commitment, no release artifact is published, and promotion to `main` remains an owner decision outside #2307.

Issue #2307 closed at Gate B no-go only because its Phase B boundary could not
remove the then-private backend/replay completion barriers. Issue #2309 has
since delivered and released those public contracts. This #2311 addendum
therefore fixes the present boundary without reopening either historical
result: the first implementation captures the **fixed Manager collector
publication subboundary**, not merely MJWarp physics, and not yet the complete
manager lifecycle for every task. The distinction is binding: the subboundary
still owns action/control transform, physics, all Manager observations and
rewards used by the canonical transition, termination, reset flags, packed
reset, post-reset observations, transition publication, and replay ingress. It
excludes only owner-declared semantic publications that preflight rejects for
this workload (interval events, metric/curriculum/recorder terms, manual reset,
and mapped multi-entity reset layouts), plus policy inference and learner
updates which must never enter this graph.

## Decision

Define an integration-only public capture boundary negotiated through `SimBackend` capabilities. The candidate consumes one preallocated action input of shape `[num_envs, action_dim]` and produces fixed-shape observation, reward, terminated, truncated, terminal-observation, and transition-publication carriers. The candidate is exactly K=1; policy inference and learner updates are outside capture.

### Public capability and preflight

`SimBackend` exposes a public capture candidate capability and preflight contract before any environment graph is constructed. Preflight requires fixed shapes/devices, declared stable buffers, supported Warp/Torch stream semantics, allocator readiness, RNG ownership, and capability support. Unsupported capabilities fail closed; UniLab never probes or branches on backend-private attributes.

For #2311 the public surface is the existing contract version 1. `SimBackend`
already declares `state_views`, `stepping`, fixed-width mask reset, deferred
fused validation, graph-completion publication, and graph lifecycle/invalidation.
`ManagerBasedRlEnv.preflight_graph_candidate()` layers on the owner carriers and
semantic-owner rejection list. No additional backend contract is required
before M1. The neutral `uni_rl` completion protocol, receipt gating, replay
ingress order/capture identity, and manifest schema are likewise sufficient for
the first publication handoff. An owner-controlled mode negotiation protocol
will be added in UniLab before graph execution; until then graph remains
disabled and this preflight is evidence only.

### Buffer and stream ownership

- UniLab policy/inference IPC owns action publication and signals action readiness to the collector graph stream.
- The collector graph owns in-graph action/control reads and all fixed output carriers.
- Warp physics and Torch manager kernels execute on the declared graph stream with explicit events at stream ownership transitions.
- Replay ingress owns fixed transition slots and consumes a graph-completion event; it may overlap only after that event and must preserve contiguous tick order.
- Learner sampling and policy inference never access in-flight carriers.
- Independent-context and MPS execution use the same ordering contract; MPS changes execution sharing, not semantics.

### #2311 M0 fixed operation inventory

**Inside the K=1 graph-eligible publication boundary**

| Operation | Owner path | Classification / disposition |
| --- | --- | --- |
| Fixed public action input `[N,29]` | `uni_rl.offpolicy.worker -> TorchEnv.step` | Graph input; inference publication remains outside. |
| Full-width action finite verdict and history shift | `TorchEnv._validate_action_contract`, `_record_action_finite`; `ActionManager.process_action` | Graph-eligible device compute; no host bool publication under deferred validation. |
| Motion affine/control transform | `MotionJointPositionAction.process_actions/apply_actions` | Graph-eligible fixed-shape device writes. |
| Backend control copy and MJWarp physics | `ManagerBasedRlEnv.apply_action`; `MjwarpBackend.step_tensor` | Graph-eligible public backend boundary. |
| Scene qpos/qvel/sensor view refresh | `SceneTensorReadPlan.refresh`, stable device views | Graph-eligible stable-address refresh; no per-term repack. |
| Termination flags, timeout/truncation, reset mask | `TerminationManager.compute`, `ManagerBasedRlEnv._update_state_in_read_phase`, `TorchEnv._compute_truncated` | Graph-eligible fixed-width compute. |
| Reward vector and 11 component carriers | `RewardManager.compute`, `_publish_reward_components` | Graph-eligible fixed compute and publication. |
| Actor/critic observations (160/289 columns) and terminal observations | `ObservationManager.compute/compute_group`, persistent output carriers, `_ensure_final_observation_scratch` | Graph-eligible persistent carriers; rotating manager scratch must be pinned for capture. |
| Versioned Torch RNG draws | `TorchManagerRng`, `TensorMotionSampler.sample_frames`, `MotionCommand._resample_command`, observation noise | Graph-eligible only in the recorded draw order; state/draw versions remain outside for rollback. |
| Full-width qpos/qvel staging and packed mask reset | `ResetStateTransaction`, `TorchPackedResetComposer`, `MjwarpBackend.set_state_tensor_mask` | Graph-eligible compute subboundary; MJWarp's current eager device-wide sync and any future full refresh remain explicitly rejected for capture. |
| Post-reset full-width observations | `ManagerBasedRlEnv.reset`, Manager reset computes, `torch.where` scatter | Graph-eligible only for the preflight-clean canonical owner. |
| Transition copy into fixed replay ingress slots | `ReplayBuffer._add_to_ingress` | Graph-eligible device copies after receipt visibility. |
| One ordered step completion receipt and capture identity | `MjwarpBackend._graph_completion`; `GraphCompletionToken` | Graph output; replay waits it before first read. |
| Ingress publication event and contiguous tick counters | `record_ingress_publication`, replay publish/release sequences | Completion-owned, not replay-capture-eligible; proves read-after-write ordering. |

**Interval-published / completion-owned**

- action/control/state/reset finite verdicts (`publish_validation_verdicts`);
- reward component and reward means (`publish_reward_log`, TensorCollectorMetrics);
- episode counters, done/timeout counts, replay ingress diagnostics, backend and
  graph lifecycle manifests;
- failure-only exact invalid-row diagnostics;
- recorder, metric, curriculum, command-metric, and semantic log publication.

**Cold-path**

- owner YAML composition, registry/env/backend construction, model/materialization;
- capture identity allocation, carrier allocation/layout checks, preflight;
- mode selection/switching, manifest serialization, invalidation and teardown;
- seeded initialization and rollback state installation.

**Rejected from capture**

- policy action publication and all learner sampling/updates;
- learner/inference ring waiting and action transport;
- D2H transfer, host sync, scalar conversion, dynamic row compaction,
  data-dependent Python branches, capture-time allocation;
- backend-private graph/event/stream inspection;
- non-autoreset manual-reset diagnostics;
- public observation carrier layout recovery after shape/device/dtype drift;
- eager reward invalid-row diagnostics at a non-deferred failure boundary;
- MJWarp's current mask-reset device-wide synchronization and full-refresh
  forward: that reset compute segment remains completion-owned/eager until a
  public capture-safe reset boundary exists.
These classified eager-only paths are pinned outside capture by
`tests/architecture/test_graph_candidate_static_audit.py`; they are not hidden
violations of the graph candidate.

### #2311 M0 fixed carrier layout

All addresses are fixed before preflight/capture and live on the selected CUDA
device unless explicitly marked CPU:

- action input: `float32[N,29]`;
- control: `float32[N,29]`;
- observations: `float32[N,160]` actor and `float32[N,289]` critic;
- reward: `float32[N]`;
- reward components: `float32[N,11]`;
- terminated/truncated/reset mask: `bool[N]` each;
- terminal observation and post-reset observation groups: one `float32[N,dim]`
  carrier for each observation group;
- validation verdicts: persistent scalar/full-width device bool carriers named by
  their owners;
- transition publication: full-width obs, critic, next-obs, next-critic, action,
  reward, done, truncated, terminal mask, and terminal observation columns copied
  into replay-owned fixed ingress slots (two slots of `N` rows);
- replay ingress handoff: ordered step receipt, active non-empty capture identity,
  monotonically nondecreasing sequence, stream identity, and ingress publication
  event;
- RNG evidence: CPU state tensor plus device label, monotonic state version and
  draw version.

The final semantic/performance parity consumer must add comparison-only clones
outside capture; those copies are evidence carriers, not candidate inputs or
outputs.

### #2311 pre-registered gates

Gate S requires eager/graph equality over observations by group, rewards and all
components, terminated/truncated, episode and reset counters, actions/commands,
recorder trace, reward means, curriculum state, metrics, terminal/reset/post-reset
observations, public qpos/qvel, RNG state and versions, lifecycle snapshot and
capture identity, transition layout/order, replay ingress commit order, failed
replay no-transition rollback, invalidation/fallback behavior, empty/partial/full
masks, contiguous final-observation publication, equal replay publish/release
counters, no stale/reordered transitions, matching capture identity, and fail-closed
malformed/unsupported diagnostics. It runs on null and MPS with eager, graph, and
capture-warmup/replay trajectories and requires all three static audits zero.

Gate P uses fixed null and MPS arms, three fresh-process paired repeats per mode,
interleaved graph/eager, 300 iterations, and the last-100 aggregate median of
`run_summary.json:tail_env_steps_per_sec`. Graph passes only if at least one arm
improves by at least +10% and every safety observation holds. A failing arm must
fail closed or explicitly fall back with a manifest reason; one passing mode does
not authorize another. Throughput can never override Gate S. No threshold, shape,
K, semantic surface, or mode rule may change after results are observed.

### Lifecycle and invalidation

The lifecycle is: allocate persistent carriers, eager warmup, preflight, capture, replay, completion publication, invalidation, fallback/fail-closed. Graph state, contract version, disable reason, fallback count, replay ordering counters, and stream/RNG evidence enter the run manifest. Capture is invalidated on model/data/view/capacity/policy-I/O change, backend graph invalidation, or stream/event failure.

Issue #2311 adds the owner-selected cold-path mode
`collector_graph = disabled | eager | graph`. `graph` may be requested only after
the public preflight verdict succeeds; unsupported/malformed states fail closed;
runtime fallback is explicit, counted, reason-bearing, and never silently enables
another mode. Mode never changes inside a control step. `enabled=true` means the
collector publication graph is active, not merely that MJWarp physics graphs
exist.

### #2311 workload fixation

The canonical frozen workload is FlashSAC / G1 motion tracking / MJWarp with:
`N=2048`; action `[2048,29]`; observation groups actor/critic `[2048,160]` and
`[2048,289]`; the eleven canonical motion/action/contact reward components;
terminated, truncated, and reset masks `[2048]`; seed `1` (one Manager Torch
generator); no interval events, metric, curriculum, recorder, manual-reset, or
mapped multi-entity reset owners; `training.env_steps_per_sync=1`;
`inference_slot_capacity=1`; FlashSAC `n_step=1`; replay capacity `512*2048`
rows; ingress depth `2`; ingress slot rows `2048`.

Benchmark commands are fixed as fresh-process, same-head null and MPS runs of:
`uv run --no-sync train --algo flashsac --task g1_motion_tracking --sim mjwarp algo.seed=1 env.seed=1 training.collector_graph=<disabled|eager|graph> algo.max_iterations=300 training.no_play=true`
plus the #2311-prescribed paired repeat/interleave orchestration. The Manager
RNG has its own `env.seed`; `algo.seed` alone does not fix it. Every run records
the resolved seed and exact draw versions. Graph and eager must not be mixed
within one distribution. The recorded host is
`GPU-db468bab-1b15-2a0b-b5e6-7716e512d422` (RTX 4090), driver `580.178.04`,
CUDA runtime `13.0`, PyTorch `2.14.0+cu130`, Warp `1.16.0`, MuJoCo/Warp
`3.11.0`; each final gate run must re-record exact values rather than relying on
this M0 snapshot.

The frozen RNG/draw order is: the initial randomized episode-length draw during
`init_state`; then on each reset joint-default bias, resampling time, adaptive
motion sampler (`multinomial` bins then uniform offsets), pose, velocity, and
joint noise; then actor observation noise during observation compute. A failed
or invalidated replay restores the complete pre-boundary generator state and
never advances this order silently.

### Validation and semantics

Hot finite/shape checks become device-resident fixed packs. A failed device verdict fails graph completion; exact host diagnostics run at the failure boundary. Reward logs, episode metrics, termination diagnostics, reset counts, and other semantic publications move to interval/completion boundaries without deletion. Final observation, terminal transition ordering, and post-reset observation semantics remain identical to eager execution.

Torch RNG state is consumed in a deterministic capture-safe order. RNG state/offset is versioned; replay or invalidation resumes from the same recorded state and never silently advances consumption.

### Fallback policy

Malformed manifests or unsupported capabilities fail closed before capture. Runtime graph failure permits explicit eager fallback only when no transition was published for the failed replay and the manifest records reason, counts, and deterministic state rollback. Otherwise execution fails closed.

### K=1 boundary

K>1, pre-committed action sequences, stale/replay inference, and speculative transitions are outside this contract. FlashSAC `n_step=1` is TD bootstrapping, not action-block K.

## Consequences

- Graph-safe cleanup must land in owner modules before MVP; no backend-private probing is permitted.
- The contract is testable independently of performance acceptance.
- Integration branches may implement the bounded canonical workload only after Gate B static and semantic audits pass.
- Successful Gate D support is integration-only until owners promote it; failing performance remains a valid no-go outcome.

- Backend capability negotiation remains public through `unisim.backend.base.SimBackend`; backend-private probing is prohibited.
- Manager-Based training semantics, reset masks, terminal observations, transition ordering, semantic logs, and episode statistics remain eager-equivalent.
- Policy inference, learner updates, K>1 action blocks, pre-committed action sequences, and stale/replay inference remain outside capture.
- Manifests fail closed and expose graph version/state/reason/fallback counters.
- UniLab package version remains `1.3.5`; no UniLab tag, PyPI publication, or
  GitHub release is permitted while #2311 is open, and no graph-enabled support
  claim may be made before both pre-registered gates pass.

## Alternatives Considered

- **Expand MJWarp physics graphs only:** rejected because Phase A shows Manager update/reset and publication overhead outside the existing physics graph.
- **Capture the learner update cycle too:** rejected because policy/model updates require outside-graph ownership and would violate the bounded collector roadmap.
- **No graph route:** remains a valid Phase B/C/D no-go if cleanup, contract, semantic, or performance gates fail.

## Evidence In Repo

- Roadmap evidence: issue #2307 Phase A aggregate
- Manager runtime: `src/unilab/envs/manager_based_rl_env.py`
- Reset transaction: `src/unilab/base/reset_state.py`
- Motion owner: `src/unilab/tasks/motion_tracking/common/manager_terms.py`
- MJWarp public backend: `../unisim/src/unisim/backend/mjwarp/backend.py`
- Collector owner: `../unilab_rl/src/uni_rl/offpolicy/worker.py`
- #2309 public lifecycle tests: `tests/envs/test_manager_based_rl_env.py`
- Canonical owner preflight: `tests/envs/test_env_configs.py`
- UniLab static audit: `tests/architecture/test_graph_candidate_static_audit.py`
- UniSim static audit: `../unisim/tests/adapters/mjwarp/test_graph_candidate_static_audit.py`
- uni_rl ingress audit: `../unilab_rl/tests/ipc/test_replay_ingress_static_audit.py`
- Frozen task owner: `src/unilab/conf/flashsac/task/g1_motion_tracking/mjwarp.yaml`

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`ADR-0012 Sole Tensor Manager And Scoped Backends </adr/ADR-0012-sole-tensor-manager-and-scoped-backends>`
- {doc}`ADR-0013 CUDA MPS Single-Rank Execution Sharing </adr/ADR-0013-cuda-mps-single-rank-execution-sharing>`

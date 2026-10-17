"""Canonical TensorBoard and Weights & Biases scalar schema.

Names that exist in upstream RSL-RL are used verbatim. Algorithm-specific
fields that RSL-RL does not expose use the same terse top-level groups:
``Train``, ``Loss``, ``Policy``, ``Episode``, and ``Perf``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

METRIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MetricSpec:
    """Persisted scalar metadata."""

    tag: str
    owner: str
    unit: str
    aggregation: str
    step_axis: str
    description: str
    backends: tuple[str, ...] = ("tensorboard", "wandb")
    distributed_aggregation: str = "cross-rank mean"


@dataclass(frozen=True)
class MetricFamilySpec:
    """Open family metadata for owner-defined terms."""

    prefix: str
    owner: str
    unit: str
    aggregation: str
    step_axis: str
    description: str
    backends: tuple[str, ...] = ("tensorboard", "wandb")
    distributed_aggregation: str = "cross-rank mean"


def _spec(
    tag: str,
    owner: str,
    unit: str,
    aggregation: str,
    description: str,
    *,
    distributed_aggregation: str = "cross-rank mean",
) -> MetricSpec:
    return MetricSpec(
        tag=tag,
        owner=owner,
        unit=unit,
        aggregation=aggregation,
        step_axis=_STEP_AXIS,
        description=description,
        distributed_aggregation=distributed_aggregation,
    )


def _count(
    tag: str,
    owner: str,
    unit: str,
    aggregation: str,
    description: str,
) -> MetricSpec:
    return _spec(
        tag,
        owner,
        unit,
        aggregation,
        description,
        distributed_aggregation="cross-rank sum",
    )


def _ms(tag: str, owner: str, aggregation: str, description: str) -> MetricSpec:
    return _spec(tag, owner, "ms", aggregation, description)


def _learner_ms(tag: str, phase: str, description: str) -> MetricSpec:
    return _ms(
        tag,
        "learner",
        f"sum in the logged iteration; {phase}",
        description,
    )


def _collector_ms(tag: str, description: str) -> MetricSpec:
    return _ms(
        tag,
        "collector",
        "latest collector-report mean (APPO uses an EMA); stale until the next report",
        description,
    )


_STEP_AXIS = (
    "collector-reported total env steps with iteration fallback for OffPolicyLogger; "
    "zero-based iteration for OnPolicyLogger"
)


METRIC_SPECS: dict[str, MetricSpec] = {
    spec.tag: spec
    for spec in (
        _spec(
            "Train/iteration",
            "runner",
            "iterations",
            "current runner iteration",
            "Iteration coordinate retained when the primary backend step is env steps.",
            distributed_aggregation="rank-0 scalar",
        ),
        _spec(
            "Train/mean_reward",
            "collector",
            "reward units",
            "latest mean of raw environment returns for the most recent 100 completed episodes",
            "Upstream-RSL-RL-compatible episode return; excludes timeout-bootstrap corrections.",
            distributed_aggregation="cross-rank mean of per-rank 100-episode means",
        ),
        _spec(
            "Train/mean_episode_length",
            "collector",
            "env steps/episode",
            "latest mean of the most recent 100 completed episode lengths",
            "Upstream-RSL-RL-compatible episode length field.",
            distributed_aggregation="cross-rank mean of per-rank 100-episode means",
        ),
        _spec(
            "Episode/timeout_rate",
            "collector",
            "ratio",
            "timeouts / completed episodes since the previous report containing a completion",
            "Omitted until the first completion; stale-latest between such reports.",
            distributed_aggregation="cross-rank mean of per-rank rates",
        ),
        _spec(
            "Loss/surrogate",
            "learner",
            "dimensionless objective",
            "mean over learner updates emitted in the logged iteration",
            "Upstream-RSL-RL-compatible PPO/APPO surrogate objective.",
        ),
        _spec(
            "Loss/value",
            "learner",
            "dimensionless objective",
            "mean over learner updates emitted in the logged iteration",
            "Upstream-RSL-RL-compatible PPO/APPO value objective.",
        ),
        _spec(
            "Loss/entropy",
            "learner",
            "nats",
            "final emitted actor update for deferred SAC learners; otherwise mean over emitted updates",
            "Upstream-RSL-RL-compatible entropy estimate.",
        ),
        _spec(
            "Loss/actor",
            "learner",
            "dimensionless objective",
            "final emitted actor update for deferred learners; otherwise mean over emitted updates",
            "SAC actor objective; RSL-RL has no SAC actor-loss name.",
        ),
        _spec(
            "Loss/critic",
            "learner",
            "dimensionless objective",
            "final emitted critic update for deferred learners; otherwise mean over emitted updates",
            "SAC/Q critic objective; kept separate from PPO value loss.",
        ),
        _spec(
            "Loss/temperature",
            "learner",
            "dimensionless objective",
            "final emitted temperature update for deferred learners; otherwise mean over emitted updates",
            "SAC temperature objective.",
        ),
        _spec(
            "Loss/learning_rate",
            "learner",
            "learning-rate hyperparameter",
            "latest learner update value in the logged iteration",
            "Upstream-RSL-RL-compatible optimizer-state tag.",
        ),
        _spec(
            "Policy/mean_std",
            "learner",
            "action units",
            "post-update policy state for the logged iteration",
            "Upstream-RSL-RL-compatible Gaussian policy standard deviation.",
        ),
        _spec(
            "Policy/temperature",
            "learner",
            "dimensionless",
            "post-update temperature state for the logged iteration",
            "SAC alpha/temperature state; RSL-RL has no SAC equivalent.",
        ),
        _spec(
            "Train/global_gradient_norm",
            "learner",
            "parameter norm",
            "mean over learner updates emitted in the logged iteration",
            "Combined-parameter pre-clip gradient norm.",
        ),
        _spec(
            "Train/actor_gradient_norm",
            "learner",
            "parameter norm",
            "final emitted actor update for deferred learners; otherwise mean over emitted updates",
            "Actor pre-clip gradient norm.",
        ),
        _spec(
            "Train/critic_gradient_norm",
            "learner",
            "parameter norm",
            "final emitted critic update for deferred learners; otherwise mean over emitted updates",
            "Critic pre-clip gradient norm.",
        ),
        _spec(
            "Train/target_q_max",
            "learner",
            "discounted-return units",
            "final emitted critic update for deferred learners; otherwise mean over emitted updates",
            "Maximum target-Q batch value.",
        ),
        _spec(
            "Train/target_q_min",
            "learner",
            "discounted-return units",
            "final emitted critic update for deferred learners; otherwise mean over emitted updates",
            "Minimum target-Q batch value.",
        ),
        _spec(
            "Train/reward_scale_std",
            "learner",
            "discounted-return units",
            "final emitted learner update value in the logged iteration",
            "Reward-normalization scale standard deviation; emitted only when enabled.",
        ),
        _count(
            "Train/ring_available_slots",
            "runner",
            "slots",
            "current value when the learner arrives at the ring",
            "Rollout-ring available slots before draining.",
        ),
        _count(
            "Train/rollouts_read",
            "runner",
            "rollouts",
            "count in the logged iteration",
            "Rollouts drained from the ring in this iteration.",
        ),
        _spec(
            "Train/replay_ingress_depth",
            "ipc/replay",
            "slots",
            "configured constant at the logged iteration",
            "Configured bounded replay-ingress ring depth.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Train/replay_ingress_occupancy",
            "ipc/replay",
            "slots",
            "published-minus-released snapshot at the logged iteration",
            "Live bounded replay-ingress occupancy, including transfers awaiting commit.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Train/replay_ingress_high_water",
            "ipc/replay",
            "slots",
            "monotonic maximum since replay ingress creation",
            "Bounded replay-ingress occupancy high-water.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Train/replay_ingress_backpressure_wait_ms",
            "ipc/replay",
            "ms",
            "cumulative producer-side bounded-ingress wait time",
            "Wall time replay producers spent waiting for an ingress slot.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Train/replay_ingress_dropped_batches",
            "ipc/replay",
            "batches",
            "cumulative early returns at closed or stopping ingress",
            "Transition batches dropped instead of published to replay.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Train/approx_kl",
            "learner",
            "nats",
            "mean over learner updates emitted in the logged iteration",
            "Target-to-current KL estimate.",
        ),
        _spec(
            "Train/clip_fraction",
            "learner",
            "ratio",
            "mean over learner updates emitted in the logged iteration",
            "Fraction of PPO ratios outside the clip range.",
        ),
        _spec(
            "Train/behavior_to_current_log_prob_delta",
            "learner",
            "nats",
            "mean over learner updates emitted in the logged iteration",
            "mean(behavior_log_prob - current_log_prob); signed diagnostic, not KL.",
        ),
        _spec(
            "Train/vtrace_rho_clip_fraction",
            "learner",
            "ratio",
            "mean over processed rollout transitions",
            "Fraction of V-trace importance ratios above the clip bound.",
        ),
        _spec(
            "Train/vtrace_rho_p99",
            "learner",
            "ratio",
            "p99 over a bounded rollout-transition sample",
            "Raw V-trace importance-ratio p99.",
        ),
        _spec(
            "Perf/total_fps",
            "runner",
            "env steps/s",
            "env steps in the logged iteration / iteration wall time",
            "Upstream-RSL-RL-compatible end-to-end runner throughput.",
            distributed_aggregation="cross-rank sum",
        ),
        _spec(
            "Perf/iteration_time",
            "runner",
            "s",
            "measured complete iteration wall time",
            "Runner iteration wall time; omitted when no complete wall timer is available.",
        ),
        _spec(
            "Perf/collection_time",
            "collector",
            "s",
            "most recently completed rollout phase",
            "Upstream-RSL-RL-compatible collection phase timing; one complete rollout "
            "(all vectorized env steps), unlike the per-step Perf/collector_env_step_ms EMA.",
        ),
        _spec(
            "Perf/learning_time",
            "learner",
            "s",
            "learner update computation wall time in the logged iteration",
            "Upstream-RSL-RL-compatible learning phase timing.",
        ),
        _ms(
            "Perf/dp_gradient_sync_ms_per_rank",
            "ipc/dp_sync",
            "sum over gradient collectives in the logged iteration",
            "Synchronous gradient-all-reduce wall time.",
        ),
        _spec(
            "Perf/dp_gradient_sync_calls_per_rank",
            "ipc/dp_sync",
            "calls",
            "count of gradient collectives in the logged iteration",
            "Gradient collectives issued by this rank.",
        ),
        _learner_ms(
            "Perf/learner_collector_wait_ms",
            "mutually exclusive phase in the selected timing profile",
            "Wait for the next collector result.",
        ),
        _learner_ms(
            "Perf/learner_inference_ms",
            "mutually exclusive phase and container of inference details",
            "Learner-side policy inference wall time.",
        ),
        _learner_ms(
            "Perf/learner_collector_release_ms",
            "mutually exclusive phase in the SAC-family profile",
            "Release a collector result after inference.",
        ),
        _learner_ms(
            "Perf/learner_replay_batch_wait_ms",
            "mutually exclusive phase in the SAC-family profile",
            "Wait for a replay batch to become ready.",
        ),
        _learner_ms(
            "Perf/learner_replay_stage_ms",
            "mutually exclusive phase in the APPO profile",
            "Stage rollout data for learner replay.",
        ),
        _learner_ms(
            "Perf/learner_replay_sample_ms",
            "mutually exclusive phase in the selected timing profile",
            "Sample replay rows for an update.",
        ),
        _learner_ms(
            "Perf/learner_weight_publish_ms",
            "mutually exclusive phase in the APPO profile",
            "Publish updated policy weights.",
        ),
        _learner_ms(
            "Perf/learner_inference_h2d_ms",
            "nested detail of learner inference",
            "Copy inference observations to the learner device.",
        ),
        _learner_ms(
            "Perf/learner_inference_forward_ms",
            "nested detail of learner inference",
            "Execute learner-side policy inference.",
        ),
        _learner_ms(
            "Perf/learner_inference_d2h_ms",
            "nested detail of learner inference",
            "Copy actions from the learner device.",
        ),
        _learner_ms(
            "Perf/replay_ingress_h2d_submit_ms",
            "nested replay ingress diagnostic",
            "Submit asynchronous replay ingress transfers.",
        ),
        _collector_ms("Perf/collector_mlp_infer_ms", "Collector MLP inference time."),
        _collector_ms("Perf/collector_inference_request_ms", "Request policy inference."),
        _collector_ms("Perf/collector_learner_action_wait_ms", "Wait for learner actions."),
        _spec(
            "Perf/collector_inference_queue_depth",
            "collector",
            "requests",
            "published observations not yet copied by the learner at the report",
            "Dynamic observation backlog; configured ring capacity is reported in the runtime manifest.",
        ),
        _spec(
            "Perf/collector_inference_action_backlog",
            "collector",
            "requests",
            "published actions not yet consumed by the collector at the report",
            "Distinguishes response backlog from observation publication backlog.",
        ),
        _spec(
            "Perf/collector_inference_max_action_backlog",
            "collector",
            "requests",
            "maximum action backlog since the previous collector report",
            "Observed response backlog without attributing speculative capacity to a serial chain.",
        ),
        _spec(
            "Perf/collector_inference_in_flight",
            "collector",
            "requests",
            "latest published-but-unconsumed inference count",
            "Current outstanding requests after the collector's latest action consumption.",
        ),
        _spec(
            "Perf/collector_inference_max_in_flight",
            "collector",
            "requests",
            "maximum in-flight count since the previous collector report",
            "Observed depth without attributing speculative capacity to a serial transition chain.",
        ),
        _spec(
            "Perf/collector_inference_publication_lag",
            "collector",
            "ticks",
            "latest published tick minus consumed tick",
            "Current ordered publication backlog in inference ticks.",
        ),
        _spec(
            "Perf/collector_inference_max_publication_lag",
            "collector",
            "ticks",
            "maximum publication lag since the previous collector report",
            "Observed ordered backlog without converting it into an out-of-order environment action.",
        ),
        _collector_ms(
            "Perf/collector_env_step_ms",
            "Single vectorized environment step wall time. One complete rollout is "
            "steps_per_env of these steps plus rollout bookkeeping; compare rollout "
            "cost against Perf/collection_time, not this per-step EMA.",
        ),
        _collector_ms(
            "Perf/collector_env_step_action_validate_ms",
            "Validate public policy actions.",
        ),
        _collector_ms(
            "Perf/collector_env_step_apply_action_ms",
            "Process actions and publish control tensors.",
        ),
        _collector_ms(
            "Perf/collector_env_step_action_backend_ms",
            "Apply actions and execute the backend step.",
        ),
        _collector_ms("Perf/collector_env_step_backend_ms", "Backend environment step."),
        _collector_ms(
            "Perf/collector_env_step_update_state_ms",
            "Update environment-owned state.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_read_boundary_ms",
            "Invalidate and refresh state at the update-state boundary.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_queue_drain_ms",
            "Synchronize pending backend work at the start of update-state.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_termination_ms",
            "Compute termination and reset flags.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_reward_ms",
            "Compute reward terms.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_metrics_ms",
            "Compute environment metrics.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_events_ms",
            "Apply runtime events.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_command_ms",
            "Compute command terms.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_observation_ms",
            "Compute observation terms.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_map_ms",
            "Map manager observations to policy groups.",
        ),
        _collector_ms(
            "Perf/collector_env_step_update_state_publish_ms",
            "Publish mapped observations and transition outputs.",
        ),
        _collector_ms(
            "Perf/collector_env_step_reset_done_ms",
            "Process resets and done observations.",
        ),
        _collector_ms(
            "Perf/collector_transition_extract_ms",
            "Extract one transition from the environment state without crossing the replay boundary.",
        ),
        _collector_ms(
            "Perf/collector_replay_write_ms",
            "Write transitions to replay storage.",
        ),
        _collector_ms(
            "Perf/collector_metrics_publish_ms",
            "Update and publish collector episode metrics.",
        ),
    )
}


METRIC_FAMILIES: dict[str, MetricFamilySpec] = {
    family.prefix: family
    for family in (
        MetricFamilySpec(
            prefix="reward/",
            owner="environment/reward",
            unit="weighted reward units/s",
            aggregation="mean over collector reports since the previous learner iteration",
            step_axis=_STEP_AXIS,
            description=(
                "Weighted environment reward-term rate, as passed through by "
                "upstream RSL-RL; not a completed-episode return."
            ),
            distributed_aggregation="cross-rank mean of per-rank report means",
        ),
        MetricFamilySpec(
            prefix="Loss/aux_",
            owner="learner",
            unit="dimensionless objective",
            aggregation="final emitted actor update for deferred learners; otherwise mean over emitted updates",
            step_axis=_STEP_AXIS,
            description=(
                "Actor-owned auxiliary scalar losses (FlashSAC actor_config custom "
                "actors); pre-weighted by the actor and already included in Loss/actor."
            ),
        ),
    )
}


LOGGER_OWNED_METRICS = frozenset(
    tag
    for tag, spec in METRIC_SPECS.items()
    if spec.owner == "collector"
    or tag
    in {
        "Train/iteration",
        "Perf/total_fps",
        "Perf/iteration_time",
        "Perf/learning_time",
    }
)
"""Canonical fields assembled by the logger from dedicated telemetry inputs."""


def metric_spec(tag: str) -> MetricSpec | MetricFamilySpec | None:
    """Return the exact or open-family specification for a canonical tag."""

    exact = METRIC_SPECS.get(tag)
    if exact is not None:
        return exact
    if tag.startswith("reward/"):
        term = tag.removeprefix("reward/")
        if not term or term in {"mean", "mean_ep100"}:
            return None
    return next(
        (family for prefix, family in METRIC_FAMILIES.items() if tag.startswith(prefix)),
        None,
    )


def normalize_metric_map(metrics: Mapping[str, float]) -> dict[str, float]:
    """Validate canonical source metrics and coerce scalar values to floats."""

    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if metric_spec(key) is None:
            raise ValueError(f"unregistered canonical training metric {key!r}")
        normalized[key] = float(value)
    return normalized


def reward_term_key(term: str) -> str:
    """Build the upstream-compatible persisted tag for one reward term."""

    if term.startswith("reward/"):
        raise ValueError("pass a reward term name, not a namespaced metric key")
    if not term:
        raise ValueError("reward term name must not be empty")
    if term in {"mean", "mean_ep100"}:
        raise ValueError(f"reserved reward component name {term!r}")
    return f"reward/{term}"


def validate_metric_tags(tags: Iterable[str]) -> None:
    """Fail closed when a backend scalar has no schema owner."""

    unknown = [tag for tag in tags if metric_spec(tag) is None]
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unregistered backend metric tags: {names}")

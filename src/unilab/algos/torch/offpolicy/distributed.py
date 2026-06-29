"""Distributed learner contract for off-policy multi-GPU training.

The runner owns process, replay, and IPC orchestration. Algorithm learners own
which model states participate in distributed synchronization. Keeping that
boundary explicit avoids generic DDP wrappers in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

MULTIGPU_SYNC_MODES = frozenset({"sync_sgd", "local_sgd"})


class DistributedOffPolicyLearner(Protocol):
    """Protocol implemented by learners that opt into multi-GPU off-policy training."""

    supports_multi_gpu: bool
    supports_multi_gpu_symmetry: bool
    supported_multi_gpu_sync_modes: frozenset[str]
    actor: Any
    update_count: int

    def update_critic(self, batch: dict[str, Any]) -> dict[str, float]: ...

    def update_actor(self, batch: dict[str, Any]) -> dict[str, float]: ...

    def soft_update_target(self) -> None: ...

    def get_state_dict(self) -> dict[str, Any]: ...

    def sync_initial_parameters(self, src: int = 0) -> None: ...

    def average_distributed_parameters(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DistributedLearnerHooks:
    """Bound distributed hooks resolved once per learner worker."""

    sync_initial_parameters: Callable[..., None]
    average_distributed_parameters: Callable[[], None]


def _learner_name(learner_cls: type[Any]) -> str:
    return getattr(learner_cls, "__name__", str(learner_cls))


def normalize_distributed_sync_mode(mode: str) -> str:
    """Return a validated distributed learner synchronization mode."""
    normalized = str(mode).strip().lower()
    if normalized not in MULTIGPU_SYNC_MODES:
        supported = ", ".join(sorted(MULTIGPU_SYNC_MODES))
        raise ValueError(f"training.multi_gpu_sync_mode must be one of: {supported}; got {mode!r}")
    return normalized


def validate_distributed_learner_capability(
    *,
    learner_cls: type[Any],
    algo_type: str,
    learner_kwargs: dict[str, Any],
    num_gpus: int,
    sync_mode: str,
) -> None:
    """Validate that a learner class has explicitly opted into multi-GPU training."""
    if int(num_gpus) <= 1:
        return

    normalized_sync_mode = normalize_distributed_sync_mode(sync_mode)

    learner_name = _learner_name(learner_cls)
    if not bool(learner_cls.__dict__.get("supports_multi_gpu", False)):
        raise ValueError(
            f"{algo_type} learner {learner_name} does not support training.num_gpus > 1"
        )

    supported_modes = set(getattr(learner_cls, "supported_multi_gpu_sync_modes", ()))
    if normalized_sync_mode not in supported_modes:
        supported = ", ".join(sorted(supported_modes)) or "<none>"
        raise ValueError(
            f"{algo_type} learner {learner_name} does not support "
            f"training.multi_gpu_sync_mode={sync_mode!r}; supported modes: {supported}"
        )

    if bool(learner_kwargs.get("use_symmetry", False)) and not bool(
        getattr(learner_cls, "supports_multi_gpu_symmetry", False)
    ):
        raise ValueError(
            "Off-policy symmetry augmentation does not support training.num_gpus > 1; "
            "set training.num_gpus=1 or algo.use_symmetry=false"
        )

    sync_initial_parameters = getattr(learner_cls, "sync_initial_parameters", None)
    if not callable(sync_initial_parameters):
        raise ValueError(
            f"{algo_type} learner {learner_name} must implement "
            "sync_initial_parameters(src=0) for multi-GPU training"
        )
    if normalized_sync_mode == "local_sgd":
        average_distributed_parameters = getattr(
            learner_cls,
            "average_distributed_parameters",
            None,
        )
        if not callable(average_distributed_parameters):
            raise ValueError(
                f"{algo_type} learner {learner_name} must implement "
                "average_distributed_parameters() for local_sgd multi-GPU training"
            )


def _noop_average_parameters() -> None:
    return None


def resolve_distributed_learner_hooks(
    learner: Any,
    *,
    sync_mode: str,
) -> DistributedLearnerHooks:
    """Resolve distributed learner hooks once, outside the learner update loop."""
    sync_initial_parameters = getattr(learner, "sync_initial_parameters", None)
    if not callable(sync_initial_parameters):
        raise ValueError(
            "Multi-GPU off-policy learner must implement sync_initial_parameters(src=0)"
        )
    sync_initial_parameters = cast(Callable[..., None], sync_initial_parameters)

    average_distributed_parameters = getattr(learner, "average_distributed_parameters", None)
    average_parameters: Callable[[], None]
    if sync_mode == "local_sgd":
        if not callable(average_distributed_parameters):
            raise ValueError(
                "Multi-GPU local_sgd requires learner.average_distributed_parameters()"
            )
        average_parameters = cast(Callable[[], None], average_distributed_parameters)
    else:
        average_parameters = _noop_average_parameters

    return DistributedLearnerHooks(
        sync_initial_parameters=sync_initial_parameters,
        average_distributed_parameters=average_parameters,
    )

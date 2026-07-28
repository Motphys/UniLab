"""Immutable compiled artifacts consumed by managed task executors."""

from __future__ import annotations

from dataclasses import dataclass

from unilab.base.backend.batch import (
    BackendIORequirements,
    BufferContract,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
)
from unilab.base.backend.mutation import MutationSpec

from .entities import CompiledSelector, ManagerContractError
from .spec import (
    FrozenParameters,
    NormalizationMode,
    QuaternionOrder,
    TensorSpec,
    TermPhase,
    TermRole,
)

MANAGER_TASK_CONTRACT_VERSION = "manager-task-contract-v1"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagerContractError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class OutputSlice:
    channel: str
    start: int
    stop: int
    tensor: TensorSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _non_empty(self.channel, "output channel"))
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
            or isinstance(self.stop, bool)
            or not isinstance(self.stop, int)
            or self.stop <= self.start
        ):
            raise ManagerContractError("output slice requires 0 <= start < stop")
        if not isinstance(self.tensor, TensorSpec):
            raise ManagerContractError("output slice tensor must be a TensorSpec")
        if self.stop - self.start != self.tensor.width:
            raise ManagerContractError("output slice width does not match its tensor")


@dataclass(frozen=True)
class OutputChannelPlan:
    key: str
    buffer: BufferContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "output channel key"))
        if not isinstance(self.buffer, BufferContract):
            raise ManagerContractError("output channel buffer must be a BufferContract")
        if len(self.buffer.row_shape) != 1:
            raise ManagerContractError("output channel buffer must be one-dimensional")
        if self.buffer.owner is not BufferOwner.RUNTIME:
            raise ManagerContractError("output channel buffers must be runtime-owned")
        if self.buffer.mutability is not BufferMutability.READ_WRITE:
            raise ManagerContractError("output channel buffers must be read-write")
        if self.buffer.lifetime is not BufferLifetime.PLAN:
            raise ManagerContractError("output channel buffers must use plan lifetime")
        if not self.buffer.address_stable:
            raise ManagerContractError("output channel buffer addresses must be plan-stable")


@dataclass(frozen=True)
class CompiledTerm:
    key: str
    definition_key: str
    definition_version: str
    phase: TermPhase
    role: TermRole
    dependency_indices: tuple[int, ...]
    state_field_indices: tuple[int, ...]
    mutation_indices: tuple[int, ...]
    parameters: FrozenParameters
    output: OutputSlice | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "compiled term key"))
        object.__setattr__(
            self,
            "definition_key",
            _non_empty(self.definition_key, "compiled term definition key"),
        )
        object.__setattr__(
            self,
            "definition_version",
            _non_empty(self.definition_version, "compiled term definition version"),
        )
        if not isinstance(self.phase, TermPhase):
            raise ManagerContractError("compiled term phase must be a TermPhase")
        if not isinstance(self.role, TermRole):
            raise ManagerContractError("compiled term role must be a TermRole")
        for name, indices in (
            ("dependency_indices", self.dependency_indices),
            ("state_field_indices", self.state_field_indices),
            ("mutation_indices", self.mutation_indices),
        ):
            if not isinstance(indices, tuple) or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            ):
                raise ManagerContractError(f"compiled term {name} must contain non-negative ids")
            if len(set(indices)) != len(indices):
                raise ManagerContractError(f"compiled term {name} must contain unique ids")
        if not isinstance(self.parameters, tuple):
            raise ManagerContractError("compiled term parameters must be a tuple")
        if self.output is not None and not isinstance(self.output, OutputSlice):
            raise ManagerContractError("compiled term output must be an OutputSlice or None")


@dataclass(frozen=True)
class ObservationOutput:
    term_index: int
    semantic_key: str
    output: OutputSlice

    def __post_init__(self) -> None:
        if (
            isinstance(self.term_index, bool)
            or not isinstance(self.term_index, int)
            or self.term_index < 0
        ):
            raise ManagerContractError("observation term_index must be non-negative")
        object.__setattr__(self, "semantic_key", _non_empty(self.semantic_key, "observation key"))
        if not isinstance(self.output, OutputSlice):
            raise ManagerContractError("observation output must be an OutputSlice")


@dataclass(frozen=True)
class ObservationGroupPlan:
    key: str
    width: int
    dtype: str
    outputs: tuple[ObservationOutput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty(self.key, "observation group key"))
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ManagerContractError("observation group width must be positive")
        object.__setattr__(self, "dtype", _non_empty(self.dtype, "observation group dtype"))
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise ManagerContractError("observation group requires at least one output")
        if any(not isinstance(item, ObservationOutput) for item in self.outputs):
            raise ManagerContractError("observation group outputs contain an invalid value")
        intervals = sorted((item.output.start, item.output.stop) for item in self.outputs)
        cursor = 0
        for start, stop in intervals:
            if start != cursor:
                raise ManagerContractError(
                    "observation group output slices must be contiguous and non-overlapping"
                )
            cursor = stop
        if cursor != self.width:
            raise ManagerContractError("observation group slices do not cover its declared width")
        if any(item.output.channel != f"obs:{self.key}" for item in self.outputs):
            raise ManagerContractError("observation output channel does not match its group")
        if any(item.output.tensor.dtype != self.dtype for item in self.outputs):
            raise ManagerContractError("observation group outputs must use one dtype")


@dataclass(frozen=True)
class PolicyABI:
    observation_groups: tuple[ObservationGroupPlan, ...]
    action_key: str
    action_dim: int
    action_dtype: str
    action_scale: tuple[float, ...]
    normalization: NormalizationMode
    quaternion_outputs: tuple[tuple[str, QuaternionOrder], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation_groups, tuple) or not self.observation_groups:
            raise ManagerContractError("policy ABI requires observation groups")
        if any(not isinstance(item, ObservationGroupPlan) for item in self.observation_groups):
            raise ManagerContractError("policy ABI observation_groups contain an invalid value")
        group_keys = [item.key for item in self.observation_groups]
        if len(set(group_keys)) != len(group_keys):
            raise ManagerContractError("policy ABI observation group keys must be unique")
        object.__setattr__(self, "action_key", _non_empty(self.action_key, "policy action key"))
        if (
            isinstance(self.action_dim, bool)
            or not isinstance(self.action_dim, int)
            or self.action_dim <= 0
        ):
            raise ManagerContractError("policy action_dim must be positive")
        object.__setattr__(
            self, "action_dtype", _non_empty(self.action_dtype, "policy action dtype")
        )
        if len(self.action_scale) != self.action_dim:
            raise ManagerContractError("policy action_scale must have one value per action")
        if not isinstance(self.normalization, NormalizationMode):
            raise ManagerContractError("policy normalization must be a NormalizationMode")
        if not isinstance(self.quaternion_outputs, tuple):
            raise ManagerContractError("policy quaternion_outputs must be a tuple")
        quaternion_keys = [key for key, _ in self.quaternion_outputs]
        if len(set(quaternion_keys)) != len(quaternion_keys):
            raise ManagerContractError("policy quaternion output keys must be unique")
        if any(order is QuaternionOrder.NONE for _, order in self.quaternion_outputs):
            raise ManagerContractError("policy quaternion outputs require an explicit order")
        object.__setattr__(
            self, "fingerprint", _non_empty(self.fingerprint, "policy ABI fingerprint")
        )


@dataclass(frozen=True)
class CompiledTaskPlan:
    task_key: str
    selectors: tuple[CompiledSelector, ...]
    terms: tuple[CompiledTerm, ...]
    backend_io: BackendIORequirements
    mutation_specs: tuple[MutationSpec, ...]
    output_channels: tuple[OutputChannelPlan, ...]
    policy_abi: PolicyABI
    executor_key: str
    required_capabilities: tuple[str, ...]
    diagnostic_signature: tuple[str, ...]
    fingerprint: str
    selector_binding_fingerprint: str
    contract_version: str = MANAGER_TASK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_key", _non_empty(self.task_key, "compiled task key"))
        if not isinstance(self.selectors, tuple) or any(
            not isinstance(item, CompiledSelector) for item in self.selectors
        ):
            raise ManagerContractError("compiled selectors must be a tuple of CompiledSelector")
        selector_keys = [item.key for item in self.selectors]
        if selector_keys != sorted(selector_keys) or len(set(selector_keys)) != len(selector_keys):
            raise ManagerContractError("compiled selectors must use unique canonical key order")
        if not isinstance(self.terms, tuple) or not self.terms:
            raise ManagerContractError("compiled task requires terms")
        if any(not isinstance(item, CompiledTerm) for item in self.terms):
            raise ManagerContractError("compiled task terms contain an invalid value")
        term_keys = [item.key for item in self.terms]
        if len(set(term_keys)) != len(term_keys):
            raise ManagerContractError("compiled task term keys must be unique")
        if not isinstance(self.backend_io, BackendIORequirements):
            raise ManagerContractError("compiled backend_io must be BackendIORequirements")
        if not isinstance(self.mutation_specs, tuple) or any(
            not isinstance(item, MutationSpec) for item in self.mutation_specs
        ):
            raise ManagerContractError("compiled mutation_specs contain an invalid value")
        mutation_keys = [item.term_key for item in self.mutation_specs]
        if mutation_keys != sorted(mutation_keys) or len(set(mutation_keys)) != len(mutation_keys):
            raise ManagerContractError(
                "compiled mutation specs must use unique canonical key order"
            )
        if not isinstance(self.output_channels, tuple) or not self.output_channels:
            raise ManagerContractError("compiled task requires output channels")
        if any(not isinstance(item, OutputChannelPlan) for item in self.output_channels):
            raise ManagerContractError("compiled output channels contain an invalid value")
        channel_keys = [item.key for item in self.output_channels]
        if channel_keys != sorted(channel_keys) or len(set(channel_keys)) != len(channel_keys):
            raise ManagerContractError("output channels must use unique canonical key order")
        if not isinstance(self.policy_abi, PolicyABI):
            raise ManagerContractError("compiled policy_abi must be a PolicyABI")
        object.__setattr__(self, "executor_key", _non_empty(self.executor_key, "executor key"))
        if not isinstance(self.required_capabilities, tuple):
            raise ManagerContractError("required_capabilities must be a tuple")
        if self.required_capabilities != tuple(sorted(set(self.required_capabilities))):
            raise ManagerContractError("required capabilities must be unique and canonical")
        if not isinstance(self.diagnostic_signature, tuple) or not self.diagnostic_signature:
            raise ManagerContractError("diagnostic_signature must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.diagnostic_signature):
            raise ManagerContractError("diagnostic_signature values must be non-empty strings")
        object.__setattr__(self, "fingerprint", _non_empty(self.fingerprint, "task fingerprint"))
        object.__setattr__(
            self,
            "selector_binding_fingerprint",
            _non_empty(self.selector_binding_fingerprint, "selector binding fingerprint"),
        )
        if self.contract_version != MANAGER_TASK_CONTRACT_VERSION:
            raise ManagerContractError(
                f"unsupported manager task contract version {self.contract_version!r}"
            )
        self._validate_indices_and_outputs()
        self._validate_policy_abi()

    def _validate_indices_and_outputs(self) -> None:
        state_count = len(self.backend_io.state_fields)
        mutation_count = len(self.mutation_specs)
        channels = {item.key: item for item in self.output_channels}
        channel_intervals: dict[str, list[tuple[int, int]]] = {}
        for term_index, term in enumerate(self.terms):
            if any(index >= term_index for index in term.dependency_indices):
                raise ManagerContractError(
                    "compiled dependency indices must refer to preceding terms"
                )
            if any(index >= state_count for index in term.state_field_indices):
                raise ManagerContractError("compiled term references an unknown state field")
            if any(index >= mutation_count for index in term.mutation_indices):
                raise ManagerContractError("compiled term references an unknown mutation")
            if term.output is not None:
                try:
                    channel = channels[term.output.channel]
                except KeyError as exc:
                    raise ManagerContractError(
                        f"compiled term references unknown output channel {term.output.channel!r}"
                    ) from exc
                if term.output.stop > channel.buffer.row_shape[0]:
                    raise ManagerContractError("compiled output slice exceeds its channel width")
                if term.output.tensor.dtype != channel.buffer.dtype:
                    raise ManagerContractError(
                        "compiled output dtype does not match its channel buffer"
                    )
                channel_intervals.setdefault(term.output.channel, []).append(
                    (term.output.start, term.output.stop)
                )
        for channel_key, intervals in channel_intervals.items():
            ordered = sorted(intervals)
            for previous, current in zip(ordered, ordered[1:]):
                if previous[1] > current[0]:
                    raise ManagerContractError(
                        f"compiled output slices overlap in channel {channel_key!r}"
                    )
        if set(channel_intervals) != set(channels):
            raise ManagerContractError(
                "compiled output channels must exactly match term output channels"
            )

    def _validate_policy_abi(self) -> None:
        control = self.backend_io.control
        if len(control.buffer.row_shape) != 1:
            raise ManagerContractError("policy ABI requires a one-dimensional control buffer")
        if (
            self.policy_abi.action_key != control.semantic_key
            or self.policy_abi.action_dim != control.buffer.row_shape[0]
            or self.policy_abi.action_dtype != control.buffer.dtype
        ):
            raise ManagerContractError("policy ABI action does not match the control contract")
        channels = {item.key: item for item in self.output_channels}
        for group in self.policy_abi.observation_groups:
            channel = channels.get(f"obs:{group.key}")
            if channel is None:
                raise ManagerContractError(f"policy ABI group {group.key!r} has no output channel")
            if channel.buffer.row_shape != (group.width,) or channel.buffer.dtype != group.dtype:
                raise ManagerContractError(
                    f"policy ABI group {group.key!r} does not match its output channel"
                )
            for observation in group.outputs:
                if observation.term_index >= len(self.terms):
                    raise ManagerContractError("policy ABI references an unknown term")
                term = self.terms[observation.term_index]
                if term.key != observation.semantic_key or term.output != observation.output:
                    raise ManagerContractError(
                        "policy ABI observation does not match its compiled term output"
                    )

    def term_index(self, key: str) -> int:
        """Cold-path/diagnostic lookup; executors retain integer indices."""
        for index, term in enumerate(self.terms):
            if term.key == key:
                return index
        raise ManagerContractError(f"compiled term {key!r} does not exist")

    def selector_index(self, key: str) -> int:
        """Cold-path/diagnostic lookup; executors retain integer indices."""
        for index, selector in enumerate(self.selectors):
            if selector.key == key:
                return index
        raise ManagerContractError(f"compiled selector {key!r} does not exist")

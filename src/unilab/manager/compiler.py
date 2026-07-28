"""Cold-path compiler from declarative managed tasks to immutable plans."""

from __future__ import annotations

from dataclasses import dataclass

from unilab.base.backend.batch import (
    BackendIORequirements,
    BoundFieldIdentity,
    BufferContract,
    BufferLayout,
    BufferLifetime,
    BufferMutability,
    BufferOwner,
    BufferPlacement,
    ExecutionProfile,
    StateFieldSpec,
)
from unilab.base.backend.mutation import MutationSpec, MutationTargetSpec

from .entities import CompiledSelector, EntityResolver, EntitySelector, ManagerContractError
from .fingerprint import canonical_digest, compiled_plan_payload, tensor_payload
from .plan import (
    MANAGER_TASK_CONTRACT_VERSION,
    CompiledTaskPlan,
    CompiledTerm,
    ObservationGroupPlan,
    ObservationOutput,
    OutputChannelPlan,
    OutputSlice,
    PolicyABI,
)
from .registry import TermRegistry
from .spec import (
    TERM_PHASE_ORDER,
    FrozenParameters,
    MutationTemplate,
    ParameterKind,
    ParameterSpec,
    ParameterValue,
    QuaternionOrder,
    StateRequirement,
    TaskSpec,
    TermDefinition,
    TermInvocation,
    TermRole,
)


@dataclass(frozen=True)
class _ResolvedInvocation:
    invocation: TermInvocation
    definition: TermDefinition
    parameters: FrozenParameters


def _validate_parameter(spec: ParameterSpec, value: ParameterValue) -> ParameterValue:
    if spec.kind is ParameterKind.BOOL:
        if not isinstance(value, bool):
            raise ManagerContractError(f"parameter {spec.key!r} requires bool")
        return value
    if spec.kind is ParameterKind.INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ManagerContractError(f"parameter {spec.key!r} requires int")
        return value
    if spec.kind is ParameterKind.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManagerContractError(f"parameter {spec.key!r} requires float")
        return float(value)
    if spec.kind is ParameterKind.STRING:
        if not isinstance(value, str):
            raise ManagerContractError(f"parameter {spec.key!r} requires string")
        return value
    if spec.kind is ParameterKind.INT_TUPLE:
        if not isinstance(value, tuple) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            raise ManagerContractError(f"parameter {spec.key!r} requires an int tuple")
        return value
    if spec.kind is ParameterKind.FLOAT_TUPLE:
        if not isinstance(value, tuple) or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            raise ManagerContractError(f"parameter {spec.key!r} requires a float tuple")
        return tuple(float(item) for item in value)
    raise ManagerContractError(f"unsupported parameter kind {spec.kind!r}")


def _compile_parameters(
    invocation: TermInvocation,
    definition: TermDefinition,
) -> FrozenParameters:
    supplied = dict(invocation.parameters)
    schema = {item.key: item for item in definition.parameters}
    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        raise ManagerContractError(
            f"term {invocation.key!r} has unknown parameters: {', '.join(unknown)}"
        )
    compiled: list[tuple[str, ParameterValue]] = []
    for key in sorted(schema):
        item = schema[key]
        if key in supplied:
            value = supplied[key]
        elif item.required:
            raise ManagerContractError(
                f"term {invocation.key!r} is missing required parameter {key!r}"
            )
        else:
            assert item.default is not None
            value = item.default
        compiled.append((key, _validate_parameter(item, value)))
    return tuple(compiled)


def _resolve_definitions(task: TaskSpec, registry: TermRegistry) -> dict[str, _ResolvedInvocation]:
    definitions: dict[str, TermDefinition] = {}
    for definition_key in sorted({item.definition_key for item in task.terms}):
        definitions[definition_key] = registry.resolve(definition_key)
    resolved: dict[str, _ResolvedInvocation] = {}
    for invocation in task.terms:
        definition = definitions[invocation.definition_key]
        if definition.role is TermRole.OBSERVATION:
            if invocation.observation_group is None:
                raise ManagerContractError(
                    f"observation term {invocation.key!r} requires an observation group"
                )
        elif invocation.observation_group is not None:
            raise ManagerContractError(
                f"non-observation term {invocation.key!r} cannot declare an observation group"
            )
        resolved[invocation.key] = _ResolvedInvocation(
            invocation=invocation,
            definition=definition,
            parameters=_compile_parameters(invocation, definition),
        )
    return resolved


def _canonical_term_order(resolved: dict[str, _ResolvedInvocation]) -> tuple[str, ...]:
    phase_index = {phase: index for index, phase in enumerate(TERM_PHASE_ORDER)}
    indegree = {key: 0 for key in resolved}
    dependents: dict[str, list[str]] = {key: [] for key in resolved}
    for key, item in resolved.items():
        for dependency in item.invocation.dependencies:
            try:
                dependency_item = resolved[dependency]
            except KeyError as exc:
                raise ManagerContractError(
                    f"term {key!r} depends on unknown term {dependency!r}"
                ) from exc
            if phase_index[dependency_item.definition.phase] > phase_index[item.definition.phase]:
                raise ManagerContractError(
                    f"term {key!r} has a dependency from later phase "
                    f"{dependency_item.definition.phase.value!r}"
                )
            indegree[key] += 1
            dependents[dependency].append(key)
    ready = sorted(
        (phase_index[item.definition.phase], key)
        for key, item in resolved.items()
        if indegree[key] == 0
    )
    ordered: list[str] = []
    while ready:
        _, key = ready.pop(0)
        ordered.append(key)
        for dependent in sorted(dependents[key]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                item = resolved[dependent]
                ready.append((phase_index[item.definition.phase], dependent))
                ready.sort()
    if len(ordered) != len(resolved):
        cyclic = ", ".join(sorted(key for key, degree in indegree.items() if degree))
        raise ManagerContractError(f"term dependency graph contains a cycle: {cyclic}")
    return tuple(ordered)


def _collect_selectors(resolved: dict[str, _ResolvedInvocation]) -> dict[str, EntitySelector]:
    selectors: dict[str, EntitySelector] = {}

    def add(selector: EntitySelector) -> None:
        previous = selectors.get(selector.key)
        if previous is not None and previous != selector:
            raise ManagerContractError(
                f"selector key {selector.key!r} has inconsistent declarations"
            )
        selectors[selector.key] = selector

    for item in resolved.values():
        for requirement in item.definition.state_requirements:
            add(requirement.selector)
        for template in item.definition.mutation_templates:
            if template.selector is not None:
                add(template.selector)
    return selectors


def _bind_selectors(
    selectors: dict[str, EntitySelector],
    resolver: EntityResolver,
) -> tuple[CompiledSelector, ...]:
    return tuple(CompiledSelector.bind(selectors[key], resolver) for key in sorted(selectors))


def _state_buffer(
    requirement: StateRequirement,
    profile: ExecutionProfile,
    placement: BufferPlacement,
) -> BufferContract:
    return BufferContract(
        row_shape=requirement.tensor.shape,
        dtype=requirement.tensor.dtype,
        layout=BufferLayout.C_CONTIGUOUS,
        placement=placement,
        owner=BufferOwner.BACKEND,
        mutability=BufferMutability.READ_ONLY,
        lifetime=BufferLifetime.BORROWED_UNTIL_MUTATION,
        dlpack_exportable=profile is ExecutionProfile.DEVICE_RESIDENT,
        address_stable=True,
    )


def _compile_state_fields(
    resolved: dict[str, _ResolvedInvocation],
    selectors: dict[str, CompiledSelector],
    profile: ExecutionProfile,
    placement: BufferPlacement,
) -> tuple[tuple[StateFieldSpec, ...], dict[str, tuple[str, ...]]]:
    fields: dict[str, StateFieldSpec] = {}
    term_state_keys: dict[str, tuple[str, ...]] = {}
    for term_key, item in resolved.items():
        keys: list[str] = []
        for requirement in item.definition.state_requirements:
            selector = selectors[requirement.selector.key]
            if requirement.entity_axis is not None:
                actual = requirement.tensor.shape[requirement.entity_axis]
                if actual != len(selector.entity_ids):
                    raise ManagerContractError(
                        f"state {requirement.semantic_key!r} entity axis has width {actual}, "
                        f"but selector {selector.key!r} resolved {len(selector.entity_ids)} ids"
                    )
            field = StateFieldSpec(
                semantic_key=requirement.semantic_key,
                identity=BoundFieldIdentity(
                    entity_kind=requirement.state_entity_kind,
                    field_kind=requirement.field_kind,
                    entity_ids=selector.entity_ids,
                ),
                frame=requirement.tensor.frame,
                unit=requirement.tensor.unit,
                buffer=_state_buffer(requirement, profile, placement),
            )
            previous = fields.get(field.semantic_key)
            if previous is not None and previous != field:
                raise ManagerContractError(
                    f"state semantic {field.semantic_key!r} has incompatible declarations"
                )
            fields[field.semantic_key] = field
            keys.append(field.semantic_key)
        term_state_keys[term_key] = tuple(keys)
    ordered = tuple(fields[key] for key in sorted(fields))
    return ordered, term_state_keys


def _mutation_key(term_key: str, template: MutationTemplate) -> str:
    return term_key if not template.key_suffix else f"{term_key}.{template.key_suffix}"


def _compile_mutations(
    resolved: dict[str, _ResolvedInvocation],
    selectors: dict[str, CompiledSelector],
    placement: BufferPlacement,
) -> tuple[tuple[MutationSpec, ...], dict[str, tuple[str, ...]]]:
    specs: dict[str, MutationSpec] = {}
    writes: dict[tuple[object, ...], str] = {}
    term_mutation_keys: dict[str, tuple[str, ...]] = {}
    for term_key, item in resolved.items():
        keys: list[str] = []
        for template in item.definition.mutation_templates:
            if template.value_template.placement != placement:
                raise ManagerContractError(
                    f"mutation {_mutation_key(term_key, template)!r} placement does not match "
                    "the task control/state placement"
                )
            key = _mutation_key(term_key, template)
            selector = selectors[template.selector.key] if template.selector is not None else None
            target = MutationTargetSpec(
                target_key=template.target_key,
                target_kind=template.target_kind,
                entity_kind=template.entity_kind,
                field_kind=template.field_kind,
                selector=selector.key if selector is not None else None,
            )
            spec = MutationSpec(
                term_key=key,
                target=target,
                trigger=template.trigger,
                commit_phase=template.commit_phase,
                operation=template.operation,
                baseline=template.baseline,
                persistence=template.persistence,
                recompute=template.recompute,
                value_template=template.value_template,
            )
            if key in specs:
                raise ManagerContractError(f"duplicate mutation key {key!r}")
            bound_ids = selector.entity_ids if selector is not None else ()
            write_key = (
                template.commit_phase,
                template.target_kind,
                template.entity_kind,
                template.field_kind,
                bound_ids,
            )
            previous = writes.get(write_key)
            if previous is not None:
                raise ManagerContractError(
                    f"mutation writes {previous!r} and {key!r} conflict at one commit barrier"
                )
            writes[write_key] = key
            specs[key] = spec
            keys.append(key)
        term_mutation_keys[term_key] = tuple(keys)
    ordered = tuple(specs[key] for key in sorted(specs))
    return ordered, term_mutation_keys


def _required_capabilities(resolved: dict[str, _ResolvedInvocation]) -> tuple[str, ...]:
    values: set[str] = set()
    for item in resolved.values():
        values.update(item.definition.required_capabilities)
        values.update(
            requirement.capability_key
            for requirement in item.definition.state_requirements
            if requirement.capability_key is not None
        )
        values.update(
            template.capability_key
            for template in item.definition.mutation_templates
            if template.capability_key is not None
        )
    return tuple(sorted(values))


def _allocate_outputs(
    ordered_keys: tuple[str, ...],
    resolved: dict[str, _ResolvedInvocation],
) -> dict[str, OutputSlice | None]:
    cursors: dict[str, int] = {}
    outputs: dict[str, OutputSlice | None] = {}
    for key in ordered_keys:
        item = resolved[key]
        tensor = item.definition.output
        if tensor is None:
            outputs[key] = None
            continue
        if item.definition.role is TermRole.OBSERVATION:
            assert item.invocation.observation_group is not None
            channel = f"obs:{item.invocation.observation_group}"
        else:
            channel = item.definition.role.value
        start = cursors.get(channel, 0)
        output = OutputSlice(channel=channel, start=start, stop=start + tensor.width, tensor=tensor)
        outputs[key] = output
        cursors[channel] = output.stop
    return outputs


def _build_output_channels(
    outputs: dict[str, OutputSlice | None],
    *,
    placement: BufferPlacement,
    profile: ExecutionProfile,
) -> tuple[OutputChannelPlan, ...]:
    widths: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    for output in outputs.values():
        if output is None:
            continue
        previous_dtype = dtypes.get(output.channel)
        if previous_dtype is not None and previous_dtype != output.tensor.dtype:
            raise ManagerContractError(f"output channel {output.channel!r} mixes output dtypes")
        dtypes[output.channel] = output.tensor.dtype
        widths[output.channel] = max(widths.get(output.channel, 0), output.stop)
    return tuple(
        OutputChannelPlan(
            key=key,
            buffer=BufferContract(
                row_shape=(widths[key],),
                dtype=dtypes[key],
                layout=BufferLayout.C_CONTIGUOUS,
                placement=placement,
                owner=BufferOwner.RUNTIME,
                mutability=BufferMutability.READ_WRITE,
                lifetime=BufferLifetime.PLAN,
                dlpack_exportable=profile is ExecutionProfile.DEVICE_RESIDENT,
                address_stable=True,
            ),
        )
        for key in sorted(widths)
    )


def _build_policy_abi(
    task: TaskSpec,
    ordered_keys: tuple[str, ...],
    resolved: dict[str, _ResolvedInvocation],
    outputs: dict[str, OutputSlice | None],
    term_indices: dict[str, int],
) -> PolicyABI:
    declared_groups = task.policy.observation_groups
    actual_groups: set[str] = set()
    for item in resolved.values():
        if item.definition.role is not TermRole.OBSERVATION:
            continue
        group = item.invocation.observation_group
        assert group is not None
        actual_groups.add(group)
    if actual_groups != set(declared_groups):
        missing = sorted(set(declared_groups) - actual_groups)
        extra = sorted(actual_groups - set(declared_groups))
        raise ManagerContractError(
            f"policy observation groups do not match observation terms; missing={missing}, extra={extra}"
        )
    groups: list[ObservationGroupPlan] = []
    quaternion_outputs: list[tuple[str, QuaternionOrder]] = []
    for group_key in declared_groups:
        group_outputs: list[ObservationOutput] = []
        dtype: str | None = None
        for key in ordered_keys:
            item = resolved[key]
            if item.invocation.observation_group != group_key:
                continue
            output = outputs[key]
            assert output is not None
            if dtype is None:
                dtype = output.tensor.dtype
            elif dtype != output.tensor.dtype:
                raise ManagerContractError(f"observation group {group_key!r} mixes output dtypes")
            semantic_key = key
            group_outputs.append(
                ObservationOutput(
                    term_index=term_indices[key],
                    semantic_key=semantic_key,
                    output=output,
                )
            )
            if output.tensor.quaternion_order is not QuaternionOrder.NONE:
                quaternion_outputs.append((semantic_key, output.tensor.quaternion_order))
        assert dtype is not None
        groups.append(
            ObservationGroupPlan(
                key=group_key,
                width=sum(item.output.tensor.width for item in group_outputs),
                dtype=dtype,
                outputs=tuple(group_outputs),
            )
        )
    if len(task.control.buffer.row_shape) != 1:
        raise ManagerContractError("managed policy action/control buffer must be one-dimensional")
    action_dim = task.control.buffer.row_shape[0]
    if len(task.policy.action_scale) == 1:
        action_scale = task.policy.action_scale * action_dim
    elif len(task.policy.action_scale) == action_dim:
        action_scale = task.policy.action_scale
    else:
        raise ManagerContractError("policy action_scale requires one value or action_dim values")
    payload = {
        "groups": [
            {
                "key": group.key,
                "width": group.width,
                "dtype": group.dtype,
                "outputs": [
                    {
                        "key": output.semantic_key,
                        "start": output.output.start,
                        "stop": output.output.stop,
                        "tensor": tensor_payload(output.output.tensor),
                    }
                    for output in group.outputs
                ],
            }
            for group in groups
        ],
        "action": {
            "key": task.control.semantic_key,
            "dim": action_dim,
            "dtype": task.control.buffer.dtype,
            "scale": list(action_scale),
        },
        "normalization": task.policy.normalization.value,
    }
    return PolicyABI(
        observation_groups=tuple(groups),
        action_key=task.control.semantic_key,
        action_dim=action_dim,
        action_dtype=task.control.buffer.dtype,
        action_scale=action_scale,
        normalization=task.policy.normalization,
        quaternion_outputs=tuple(sorted(quaternion_outputs, key=lambda item: item[0])),
        fingerprint=f"managed-policy-abi-v1:{canonical_digest(payload)}",
    )


class TaskCompiler:
    """Compile task metadata and selectors once; runtime execution is lookup-free."""

    def __init__(self, registry: TermRegistry) -> None:
        if not isinstance(registry, TermRegistry):
            raise ManagerContractError("TaskCompiler requires a TermRegistry")
        self._registry = registry

    def compile(
        self,
        task: TaskSpec,
        *,
        resolver: EntityResolver,
        capabilities: frozenset[str],
    ) -> CompiledTaskPlan:
        if not isinstance(task, TaskSpec):
            raise ManagerContractError("task must be a TaskSpec")
        if not isinstance(capabilities, frozenset) or any(
            not isinstance(item, str) or not item for item in capabilities
        ):
            raise ManagerContractError("capabilities must be a frozenset of non-empty strings")
        resolved = _resolve_definitions(task, self._registry)
        ordered_keys = _canonical_term_order(resolved)
        required_capabilities = _required_capabilities(resolved)
        missing = sorted(set(required_capabilities) - capabilities)
        if missing:
            raise ManagerContractError(
                f"task {task.key!r} requires unsupported capabilities: {', '.join(missing)}"
            )
        selectors = _bind_selectors(_collect_selectors(resolved), resolver)
        selector_map = {item.key: item for item in selectors}
        state_fields, term_state_keys = _compile_state_fields(
            resolved,
            selector_map,
            task.execution_profile,
            task.control.buffer.placement,
        )
        mutations, term_mutation_keys = _compile_mutations(
            resolved,
            selector_map,
            task.control.buffer.placement,
        )
        backend_io = BackendIORequirements(
            state_fields=state_fields,
            control=task.control,
            execution_profile=task.execution_profile,
            hot_path_budget=task.hot_path_budget,
            reset_hot_path_budget=task.reset_hot_path_budget,
        )
        outputs = _allocate_outputs(ordered_keys, resolved)
        output_channels = _build_output_channels(
            outputs,
            placement=task.control.buffer.placement,
            profile=task.execution_profile,
        )
        term_indices = {key: index for index, key in enumerate(ordered_keys)}
        state_indices = {item.semantic_key: index for index, item in enumerate(state_fields)}
        mutation_indices = {item.term_key: index for index, item in enumerate(mutations)}
        compiled_terms = tuple(
            CompiledTerm(
                key=key,
                definition_key=resolved[key].definition.key,
                definition_version=resolved[key].definition.version,
                phase=resolved[key].definition.phase,
                role=resolved[key].definition.role,
                dependency_indices=tuple(
                    sorted(term_indices[item] for item in resolved[key].invocation.dependencies)
                ),
                state_field_indices=tuple(state_indices[item] for item in term_state_keys[key]),
                mutation_indices=tuple(mutation_indices[item] for item in term_mutation_keys[key]),
                parameters=resolved[key].parameters,
                output=outputs[key],
            )
            for key in ordered_keys
        )
        policy_abi = _build_policy_abi(task, ordered_keys, resolved, outputs, term_indices)
        diagnostic_signature = (
            MANAGER_TASK_CONTRACT_VERSION,
            f"task={task.key}",
            f"executor={task.executor_key}",
            f"profile={task.execution_profile.value}",
            f"terms={len(compiled_terms)}",
            f"state_fields={len(state_fields)}",
            f"mutations={len(mutations)}",
            f"output_channels={len(output_channels)}",
            f"policy_abi={policy_abi.fingerprint}",
        )
        provisional = CompiledTaskPlan(
            task_key=task.key,
            selectors=selectors,
            terms=compiled_terms,
            backend_io=backend_io,
            mutation_specs=mutations,
            output_channels=output_channels,
            policy_abi=policy_abi,
            executor_key=task.executor_key,
            required_capabilities=required_capabilities,
            diagnostic_signature=diagnostic_signature,
            fingerprint="pending",
            selector_binding_fingerprint="pending",
        )
        semantic_fingerprint = (
            f"{MANAGER_TASK_CONTRACT_VERSION}:"
            f"{canonical_digest(compiled_plan_payload(provisional, include_bindings=False))}"
        )
        binding_fingerprint = (
            "manager-selector-binding-v1:"
            f"{canonical_digest(compiled_plan_payload(provisional, include_bindings=True))}"
        )
        return CompiledTaskPlan(
            task_key=provisional.task_key,
            selectors=provisional.selectors,
            terms=provisional.terms,
            backend_io=provisional.backend_io,
            mutation_specs=provisional.mutation_specs,
            output_channels=provisional.output_channels,
            policy_abi=provisional.policy_abi,
            executor_key=provisional.executor_key,
            required_capabilities=provisional.required_capabilities,
            diagnostic_signature=(
                *provisional.diagnostic_signature,
                f"selector_binding={binding_fingerprint}",
            ),
            fingerprint=semantic_fingerprint,
            selector_binding_fingerprint=binding_fingerprint,
        )

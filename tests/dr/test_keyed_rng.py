from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from unilab.dr import (
    KEYED_RNG_ALGORITHM,
    KeyedRandomContractError,
    KeyedRandomSpec,
    KeyedRandomStream,
    RandomCorrelation,
    RandomDistribution,
    StaleKeyedRandomBatchError,
    keyed_random_reference,
)


def _spec(
    *,
    term_key: str = "event.randomize_friction",
    row_shape: tuple[int, ...] = (3, 2),
    distribution: RandomDistribution = RandomDistribution.UNIFORM,
    correlation: RandomCorrelation = RandomCorrelation.PER_ELEMENT,
    parameters: tuple[float, float] = (-0.75, 1.25),
) -> KeyedRandomSpec:
    return KeyedRandomSpec(
        term_key=term_key,
        term_version="v2",
        row_shape=row_shape,
        distribution=distribution,
        correlation=correlation,
        parameters=parameters,
    )


def _mask(num_envs: int, rows: tuple[int, ...], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if rows:
        mask[torch.tensor(rows, dtype=torch.int64, device=device)] = True
    return mask


def _run_schedule(
    *,
    spec: KeyedRandomSpec,
    run_seed: int,
    num_envs: int,
    device: torch.device,
    schedule: tuple[tuple[int, ...], ...],
    insert_unrelated: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    stream = KeyedRandomStream(
        spec,
        run_seed=run_seed,
        num_envs=num_envs,
        device=device,
    )
    unrelated = KeyedRandomStream(
        _spec(term_key="event.unrelated", row_shape=(5,)),
        run_seed=run_seed,
        num_envs=num_envs,
        device=device,
    )
    latest: torch.Tensor | None = None
    for index, rows in enumerate(schedule):
        if insert_unrelated:
            unrelated.sample(_mask(num_envs, tuple(range(index, num_envs, 3)), device))
        latest = stream.sample(_mask(num_envs, rows, device)).values.clone()
        if insert_unrelated:
            unrelated.sample(_mask(num_envs, tuple(range(0, num_envs, 2)), device))
    assert latest is not None
    return latest, stream.capture_trigger_counts()


@pytest.mark.slow
def test_rng_is_invariant_to_row_order_and_unrelated_terms() -> None:
    """Mandatory Phase-6 metamorphic oracle on CPU and real CUDA."""

    if not torch.cuda.is_available():
        pytest.fail("P6-RNG-REPRODUCIBILITY requires a real CUDA device")
    canonical = ((0, 5, 9, 16), (2, 5, 11), (16, 1, 2), (7, 5, 0))
    permuted = ((16, 9, 0, 5), (11, 2, 5), (2, 1, 16), (0, 7, 5))

    for seed in (0, 7, 29):
        for num_envs in (17, 128):
            spec = _spec()
            expected_values, expected_counts = _run_schedule(
                spec=spec,
                run_seed=seed,
                num_envs=num_envs,
                device=torch.device("cpu"),
                schedule=canonical,
                insert_unrelated=False,
            )
            for device, schedule, unrelated in (
                (torch.device("cpu"), permuted, True),
                (torch.device("cuda", torch.cuda.current_device()), canonical, True),
                (torch.device("cuda", torch.cuda.current_device()), permuted, False),
            ):
                actual_values, actual_counts = _run_schedule(
                    spec=spec,
                    run_seed=seed,
                    num_envs=num_envs,
                    device=device,
                    schedule=schedule,
                    insert_unrelated=unrelated,
                )
                np.testing.assert_array_equal(actual_counts, expected_counts)
                torch.testing.assert_close(actual_values.cpu(), expected_values, rtol=0, atol=0)

            active = np.flatnonzero(expected_counts)
            reference = keyed_random_reference(
                spec,
                run_seed=seed,
                env_ids=active,
                trigger_counts=expected_counts[active] - 1,
            )
            np.testing.assert_allclose(
                expected_values.numpy()[active],
                reference,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            np.testing.assert_array_equal(
                expected_values.numpy()[expected_counts == 0],
                np.zeros((int(np.count_nonzero(expected_counts == 0)), *spec.row_shape)),
            )


@pytest.mark.parametrize(
    ("correlation", "row_shape"),
    [
        (RandomCorrelation.PER_ELEMENT, (3, 4)),
        (RandomCorrelation.PER_ENTITY, (3, 4)),
        (RandomCorrelation.PER_ENV, (3, 4)),
        (RandomCorrelation.GLOBAL, (3, 4)),
    ],
)
def test_correlation_scope_matches_the_compiled_shape(
    correlation: RandomCorrelation,
    row_shape: tuple[int, ...],
) -> None:
    spec = _spec(correlation=correlation, row_shape=row_shape)
    stream = KeyedRandomStream(spec, run_seed=11, num_envs=5, device="cpu")
    values = stream.sample(torch.ones(5, dtype=torch.bool)).values.numpy()
    assert np.all(values >= spec.parameters[0])
    assert np.all(values <= spec.parameters[1])
    if correlation is RandomCorrelation.PER_ELEMENT:
        assert len(np.unique(values[0])) > 1
    elif correlation is RandomCorrelation.PER_ENTITY:
        for row in values:
            for entity in row:
                np.testing.assert_array_equal(entity, np.full_like(entity, entity[0]))
        assert len(np.unique(values[0, :, 0])) > 1
    elif correlation is RandomCorrelation.PER_ENV:
        for row in values:
            np.testing.assert_array_equal(row, np.full_like(row, row.flat[0]))
        assert len(np.unique(values[:, 0, 0])) > 1
    else:
        np.testing.assert_array_equal(values, np.full_like(values, values.flat[0]))


def test_normal_distribution_and_zero_variance_are_explicit() -> None:
    normal = _spec(
        distribution=RandomDistribution.NORMAL,
        parameters=(1.5, 0.25),
        row_shape=(4096,),
    )
    values = (
        KeyedRandomStream(normal, run_seed=3, num_envs=2, device="cpu")
        .sample(torch.ones(2, dtype=torch.bool))
        .values
    )
    assert abs(float(values.mean()) - 1.5) < 0.02
    assert abs(float(values.std()) - 0.25) < 0.02

    constant = replace(normal, parameters=(-4.0, 0.0), row_shape=(3,))
    constant_values = (
        KeyedRandomStream(constant, run_seed=3, num_envs=2, device="cpu")
        .sample(torch.ones(2, dtype=torch.bool))
        .values
    )
    torch.testing.assert_close(constant_values, torch.full((2, 3), -4.0))


def test_partitioned_disjoint_rows_match_one_all_world_trigger() -> None:
    spec = _spec(row_shape=(2,))
    all_worlds = KeyedRandomStream(spec, run_seed=17, num_envs=8, device="cpu")
    expected = all_worlds.sample(torch.ones(8, dtype=torch.bool)).values.clone()

    partitioned = KeyedRandomStream(spec, run_seed=17, num_envs=8, device="cpu")
    partitioned.sample(_mask(8, (0, 2, 4, 6), torch.device("cpu")))
    actual = partitioned.sample(_mask(8, (1, 3, 5, 7), torch.device("cpu"))).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(partitioned.capture_trigger_counts(), np.ones(8, dtype=np.int64))


def test_frozen_algorithm_vector_and_fingerprint_detect_semantic_changes() -> None:
    spec = KeyedRandomSpec(
        term_key="x",
        term_version="v1",
        row_shape=(2, 3),
        distribution=RandomDistribution.UNIFORM,
        correlation=RandomCorrelation.PER_ELEMENT,
        parameters=(-1.0, 1.0),
    )
    values = (
        KeyedRandomStream(spec, run_seed=7, num_envs=4, device="cpu")
        .sample(torch.tensor([True, False, True, False]))
        .values
    )
    torch.testing.assert_close(
        values[0].flatten(),
        torch.tensor([0.92262006, -0.19305289, 0.66231829, -0.10574204, 0.82954270, -0.69017231]),
        rtol=0,
        atol=1.0e-7,
    )
    assert spec.algorithm == KEYED_RNG_ALGORITHM
    assert spec.fingerprint == (
        "splitmix32-v1:01909a6dec3871441b42e39d7ffd0d99d906689a52e87e3354765b8dac587200"
    )
    variants = (
        replace(spec, term_key="y"),
        replace(spec, term_version="v2"),
        replace(spec, row_shape=(6,)),
        replace(spec, correlation=RandomCorrelation.PER_ENV),
        replace(spec, distribution=RandomDistribution.NORMAL),
        replace(spec, parameters=(-2.0, 1.0)),
    )
    assert all(item.fingerprint != spec.fingerprint for item in variants)


def test_borrowed_batch_lifetime_and_output_address_are_stable() -> None:
    stream = KeyedRandomStream(_spec(), run_seed=0, num_envs=17, device="cpu")
    active = torch.ones(17, dtype=torch.bool)
    address = stream.output_address
    first = stream.sample(active)
    assert first.values.data_ptr() == address
    stream.sample(active)
    with pytest.raises(StaleKeyedRandomBatchError, match="stream advanced"):
        _ = first.values
    for _ in range(32):
        assert stream.sample(active).values.data_ptr() == address


@pytest.mark.parametrize(
    "kwargs",
    [
        {"term_key": ""},
        {"term_version": ""},
        {"row_shape": [2]},
        {"row_shape": (0,)},
        {"distribution": "uniform"},
        {"correlation": "per_env"},
        {"parameters": (2.0, 1.0)},
        {"parameters": (0.0, float("nan"))},
        {"algorithm": "mutable-generator-v1"},
    ],
)
def test_random_spec_rejects_malformed_identity(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "term_key": "event.mass",
        "term_version": "v1",
        "row_shape": (2,),
        "distribution": RandomDistribution.UNIFORM,
        "correlation": RandomCorrelation.PER_ELEMENT,
        "parameters": (0.0, 1.0),
    }
    values.update(kwargs)
    with pytest.raises(KeyedRandomContractError):
        KeyedRandomSpec(**values)  # type: ignore[arg-type]


def test_stream_rejects_wrong_mask_and_stream_metadata() -> None:
    spec = _spec()
    with pytest.raises(KeyedRandomContractError, match="run_seed"):
        KeyedRandomStream(spec, run_seed=True, num_envs=2, device="cpu")
    with pytest.raises(KeyedRandomContractError, match="num_envs"):
        KeyedRandomStream(spec, run_seed=0, num_envs=0, device="cpu")
    with pytest.raises(KeyedRandomContractError, match="float32 or float64"):
        KeyedRandomStream(spec, run_seed=0, num_envs=2, device="cpu", dtype=torch.float16)
    stream = KeyedRandomStream(spec, run_seed=0, num_envs=2, device="cpu")
    for mask in (
        np.ones(2, dtype=np.bool_),
        torch.ones(2, dtype=torch.int64),
        torch.ones(3, dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
    ):
        with pytest.raises(KeyedRandomContractError, match="active_mask"):
            stream.sample(mask)  # type: ignore[arg-type]


def test_reference_rejects_invalid_rows_and_matches_torch_float64() -> None:
    spec = _spec(
        distribution=RandomDistribution.NORMAL,
        parameters=(0.25, 1.5),
        row_shape=(4,),
    )
    with pytest.raises(KeyedRandomContractError, match="equal-length"):
        keyed_random_reference(
            spec,
            run_seed=0,
            env_ids=np.array([0, 1]),
            trigger_counts=np.array([0]),
        )
    with pytest.raises(KeyedRandomContractError, match="non-negative"):
        keyed_random_reference(
            spec,
            run_seed=0,
            env_ids=np.array([-1]),
            trigger_counts=np.array([0]),
        )
    expected = keyed_random_reference(
        spec,
        run_seed=5,
        env_ids=np.arange(3),
        trigger_counts=np.zeros(3, dtype=np.int64),
        dtype=np.float64,
    )
    actual = (
        KeyedRandomStream(
            spec,
            run_seed=5,
            num_envs=3,
            device="cpu",
            dtype=torch.float64,
        )
        .sample(torch.ones(3, dtype=torch.bool))
        .values.numpy()
    )
    # Counter/hash bits are identical; Box-Muller log/cos implementations are
    # numerically, not bitwise, portable across NumPy and Torch kernels.
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.slow
def test_cuda_sampling_has_stable_storage_and_no_transfer_sync_or_allocation() -> None:
    if not torch.cuda.is_available():
        pytest.fail("keyed RNG CUDA traffic oracle requires a real CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    stream = KeyedRandomStream(_spec(row_shape=(12,)), run_seed=9, num_envs=128, device=device)
    active = torch.ones(128, dtype=torch.bool, device=device)
    for _ in range(3):
        stream.sample(active)
    torch.cuda.synchronize(device)  # Warmup/oracle boundary, outside the measured scope.
    before = torch.cuda.memory_allocated(device)
    address = stream.output_address
    with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("hot sync")):
        for _ in range(64):
            batch = stream.sample(active)
            assert batch.values.data_ptr() == address
    after = torch.cuda.memory_allocated(device)
    assert after == before
    assert stream.traffic_diagnostics.host_to_device_transfers == 0
    assert stream.traffic_diagnostics.device_to_host_transfers == 0
    assert stream.traffic_diagnostics.global_synchronizations == 0
    assert stream.traffic_diagnostics.sample_allocations == 0

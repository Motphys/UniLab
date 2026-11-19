from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.managers import ManagerRng


def test_rng_replays_streams_and_keeps_bounds_on_device() -> None:
    rng = ManagerRng.seeded(17)
    reference = ManagerRng.seeded(17)

    uniform = rng.uniform(-2.0, 3.0, (2, 3))
    integers = rng.integers(0, 7, (5,), dtype=torch.int32)
    normal = rng.normal(1.0, 0.25, (2,))

    torch.testing.assert_close(uniform, reference.uniform(-2.0, 3.0, (2, 3)))
    torch.testing.assert_close(integers, reference.integers(0, 7, (5,), dtype=torch.int32))
    torch.testing.assert_close(normal, reference.normal(1.0, 0.25, (2,)))
    assert uniform.device == rng.device
    assert uniform.dtype == torch.float32
    assert bool((uniform >= -2.0).all() and (uniform < 3.0).all())
    assert integers.dtype == torch.int32
    assert bool((integers >= 0).all() and (integers < 7).all())


def test_rng_broadcasts_tensor_bounds_without_host_return() -> None:
    rng = ManagerRng.seeded(3)
    low = torch.tensor([-1.0, -2.0, -3.0])
    high = torch.tensor([0.0, 0.5, 1.0])

    values = rng.uniform(low, high, (4, 3))

    assert values.shape == (4, 3)
    assert values.device == low.device
    assert bool((values >= low).all() and (values <= high + 1e-6).all())


def test_rng_choice_samples_device_probabilities() -> None:
    rng = ManagerRng.seeded(9)
    probabilities = torch.tensor([0.0, 1.0, 0.0])

    sampled = rng.choice(3, (8,), p=probabilities)

    torch.testing.assert_close(sampled, torch.full((8,), 1, dtype=torch.long))


@pytest.mark.parametrize(
    ("seed", "match"),
    ((-1, "non-negative integer"), (True, "non-negative integer"), (1.5, "non-negative integer")),
)
def test_rng_rejects_invalid_seeds(seed: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ManagerRng(device="cpu").manual_seed(seed)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("low", "high"),
    ((1, 1), (2, 1), (-1, 2)),
)
def test_rng_rejects_invalid_integer_range(low: int, high: int) -> None:
    with pytest.raises(ValueError, match="integers"):
        ManagerRng.seeded(1).integers(low, high, (1,))


def test_rng_rejects_unsupported_choice_semantics() -> None:
    rng = ManagerRng.seeded(1)

    with pytest.raises(TypeError, match="probabilities must be a torch.Tensor"):
        rng.choice(2, (2,), p=[0.5, 0.5])  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="probabilities must live on"):
        rng.choice(2, (2,), p=torch.ones(2, device="meta"))
    with pytest.raises(ValueError, match="probabilities must have shape"):
        rng.choice(3, (2,), p=torch.ones(2))


def test_rng_restores_generator_state_in_place() -> None:
    rng = ManagerRng.seeded(23)
    rng.random((3,))
    state = rng.get_state()
    second = rng.random((3,))
    rng.set_state(state.clone())

    torch.testing.assert_close(second, rng.random((3,)))


def test_rng_reseeding_reproduces_the_stream() -> None:
    rng = ManagerRng.seeded(7)
    first = rng.random(4)

    rng.manual_seed(7)
    torch.testing.assert_close(first, rng.random(4))


def test_rng_set_state_restores_stream_position_across_draw_kinds() -> None:
    rng = ManagerRng.seeded(7)
    rng.uniform(0.0, 1.0, (4,))
    state = rng.get_state()

    draws = (
        rng.standard_normal(4),
        rng.normal(0.0, 1.0, (4,)),
        rng.integers(0, 2, (4,)),
        rng.choice(2, 4, p=torch.tensor((0.5, 0.5))),
    )
    rng.set_state(state)
    redraws = (
        rng.standard_normal(4),
        rng.normal(0.0, 1.0, (4,)),
        rng.integers(0, 2, (4,)),
        rng.choice(2, 4, p=torch.tensor((0.5, 0.5))),
    )
    for draw, redraw in zip(draws, redraws, strict=True):
        torch.testing.assert_close(draw, redraw)


def test_numpy_manager_stream_stays_independent() -> None:
    # Contract guard: adding the Manager RNG adapter must not alter NumPy RNG.
    ManagerRng.seeded(11).random((2,))

    np.testing.assert_allclose(
        np.random.default_rng(11).random(2), np.random.default_rng(11).random(2)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rng_can_own_cuda_generator() -> None:
    device = torch.device("cuda", index=torch.cuda.current_device())
    rng = ManagerRng.seeded(5, device="cuda")

    assert rng.device == device
    assert rng.generator.device == device
    values = rng.uniform(-1.0, 1.0, (4,))
    assert values.is_cuda

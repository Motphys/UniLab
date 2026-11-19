"""Custom persistent step-output carriers (issue #2628).

Tasks that materialize per-step data (transition snapshots, custom metric
tensors) register a fixed-address carrier with ``TorchEnv.register_step_carrier``;
the env publishes it through the same publication ring as obs/reward/done, so
consumers hold the published slot view without a defensive per-step clone.
"""

from __future__ import annotations

import pytest
import torch

from tests.envs.test_fixed_shape_autoreset import CUDA, DEVICES, NUM_ENVS, _make_env
from unilab.base.torch_env import TorchEnvState


def _zeros(device: str) -> torch.Tensor:
    return torch.zeros((NUM_ENVS, 1), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_register_step_carrier_requires_persistent_step_buffers() -> None:
    env, _ = _make_env("cpu")
    carrier = torch.zeros((NUM_ENVS, 2), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="persistent step buffers"):
        env.register_step_carrier("transition", carrier)


def test_register_step_carrier_validates_name_tensor_and_device() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True, persistent_step_buffers=True)
    carrier = torch.zeros((NUM_ENVS, 2), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="non-empty string"):
        env.register_step_carrier("", carrier)
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        env.register_step_carrier("bad", carrier.tolist())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be on"):
        env.register_step_carrier("bad", torch.zeros((NUM_ENVS, 2), device="meta"))
    env.register_step_carrier("transition", carrier)
    with pytest.raises(ValueError, match="already registered"):
        env.register_step_carrier("transition", torch.zeros_like(carrier))


def test_register_step_carrier_fails_closed_after_the_first_publish() -> None:
    env, _ = _make_env("cpu", masked_autoreset=True, persistent_step_buffers=True)
    env.reset()
    carrier = torch.zeros((NUM_ENVS, 2), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="before the first reset"):
        env.register_step_carrier("transition", carrier)


# ---------------------------------------------------------------------------
# Ring publication semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_registered_carriers_publish_through_the_ring(device: str) -> None:
    env, _ = _make_env(
        device, masked_autoreset=True, persistent_step_buffers=True, published_buffer_depth=2
    )
    working = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)
    env.register_step_carrier("transition", working)
    env.reset()

    published: list[tuple[TorchEnvState, torch.Tensor]] = []
    for step in range(4):
        # Simulate the task writing the carrier in place during update_state.
        working.fill_(float(step + 1))
        state = env.step(_zeros(device))
        published.append((state, state.carriers["transition"].clone()))

    # The slot objects alternate with a two-deep ring.
    assert published[0][0] is published[2][0]
    assert published[1][0] is published[3][0]
    assert published[0][0] is not published[1][0]
    # The ring slot tensor is storage the env owns, not the working carrier.
    latest = env.state
    assert latest is not None
    assert latest.carriers["transition"].data_ptr() != working.data_ptr()
    # Only the freshest two slots still hold their publish-time content; older
    # slots were rotated and overwritten.
    for step, (state, values) in enumerate(published[-2:], start=len(published) - 2):
        torch.testing.assert_close(
            state.carriers["transition"], torch.full_like(values, float(step + 1))
        )

    # The off-policy contract: the latest published view survives one further
    # step (the other ring slot is published) and is rotated onto at the
    # second.
    held_view = latest.carriers["transition"]
    held_values = held_view.clone()
    working.fill_(99.0)
    env.step(_zeros(device))
    torch.testing.assert_close(held_view, held_values)
    working.fill_(100.0)
    env.step(_zeros(device))
    torch.testing.assert_close(held_view, torch.full_like(held_values, 100.0))


@pytest.mark.parametrize("device", DEVICES)
def test_unregistered_env_publishes_empty_carriers(device: str) -> None:
    env, _ = _make_env(
        device, masked_autoreset=True, persistent_step_buffers=True, published_buffer_depth=2
    )
    env.reset()
    state = env.step(_zeros(device))
    assert state.carriers == {}


def test_registered_carriers_are_zero_copy_views_at_ring_depth_zero() -> None:
    env, _ = _make_env(
        "cpu", masked_autoreset=True, persistent_step_buffers=True, published_buffer_depth=0
    )
    working = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device="cpu")
    env.register_step_carrier("transition", working)
    env.reset()
    working.fill_(7.0)
    state = env.step(_zeros("cpu"))
    # Depth 0 publishes the working carriers themselves: no ring, no copies.
    assert state.carriers["transition"] is working
    torch.testing.assert_close(state.carriers["transition"], torch.full((NUM_ENVS, 3), 7.0))


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
def test_registered_carriers_publish_under_autoreset_graph_replay() -> None:
    env, _ = _make_env("cuda", autoreset_cuda_graph=True)
    working = torch.zeros((NUM_ENVS, 2), dtype=torch.float32, device="cuda")
    env.register_step_carrier("transition", working)
    env.reset()

    published: list[tuple[TorchEnvState, torch.Tensor]] = []
    for step in range(6):
        working.fill_(float(step + 1))
        state = env.step(_zeros("cuda"))
        published.append((state, state.carriers["transition"].clone()))

    graph = env.autoreset_graph
    assert graph is not None and graph.captured
    # The freshest two slots hold their publish-time content across the graph
    # replays; the captured autoreset segment does not touch the carriers.
    for step, (state, values) in enumerate(published[-2:], start=len(published) - 2):
        torch.testing.assert_close(
            state.carriers["transition"], torch.full_like(values, float(step + 1))
        )

    # A held view stays valid across one further replayed step.
    latest = env.state
    assert latest is not None
    held_view = latest.carriers["transition"]
    held_values = held_view.clone()
    working.fill_(42.0)
    env.step(_zeros("cuda"))
    torch.testing.assert_close(held_view, held_values)

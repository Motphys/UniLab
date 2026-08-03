from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from unilab.envs.locomotion.g1.managed_reward_terms import (
    G1_REWARD_TERM_REGISTRY,
    NUMPY_G1_REWARD_MATH,
    TORCH_G1_REWARD_MATH,
    G1RewardContext,
    G1RewardScratch,
    bind_g1_reward_terms,
)


def _scratch(
    *,
    num_envs: int,
    action_dim: int,
    torch_device: torch.device | None,
) -> G1RewardScratch:
    def empty(shape: tuple[int, ...], *, boolean: bool = False) -> Any:
        if torch_device is not None:
            return torch.empty(
                shape,
                dtype=torch.bool if boolean else torch.float32,
                device=torch_device,
            )
        return np.empty(shape, dtype=bool if boolean else np.float32)

    return G1RewardScratch(
        bool_a=empty((num_envs,), boolean=True),
        bool_b=empty((num_envs,), boolean=True),
        scalar_b=empty((num_envs,)),
        scalar_c=empty((num_envs,)),
        scalar_d=empty((num_envs,)),
        vector2=empty((num_envs, 2)),
        action=empty((num_envs, action_dim)),
        left_height=empty((num_envs,)),
        right_height=empty((num_envs,)),
    )


def _contexts(
    *, torch_device: torch.device | None = None
) -> tuple[G1RewardContext, G1RewardContext]:
    if torch_device is None:
        torch_device = torch.device("cpu")
    num_envs = 7
    action_dim = 29
    rng = np.random.default_rng(871)
    arrays = {
        "commands": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "current_actions": rng.normal(size=(num_envs, action_dim)).astype(np.float32),
        "last_actions": rng.normal(size=(num_envs, action_dim)).astype(np.float32),
        "gait_phase": rng.uniform(0.0, 2.0 * np.pi, size=(num_envs, 2)).astype(np.float32),
        "root_position": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "dof_position": rng.normal(size=(num_envs, action_dim)).astype(np.float32),
        "linear_velocity": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "gyro": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "upvector": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "left_foot_position": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "right_foot_position": rng.normal(size=(num_envs, 3)).astype(np.float32),
        "default_angles": rng.normal(size=(action_dim,)).astype(np.float32),
        "pose_weights": rng.uniform(0.1, 2.0, size=(action_dim,)).astype(np.float32),
        "upper_body_pose_weights": rng.uniform(0.0, 2.0, size=(action_dim,)).astype(np.float32),
    }
    arrays["commands"][:, 0] = np.asarray(
        (-0.2, 0.0, 1.0e-7, 0.03, 0.2, 0.8, 1.5), dtype=np.float32
    )
    arrays["linear_velocity"][:, 0] = np.asarray(
        (-0.1, 0.0, 0.02, 0.049, 0.05, 0.4, 2.0), dtype=np.float32
    )
    arrays["right_foot_position"][:, :2] = arrays["left_foot_position"][:, :2]
    arrays["right_foot_position"][::2, 0] += np.float32(0.3)

    common = {
        **arrays,
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "min_forward_speed_for_gait_reward": 0.05,
        "close_feet_threshold": 0.15,
    }
    numpy_context = G1RewardContext(
        **common,
        scratch=_scratch(num_envs=num_envs, action_dim=action_dim, torch_device=None),
    )
    torch_context = G1RewardContext(
        **{
            key: torch.from_numpy(value.copy()).to(torch_device)
            if isinstance(value, np.ndarray)
            else value
            for key, value in common.items()
        },
        scratch=_scratch(
            num_envs=num_envs,
            action_dim=action_dim,
            torch_device=torch_device,
        ),
    )
    return numpy_context, torch_context


@pytest.mark.parametrize("name", tuple(G1_REWARD_TERM_REGISTRY))
def test_all_registered_terms_match_numpy_and_torch_individually(name: str) -> None:
    numpy_context, torch_context = _contexts()
    numpy_out = np.empty((7,), dtype=np.float32)
    torch_out = torch.empty((7,), dtype=torch.float32)

    term = G1_REWARD_TERM_REGISTRY[name]
    term.evaluate(NUMPY_G1_REWARD_MATH, numpy_context, out=numpy_out)
    term.evaluate(TORCH_G1_REWARD_MATH, torch_context, out=torch_out)

    np.testing.assert_allclose(torch_out.numpy(), numpy_out, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.slow
@pytest.mark.parametrize("name", tuple(G1_REWARD_TERM_REGISTRY))
def test_all_registered_terms_match_numpy_and_cuda_individually(name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA reward parity requires a CUDA-capable Torch runtime")
    device = torch.device("cuda:0")
    numpy_context, torch_context = _contexts(torch_device=device)
    numpy_out = np.empty((7,), dtype=np.float32)
    torch_out = torch.empty((7,), dtype=torch.float32, device=device)

    term = G1_REWARD_TERM_REGISTRY[name]
    term.evaluate(NUMPY_G1_REWARD_MATH, numpy_context, out=numpy_out)
    term.evaluate(TORCH_G1_REWARD_MATH, torch_context, out=torch_out)

    np.testing.assert_allclose(torch_out.cpu().numpy(), numpy_out, rtol=1.0e-5, atol=1.0e-6)


def test_registry_is_the_single_supported_term_inventory() -> None:
    assert set(G1_REWARD_TERM_REGISTRY) == {
        "action_rate",
        "alive",
        "ang_vel_xy",
        "base_height",
        "feet_phase",
        "feet_phase_contrast",
        "forward_progress",
        "lin_vel_z",
        "orientation",
        "penalty_action_rate",
        "penalty_ang_vel_xy",
        "penalty_close_feet_xy",
        "penalty_orientation",
        "pose",
        "tracking_ang_vel",
        "tracking_lin_vel",
        "under_speed",
        "upper_body_pose",
    }
    assert G1_REWARD_TERM_REGISTRY["orientation"].evaluate is (
        G1_REWARD_TERM_REGISTRY["penalty_orientation"].evaluate
    )
    assert G1_REWARD_TERM_REGISTRY["ang_vel_xy"].evaluate is (
        G1_REWARD_TERM_REGISTRY["penalty_ang_vel_xy"].evaluate
    )
    assert G1_REWARD_TERM_REGISTRY["action_rate"].evaluate is (
        G1_REWARD_TERM_REGISTRY["penalty_action_rate"].evaluate
    )


def test_binding_rejects_an_unregistered_term_before_runtime_dispatch() -> None:
    with pytest.raises(ValueError, match="not registered"):
        bind_g1_reward_terms((("missing_reward", 1.0),))

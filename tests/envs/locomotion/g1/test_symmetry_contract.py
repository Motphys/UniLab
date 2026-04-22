from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from unilab.base import registry
from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig
from unilab.utils.algo_utils import ensure_registries

pytest.importorskip("mujoco", reason="mujoco is required for G1 symmetry contract tests")


def _reward_config() -> G1WalkRewardConfig:
    return G1WalkRewardConfig(
        scales={"tracking_lin_vel": 2.0, "alive": 10.0},
        tracking_sigma=0.25,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        close_feet_threshold=0.15,
        pose_weights=[0.01] * 29,
    )


def test_g1_walk_flat_symmetry_contract_matches_obs_groups():
    ensure_registries()
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": _reward_config()},
        ),
    )

    try:
        layouts = env.get_symmetry_obs_layouts()
        assert set(layouts) == {"obs", "critic"}
        for group_name, layout in layouts.items():
            assert sum(dim for _, dim in layout) == env.obs_groups_spec[group_name]
    finally:
        env.close()


def test_g1_walk_flat_symmetry_can_augment_critic_group():
    ensure_registries()
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": _reward_config()},
        ),
    )

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        action_dim = env.action_space.shape[0]
        obs = torch.zeros((1, env.obs_groups_spec["obs"]))
        critic = torch.zeros((1, env.obs_groups_spec["critic"]))
        actions = torch.zeros((1, action_dim))

        actor_aug, action_aug = augmentation.augment_obs_and_actions(obs, actions, obs_group="obs")
        critic_aug, critic_action_aug = augmentation.augment_obs_and_actions(
            critic,
            actions,
            obs_group="critic",
        )

        assert actor_aug.shape == (2, env.obs_groups_spec["obs"])
        assert critic_aug.shape == (2, env.obs_groups_spec["critic"])
        assert action_aug.shape == (2, action_dim)
        assert critic_action_aug.shape == (2, action_dim)
    finally:
        env.close()

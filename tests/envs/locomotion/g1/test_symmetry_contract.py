"""G1 symmetry contract on the Manager-Based runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.training.backend_adapter import BackendAdapter

pytest.importorskip("mujoco", reason="mujoco is required for G1 symmetry contract tests")

ROOT_DIR = Path(__file__).parents[4]
CONF_DIR = ROOT_DIR / "conf"


def _make_env(task_name: str = "G1WalkFlat", num_envs: int = 1) -> Any:
    owner = "g1_23dof_walk_flat/mujoco" if "23Dof" in task_name else "g1_walk_flat/mujoco"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}"])
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    return registry.make(
        task_name,
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=env_cfg_override,
    )


def test_g1_walk_flat_symmetry_contract_matches_obs_groups():
    ensure_registries()
    env = cast(Any, _make_env())

    try:
        layouts = env.get_symmetry_obs_layouts()
        assert set(layouts) == {"obs", "critic"}
        assert [name for name, _ in layouts["obs"]] == [
            "gyro",
            "gravity",
            "dof_pos",
            "dof_vel",
            "actions",
            "command",
            "gait_phase",
        ]
        assert [name for name, _ in layouts["critic"]][-1] == "linvel"
        for group_name, layout in layouts.items():
            assert sum(dim for _, dim in layout) == env.obs_groups_spec[group_name]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_name", "obs_dim", "critic_dim", "action_dim"),
    [
        ("G1WalkFlat", 98, 101, 29),
        ("G1Walk23DofFlat", 80, 83, 23),
    ],
)
def test_g1_walk_symmetry_can_augment_critic_group(
    task_name: str, obs_dim: int, critic_dim: int, action_dim: int
):
    ensure_registries()
    env = cast(Any, _make_env(task_name))

    try:
        augmentation = env.build_symmetry_augmentation(device="cpu")
        assert augmentation is not None

        assert env.action_space.shape[0] == action_dim
        obs = torch.zeros((1, env.obs_groups_spec["obs"]))
        critic = torch.zeros((1, env.obs_groups_spec["critic"]))
        actions = torch.zeros((1, action_dim))

        assert env.obs_groups_spec["obs"] == obs_dim
        assert env.obs_groups_spec["critic"] == critic_dim

        actor_aug, action_aug = augmentation.augment_obs_and_actions(obs, actions, obs_group="obs")
        critic_aug, critic_action_aug = augmentation.augment_obs_and_actions(
            critic,
            actions,
            obs_group="critic",
        )
        actor_obs_aug = augmentation.augment_obs(obs, obs_group="obs")
        critic_obs_aug = augmentation.augment_obs(critic, obs_group="critic")

        assert actor_aug.shape == (2, obs_dim)
        assert critic_aug.shape == (2, critic_dim)
        assert action_aug.shape == (2, action_dim)
        assert critic_action_aug.shape == (2, action_dim)
        assert torch.equal(actor_obs_aug, actor_aug)
        assert torch.equal(critic_obs_aug, critic_aug)
    finally:
        env.close()

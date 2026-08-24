from __future__ import annotations

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.migration_matrix import migration_record


def test_a2arm_is_a_manager_based_registered_owner() -> None:
    registry.ensure_registries()
    assert registry.contains("A2ArmPosForce")
    cfg = registry.materialize_env_config("A2ArmPosForce")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    assert cfg.policy_observation_group == "policy"
    assert cfg.critic_observation_group == "critic"
    assert migration_record("A2ArmPosForce").target == "complete"

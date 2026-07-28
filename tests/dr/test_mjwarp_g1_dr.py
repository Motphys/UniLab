"""Owner-level DR semantics for the first independent ``mjwarp`` G1 profile."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_g1_kp_kd_owner_semantics_have_physics_effect_or_are_disabled() -> None:
    """The Phase-2 owner chooses the explicit disabled branch, not silent filtering.

    Actuator-gain DR becomes an effect-tested capability in the typed mutation
    phase.  Until then the owner YAML itself must override G1's true defaults.
    """
    conf_dir = Path(__file__).resolve().parents[2] / "conf" / "ppo"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_walk_flat/mjwarp"])

    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False

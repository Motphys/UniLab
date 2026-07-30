"""Owner-level DR semantics for the first independent ``mjwarp`` G1 profile."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.dr.dr_utils import build_common_reset_randomization
from unilab.envs.locomotion.g1.joystick import G1WalkDomainRandomizationProvider


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


def test_g1_armature_baseline_builds_reset_payload_from_cold_snapshot() -> None:
    baseline = np.asarray([0.0, 0.01, 0.02], dtype=np.float32)
    provider = G1WalkDomainRandomizationProvider(base_dof_armature=baseline.copy())
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_rand=SimpleNamespace(
                randomize_base_mass=False,
                randomize_body_mass=False,
                random_com=False,
                randomize_gravity=False,
                randomize_ground_friction=False,
                randomize_dof_armature=True,
                dof_armature_multiplier_range=[1.0, 1.0],
                randomize_kp=False,
                randomize_kd=False,
            )
        ),
        _num_action=2,
    )

    _, _, _, cached = provider._get_reset_randomization_baselines(env)
    assert cached is not None
    payload = build_common_reset_randomization(env, 2, base_dof_armature=cached)

    assert payload is not None
    assert payload.dof_armature is not None
    np.testing.assert_allclose(payload.dof_armature, np.broadcast_to(baseline, (2, 3)))
    baseline.fill(99.0)
    np.testing.assert_allclose(payload.dof_armature, [[0.0, 0.01, 0.02]] * 2)

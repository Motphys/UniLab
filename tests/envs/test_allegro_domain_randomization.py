from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
except Exception:
    pytest.skip(
        "mujoco.batch_env not available (platform/libstdc++ issue)", allow_module_level=True
    )

from unilab.utils.algo_utils import ensure_registries


@pytest.mark.slow
def test_allegro_mujoco_reset_applies_base_mass_and_com_domain_randomization(
    default_allegro_reward_config: dict[str, Any],
) -> None:
    ensure_registries()

    from unilab.base import registry

    env = registry.make(
        "AllegroInhandRotation",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={
            "reward_config": default_allegro_reward_config,
            "domain_rand": {
                "randomize_base_mass": True,
                "added_mass_range": [-0.02, 0.02],
                "random_com": True,
                "com_offset_x": [-0.005, 0.005],
            },
        },
    )
    env_obj: Any = env
    try:
        env_obj.init_state()
        backend: Any = env_obj._backend
        pool = backend._pool
        base_body_id = int(backend._base_body_id)
        body_mass = np.stack([pool.get_field(i, "body_mass") for i in range(env_obj.num_envs)])
        body_ipos = np.stack([pool.get_field(i, "body_ipos") for i in range(env_obj.num_envs)])
        body_ipos = body_ipos.reshape(env_obj.num_envs, -1, 3)

        base_mass = float(backend._base_body_mass[base_body_id])
        base_ipos = np.asarray(backend._base_body_ipos[base_body_id])

        randomized_base_mass = body_mass[:, base_body_id]
        randomized_base_ipos = body_ipos[:, base_body_id]

        delta_mass = randomized_base_mass - base_mass
        delta_x = randomized_base_ipos[:, 0] - base_ipos[0]
        delta_yz = randomized_base_ipos[:, 1:] - base_ipos[1:]

        assert np.unique(np.round(delta_mass, 6)).size > 1
        assert np.unique(np.round(delta_x, 6)).size > 1

        assert np.all(delta_mass >= -0.02)
        assert np.all(delta_mass <= 0.02)
        assert np.all(delta_x >= -0.005)
        assert np.all(delta_x <= 0.005)
        np.testing.assert_allclose(delta_yz, 0.0)
    finally:
        env_obj.close()

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

from unilab.base.registry import ensure_registries


@pytest.mark.slow
def test_go1_mujoco_reset_applies_base_mass_and_com_domain_randomization(
    default_go1_reward_config: dict[str, Any],
) -> None:
    ensure_registries()

    from unilab.base import registry

    env = registry.make(
        "Go1JoystickFlat",
        num_envs=4,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": default_go1_reward_config},
    )
    env_obj: Any = env
    try:
        env_obj.init_state()
        backend: Any = env_obj._backend
        base_body_id = int(backend._base_body_id)
        body_mass = np.stack(
            [backend._pool.get_field(i, "body_mass") for i in range(env_obj.num_envs)]
        )
        body_ipos = np.stack(
            [backend._pool.get_field(i, "body_ipos") for i in range(env_obj.num_envs)]
        )
        body_ipos = body_ipos.reshape(env_obj.num_envs, -1, 3)

        base_mass = float(backend._base_body_mass[base_body_id])
        base_ipos = np.asarray(backend._base_body_ipos[base_body_id])
        randomized_base_mass = body_mass[:, base_body_id]
        randomized_base_ipos = body_ipos[:, base_body_id]
        delta_x = randomized_base_ipos[:, 0] - base_ipos[0]
        delta_yz = randomized_base_ipos[:, 1:] - base_ipos[1:]

        assert np.unique(np.round(randomized_base_mass, 6)).size > 1
        assert np.unique(np.round(delta_x, 6)).size > 1
        assert np.all(randomized_base_mass >= base_mass - 1.5)
        assert np.all(randomized_base_mass <= base_mass + 1.5)
        assert np.all(delta_x >= -0.05)
        assert np.all(delta_x <= 0.05)
        np.testing.assert_allclose(delta_yz, 0.0)
    finally:
        env_obj.close()

from __future__ import annotations

import numpy as np

from benchmark import benchmark_sharpa_init_dr_construct as sharpa_benchmark


def test_sharpa_init_dr_benchmark_uses_owner_scale_list_key() -> None:
    cfg = sharpa_benchmark._compose_cfg(
        "sharpa_inhand/mujoco",
        lower=0.5,
        upper=0.8,
        variant_count=4,
    )

    np.testing.assert_allclose(
        np.asarray(cfg.env.domain_rand.scale_list, dtype=np.float64),
        np.linspace(0.5, 0.8, 4, dtype=np.float64),
    )
    assert "scale_list" not in cfg.env

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.manipulation.sharpa_inhand.rotation import SharpaInhandRotationDRProvider


def test_sharpa_provider_builds_mujoco_init_geom_scale_plan() -> None:
    env = SimpleNamespace(
        _backend=SimpleNamespace(backend_type="mujoco"),
        _object_geom_base_size=np.array([0.02, 0.016, 0.0], dtype=np.float64),
        scale_values=np.array([0.5, 0.8], dtype=np.float64),
        scale_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        cfg=SimpleNamespace(object_geom_name="object"),
    )

    plan = SharpaInhandRotationDRProvider().build_init_randomization_plan(env)

    assert plan is not None
    np.testing.assert_array_equal(
        plan.model_assignments,
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    assert len(plan.model_variants) == 2
    assert plan.model_variants[0].geom_size_overrides[0].geom_name == "object"
    np.testing.assert_allclose(
        plan.model_variants[0].geom_size_overrides[0].size,
        [0.01, 0.008, 0.0],
    )
    np.testing.assert_allclose(
        plan.model_variants[1].geom_size_overrides[0].size,
        [0.016, 0.0128, 0.0],
    )


def test_sharpa_provider_skips_non_mujoco_init_geom_scale_plan() -> None:
    env = SimpleNamespace(
        _backend=SimpleNamespace(backend_type="motrix"),
        _object_geom_base_size=np.array([0.02, 0.016, 0.0], dtype=np.float64),
        scale_values=np.array([0.5], dtype=np.float64),
        scale_ids=np.array([0], dtype=np.int32),
        cfg=SimpleNamespace(object_geom_name="object"),
    )

    plan = SharpaInhandRotationDRProvider().build_init_randomization_plan(env)

    assert plan is None

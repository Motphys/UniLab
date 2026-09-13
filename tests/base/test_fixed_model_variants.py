from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from unisim.dr.types import DomainRandomizationCapabilities, FixedVariantLayout

from unilab.base.base import EnvCfg
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.base.variants import (
    FixedModelVariantCatalogCfg,
    FixedModelVariantCfg,
    _build_fixed_variant_plan,
    _require_fixed_variant_support,
)
from unilab.envs import manager_based_rl_env
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg, make_manager_based_rl_env


def _catalog(explicit_variant_names: tuple[str, ...] = ()) -> FixedModelVariantCatalogCfg:
    return FixedModelVariantCatalogCfg(
        variants=(
            FixedModelVariantCfg("tool_a", "tools/a.xml"),
            FixedModelVariantCfg("tool_b", "tools/b.xml"),
        ),
        explicit_variant_names=explicit_variant_names,
    )


def _capabilities() -> DomainRandomizationCapabilities:
    return DomainRandomizationCapabilities(
        supports_fixed_variants=True,
        supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
    )


def test_hydra_materializes_typed_catalog_and_explicit_names() -> None:
    cfg = EnvCfg()
    apply_cfg_overrides(
        cfg,
        OmegaConf.create(
            {
                "fixed_model_variants": {
                    "variants": [
                        {"name": "tool_a", "source_model_file": "tools/a.xml"},
                        {"name": "tool_b", "source_model_file": "tools/b.xml"},
                    ],
                    "explicit_variant_names": ["tool_b", "tool_a"],
                }
            }
        ),
    )
    cfg.validate()

    assert cfg.fixed_model_variants == _catalog(("tool_b", "tool_a"))


def test_default_assignment_is_round_robin_and_immutable() -> None:
    plan = _build_fixed_variant_plan(_catalog(), num_envs=5)

    np.testing.assert_array_equal(plan.assignment, np.array([0, 1, 0, 1, 0], dtype=np.int32))
    assert not plan.assignment.flags.writeable
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        plan.assignment[0] = 1


def test_explicit_assignment_uses_names_not_engine_objects() -> None:
    plan = _build_fixed_variant_plan(_catalog(("tool_b", "tool_a", "tool_b")), num_envs=3)

    np.testing.assert_array_equal(plan.assignment, np.array([1, 0, 1], dtype=np.int32))
    assert tuple(variant.model_file for variant in plan.variants) == (
        "tools/a.xml",
        "tools/b.xml",
    )


def test_catalog_and_assignment_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate 'tool_a'"):
        FixedModelVariantCatalogCfg(
            variants=(
                FixedModelVariantCfg("tool_a", "a.xml"),
                FixedModelVariantCfg("tool_a", "b.xml"),
            )
        )
    with pytest.raises(ValueError, match="must not be empty"):
        FixedModelVariantCatalogCfg()
    with pytest.raises(ValueError, match="source_model_file must be a non-empty string"):
        FixedModelVariantCatalogCfg(variants=(FixedModelVariantCfg("tool_a", " "),))
    with pytest.raises(ValueError, match="unknown variants"):
        _catalog(("tool_a", "missing"))
    with pytest.raises(ValueError, match="exactly num_envs"):
        _build_fixed_variant_plan(_catalog(("tool_a",)), num_envs=2)


def test_support_negotiation_requires_the_complete_plan_contract() -> None:
    plan = _build_fixed_variant_plan(_catalog(), num_envs=2)
    _require_fixed_variant_support(_capabilities(), plan)

    unsupported = DomainRandomizationCapabilities()
    with pytest.raises(NotImplementedError, match="cannot realize"):
        _require_fixed_variant_support(unsupported, plan)

    partial = DomainRandomizationCapabilities(supports_fixed_variants=True)
    with pytest.raises(NotImplementedError, match="layout 'same_layout' is unsupported"):
        _require_fixed_variant_support(partial, plan)


def test_variant_owner_module_does_not_reference_engine_internals() -> None:
    repo_root = Path(__file__).parents[2]
    source = (repo_root / "src" / "unilab" / "base" / "variants.py").read_text(encoding="utf-8")
    manager_source = (repo_root / "src" / "unilab" / "envs" / "manager_based_rl_env.py").read_text(
        encoding="utf-8"
    )

    assert "mjbatch" not in source
    assert "MjSpec" not in source
    assert "mujoco" not in source
    assert "mjbatch" not in manager_source
    assert "MjSpec" not in manager_source


def test_manager_env_fails_closed_before_variant_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedBackend:
        backend_type = "test"
        num_envs = 2
        cleaned = False

        def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
            return DomainRandomizationCapabilities()

        def cleanup_scene_assets(self) -> None:
            self.cleaned = True

    backend = UnsupportedBackend()
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="scene.xml",
            entities={"robot": EntityCfg(root_body_name="base")},
        ),
        max_episode_seconds=1.0,
        fixed_model_variants=_catalog(),
    )
    monkeypatch.setattr(manager_based_rl_env, "env_backend_kwargs", lambda _cfg: {})
    monkeypatch.setattr(
        manager_based_rl_env,
        "create_backend",
        lambda *_args, **_kwargs: backend,
    )

    def _fail_if_env_is_constructed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsupported variants must fail before env construction")

    monkeypatch.setattr(manager_based_rl_env, "ManagerBasedRlEnv", _fail_if_env_is_constructed)

    with pytest.raises(NotImplementedError, match="cannot realize"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")

    assert backend.cleaned is True
    assert cfg.scene is not None
    assert cfg.scene.fixed_variant_plan is not None
    np.testing.assert_array_equal(
        cfg.scene.fixed_variant_plan.assignment, np.array([0, 1], dtype=np.int32)
    )

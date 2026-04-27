"""Tests for registry bootstrap and torch actor factory helpers."""

from __future__ import annotations

import importlib
import logging
import sys

import pytest

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.base.registry import ensure_registries


class TestEnsureRegistries:
    """Tests for ensure_registries."""

    def test_runs_without_error(self) -> None:
        """Default call should work with standard packages."""
        ensure_registries()

    def test_is_idempotent(self) -> None:
        """Multiple calls should be safe."""
        ensure_registries()
        ensure_registries()
        ensure_registries()

    def test_fail_on_error_default_raises(self) -> None:
        """By default, invalid packages should raise ImportError."""
        with pytest.raises(ImportError):
            ensure_registries(["nonexistent_package_12345"])

    def test_fail_on_error_false_logs_warning(self, caplog) -> None:
        """With fail_on_error=False, invalid packages log warning."""
        with caplog.at_level(logging.WARNING):
            ensure_registries(["nonexistent_package_12345"], fail_on_error=False)

        assert "nonexistent_package_12345" in caplog.text

    def test_optional_package_logs_warning(self, caplog) -> None:
        """Optional packages that fail to import log warning instead of raising."""
        with caplog.at_level(logging.WARNING):
            ensure_registries(
                ["nonexistent_package_12345"],
                optional_packages=["nonexistent_package_12345"],
            )

        assert "nonexistent_package_12345" in caplog.text
        assert "Optional" in caplog.text

    def test_mixed_optional_and_required(self, caplog) -> None:
        """Mix of optional (fails gracefully) and required (works)."""
        # Use real package + fake optional package
        with caplog.at_level(logging.WARNING):
            ensure_registries(
                ["unilab.envs.locomotion", "nonexistent_optional_12345"],
                optional_packages=["nonexistent_optional_12345"],
            )

        # Should warn about optional but not raise
        assert "nonexistent_optional_12345" in caplog.text

    def test_empty_packages_list(self) -> None:
        """Empty packages list should be a no-op."""
        ensure_registries([])

    def test_non_package_module_import_is_supported(self) -> None:
        """A plain module path should be accepted without package scanning."""
        ensure_registries(["unilab.base.registry"])

    def test_declared_registry_module_failure_raises(self, tmp_path, monkeypatch) -> None:
        """Required packages should fail fast when declared registry modules fail to import."""
        pkg_dir = tmp_path / "registry_fail_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "__unilab_registry_modules__ = ('registry_fail_pkg.broken',)\n",
            encoding="utf-8",
        )
        (pkg_dir / "broken.py").write_text(
            "raise RuntimeError('bootstrap failure')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        sys.modules.pop("registry_fail_pkg", None)
        sys.modules.pop("registry_fail_pkg.broken", None)

        with pytest.raises(RuntimeError, match="registry_fail_pkg.broken"):
            ensure_registries(["registry_fail_pkg"])

    def test_optional_package_registry_module_failure_logs_warning(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        """Optional package declared-module failures should log warnings and continue."""
        pkg_dir = tmp_path / "registry_optional_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "__unilab_registry_modules__ = ('registry_optional_pkg.broken',)\n",
            encoding="utf-8",
        )
        (pkg_dir / "broken.py").write_text(
            "raise RuntimeError('optional failure')\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        sys.modules.pop("registry_optional_pkg", None)
        sys.modules.pop("registry_optional_pkg.broken", None)

        with caplog.at_level(logging.WARNING):
            ensure_registries(
                ["registry_optional_pkg"],
                optional_packages=["registry_optional_pkg"],
            )

        assert "registry_optional_pkg.broken" in caplog.text
        assert "Failed to import declared registry module" in caplog.text

    def test_invalid_registry_modules_contract_raises(self, tmp_path, monkeypatch) -> None:
        """Package bootstrap contract must be a sequence of module names."""
        pkg_dir = tmp_path / "registry_invalid_contract_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "__unilab_registry_modules__ = 'registry_invalid_contract_pkg.child'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        sys.modules.pop("registry_invalid_contract_pkg", None)

        with pytest.raises(TypeError, match="__unilab_registry_modules__"):
            ensure_registries(["registry_invalid_contract_pkg"])


class TestBuildActor:
    """Tests for build_actor."""

    def test_builds_sac_actor(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
        )
        assert hasattr(actor, "forward")

    def test_builds_sac_actor_without_layer_norm(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
        )
        assert hasattr(actor, "forward")

    def test_builds_td3_actor(self) -> None:
        actor = build_actor(
            algo_type="td3",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
            num_envs=1,
        )
        assert hasattr(actor, "forward")

    def test_builds_td3_actor_for_multiple_envs(self) -> None:
        actor = build_actor(
            algo_type="td3",
            obs_dim=8,
            action_dim=2,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
            num_envs=4,
        )
        assert hasattr(actor, "forward")

    def test_builds_flashsac_actor(self) -> None:
        actor = build_actor(
            algo_type="flashsac",
            obs_dim=98,
            action_dim=29,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
            actor_num_blocks=2,
            actor_noise_zeta_mu=2.0,
            actor_noise_zeta_max=16,
        )
        assert hasattr(actor, "forward")
        assert hasattr(actor, "explore")

    def test_raises_for_unknown_algo_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown algo_type"):
            build_actor(
                algo_type="unknown",
                obs_dim=10,
                action_dim=4,
                actor_hidden_dim=256,
                use_layer_norm=True,
                device="cpu",
            )

    def test_builds_sac_actor_with_different_dims(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=49,
            action_dim=12,
            actor_hidden_dim=512,
            use_layer_norm=True,
            device="cpu",
        )
        assert hasattr(actor, "forward")

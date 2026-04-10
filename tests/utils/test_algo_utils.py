"""Tests for unilab.utils.algo_utils."""

from __future__ import annotations

import pytest

from unilab.utils.algo_utils import build_actor, ensure_registries


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
        import logging

        with caplog.at_level(logging.WARNING):
            ensure_registries(["nonexistent_package_12345"], fail_on_error=False)

        assert "nonexistent_package_12345" in caplog.text

    def test_optional_package_logs_warning(self, caplog) -> None:
        """Optional packages that fail to import log warning instead of raising."""
        import logging

        with caplog.at_level(logging.WARNING):
            ensure_registries(
                ["nonexistent_package_12345"],
                optional_packages=["nonexistent_package_12345"],
            )

        assert "nonexistent_package_12345" in caplog.text
        assert "Optional" in caplog.text

    def test_mixed_optional_and_required(self, caplog) -> None:
        """Mix of optional (fails gracefully) and required (works)."""
        import logging

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

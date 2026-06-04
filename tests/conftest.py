"""Shared fixtures for UniLab tests.

Spawn-based collector subprocesses (off-policy / APPO runners) start as fresh
Python interpreters and therefore never execute this conftest. To make
test-only envs (e.g. ``DummyFlatTest``) discoverable in those subprocesses,
this module:

1. Hosts the env class in ``tests._test_registry.dummy_flat_env`` so it can be
   imported from anywhere — not just from a pytest conftest.
2. Sets the ``UNILAB_EXTRA_REGISTRY_PACKAGES`` environment variable so that
   ``unilab.base.registry.ensure_registries`` (called inside collector
   subprocesses) imports the test registry package and re-registers the env.
3. Prepends the repo root to ``PYTHONPATH`` so that ``tests._test_registry``
   resolves inside spawn subprocesses.

If you remove or rename this hook, ``make test-slow`` will fail with
``ValueError: Environment 'DummyFlatTest' is not registered.`` inside the
collector subprocess. See ``docs/sphinx/source/{lang}/4-developer_guide/
4-contributing.md`` ("Notes for ``make test-slow``") for the rationale.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

# ---------------------------------------------------------------------------
# Dummy flat env — no MuJoCo required
#
# The actual class + registry call live in ``tests._test_registry.dummy_flat_env``
# so that spawn-based collector subprocesses can re-register the env via
# ``ensure_registries`` + the ``UNILAB_EXTRA_REGISTRY_PACKAGES`` env var
# (subprocesses do not execute conftest.py).
# ---------------------------------------------------------------------------
from tests._test_registry.dummy_flat_env import (  # noqa: E402  (side-effect import)
    DUMMY_ENV_NAME as _DUMMY_ENV_NAME,
)
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.ipc.rollout_ring_buffer import RolloutRingBuffer

_DUMMY_OBS_DIM = 8
_DUMMY_ACT_DIM = 3

# Make the dummy env discoverable inside spawn collector subprocesses.
_existing = os.environ.get("UNILAB_EXTRA_REGISTRY_PACKAGES", "")
_pkgs = [p.strip() for p in _existing.split(",") if p.strip()]
if "tests._test_registry" not in _pkgs:
    _pkgs.append("tests._test_registry")
os.environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] = ",".join(_pkgs)

# Spawn-based collector subprocesses start as fresh interpreters and need to
# import ``tests._test_registry``. Make sure the repo root is on PYTHONPATH so
# that import resolves there.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_existing_pp = os.environ.get("PYTHONPATH", "")
_pp_parts = [p for p in _existing_pp.split(os.pathsep) if p]
if _repo_root not in _pp_parts:
    _pp_parts.insert(0, _repo_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(_pp_parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_SESSION_FAILED = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        global _TEST_SESSION_FAILED
        _TEST_SESSION_FAILED = True


@pytest.fixture(scope="session", autouse=True)
def _isolate_training_logs_for_tests(tmp_path_factory: pytest.TempPathFactory):
    """Keep training smoke-test artifacts out of the repository log tree."""
    log_root = tmp_path_factory.mktemp("unilab-training-logs")
    previous = pytest.MonkeyPatch()
    previous.setenv("UNILAB_TEST_LOG_ROOT", str(log_root))
    yield log_root
    previous.undo()
    if _TEST_SESSION_FAILED:
        print(f"Preserving UniLab test training logs after failure: {log_root}")
    else:
        shutil.rmtree(log_root, ignore_errors=True)


@pytest.fixture
def mp_ctx():
    return torch.multiprocessing.get_context("spawn")


@pytest.fixture
def tiny_replay_buffer():
    buf = ReplayBuffer(
        capacity=128, obs_dim=_DUMMY_OBS_DIM, action_dim=_DUMMY_ACT_DIM, device="cpu"
    )
    yield buf


@pytest.fixture
def tiny_storage():
    storage = RolloutRingBuffer(
        num_envs=4,
        num_steps=10,
        obs_dim=_DUMMY_OBS_DIM,
        action_dim=_DUMMY_ACT_DIM,
        num_slots=2,
        create=True,
    )
    yield storage
    storage.cleanup()


@pytest.fixture
def tiny_weight_shapes():
    """Small MLP param shapes dict — linear(8,16) + bias, linear(16,3) + bias."""
    return {
        "layer1.weight": torch.Size([16, 8]),
        "layer1.bias": torch.Size([16]),
        "layer2.weight": torch.Size([3, 16]),
        "layer2.bias": torch.Size([3]),
    }


@pytest.fixture
def mock_env_name() -> str:
    return _DUMMY_ENV_NAME


@pytest.fixture
def default_go1_reward_config():
    """Default reward config for Go1 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }


@pytest.fixture
def default_go2_reward_config():
    """Default reward config for Go2 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }


@pytest.fixture
def default_g1_reward_config():
    """Default reward config for G1 testing."""
    return {
        "scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.25,
            "forward_progress": 0.0,
            "under_speed": -0.2,
            "upper_body_pose": -0.05,
            "penalty_feet_ori": 0.0,
            "feet_phase": 1.0,
            "feet_phase_contrast": 1.0,
            "feet_phase_contact": 0.5,
            "feet_double_stance": -0.5,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.2,
            "base_height": -120.0,
            "orientation": -2.5,
            "action_rate": -0.005,
            "pose": -0.05,
        },
        "tracking_sigma": 0.25,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "base_height_target": 0.765,
        "min_forward_speed_for_gait_reward": 0.05,
        "min_base_height": 0.5,
        "max_tilt_deg": 35.0,
        "pose_weights": [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ],
    }


@pytest.fixture
def default_allegro_reward_config():
    """Default reward config for AllegroInhandRotation testing."""
    return {
        "scales": {
            "rotate": 1.25,
            "obj_linvel": -0.3,
            "pose_diff": -0.3,
            "torque": -0.1,
            "work": -2.0,
            "drop": 0.0,
        },
        "angvel_clip_min": -0.5,
        "angvel_clip_max": 0.5,
        "reset_z_threshold": 0.125,
    }


@pytest.fixture
def default_g1_walk_flat_reward_config():
    """Default reward config for G1 SAC testing."""
    return {
        "scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.5,
            "penalty_ang_vel_xy": -1.0,
            "penalty_orientation": -10.0,
            "penalty_action_rate": -2.0,
            "pose": -0.5,
            "penalty_feet_ori": -25.0,
            "feet_phase": 5.0,
            "alive": 10.0,
        },
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "min_base_height": 0.3,
        "max_tilt_deg": 65.0,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "close_feet_threshold": 0.15,
        "pose_weights": [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ],
    }

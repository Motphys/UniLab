"""G1Bfm cross-backend contracts.

The aux rewards are the risk here. Three of the eight depend on contact force and
together carry ~16% of the weighted aux signal while being non-zero on 98.5-99.5% of
steps; if a backend cannot read contact force they silently read zero, training still
runs, and the run is quietly wrong. These tests assert they are live.

See docs/BFM_root_cause_ledger.md §9.6-9.8, §9.11.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.base import registry
from unilab.training import ensure_registries

NUM_ENVS = 4
ROLLOUT_STEPS = 60
CONTACT_DEPENDENT_AUX = ("penalty_feet_ori", "penalty_slippage", "penalty_undesired_contact")
EXPECTED_OBS_GROUPS = {
    "state": 64,
    "last_action": 29,
    "privileged_state": 463,
    "history_actor": 372,
}


def _make(backend: str):
    if backend == "motrix":
        pytest.importorskip("motrixsim")
    ensure_registries()
    env = registry.make("G1Bfm", sim_backend=backend, num_envs=NUM_ENVS)
    env.init_state()
    return env


def _rollout_aux(env, steps: int = ROLLOUT_STEPS) -> dict[str, np.ndarray]:
    """Drive the env with a fixed action sequence and stack the per-step aux means."""
    rng = np.random.RandomState(0)
    acc: dict[str, list[float]] = {}
    for _ in range(steps):
        actions = rng.uniform(-0.3, 0.3, size=(NUM_ENVS, env._num_action)).astype(np.float32)
        state = env.step(actions)
        for key, value in state.info["aux_rewards"].items():
            acc.setdefault(key, []).append(float(np.mean(value)))
    return {key: np.asarray(values, np.float64) for key, values in acc.items()}


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
class TestBfmBackendParity:
    def test_registered_and_constructs(self, backend):
        env = _make(backend)
        assert env.num_envs == NUM_ENVS
        assert env._num_action == 29

    def test_obs_groups_spec_is_backend_independent(self, backend):
        """Policy I/O width must not depend on the backend, or checkpoints stop loading."""
        assert _make(backend).obs_groups_spec == EXPECTED_OBS_GROUPS

    def test_obs_shapes_and_finiteness(self, backend):
        env = _make(backend)
        state = env.step(np.zeros((NUM_ENVS, env._num_action), np.float32))
        assert set(state.obs) == set(EXPECTED_OBS_GROUPS)
        for key, width in EXPECTED_OBS_GROUPS.items():
            obs = np.asarray(state.obs[key])
            assert obs.shape == (NUM_ENVS, width)
            assert np.all(np.isfinite(obs))

    def test_contact_dependent_aux_rewards_are_live(self, backend):
        """The silent-failure guard: these must not be identically zero.

        A backend without contact force, or a threshold compared against a 0/1 found flag
        instead of newtons, leaves all three at zero for the whole rollout without
        raising anything.
        """
        aux = _rollout_aux(_make(backend))
        dead = [key for key in CONTACT_DEPENDENT_AUX if not np.any(aux[key] > 0.0)]
        assert not dead, f"{backend}: contact-dependent aux rewards never fired: {dead}"

    def test_all_aux_reward_keys_present_and_finite(self, backend):
        aux = _rollout_aux(_make(backend), steps=10)
        from unilab.algos.torch.bfm.runner import DEFAULT_AUX_REWARDS

        assert set(aux) == set(DEFAULT_AUX_REWARDS)
        for key, values in aux.items():
            assert np.all(np.isfinite(values)), key

    def test_reset_returns_finite_obs(self, backend):
        env = _make(backend)
        obs, _ = env.reset(np.arange(NUM_ENVS))
        assert set(obs) == set(EXPECTED_OBS_GROUPS)
        for value in obs.values():
            assert np.all(np.isfinite(np.asarray(value)))


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["mujoco", "motrix"])
def test_action_space_is_finite_and_samplable(backend):
    """The runner samples ``rng.uniform(low, high)`` for its seed steps.

    ``g1_bfm.xml`` sets no ``ctrlrange``, so both backends report "unlimited" but spell
    it differently -- MuJoCo ``[0, 0]``, Motrix ``[-inf, +inf]``. A non-finite bound
    makes ``rng.uniform`` raise ``OverflowError``, which only surfaces in a real training
    run: every env-level test drives ``env.step`` with actions of its own.
    """
    env = _make(backend)
    low = np.asarray(env.action_space.low, np.float64)
    high = np.asarray(env.action_space.high, np.float64)
    assert np.all(np.isfinite(low)), f"{backend} action_space.low is not finite: {low}"
    assert np.all(np.isfinite(high)), f"{backend} action_space.high is not finite: {high}"
    sampled = np.random.RandomState(0).uniform(low, high, (NUM_ENVS, env._num_action))
    assert np.all(np.isfinite(sampled))


@pytest.mark.slow
def test_action_space_agrees_across_backends():
    pytest.importorskip("motrixsim")
    mj, mx = _make("mujoco"), _make("motrix")
    np.testing.assert_array_equal(mj.action_space.low, mx.action_space.low)
    np.testing.assert_array_equal(mj.action_space.high, mx.action_space.high)


@pytest.mark.slow
def test_unsupported_backend_rejected():
    from unilab.envs.locomotion.g1.bfm import G1BfmCfg, G1BfmEnv

    with pytest.raises(ValueError, match="mujoco and motrix"):
        G1BfmEnv(G1BfmCfg(), num_envs=1, backend_type="isaac")


@pytest.mark.slow
def test_action_pipeline_is_backend_independent():
    """penalty_action_rate depends only on actions, so it must match exactly."""
    pytest.importorskip("motrixsim")
    mj = _rollout_aux(_make("mujoco"), steps=20)
    mx = _rollout_aux(_make("motrix"), steps=20)
    np.testing.assert_allclose(
        mj["penalty_action_rate"], mx["penalty_action_rate"], rtol=0, atol=1e-9
    )


@pytest.mark.slow
def test_geomless_bodies_report_zero_contact_on_both_backends():
    """Bodies with only visual geoms cannot collide; both backends must read 0 (§9.11)."""
    pytest.importorskip("motrixsim")
    geomless = (
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    )
    for backend in ("mujoco", "motrix"):
        env = _make(backend)
        rng = np.random.RandomState(0)
        for _ in range(20):
            env.step(rng.uniform(-0.4, 0.4, size=(NUM_ENVS, env._num_action)).astype(np.float32))
        for body in geomless:
            if body not in env._contact_bodies:
                continue
            force = np.asarray(env._touch(body))
            np.testing.assert_allclose(force, 0.0, atol=0.0, err_msg=f"{backend} {body}")

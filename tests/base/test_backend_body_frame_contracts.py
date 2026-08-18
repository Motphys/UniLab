"""Contracts added for motrix BFM support: ``get_base_ang_vel_b`` / ``get_dof_pos_limits``.

Both are additive: they must not change any pre-existing method's behaviour. The
MuJoCo assertions below are the regression gate for that claim — ``get_base_ang_vel_b``
has to be bit-identical to ``get_base_ang_vel`` there, because MuJoCo already stores a
free joint's angular velocity body-frame in ``qvel[3:6]``.

See docs/BFM_root_cause_ledger.md §9.3 / §9.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.scene import SceneCfg

MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_bfm.xml")
NUM_ENVS = 4
SIM_DT = 0.005
BASE_NAME = "pelvis"
NQ = 36  # 7 root + 29 dof
NV = 35  # 6 root + 29 dof

# Non-trivial attitudes: a frame error cannot hide behind an identity rotation, which
# is exactly why the pre-existing cross-backend test (zero qvel + identity quat) stayed
# green through the divergence recorded in the ledger.
QUATS_WXYZ = np.array(
    [
        [0.9689124, 0.0, 0.2474040, 0.0],
        [0.8253356, 0.2620026, 0.4396797, 0.2273244],
        [0.7071068, 0.7071068, 0.0, 0.0],
        [0.6087614, 0.3299871, -0.5566704, 0.4520139],
    ],
    dtype=np.float64,
)


def _state(seed: int = 7):
    """Return ``(qpos, qvel, w_body)`` with a known body-frame base angular velocity."""
    rng = np.random.RandomState(seed)
    quats = QUATS_WXYZ / np.linalg.norm(QUATS_WXYZ, axis=-1, keepdims=True)

    qpos = np.zeros((NUM_ENVS, NQ), np.float32)
    qpos[:, :3] = [0.0, 0.0, 0.80]
    qpos[:, 3:7] = quats
    qpos[:, 7:] = rng.uniform(-0.3, 0.3, size=(NUM_ENVS, 29))

    # MuJoCo free-joint convention: qvel[3:6] holds body-frame angular velocity.
    w_body = rng.uniform(-1.5, 1.5, size=(NUM_ENVS, 3)).astype(np.float32)
    qvel = np.zeros((NUM_ENVS, NV), np.float32)
    qvel[:, :3] = rng.uniform(-0.4, 0.4, size=(NUM_ENVS, 3))
    qvel[:, 3:6] = w_body
    qvel[:, 6:] = rng.uniform(-0.5, 0.5, size=(NUM_ENVS, 29))
    return qpos, qvel, w_body


def _mujoco_backend():
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = MuJoCoBackend(SceneCfg(model_file=MODEL_FILE), NUM_ENVS, SIM_DT, base_name=BASE_NAME)
    backend.materialize()
    return backend


def _motrix_backend():
    pytest.importorskip("motrixsim")
    from unilab.base.backend.motrix.backend import MotrixBackend

    return MotrixBackend(SceneCfg(model_file=MODEL_FILE), NUM_ENVS, SIM_DT, base_name=BASE_NAME)


class TestMuJoCoAdditiveContracts:
    """MuJoCo must be numerically untouched by the new methods."""

    def test_base_ang_vel_b_is_bit_identical_to_base_ang_vel(self):
        """The regression gate: adding the method must not change a single bit."""
        backend = _mujoco_backend()
        qpos, qvel, _ = _state()
        backend.set_state(np.arange(NUM_ENVS), qpos, qvel)

        np.testing.assert_array_equal(backend.get_base_ang_vel_b(), backend.get_base_ang_vel())

        ctrl = np.zeros((NUM_ENVS, backend.num_actuators), np.float32)
        for _ in range(5):
            backend.step(ctrl, nsteps=4)
            np.testing.assert_array_equal(backend.get_base_ang_vel_b(), backend.get_base_ang_vel())

    def test_base_ang_vel_b_recovers_written_body_frame_velocity(self):
        backend = _mujoco_backend()
        qpos, qvel, w_body = _state()
        backend.set_state(np.arange(NUM_ENVS), qpos, qvel)
        np.testing.assert_allclose(backend.get_base_ang_vel_b(), w_body, atol=1e-6)

    def test_dof_pos_limits_matches_joint_range(self):
        backend = _mujoco_backend()
        joint_range = backend.get_joint_range()
        assert joint_range is not None
        np.testing.assert_array_equal(backend.get_dof_pos_limits(), joint_range)
        assert backend.get_dof_pos_limits().shape == (backend.num_actuators, 2)


@pytest.mark.slow
class TestMotrixAdditiveContracts:
    def test_base_ang_vel_b_recovers_written_body_frame_velocity(self):
        """motrix's native base ang vel is world-frame; the _b variant must rotate back."""
        backend = _motrix_backend()
        qpos, qvel, w_body = _state()
        backend.set_state(np.arange(NUM_ENVS), qpos, qvel)
        np.testing.assert_allclose(backend.get_base_ang_vel_b(), w_body, atol=1e-5)

    def test_base_ang_vel_stays_world_frame(self):
        """Pre-existing behaviour must be untouched: the plain getter stays world-frame."""
        backend = _motrix_backend()
        qpos, qvel, w_body = _state()
        backend.set_state(np.arange(NUM_ENVS), qpos, qvel)
        # A non-trivial attitude makes world and body frames genuinely differ.
        assert np.abs(np.asarray(backend.get_base_ang_vel()) - w_body).max() > 1e-2

    def test_joint_range_still_returns_none(self):
        """Load-bearing: envs treat ``None`` as "skip joint clamping" (ledger §9.4)."""
        assert _motrix_backend().get_joint_range() is None

    def test_dof_pos_limits_is_available_and_shaped(self):
        backend = _motrix_backend()
        limits = backend.get_dof_pos_limits()
        assert limits.shape == (backend.num_actuators, 2)
        assert np.all(limits[:, 0] <= limits[:, 1])


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_BODIES = (*FEET, "pelvis")


def _with_contact_sensors(backend_type: str):
    from unilab.base.backend import create_backend

    if backend_type == "motrix":
        pytest.importorskip("motrixsim")
    backend = create_backend(
        backend_type,
        SceneCfg(model_file=MODEL_FILE),
        NUM_ENVS,
        SIM_DT,
        base_name=BASE_NAME,
        add_body_sensors=True,
        contact_sensor_bodies=list(CONTACT_BODIES),
    )
    backend.materialize()
    return backend


def _settled_on_ground(backend, steps: int = 8):
    """Put the robot in the stand keyframe and hold it so the feet carry load."""
    keyframe_qpos = np.asarray(backend.get_keyframe_qpos("stand"), np.float32)
    backend.set_state(
        np.arange(NUM_ENVS),
        np.tile(keyframe_qpos, (NUM_ENVS, 1)),
        np.zeros((NUM_ENVS, NV), np.float32),
    )
    ctrl = np.tile(keyframe_qpos[7:], (NUM_ENVS, 1)).astype(np.float32)
    for _ in range(steps):
        backend.step(ctrl, nsteps=4)
    return ctrl


@pytest.mark.slow
class TestContactForceContract:
    """``get_body_contact_force`` must read newtons on both backends."""

    @pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
    def test_loaded_feet_report_force_above_threshold(self, backend_type):
        backend = _with_contact_sensors(backend_type)
        _settled_on_ground(backend)
        for body in FEET:
            force = np.asarray(backend.get_body_contact_force(body))
            assert force.shape == (NUM_ENVS,)
            assert np.all(force > 1.0), f"{backend_type} {body} reported {force}"

    @pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
    def test_total_foot_force_is_the_right_order_as_body_weight(self, backend_type):
        """Catches under-enumerated contact geoms, which read low but never zero.

        A G1 foot has 4 sphere geoms plus 7 capsules; summing only the spheres yields
        ~0.2x body weight, which still passes a "> 1.0 N" check (ledger §9.8).
        """
        backend = _with_contact_sensors(backend_type)
        _settled_on_ground(backend)
        total = sum(float(np.asarray(backend.get_body_contact_force(b))[0]) for b in FEET)
        weight = float(np.asarray(backend.get_body_mass()).sum()) * 9.81
        assert 0.7 < total / weight < 1.4, (
            f"{backend_type}: feet carry {total:.1f} N vs weight {weight:.1f} N "
            f"(ratio {total / weight:.3f})"
        )

    @pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
    def test_unregistered_body_raises(self, backend_type):
        backend = _with_contact_sensors(backend_type)
        with pytest.raises(ValueError, match="torso_link"):
            backend.get_body_contact_force("torso_link")

    def test_boolean_contact_decision_agrees_across_backends(self):
        mj, mx = _with_contact_sensors("mujoco"), _with_contact_sensors("motrix")
        mj_ctrl = _settled_on_ground(mj, steps=1)
        _settled_on_ground(mx, steps=1)
        for _ in range(20):
            mj.step(mj_ctrl, nsteps=4)
            mx.step(mj_ctrl, nsteps=4)
            for body in FEET:
                assert (float(np.asarray(mj.get_body_contact_force(body))[0]) > 1.0) == (
                    float(np.asarray(mx.get_body_contact_force(body))[0]) > 1.0
                )


@pytest.mark.slow
class TestMotrixDerivedContracts:
    """Force range and physics-state snapshot, both parsed/assembled by the backend."""

    def test_actuator_force_range_matches_mujoco(self):
        mj, mx = _mujoco_backend(), _motrix_backend()
        np.testing.assert_allclose(
            mj.get_actuator_force_range(), mx.get_actuator_force_range(), atol=1e-9
        )

    def test_physics_state_layout_matches_mujoco(self):
        mj, mx = _mujoco_backend(), _motrix_backend()
        qpos, qvel, _ = _state()
        mj.set_state(np.arange(NUM_ENVS), qpos, qvel)
        mx.set_state(np.arange(NUM_ENVS), qpos, qvel)
        mj_state = np.asarray(mj.get_physics_state())
        mx_state = np.asarray(mx.get_physics_state())
        assert mj_state.shape == mx_state.shape
        # [time, pos(3), quat(4), dof(29)] -- the root+dof block must agree exactly.
        np.testing.assert_allclose(mj_state[:, :NQ], mx_state[:, :NQ], atol=1e-6)

    def test_physics_state_index_3_is_root_height(self):
        """runner.py reads root height as get_physics_state()[:, 3]."""
        for backend in (_mujoco_backend(), _motrix_backend()):
            qpos, qvel, _ = _state()
            qpos[:, 2] = 0.6123
            backend.set_state(np.arange(NUM_ENVS), qpos, qvel)
            np.testing.assert_allclose(
                np.asarray(backend.get_physics_state())[:, 3], 0.6123, atol=1e-5
            )

    def test_supports_physics_state_playback_stays_false(self):
        """Flipping this would change playback behaviour for every Motrix task (§9.5)."""
        assert _motrix_backend().get_play_capabilities().supports_physics_state_playback is False


@pytest.mark.slow
class TestCrossBackendBodyFrame:
    """The new contract must agree across backends at t=0 under a non-trivial attitude."""

    ATOL = 2e-3

    def test_base_ang_vel_b_agrees(self):
        mj, mx = _mujoco_backend(), _motrix_backend()
        qpos, qvel, _ = _state()
        mj.set_state(np.arange(NUM_ENVS), qpos, qvel)
        mx.set_state(np.arange(NUM_ENVS), qpos, qvel)
        np.testing.assert_allclose(mj.get_base_ang_vel_b(), mx.get_base_ang_vel_b(), atol=self.ATOL)

    def test_dof_pos_limits_agree(self):
        mj, mx = _mujoco_backend(), _motrix_backend()
        np.testing.assert_allclose(mj.get_dof_pos_limits(), mx.get_dof_pos_limits(), atol=1e-6)

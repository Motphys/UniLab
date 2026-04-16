"""Tests for SimBackend implementations (MuJoCo and MotrixSim).

All tests are @pytest.mark.slow — they require MuJoCo (and optionally
motrixsim) to be installed, and are excluded from the default CI run.

Run with:
    uv run pytest -m slow tests/base/test_sim_backend.py -v
"""

from typing import Any, cast

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.dr import (
    GeomSizeOverride,
    InitRandomizationPlan,
    ModelVariantSpec,
    ResetRandomizationPayload,
)
from unilab.utils.xml_utils import get_named_body_ids


# ---------------------------------------------------------------------------
def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


BASIC_ROBOTS = [
    pytest.param(dict(model_file=_xml("g1"), base_name="pelvis"), id="g1"),
    pytest.param(dict(model_file=_xml("go1"), base_name="trunk"), id="go1"),
    pytest.param(dict(model_file=_xml("go2"), base_name="base"), id="go2"),
]

_G1 = dict(model_file=_xml("g1"), base_name="pelvis")
_ALLEGRO = dict(model_file=_xml("allegro_hand", "scene.xml"), base_name="palm")
_SHARPA = dict(model_file=_xml("sharpa_wave", "scene.xml"), base_name="right_hand_C_MC")

NUM_ENVS = 2
SIM_DT = 0.005


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _shape(arr: np.ndarray, *expected: int) -> None:
    assert arr.shape == expected, f"expected shape {expected}, got {arr.shape}"


def _mujoco_module() -> Any:
    import mujoco

    return cast(Any, mujoco)


def _unit_quat(q: np.ndarray, tag: str = "") -> None:
    np.testing.assert_allclose(
        np.linalg.norm(q, axis=-1),
        1.0,
        atol=1e-5,
        err_msg=f"quaternion not unit — {tag}",
    )


def _identity_qpos_mujoco(nq: int, xyz=(0.0, 0.0, 0.8)) -> np.ndarray:
    """Single-env qpos in MuJoCo format: [x,y,z, qw=1,qx,qy,qz, dofs...]."""
    q = np.zeros((1, nq))
    q[0, :3] = xyz
    q[0, 3] = 1.0  # qw — identity rotation (wxyz)
    return q


def _mujoco_expected_dof_dims(model) -> tuple[int, int]:
    mujoco = _mujoco_module()

    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return model.nq - 7, model.nv - 6
    return model.nq, model.nv


def _allegro_state() -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros((1, 23), dtype=np.float64)
    qvel = np.zeros((1, 22), dtype=np.float64)
    qpos[0, :16] = np.linspace(-0.2, 0.2, 16)
    qpos[0, 16:19] = np.array([-0.01, 0.02, 0.16])
    qpos[0, 19:23] = np.array([0.92387953, 0.0, 0.38268343, 0.0])
    qvel[0, :16] = np.linspace(-0.15, 0.15, 16)
    qvel[0, 16:22] = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
    return qpos, qvel


# ---------------------------------------------------------------------------
# MuJoCo — basic, 3 robots, no body sensors
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMuJoCoBasic:
    @pytest.fixture(params=BASIC_ROBOTS)
    def bkd(self, request):
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        p = request.param
        return MuJoCoBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])

    # properties

    def test_num_envs(self, bkd):
        assert bkd.num_envs == NUM_ENVS

    def test_model_not_none(self, bkd):
        assert bkd.model is not None

    def test_apply_init_randomization_rebuilds_model_pool_from_variant_sequence(self):
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        bkd = MuJoCoBackend(_SHARPA["model_file"], 4, SIM_DT, base_name=_SHARPA["base_name"])
        assert bkd._pool is None
        mujoco = _mujoco_module()
        geom_id = mujoco.mj_name2id(bkd.model, mujoco.mjtObj.mjOBJ_GEOM, "object")
        base_size = np.asarray(bkd.model.geom_size[geom_id], dtype=np.float64).copy()

        bkd.apply_init_randomization(
            InitRandomizationPlan(
                model_assignments=np.array([0, 1, 0, 1], dtype=np.int32),
                model_variants=(
                    ModelVariantSpec(
                        geom_size_overrides=(GeomSizeOverride("object", tuple(base_size * 0.5)),)
                    ),
                    ModelVariantSpec(
                        geom_size_overrides=(GeomSizeOverride("object", tuple(base_size * 0.75)),)
                    ),
                ),
            )
        )

        assert bkd._pool is None
        np.testing.assert_array_equal(
            bkd._model_assignments,
            np.array([0, 1, 0, 1], dtype=np.int32),
        )
        np.testing.assert_allclose(bkd._model_variants[0].geom_size[geom_id], base_size * 0.5)
        np.testing.assert_allclose(bkd._model_variants[1].geom_size[geom_id], base_size * 0.75)

    # simulation control

    def test_step(self, bkd):
        bkd.step(np.zeros((NUM_ENVS, bkd.model.nu)), nsteps=2)

    def test_set_state_moves_base(self, bkd):
        nq, nv = bkd.model.nq, bkd.model.nv
        target = (1.0, 2.0, 0.8)
        qpos = _identity_qpos_mujoco(nq, xyz=target)
        bkd.set_state(np.array([0]), qpos, np.zeros((1, nv)))
        np.testing.assert_allclose(bkd.get_base_pos()[0], target, atol=1e-5)

    def test_set_state_only_affects_target_envs(self, bkd):
        nq, nv = bkd.model.nq, bkd.model.nv
        pos_before = bkd.get_base_pos()[1].copy()
        qpos = _identity_qpos_mujoco(nq, xyz=(5.0, 5.0, 1.0))
        bkd.set_state(np.array([0]), qpos, np.zeros((1, nv)))
        np.testing.assert_allclose(bkd.get_base_pos()[1], pos_before, atol=1e-5)

    def test_set_state_randomization_only_affects_target_envs(self, bkd):
        original = [bkd._pool.get_field(i, "body_mass").copy() for i in range(NUM_ENVS)]
        qpos = _identity_qpos_mujoco(bkd.model.nq)
        qvel = np.zeros((1, bkd.model.nv))
        base_body_id = bkd._base_body_id
        delta = np.array([original[1][base_body_id] * 0.5])
        randomization = ResetRandomizationPayload(base_mass_delta=delta)

        bkd.set_state(np.array([1]), qpos, qvel, randomization=randomization)

        np.testing.assert_array_equal(bkd._pool.get_field(0, "body_mass"), original[0])
        updated = bkd._pool.get_field(1, "body_mass")
        np.testing.assert_allclose(updated[:base_body_id], original[1][:base_body_id])
        np.testing.assert_allclose(updated[base_body_id], original[1][base_body_id] + delta[0])
        np.testing.assert_allclose(updated[base_body_id + 1 :], original[1][base_body_id + 1 :])

    def test_get_dr_capabilities_include_extended_reset_terms(self, bkd):
        caps = bkd.get_dr_capabilities()
        assert {
            "base_mass_delta",
            "base_com_offset",
            "body_iquat",
            "body_inertia",
            "kp",
            "kd",
        }.issubset(caps.supported_reset_terms)
        assert caps.supports_interval_push

    def test_set_state_body_iquat_randomization_only_affects_target_envs(self, bkd):
        original = [bkd._pool.get_field(i, "body_iquat").copy() for i in range(NUM_ENVS)]
        qpos = _identity_qpos_mujoco(bkd.model.nq)
        qvel = np.zeros((1, bkd.model.nv))
        updated = original[1].reshape(bkd.model.nbody, 4).copy()
        updated[bkd._base_body_id] = np.array([0.92387953, 0.0, 0.38268343, 0.0])

        bkd.set_state(
            np.array([1]),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(body_iquat=updated[None, :, :]),
        )

        np.testing.assert_array_equal(bkd._pool.get_field(0, "body_iquat"), original[0])
        np.testing.assert_allclose(
            bkd._pool.get_field(1, "body_iquat").reshape(bkd.model.nbody, 4), updated
        )

    def test_set_state_body_inertia_randomization_only_affects_target_envs(self, bkd):
        original = [bkd._pool.get_field(i, "body_inertia").copy() for i in range(NUM_ENVS)]
        qpos = _identity_qpos_mujoco(bkd.model.nq)
        qvel = np.zeros((1, bkd.model.nv))
        updated = original[1].reshape(bkd.model.nbody, 3).copy()
        updated[bkd._base_body_id] *= 1.5

        bkd.set_state(
            np.array([1]),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(body_inertia=updated[None, :, :]),
        )

        np.testing.assert_array_equal(bkd._pool.get_field(0, "body_inertia"), original[0])
        np.testing.assert_allclose(
            bkd._pool.get_field(1, "body_inertia").reshape(bkd.model.nbody, 3), updated
        )

    def test_set_state_kp_kd_randomization_only_affects_target_envs(self, bkd):
        original_kp = [bkd._pool.get_field(i, "kp").copy() for i in range(NUM_ENVS)]
        original_kd = [bkd._pool.get_field(i, "kd").copy() for i in range(NUM_ENVS)]
        qpos = _identity_qpos_mujoco(bkd.model.nq)
        qvel = np.zeros((1, bkd.model.nv))
        new_kp = original_kp[1] + 1.25
        new_kd = np.maximum(original_kd[1] + 0.25, 0.25)

        bkd.set_state(
            np.array([1]),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(kp=new_kp[None, :], kd=new_kd[None, :]),
        )

        np.testing.assert_array_equal(bkd._pool.get_field(0, "kp"), original_kp[0])
        np.testing.assert_array_equal(bkd._pool.get_field(0, "kd"), original_kd[0])
        np.testing.assert_allclose(bkd._pool.get_field(1, "kp"), new_kp)
        np.testing.assert_allclose(bkd._pool.get_field(1, "kd"), new_kd)

    # base kinematics

    def test_get_base_pos_shape(self, bkd):
        _shape(bkd.get_base_pos(), NUM_ENVS, 3)

    def test_get_base_quat_shape(self, bkd):
        _shape(bkd.get_base_quat(), NUM_ENVS, 4)

    def test_get_base_quat_unit_norm(self, bkd):
        _unit_quat(bkd.get_base_quat(), "MuJoCo base quat")

    def test_get_base_lin_vel_shape(self, bkd):
        _shape(bkd.get_base_lin_vel(), NUM_ENVS, 3)

    def test_get_base_ang_vel_shape(self, bkd):
        _shape(bkd.get_base_ang_vel(), NUM_ENVS, 3)

    # DOF state

    def test_get_dof_pos_shape(self, bkd):
        expected_nq, _ = _mujoco_expected_dof_dims(bkd.model)
        _shape(bkd.get_dof_pos(), NUM_ENVS, expected_nq)

    def test_get_dof_vel_shape(self, bkd):
        _, expected_nv = _mujoco_expected_dof_dims(bkd.model)
        _shape(bkd.get_dof_vel(), NUM_ENVS, expected_nv)

    def test_dof_pos_finite_after_step(self, bkd):
        bkd.step(np.zeros((NUM_ENVS, bkd.model.nu)))
        assert np.all(np.isfinite(bkd.get_dof_pos()))


@pytest.mark.slow
def test_mujoco_backend_discards_visual_assets():
    mujoco = _mujoco_module()

    from unilab.base.backend.mujoco_backend import MuJoCoBackend

    model_file = _xml("go2")
    full = mujoco.MjModel.from_xml_path(model_file)
    trimmed = MuJoCoBackend(model_file, 1, SIM_DT, base_name="base")

    assert trimmed.model.ngeom < full.ngeom
    assert trimmed.model.nmesh == 0
    assert trimmed.model.ntex == 0
    assert trimmed.model.nmat == 0


@pytest.mark.slow
def test_mujoco_backend_fixed_base_dof_views_do_not_skip_first_joint():
    mujoco = _mujoco_module()

    from unilab.base.backend.mujoco_backend import MuJoCoBackend

    model_file = _xml("allegro_hand", "scene.xml")
    bkd = MuJoCoBackend(model_file, NUM_ENVS, SIM_DT, base_name="palm")
    assert int(bkd.model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
    _shape(bkd.get_dof_pos(), NUM_ENVS, bkd.model.nq)
    _shape(bkd.get_dof_vel(), NUM_ENVS, bkd.model.nv)
    _shape(bkd.get_base_pos(), NUM_ENVS, 3)
    _shape(bkd.get_base_quat(), NUM_ENVS, 4)
    np.testing.assert_allclose(bkd.get_base_lin_vel(), 0.0, atol=1e-8)
    np.testing.assert_allclose(bkd.get_base_ang_vel(), 0.0, atol=1e-8)
    _unit_quat(bkd.get_base_quat(), "MuJoCo fixed-base quat")


@pytest.mark.slow
def test_motrix_backend_fixed_base_base_views_are_available():
    from unilab.base.backend.motrix_backend import MotrixBackend

    pytest.importorskip("motrixsim")
    bkd = MotrixBackend(_ALLEGRO["model_file"], NUM_ENVS, SIM_DT, base_name=_ALLEGRO["base_name"])
    _shape(bkd.get_base_pos(), NUM_ENVS, 3)
    _shape(bkd.get_base_quat(), NUM_ENVS, 4)
    np.testing.assert_allclose(bkd.get_base_lin_vel(), 0.0, atol=1e-8)
    np.testing.assert_allclose(bkd.get_base_ang_vel(), 0.0, atol=1e-8)
    _unit_quat(bkd.get_base_quat(), "Motrix fixed-base quat")


@pytest.mark.slow
def test_motrix_backend_fixed_base_set_state_matches_mujoco_for_hand_and_ball():
    from unilab.base.backend.motrix_backend import MotrixBackend
    from unilab.base.backend.mujoco_backend import MuJoCoBackend

    pytest.importorskip("motrixsim")
    mj = MuJoCoBackend(
        _ALLEGRO["model_file"],
        NUM_ENVS,
        SIM_DT,
        base_name=_ALLEGRO["base_name"],
        add_body_sensors=True,
    )
    mx = MotrixBackend(
        _ALLEGRO["model_file"],
        NUM_ENVS,
        SIM_DT,
        base_name=_ALLEGRO["base_name"],
        add_body_sensors=True,
    )
    qpos, qvel = _allegro_state()
    env_idx = np.array([0])

    mj.set_state(env_idx, qpos, qvel)
    mx.set_state(env_idx, qpos, qvel)

    np.testing.assert_allclose(mx.get_dof_pos()[0], qpos[0, :16], atol=1e-6)
    np.testing.assert_allclose(mx.get_dof_vel()[0], qvel[0, :16], atol=1e-6)
    np.testing.assert_allclose(np.asarray(mx.data.actuator_ctrls[0]), qpos[0, :16], atol=1e-6)
    np.testing.assert_allclose(mx.get_base_pos(), mj.get_base_pos(), atol=2e-3)
    np.testing.assert_allclose(np.abs(mx.get_base_quat()), np.abs(mj.get_base_quat()), atol=2e-3)

    mj_ball_id = _mj_body_id(mj.model, "ball")
    mx_ball_id = _mx_link_id(mx.model, "ball")
    np.testing.assert_allclose(
        mx.get_body_pos_w(np.array([mx_ball_id])),
        mj.get_body_pos_w(np.array([mj_ball_id])),
        atol=2e-3,
    )
    np.testing.assert_allclose(
        mx.get_body_quat_w(np.array([mx_ball_id])),
        mj.get_body_quat_w(np.array([mj_ball_id])),
        atol=2e-3,
    )


# ---------------------------------------------------------------------------
# MuJoCo — body sensors, G1 only
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMuJoCoBodySensors:
    @pytest.fixture
    def bkd(self):
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        return MuJoCoBackend(
            _G1["model_file"],
            NUM_ENVS,
            SIM_DT,
            base_name=_G1["base_name"],
            add_body_sensors=True,
        )

    @pytest.fixture
    def body_ids(self, bkd):
        return np.array(bkd._tracked_body_ids[:2])

    def _bname(self, bkd, bid: int) -> str:
        mujoco = _mujoco_module()

        return cast(str, mujoco.mj_id2name(bkd.model, mujoco.mjtObj.mjOBJ_BODY, bid))

    # world frame

    def test_get_body_pos_w_shape(self, bkd, body_ids):
        _shape(bkd.get_body_pos_w(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_quat_w_shape(self, bkd, body_ids):
        _shape(bkd.get_body_quat_w(body_ids), NUM_ENVS, len(body_ids), 4)

    def test_get_body_quat_w_unit_norm(self, bkd, body_ids):
        _unit_quat(bkd.get_body_quat_w(body_ids), "body quat_w")

    def test_get_body_lin_vel_w_shape(self, bkd, body_ids):
        _shape(bkd.get_body_lin_vel_w(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_ang_vel_w_shape(self, bkd, body_ids):
        _shape(bkd.get_body_ang_vel_w(body_ids), NUM_ENVS, len(body_ids), 3)

    # baselink frame

    def test_get_body_pos_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_pos_b(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_quat_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_quat_b(body_ids), NUM_ENVS, len(body_ids), 4)

    def test_get_body_quat_b_unit_norm(self, bkd, body_ids):
        _unit_quat(bkd.get_body_quat_b(body_ids), "body quat_b")

    def test_get_body_lin_vel_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_lin_vel_b(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_ang_vel_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_ang_vel_b(body_ids), NUM_ENVS, len(body_ids), 3)

    # sensors

    def test_get_sensor_data_w_shape(self, bkd, body_ids):
        bname = self._bname(bkd, int(body_ids[0]))
        _shape(bkd.get_sensor_data(f"track_pos_w_{bname}"), NUM_ENVS, 3)

    def test_get_sensor_data_b_shape(self, bkd, body_ids):
        bname = self._bname(bkd, int(body_ids[0]))
        _shape(bkd.get_sensor_data(f"track_pos_b_{bname}"), NUM_ENVS, 3)

    def test_get_sensor_data_unknown_raises(self, bkd):
        with pytest.raises((ValueError, KeyError)):
            bkd.get_sensor_data("__nonexistent__")

    # semantic correctness

    def test_base_body_pos_b_is_zero(self, bkd):
        """Base body position relative to itself must be [0,0,0]."""
        mujoco = _mujoco_module()

        pelvis_id = mujoco.mj_name2id(bkd.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pos_b = bkd.get_body_pos_b(np.array([pelvis_id]))
        np.testing.assert_allclose(pos_b, 0.0, atol=1e-5)

    def test_base_body_pos_w_matches_get_base_pos(self, bkd):
        """get_body_pos_w for the base body must equal get_base_pos()."""
        mujoco = _mujoco_module()

        pelvis_id = mujoco.mj_name2id(bkd.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pos_w = bkd.get_body_pos_w(np.array([pelvis_id]))[:, 0, :]
        np.testing.assert_allclose(pos_w, bkd.get_base_pos(), atol=1e-5)

    def test_base_body_quat_b_is_identity(self, bkd):
        """Base body quaternion relative to itself must be identity [1,0,0,0]."""
        mujoco = _mujoco_module()

        pelvis_id = mujoco.mj_name2id(bkd.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        quat_b = bkd.get_body_quat_b(np.array([pelvis_id]))[:, 0, :]  # (N, 4)
        np.testing.assert_allclose(
            np.abs(quat_b),
            np.tile([1.0, 0.0, 0.0, 0.0], (quat_b.shape[0], 1)),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# MotrixSim — basic, 3 robots, no body sensors
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMotrixBasic:
    @pytest.fixture(autouse=True)
    def _require_motrix(self):
        pytest.importorskip("motrixsim")

    # Single parametrized fixture returning (backend, base_name) so that
    # dependent fixtures can share the same parameter value.
    @pytest.fixture(params=BASIC_ROBOTS)
    def _ctx(self, request):
        from unilab.base.backend.motrix_backend import MotrixBackend

        p = request.param
        bkd = MotrixBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])
        return bkd, p["base_name"]

    @pytest.fixture
    def bkd(self, _ctx):
        return _ctx[0]

    @pytest.fixture
    def one_body_id(self, _ctx):
        """Array containing just the base link index (motrixsim link index)."""
        bkd, base_name = _ctx
        return np.array([bkd.model.get_link_index(base_name)])

    # properties

    def test_num_envs(self, bkd):
        assert bkd.num_envs == NUM_ENVS

    def test_model_not_none(self, bkd):
        assert bkd.model is not None

    def test_data_not_none(self, bkd):
        assert bkd.data is not None

    # simulation control

    def test_step(self, bkd):
        ctrl = np.zeros_like(bkd.data.actuator_ctrls)
        bkd.step(ctrl, nsteps=2)

    def test_set_state_moves_base(self, _ctx):
        bkd, _ = _ctx
        nq = bkd.get_dof_pos().shape[-1] + 7  # 7 base DOFs + joint DOFs
        nv = bkd.get_dof_vel().shape[-1] + 6  # joint vels + base (3 lin + 3 ang)
        target = (1.0, 2.0, 0.8)
        qpos = _identity_qpos_mujoco(nq, xyz=target)
        bkd.set_state(np.array([0]), qpos, np.zeros((1, nv)))
        np.testing.assert_allclose(bkd.get_base_pos()[0], target, atol=1e-4)

    def test_set_state_randomization_only_affects_target_envs(self, _ctx):
        bkd, _ = _ctx
        nq = bkd.get_dof_pos().shape[-1] + 7
        nv = bkd.get_dof_vel().shape[-1] + 6
        qpos = _identity_qpos_mujoco(nq)
        qvel = np.zeros((1, nv))
        original_mass = np.asarray(bkd._body_link.get_mass_override(bkd.data)).copy()
        delta = np.array([0.25])

        bkd.set_state(
            np.array([0]),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(base_mass_delta=delta),
        )

        updated_mass = np.asarray(bkd._body_link.get_mass_override(bkd.data))
        np.testing.assert_allclose(updated_mass[1], original_mass[1], atol=1e-6)
        np.testing.assert_allclose(updated_mass[0], original_mass[0] + delta[0], atol=1e-6)

    def test_set_state_unsupported_randomization_raises(self, _ctx):
        bkd, _ = _ctx
        nq = bkd.get_dof_pos().shape[-1] + 7
        nv = bkd.get_dof_vel().shape[-1] + 6
        qpos = _identity_qpos_mujoco(nq)
        qvel = np.zeros((1, nv))

        with pytest.raises(NotImplementedError, match="kp"):
            bkd.set_state(
                np.array([0]),
                qpos,
                qvel,
                randomization=ResetRandomizationPayload(kp=np.zeros((1, 1))),
            )

    # base kinematics

    def test_get_base_pos_shape(self, bkd):
        _shape(bkd.get_base_pos(), NUM_ENVS, 3)

    def test_get_base_quat_shape(self, bkd):
        _shape(bkd.get_base_quat(), NUM_ENVS, 4)

    def test_get_base_quat_unit_norm(self, bkd):
        _unit_quat(bkd.get_base_quat(), "Motrix base quat")

    def test_get_base_lin_vel_shape(self, bkd):
        _shape(bkd.get_base_lin_vel(), NUM_ENVS, 3)

    def test_get_base_ang_vel_shape(self, bkd):
        _shape(bkd.get_base_ang_vel(), NUM_ENVS, 3)

    # DOF state

    def test_get_dof_pos_shape(self, bkd):
        d = bkd.get_dof_pos()
        assert d.ndim == 2 and d.shape[0] == NUM_ENVS and d.shape[1] > 0

    def test_get_dof_vel_shape(self, bkd):
        d = bkd.get_dof_vel()
        assert d.ndim == 2 and d.shape[0] == NUM_ENVS and d.shape[1] > 0

    def test_dof_pos_finite_after_step(self, bkd):
        ctrl = np.zeros_like(bkd.data.actuator_ctrls)
        bkd.step(ctrl)
        assert np.all(np.isfinite(bkd.get_dof_pos()))

    # body kinematics — world frame (available without sensors)

    def test_get_body_pos_w_shape(self, bkd, one_body_id):
        _shape(bkd.get_body_pos_w(one_body_id), NUM_ENVS, 1, 3)

    def test_get_body_quat_w_shape(self, bkd, one_body_id):
        _shape(bkd.get_body_quat_w(one_body_id), NUM_ENVS, 1, 4)

    def test_get_body_quat_w_unit_norm(self, bkd, one_body_id):
        _unit_quat(bkd.get_body_quat_w(one_body_id), "body quat_w")

    def test_get_body_lin_vel_w_shape(self, bkd, one_body_id):
        _shape(bkd.get_body_lin_vel_w(one_body_id), NUM_ENVS, 1, 3)

    def test_get_body_ang_vel_w_shape(self, bkd, one_body_id):
        _shape(bkd.get_body_ang_vel_w(one_body_id), NUM_ENVS, 1, 3)

    def test_base_body_pos_w_matches_get_base_pos(self, bkd, one_body_id):
        pos_w = bkd.get_body_pos_w(one_body_id)[:, 0, :]
        np.testing.assert_allclose(pos_w, bkd.get_base_pos(), atol=1e-5)


# ---------------------------------------------------------------------------
# MotrixSim — body sensors, G1 only
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMotrixBodySensors:
    @pytest.fixture(autouse=True)
    def _require_motrix(self):
        pytest.importorskip("motrixsim")

    @pytest.fixture
    def bkd(self):
        from unilab.base.backend.motrix_backend import MotrixBackend

        return MotrixBackend(
            _G1["model_file"],
            NUM_ENVS,
            SIM_DT,
            base_name=_G1["base_name"],
            add_body_sensors=True,
        )

    @pytest.fixture
    def body_ids(self, bkd):
        return np.array(list(bkd._body_id_to_name.keys())[:2])

    # baselink frame

    def test_get_body_pos_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_pos_b(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_quat_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_quat_b(body_ids), NUM_ENVS, len(body_ids), 4)

    def test_get_body_quat_b_unit_norm(self, bkd, body_ids):
        _unit_quat(bkd.get_body_quat_b(body_ids), "body quat_b")

    def test_get_body_lin_vel_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_lin_vel_b(body_ids), NUM_ENVS, len(body_ids), 3)

    def test_get_body_ang_vel_b_shape(self, bkd, body_ids):
        _shape(bkd.get_body_ang_vel_b(body_ids), NUM_ENVS, len(body_ids), 3)

    # sensors

    def test_get_sensor_data_b_shape(self, bkd, body_ids):
        bid = int(body_ids[0])
        bname = bkd._body_id_to_name[bid]
        result = bkd.get_sensor_data(f"track_pos_b_{bname}")
        assert result.shape[0] == NUM_ENVS

    # semantic correctness

    def test_base_body_pos_b_is_zero(self, bkd):
        """Base body position relative to itself must be [0,0,0]."""
        pelvis_id = bkd.model.get_body_index(_G1["base_name"])
        pos_b = bkd.get_body_pos_b(np.array([pelvis_id]))
        np.testing.assert_allclose(pos_b, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Cross-backend consistency: MuJoCo ↔ MotrixSim
# ---------------------------------------------------------------------------


def _mj_body_id(mj_model, name: str) -> int:
    mujoco = _mujoco_module()

    return int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name))


def _mx_link_id(mx_model, name: str) -> int:
    return int(mx_model.get_link_index(name))


@pytest.mark.slow
class TestCrossBackend:
    """相同 set_state 后，MuJoCo 与 MotrixSim 所有基础接口数值必须一致。"""

    ATOL = 2e-3  # float64 vs float32 + 不同 qpos0 的积累误差

    @pytest.fixture(params=BASIC_ROBOTS)
    def synced(self, request):
        """创建并同步两后端初始状态，返回 (mj, mx, base_name)。"""
        from unilab.base.backend.motrix_backend import MotrixBackend
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        pytest.importorskip("motrixsim")
        p = request.param
        mj = MuJoCoBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])
        mx = MotrixBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])
        nq, nv = mj.model.nq, mj.model.nv
        qpos = np.tile(_identity_qpos_mujoco(nq), (NUM_ENVS, 1))
        qvel = np.zeros((NUM_ENVS, nv))
        env_idx = np.arange(NUM_ENVS)
        mj.set_state(env_idx, qpos, qvel)
        mx.set_state(env_idx, qpos, qvel)
        return mj, mx, p["base_name"]

    # --- base kinematics ---

    def test_base_pos(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_base_pos(), mx.get_base_pos(), atol=self.ATOL)

    def test_base_quat(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_base_quat(), mx.get_base_quat(), atol=self.ATOL)

    def test_base_lin_vel(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_base_lin_vel(), mx.get_base_lin_vel(), atol=self.ATOL)

    def test_base_ang_vel(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_base_ang_vel(), mx.get_base_ang_vel(), atol=self.ATOL)

    # --- DOF state ---

    def test_dof_pos(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_dof_pos(), mx.get_dof_pos(), atol=self.ATOL)

    def test_dof_vel(self, synced):
        mj, mx, _ = synced
        np.testing.assert_allclose(mj.get_dof_vel(), mx.get_dof_vel(), atol=self.ATOL)

    # --- after step ---

    # def test_base_pos_after_step(self, synced):
    #     mj, mx, _ = synced
    #     ctrl = np.zeros((NUM_ENVS, mj.model.nu))
    #     mj.step(ctrl, nsteps=5)
    #     mx.step(ctrl, nsteps=5)
    #     np.testing.assert_allclose(mj.get_base_pos(), mx.get_base_pos(), atol=self.ATOL)

    # def test_dof_pos_after_step(self, synced):
    #     mj, mx, _ = synced
    #     ctrl = np.zeros((NUM_ENVS, mj.model.nu))
    #     mj.step(ctrl, nsteps=5)
    #     mx.step(ctrl, nsteps=5)
    #     np.testing.assert_allclose(mj.get_dof_pos(), mx.get_dof_pos(), atol=self.ATOL)


@pytest.mark.slow
class TestCrossBackendBodySensors:
    """body 传感器接口双后端对测（G1，add_body_sensors=True）。"""

    ATOL = 2e-3

    @pytest.fixture
    def synced(self):
        from unilab.base.backend.motrix_backend import MotrixBackend
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        pytest.importorskip("motrixsim")
        mj = MuJoCoBackend(
            _G1["model_file"],
            NUM_ENVS,
            SIM_DT,
            base_name=_G1["base_name"],
            add_body_sensors=True,
        )
        mx = MotrixBackend(
            _G1["model_file"],
            NUM_ENVS,
            SIM_DT,
            base_name=_G1["base_name"],
            add_body_sensors=True,
        )
        nq, nv = mj.model.nq, mj.model.nv
        qpos = np.tile(_identity_qpos_mujoco(nq), (NUM_ENVS, 1))
        qvel = np.zeros((NUM_ENVS, nv))
        env_idx = np.arange(NUM_ENVS)
        mj.set_state(env_idx, qpos, qvel)
        mx.set_state(env_idx, qpos, qvel)
        return mj, mx

    @pytest.fixture
    def body_pairs(self, synced):
        """选前 3 个 body（pelvis + 两个髋关节），分别返回 mujoco IDs 和 motrixsim link IDs。"""
        mujoco = _mujoco_module()

        mj, mx = synced
        names = [
            mujoco.mj_id2name(mj.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, 4)  # skip world (0)
        ]
        mj_ids = np.array([_mj_body_id(mj.model, n) for n in names])
        mx_ids = np.array([_mx_link_id(mx.model, n) for n in names])
        return mj_ids, mx_ids

    # --- world frame ---

    def test_body_pos_w(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_pos_w(mj_ids),
            mx.get_body_pos_w(mx_ids),
            atol=self.ATOL,
        )

    def test_body_quat_w(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_quat_w(mj_ids),
            mx.get_body_quat_w(mx_ids),
            atol=self.ATOL,
        )

    def test_body_lin_vel_w(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_lin_vel_w(mj_ids),
            mx.get_body_lin_vel_w(mx_ids),
            atol=self.ATOL,
        )

    def test_body_ang_vel_w(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_ang_vel_w(mj_ids),
            mx.get_body_ang_vel_w(mx_ids),
            atol=self.ATOL,
        )

    # --- baselink frame ---

    def test_body_pos_b(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_pos_b(mj_ids),
            mx.get_body_pos_b(mx_ids),
            atol=self.ATOL,
        )

    def test_body_quat_b(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_quat_b(mj_ids),
            mx.get_body_quat_b(mx_ids),
            atol=self.ATOL,
        )

    def test_body_lin_vel_b(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_lin_vel_b(mj_ids),
            mx.get_body_lin_vel_b(mx_ids),
            atol=self.ATOL,
        )

    def test_body_ang_vel_b(self, synced, body_pairs):
        mj, mx = synced
        mj_ids, mx_ids = body_pairs
        np.testing.assert_allclose(
            mj.get_body_ang_vel_b(mj_ids),
            mx.get_body_ang_vel_b(mx_ids),
            atol=self.ATOL,
        )


# ---------------------------------------------------------------------------
# Unified model properties — MuJoCo
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMuJoCoModelProperties:
    @pytest.fixture(params=BASIC_ROBOTS)
    def bkd(self, request):
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        p = request.param
        return MuJoCoBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])

    def test_num_actuators(self, bkd):
        assert bkd.num_actuators == bkd.model.nu
        assert bkd.num_actuators > 0

    def test_num_dof_vel(self, bkd):
        assert bkd.num_dof_vel > 0
        assert bkd.num_dof_vel <= bkd.model.nv

    def test_get_actuator_ctrl_range(self, bkd):
        ctrl_range = bkd.get_actuator_ctrl_range()
        _shape(ctrl_range, bkd.num_actuators, 2)
        assert np.all(ctrl_range[:, 0] <= ctrl_range[:, 1])

    def test_get_keyframe_qpos(self, bkd):
        for name in ("stand", "home"):
            try:
                qpos = bkd.get_keyframe_qpos(name)
                assert qpos.ndim == 1
                assert len(qpos) == bkd.model.nq
                return
            except ValueError:
                continue
        pytest.skip("No 'stand' or 'home' keyframe in model")

    def test_get_keyframe_qpos_missing(self, bkd):
        with pytest.raises(ValueError, match="not found"):
            bkd.get_keyframe_qpos("nonexistent_keyframe_xyz")

    def test_get_init_qvel(self, bkd):
        qvel = bkd.get_init_qvel()
        assert qvel.ndim == 1
        assert len(qvel) == bkd.model.nv
        np.testing.assert_array_equal(qvel, 0.0)

    def test_get_body_ids(self, bkd):
        mujoco = _mujoco_module()

        base_name = mujoco.mj_id2name(bkd.model, mujoco.mjtObj.mjOBJ_BODY, 1)
        ids = bkd.get_body_ids([base_name])
        assert ids.dtype == np.int32
        assert ids[0] == 1

    def test_get_body_ids_missing(self, bkd):
        with pytest.raises(ValueError, match="not found"):
            bkd.get_body_ids(["nonexistent_body_xyz"])

    def test_get_joint_range(self, bkd):
        jr = bkd.get_joint_range()
        assert jr is not None
        assert jr.ndim == 2
        assert jr.shape[1] == 2


# ---------------------------------------------------------------------------
# Unified model properties — Motrix
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMotrixModelProperties:
    @pytest.fixture(autouse=True)
    def _require_motrix(self):
        pytest.importorskip("motrixsim")

    @pytest.fixture(params=BASIC_ROBOTS)
    def _ctx(self, request):
        from unilab.base.backend.motrix_backend import MotrixBackend

        p = request.param
        bkd = MotrixBackend(p["model_file"], NUM_ENVS, SIM_DT, base_name=p["base_name"])
        return bkd, p["base_name"]

    @pytest.fixture
    def bkd(self, _ctx):
        return _ctx[0]

    def test_num_actuators(self, bkd):
        assert bkd.num_actuators > 0

    def test_num_dof_vel(self, bkd):
        assert bkd.num_dof_vel > 0

    def test_get_actuator_ctrl_range(self, bkd):
        ctrl_range = bkd.get_actuator_ctrl_range()
        _shape(ctrl_range, bkd.num_actuators, 2)

    def test_get_keyframe_qpos(self, bkd):
        # Motrix ignores the name but should not raise
        qpos = bkd.get_keyframe_qpos("home")
        assert qpos.ndim == 1
        assert len(qpos) > 0

    def test_get_init_qvel(self, bkd):
        qvel = bkd.get_init_qvel()
        assert qvel.ndim == 1
        np.testing.assert_array_equal(qvel, 0.0)

    def test_get_body_ids(self, _ctx):
        bkd, base_name = _ctx
        ids = bkd.get_body_ids([base_name])
        assert ids.dtype == np.int32
        assert len(ids) == 1

    def test_get_body_ids_missing(self, bkd):
        with pytest.raises(ValueError, match="not found"):
            bkd.get_body_ids(["nonexistent_body_xyz"])

    def test_get_joint_range(self, bkd):
        assert bkd.get_joint_range() is None


# ---------------------------------------------------------------------------
# Cross-backend model properties consistency
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCrossBackendModelProperties:
    @pytest.fixture(autouse=True)
    def _require_motrix(self):
        pytest.importorskip("motrixsim")

    @pytest.fixture
    def backends(self):
        from unilab.base.backend.motrix_backend import MotrixBackend
        from unilab.base.backend.mujoco_backend import MuJoCoBackend

        mj = MuJoCoBackend(_G1["model_file"], NUM_ENVS, SIM_DT, base_name=_G1["base_name"])
        mx = MotrixBackend(_G1["model_file"], NUM_ENVS, SIM_DT, base_name=_G1["base_name"])
        return mj, mx

    def test_num_actuators_match(self, backends):
        mj, mx = backends
        assert mj.num_actuators == mx.num_actuators

    def test_num_dof_vel_match(self, backends):
        mj, mx = backends
        assert mj.num_dof_vel == mx.num_dof_vel

    def test_actuator_ctrl_range_shape_match(self, backends):
        mj, mx = backends
        assert mj.get_actuator_ctrl_range().shape == mx.get_actuator_ctrl_range().shape

    def test_motion_body_ids_match_motion_xml(self, backends):
        mj, mx = backends
        body_names = ["pelvis", "torso_link"]
        expected = np.asarray(get_named_body_ids(_G1["model_file"], body_names), dtype=np.int32)
        np.testing.assert_array_equal(mj.get_motion_body_ids(body_names), expected)
        np.testing.assert_array_equal(mx.get_motion_body_ids(body_names), expected)

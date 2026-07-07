"""Faithful, single-source spec of the ``g1_motion_tracking`` update_state.

Every constant here is copied verbatim from the real task so the profile
stays honest.  When ``tracking.py`` changes, this is the *one* file to
resync.  Provenance (all in ``src/unilab/envs/motion_tracking/g1/tracking.py``):

  - ``RewardConfig`` scales / std_*        -> lines 44-70
  - ``G1MotionTrackingCfg.body_names``      -> lines 151-166  (n_body = 14)
  - ``anchor_body_name = "torso_link"``     -> line 150  (index 7)
  - ``ee_body_names``                       -> lines 180-185 (indices 3,6,10,13)
  - termination thresholds                  -> lines 177-187
  - ``ctrl_dt = 0.02``                      -> g1/base.py:51
  - ``num_action = 29``                     -> G1 action space (issue #665 act_dim=29)

The reward loop in ``_compute_reward`` (tracking.py:1243) iterates
``cfg.scales.items()`` and *skips* ``scale == 0`` terms.  We preserve the term
identity and ordering here in ``TERM_ORDER`` so the numpy oracle, the numba
kernel, and the per-term log all agree on index <-> name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── robot / body dimensions (faithful to G1MotionTrackingCfg) ───────────────
NUM_ACTION = 29  # G1 actuated DoF
BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
N_BODY = len(BODY_NAMES)  # 14
ANCHOR_BODY_NAME = "torso_link"
ANCHOR_BODY_IDX = BODY_NAMES.index(ANCHOR_BODY_NAME)  # 7
EE_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
EE_BODY_INDICES = np.array(
    [BODY_NAMES.index(n) for n in EE_BODY_NAMES], dtype=np.int32
)  # 3,6,10,13
# undesired-contact bodies = every tracked body that is not an end-effector
UNDESIRED_CONTACT_INDICES = np.array(
    [i for i, n in enumerate(BODY_NAMES) if n not in set(EE_BODY_NAMES)], dtype=np.int32
)

CTRL_DT = 0.02

# ── termination thresholds (faithful) ───────────────────────────────────────
ANCHOR_POS_Z_THRESHOLD = 0.25
ANCHOR_ORI_THRESHOLD = 0.8  # < 2.0 so the gravity-z check is active
EE_BODY_POS_Z_THRESHOLD = 0.25
UNDESIRED_CONTACT_Z_THRESHOLD = 0.05
TERMINATE_ON_UNDESIRED_CONTACTS = False  # default: off


@dataclass(frozen=True)
class RewardSpec:
    """Mirror of ``RewardConfig`` (tracking.py:44)."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "motion_global_root_pos": 0.5,
            "motion_global_root_ori": 0.5,
            "motion_body_pos": 1.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "motion_ee_body_pos_z": 0.0,
            "motion_joint_pos": 0.0,
            "motion_joint_vel": 0.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
        }
    )
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_joint_pos: float = 0.2
    std_joint_vel: float = 1.0


REWARD_SPEC = RewardSpec()

# Canonical ordering shared by numpy oracle, numba kernel, and log aggregation.
# Same keys/order as RewardConfig.scales -> the kernel is a static superset and
# per-term scale==0 simply contributes 0 (matching the numpy ``continue``).
TERM_ORDER: tuple[str, ...] = tuple(REWARD_SPEC.scales.keys())
N_TERMS = len(TERM_ORDER)
TERM_INDEX = {name: i for i, name in enumerate(TERM_ORDER)}


def scale_vector(scales: dict[str, float] | None = None) -> np.ndarray:
    """Build the dense ``(N_TERMS,)`` scale vector in ``TERM_ORDER``.

    Assembled once on a cold path (reset/init) and handed to the kernel —
    weights stay owned by config, never baked into compiled code.
    """
    src = REWARD_SPEC.scales if scales is None else scales
    return np.array([src.get(name, 0.0) for name in TERM_ORDER], dtype=np.float64)


def std_vector(spec: RewardSpec = REWARD_SPEC) -> np.ndarray:
    """Per-term exponential std in ``TERM_ORDER`` (0 where a term has no std).

    Matches which ``std_*`` each ``_reward_motion_*`` reads in tracking.py.
    """
    per_term = {
        "motion_global_root_pos": spec.std_root_pos,
        "motion_global_root_ori": spec.std_root_ori,
        "motion_body_pos": spec.std_body_pos,
        "motion_body_ori": spec.std_body_ori,
        "motion_body_lin_vel": spec.std_body_lin_vel,
        "motion_body_ang_vel": spec.std_body_ang_vel,
        "motion_ee_body_pos_z": spec.std_body_pos,  # ee uses std_body_pos
        "motion_joint_pos": spec.std_joint_pos,
        "motion_joint_vel": spec.std_joint_vel,
        "action_rate_l2": 0.0,  # not an exp reward
        "joint_limit": 0.0,
    }
    return np.array([per_term[name] for name in TERM_ORDER], dtype=np.float64)

"""MicroDuck × upstream microduck_rl alignment contract (issue #1453, child 1/5).

Machine-checkable comparison table between UniLab's three MicroDuck locomotion
tasks (``microduck_velocity_flat`` / ``microduck_sitstand_flat`` /
``microduck_ground_pick_flat``, ppo tree, mjwarp owner) and the upstream
training recipe.

Upstream anchor:
    pollen-robotics/microduck_rl @ 29e887ecfbf5d37144759e5a9f8a176dfb83d547
    (2026-09-02, mjlab 1.3.0 + rsl_rl 5.0.1 PPO, mjwarp backend)

Entry semantics:
    - ``status="match"``: already aligned; the drift-guard test fails if the
      current value stops matching the upstream target.
    - ``status="gap"``: known divergence with an upstream target; child 2/3/4
      flip entries to ``match`` as they land, and must update this table in the
      same change (the drift-guard test is the explicit checklist).
    - ``status="note"``: informational entry, not judged (e.g. environment
      version skew outside this roadmap, or a training budget still to be
      decided by child 5).

Reward-stack entries are scoped to ``microduck_velocity_flat``: the vetted
upstream reward specification is the velocity recipe at the anchor commit;
sitstand / ground_pick reward targets are decided by their own child issues.
"""

from __future__ import annotations

import importlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

UPSTREAM_REPO = "pollen-robotics/microduck_rl"
UPSTREAM_COMMIT = "29e887ecfbf5d37144759e5a9f8a176dfb83d547"
UPSTREAM_DATE = "2026-09-02"

MICRODUCK_TASKS: tuple[str, ...] = (
    "microduck_velocity_flat",
    "microduck_sitstand_flat",
    "microduck_ground_pick_flat",
)
CONF_TREE = "ppo"
BACKEND = "mjwarp"

REPO_ROOT = Path(__file__).resolve().parents[5]
CONF_DIR = REPO_ROOT / "src" / "unilab" / "conf" / CONF_TREE
# Files probed for the MuJoCo ``<option>`` solver element, in priority order.
# ``scene_flat.xml`` includes ``microduck.xml``; both are parsed standalone.
XML_OPTION_FILES: tuple[str, ...] = (
    "src/unilab/assets/robots/microduck/scene_flat.xml",
    "src/unilab/assets/robots/microduck/microduck.xml",
)

ABSENT = "<absent>"
PRESENT = "<present>"

VELOCITY_TASK = ("microduck_velocity_flat",)


@dataclass(frozen=True)
class ContractSource:
    """Where the current UniLab value is read from.

    kind="hydra": dotted path into the composed ppo/<task>/mjwarp config.
    kind="hydra_keys": sorted keys of the node at the dotted path.
    kind="xml_option": attribute of the first ``<option>`` element found in
        ``XML_OPTION_FILES``.
    kind="module": dotted ``module.attr`` path imported from the environment.
    """

    kind: str
    path: str


@dataclass(frozen=True)
class AlignmentEntry:
    name: str
    category: str  # "physics" | "mdp" | "infra"
    source: ContractSource
    target: Any  # ABSENT / PRESENT sentinels, a concrete value, or None (note)
    status: str  # "match" | "gap" | "note"
    note: str = ""
    tasks: tuple[str, ...] = field(default=MICRODUCK_TASKS)


def _hydra(path: str) -> ContractSource:
    return ContractSource("hydra", path)


def _hydra_keys(path: str) -> ContractSource:
    return ContractSource("hydra_keys", path)


def _xml_option(attr: str) -> ContractSource:
    return ContractSource("xml_option", attr)


def _module(path: str) -> ContractSource:
    return ContractSource("module", path)


ENTRIES: tuple[AlignmentEntry, ...] = (
    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    AlignmentEntry(
        "physics.sim_dt",
        "physics",
        _hydra("env.sim_dt"),
        0.005,
        "gap",
        "当前 0.01（substeps=2）；上游 ctrl 50 Hz 下 sim_dt=0.005（substeps=4）。",
    ),
    AlignmentEntry("physics.ctrl_dt", "physics", _hydra("env.ctrl_dt"), 0.02, "match"),
    AlignmentEntry(
        "physics.max_episode_seconds",
        "physics",
        _hydra("env.max_episode_seconds"),
        20.0,
        "match",
    ),
    AlignmentEntry(
        "physics.solver.integrator",
        "physics",
        _xml_option("integrator"),
        "implicitfast",
        "gap",
        "microduck.xml / scene_flat.xml 当前均无 <option>（MuJoCo 默认 Euler）。",
    ),
    AlignmentEntry(
        "physics.solver.solver",
        "physics",
        _xml_option("solver"),
        "newton",
        "gap",
        "上游 mjlab SimCfg 默认 newton。",
    ),
    AlignmentEntry(
        "physics.solver.cone",
        "physics",
        _xml_option("cone"),
        "pyramidal",
        "gap",
    ),
    AlignmentEntry(
        "physics.solver.iterations",
        "physics",
        _xml_option("iterations"),
        10,
        "gap",
        "无 <option> 时 MuJoCo 默认 100；上游 velocity flat 配方为 10。",
    ),
    AlignmentEntry(
        "physics.solver.ls_iterations",
        "physics",
        _xml_option("ls_iterations"),
        20,
        "gap",
        "无 <option> 时 MuJoCo 默认 50；上游为 20。",
    ),
    AlignmentEntry(
        "physics.solver.tolerance",
        "physics",
        _xml_option("tolerance"),
        1e-8,
        "gap",
    ),
    AlignmentEntry(
        "physics.solver.ls_tolerance",
        "physics",
        _xml_option("ls_tolerance"),
        0.01,
        "gap",
    ),
    # ------------------------------------------------------------------
    # mdp: actions / observations
    # ------------------------------------------------------------------
    AlignmentEntry(
        "mdp.action_scale",
        "mdp",
        _hydra("env.actions.joint_pos.scale"),
        1.0,
        "match",
    ),
    AlignmentEntry(
        "mdp.action_use_default_offset",
        "mdp",
        _hydra("env.actions.joint_pos.use_default_offset"),
        True,
        "match",
        "JointPositionActionCfg 围绕默认姿态（HOME）偏移。",
    ),
    AlignmentEntry(
        "mdp.obs_dim.policy",
        "mdp",
        _module("unilab.tasks.locomotion.microduck.deploy_contract.MICRODUCK_ACTOR_OBS_DIM"),
        61,
        "match",
    ),
    AlignmentEntry(
        "mdp.obs_dim.critic",
        "mdp",
        _module("unilab.tasks.locomotion.microduck.deploy_contract.MICRODUCK_CRITIC_OBS_DIM"),
        76,
        "match",
        "actor 61 + base_lin_vel 3 + privileged foot terms 12。",
    ),
    # ------------------------------------------------------------------
    # mdp: commands
    # ------------------------------------------------------------------
    AlignmentEntry(
        "mdp.commands.twist.lin_vel_x",
        "mdp",
        _hydra("env.commands.twist.ranges.lin_vel_x"),
        [-0.4, 0.4],
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.twist.lin_vel_y",
        "mdp",
        _hydra("env.commands.twist.ranges.lin_vel_y"),
        [-0.3, 0.3],
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.twist.ang_vel_z",
        "mdp",
        _hydra("env.commands.twist.ranges.ang_vel_z"),
        [-1.0, 1.0],
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.twist.resampling_time_range",
        "mdp",
        _hydra("env.commands.twist.resampling_time_range"),
        [3.0, 8.0],
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.twist.rel_forward_envs",
        "mdp",
        _hydra("env.commands.twist.rel_forward_envs"),
        0.2,
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.twist.rel_standing_envs",
        "mdp",
        _hydra("env.commands.twist.rel_standing_envs"),
        0.02,
        "match",
        "stage-0 值；standing_envs curriculum 爬升至 0.25（见 curriculum 条目）。",
    ),
    AlignmentEntry(
        "mdp.commands.twist.turn_in_place_fraction",
        "mdp",
        _hydra("env.commands.twist.turn_in_place_fraction"),
        0.15,
        "match",
        "turn-in-place 桶 |wz| ∈ [0.4, 1.0]。",
    ),
    AlignmentEntry(
        "mdp.commands.twist.turn_in_place_ang_min",
        "mdp",
        _hydra("env.commands.twist.turn_in_place_ang_min"),
        0.4,
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.head_pose.resampling_time_range",
        "mdp",
        _hydra("env.commands.head_pose.resampling_time_range"),
        [2.0, 5.0],
        "match",
    ),
    AlignmentEntry(
        "mdp.commands.body_pose.resampling_time_range",
        "mdp",
        _hydra("env.commands.body_pose.resampling_time_range"),
        [2.0, 5.0],
        "match",
    ),
    # ------------------------------------------------------------------
    # mdp: curriculum stage steps（= 上游 iteration × 24 env steps）
    # ------------------------------------------------------------------
    AlignmentEntry(
        "mdp.curriculum.action_rate_weight",
        "mdp",
        _hydra("env.curriculum.action_rate_weight.params.stages"),
        [
            {"step": 0, "weight": -0.1},
            {"step": 12000, "weight": -0.2},
            {"step": 18000, "weight": -0.4},
            {"step": 24000, "weight": -0.6},
            {"step": 30000, "weight": -0.8},
            {"step": 36000, "weight": -1.0},
        ],
        "match",
    ),
    AlignmentEntry(
        "mdp.curriculum.head_pose_bias_weight",
        "mdp",
        _hydra("env.curriculum.head_pose_bias_weight.params.stages"),
        [
            {"step": 0, "weight": 0.0},
            {"step": 14400, "weight": 1.0},
            {"step": 24000, "weight": 2.0},
            {"step": 36000, "weight": 3.0},
        ],
        "match",
    ),
    AlignmentEntry(
        "mdp.curriculum.standing_envs",
        "mdp",
        _hydra("env.curriculum.standing_envs.params.stages"),
        [
            {"step": 0, "rel_standing_envs": 0.02},
            {"step": 12000, "rel_standing_envs": 0.05},
            {"step": 18000, "rel_standing_envs": 0.1},
            {"step": 24000, "rel_standing_envs": 0.15},
            {"step": 36000, "rel_standing_envs": 0.2},
            {"step": 48000, "rel_standing_envs": 0.25},
        ],
        "match",
    ),
    # ------------------------------------------------------------------
    # mdp: events / terminations
    # ------------------------------------------------------------------
    AlignmentEntry(
        "mdp.events.reset_base",
        "mdp",
        _hydra("env.events.reset_base"),
        PRESENT,
        "gap",
        "当前严格回 keyframe（reset_scene_to_default，零速度）；上游 reset_base："
        "base xy ±0.5 m、z ∈ [0.12, 0.13]、yaw ±π、速度全零、关节零扰动回 HOME。",
    ),
    AlignmentEntry(
        "mdp.events.startup_mass_inertia",
        "mdp",
        _hydra("env.events.randomize_mass_inertia"),
        PRESENT,
        "gap",
        "当前无 startup 质量/惯量 DR；上游对 trunk_base 施加 pseudo_inertia "
        "（质量+惯量整体 ×[0.95, 1.05]，startup 一次性，CoM 不变）。",
    ),
    AlignmentEntry(
        "mdp.events.push_robot.is_global_time",
        "mdp",
        _hydra("env.events.push_robot.is_global_time"),
        False,
        "gap",
        "当前全局时钟同步推挤；上游 per-env 独立 interval。",
    ),
    AlignmentEntry(
        "mdp.events.push_robot.interval_range_s",
        "mdp",
        _hydra("env.events.push_robot.interval_range_s"),
        [3.0, 6.0],
        "match",
    ),
    AlignmentEntry(
        "mdp.events.push_robot.velocity_xy",
        "mdp",
        _hydra("env.events.push_robot.params.velocity_range.x"),
        [-0.3, 0.3],
        "match",
        "y 方向同为 ±0.3。",
    ),
    AlignmentEntry(
        "mdp.terminations.set",
        "mdp",
        _hydra_keys("env.terminations"),
        ["nan_state", "tilt", "time_out"],
        "gap",
        "当前多 base_height<0.08；上游集合 = time_out + tilt 70° + nan_detection。",
    ),
    AlignmentEntry(
        "mdp.terminations.tilt_limit",
        "mdp",
        _hydra("env.terminations.tilt.params.limit_angle"),
        1.2217304763960306,
        "match",
        "70°。",
    ),
    # ------------------------------------------------------------------
    # mdp: reward stack（velocity_flat；上游 HEAD 配方）
    # ------------------------------------------------------------------
    AlignmentEntry(
        "mdp.reward.track_linear_velocity.weight",
        "mdp",
        _hydra("reward.tracking_lin_vel.weight"),
        2.0,
        "gap",
        "当前 3.0；上游 track_linear_velocity 权重 2.0。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.track_linear_velocity.std",
        "mdp",
        _hydra("reward.tracking_lin_vel.params.tracking_sigma"),
        math.sqrt(0.1),
        "gap",
        "当前 tracking_sigma=0.25 且只跟踪 xy；上游 std=√0.1，指数内含 vz²，body frame。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.track_angular_velocity.weight",
        "mdp",
        _hydra("reward.tracking_ang_vel.weight"),
        2.0,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.track_angular_velocity.std",
        "mdp",
        _hydra("reward.tracking_ang_vel.params.std"),
        math.sqrt(0.5),
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.upright",
        "mdp",
        _hydra("reward.upright.weight"),
        2.0,
        "gap",
        "缺失；上游 upright 2.0（std=√0.05，trunk_base）。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.variable_posture",
        "mdp",
        _hydra("reward.leg_pose.params.std_walking"),
        PRESENT,
        "gap",
        "上游 pose（variable_posture）站立/行走两套 std 并在行走时继续评分；"
        "UniLab leg_pose 仅 std_standing，行走时置零。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.variable_posture.weight",
        "mdp",
        _hydra("reward.leg_pose.weight"),
        1.0,
        "match",
        "权重一致；形态差异见 variable_posture 条目。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.body_ang_vel",
        "mdp",
        _hydra("reward.body_ang_vel.weight"),
        -0.05,
        "gap",
        "缺失；上游惩罚 trunk_base 全三轴角速度。UniLab ang_vel_xy(-0.05) 只覆盖 xy。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.angular_momentum",
        "mdp",
        _hydra("reward.angular_momentum.weight"),
        -0.02,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.dof_pos_limits",
        "mdp",
        _hydra("reward.dof_pos_limits.weight"),
        -1.0,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.action_rate.stage0",
        "mdp",
        _hydra("reward.action_rate.weight"),
        -0.1,
        "match",
        "curriculum -0.1 → -1.0（见 curriculum 条目）。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.air_time.weight",
        "mdp",
        _hydra("reward.air_time.weight"),
        3.0,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.air_time.window_min",
        "mdp",
        _hydra("reward.air_time.params.threshold_min"),
        0.125,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.air_time.window_max",
        "mdp",
        _hydra("reward.air_time.params.threshold_max"),
        0.3,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.foot_clearance.weight",
        "mdp",
        _hydra("reward.foot_clearance.weight"),
        -2.0,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.foot_clearance.target_height",
        "mdp",
        _hydra("reward.foot_clearance.params.target_height"),
        0.02,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.foot_swing_height",
        "mdp",
        _hydra("reward.foot_swing_height.weight"),
        -0.25,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.foot_slip",
        "mdp",
        _hydra("reward.foot_slip.weight"),
        -0.1,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.self_collisions",
        "mdp",
        _hydra("reward.self_collisions.weight"),
        -1.0,
        "gap",
        "缺失；上游 self_collision_cost（10 N 接触力阈值）。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.head_pose_tracking.weight",
        "mdp",
        _hydra("reward.head_pose_tracking.weight"),
        2.0,
        "gap",
        "当前 0.5；上游 2.0。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.head_pose_tracking.std",
        "mdp",
        _hydra("reward.head_pose_tracking.params.std"),
        0.5,
        "match",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.body_pose_tracking",
        "mdp",
        _hydra("reward.body_pose_tracking.weight"),
        0.0,
        "gap",
        "缺失；上游挂起态（weight 0.0，保持 command/obs 通道活性）。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.head_pose_bias.stage0",
        "mdp",
        _hydra("reward.head_pose_bias.weight"),
        0.0,
        "match",
        "curriculum 0 → 3.0（见 curriculum 条目）。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.extra.orientation",
        "mdp",
        _hydra("reward.orientation.weight"),
        ABSENT,
        "gap",
        "上游无此 term（当前 -3.0）；对齐时需移除。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.extra.base_height",
        "mdp",
        _hydra("reward.base_height.weight"),
        ABSENT,
        "gap",
        "上游无此 term（当前 -50.0）；对齐时需移除。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.extra.alive",
        "mdp",
        _hydra("reward.alive.weight"),
        ABSENT,
        "gap",
        "上游无此 term（当前 +0.1）；对齐时需移除。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.extra.flight_phase",
        "mdp",
        _hydra("reward.flight_phase.weight"),
        ABSENT,
        "gap",
        "上游无此 term（当前 -2.0）；对齐时需移除。",
        tasks=VELOCITY_TASK,
    ),
    AlignmentEntry(
        "mdp.reward.extra.lin_vel_z",
        "mdp",
        _hydra("reward.lin_vel_z.weight"),
        ABSENT,
        "gap",
        "上游无独立 lin_vel_z term（vz² 已并入 track_linear_velocity 指数）。",
        tasks=VELOCITY_TASK,
    ),
    # ------------------------------------------------------------------
    # infra: PPO 超参数（compose 后）
    # ------------------------------------------------------------------
    AlignmentEntry(
        "infra.algo.actor_hidden_dims",
        "infra",
        _hydra("algo.policy.actor_hidden_dims"),
        [512, 256, 128],
        "match",
    ),
    AlignmentEntry(
        "infra.algo.critic_hidden_dims",
        "infra",
        _hydra("algo.policy.critic_hidden_dims"),
        [512, 256, 128],
        "match",
    ),
    AlignmentEntry(
        "infra.algo.activation",
        "infra",
        _hydra("algo.policy.activation"),
        "elu",
        "match",
    ),
    AlignmentEntry(
        "infra.algo.init_noise_std",
        "infra",
        _hydra("algo.policy.init_noise_std"),
        1.0,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.num_learning_epochs",
        "infra",
        _hydra("algo.algorithm.num_learning_epochs"),
        5,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.num_mini_batches",
        "infra",
        _hydra("algo.algorithm.num_mini_batches"),
        4,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.learning_rate",
        "infra",
        _hydra("algo.algorithm.learning_rate"),
        1e-3,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.schedule",
        "infra",
        _hydra("algo.algorithm.schedule"),
        "adaptive",
        "match",
    ),
    AlignmentEntry(
        "infra.algo.desired_kl",
        "infra",
        _hydra("algo.algorithm.desired_kl"),
        0.01,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.gamma",
        "infra",
        _hydra("algo.algorithm.gamma"),
        0.99,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.lam",
        "infra",
        _hydra("algo.algorithm.lam"),
        0.95,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.clip_param",
        "infra",
        _hydra("algo.algorithm.clip_param"),
        0.2,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.entropy_coef",
        "infra",
        _hydra("algo.algorithm.entropy_coef"),
        0.01,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.num_steps_per_env",
        "infra",
        _hydra("algo.num_steps_per_env"),
        24,
        "match",
    ),
    AlignmentEntry(
        "infra.algo.empirical_normalization",
        "infra",
        _hydra("algo.empirical_normalization"),
        True,
        "match",
    ),
    AlignmentEntry(
        "infra.num_envs",
        "infra",
        _hydra("algo.num_envs"),
        4096,
        "gap",
        "当前 2048。",
    ),
    AlignmentEntry(
        "infra.seed",
        "infra",
        _hydra("algo.seed"),
        42,
        "gap",
        "当前 1。",
    ),
    AlignmentEntry(
        "infra.max_iterations",
        "infra",
        _hydra("algo.max_iterations"),
        None,
        "note",
        "当前 500，上游 50000；训练预算由 child 5 决策，不参与 match/gap 判定。",
    ),
    AlignmentEntry(
        "infra.mujoco_warp_version",
        "infra",
        _module("mujoco_warp.__version__"),
        "3.8.1",
        "note",
        "UniLab pin 3.10.0.3 vs 上游 3.8.1；环境项，本 roadmap 不修复。",
    ),
)


@dataclass(frozen=True)
class EntryResult:
    task: str
    name: str
    category: str
    status: str
    current: Any
    target: Any
    matches: bool | None  # None for status="note"
    note: str


def _normalize(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _values_equal(current: Any, target: Any) -> bool:
    current, target = _normalize(current), _normalize(target)
    if isinstance(current, float) or isinstance(target, float):
        if not isinstance(current, (int, float)) or not isinstance(target, (int, float)):
            return False
        return math.isclose(float(current), float(target), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(current, list) and isinstance(target, list):
        return len(current) == len(target) and all(
            _values_equal(c, t) for c, t in zip(current, target, strict=True)
        )
    if isinstance(current, dict) and isinstance(target, dict):
        return current.keys() == target.keys() and all(
            _values_equal(current[k], target[k]) for k in current
        )
    return bool(current == target)


def _xml_option_attr(attr: str) -> Any:
    for rel_path in XML_OPTION_FILES:
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        option = ET.parse(path).getroot().find(".//option")
        if option is not None and attr in option.attrib:
            return option.attrib[attr]
    return ABSENT


def read_current(source: ContractSource, cfg: Any) -> Any:
    """Read the current UniLab value for one entry source."""
    if source.kind == "hydra":
        value = OmegaConf.select(cfg, source.path)
        return ABSENT if value is None else _normalize(value)
    if source.kind == "hydra_keys":
        value = OmegaConf.select(cfg, source.path)
        return ABSENT if value is None else sorted(_normalize(value).keys())
    if source.kind == "xml_option":
        raw = _xml_option_attr(source.path)
        if raw is ABSENT:
            return ABSENT
        try:
            return float(raw)
        except ValueError:
            return raw
    if source.kind == "module":
        module_path, _, attr = source.path.rpartition(".")
        try:
            return _normalize(getattr(importlib.import_module(module_path), attr))
        except (ImportError, AttributeError):
            return ABSENT
    raise ValueError(f"Unknown contract source kind: {source.kind}")


def evaluate_entry(entry: AlignmentEntry, task: str, cfg: Any) -> EntryResult:
    current = read_current(entry.source, cfg)
    if entry.status == "note":
        matches: bool | None = None
    elif entry.target is PRESENT:
        matches = current is not ABSENT
    elif entry.target is ABSENT:
        matches = current is ABSENT
    else:
        matches = current is not ABSENT and _values_equal(current, entry.target)
    return EntryResult(
        task=task,
        name=entry.name,
        category=entry.category,
        status=entry.status,
        current=current,
        target=entry.target if entry.target in (ABSENT, PRESENT) else _normalize(entry.target),
        matches=matches,
        note=entry.note,
    )


def compose_task(task: str) -> Any:
    """Compose the ppo/<task>/mjwarp owner config."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=[f"task={task}/{BACKEND}"])


def evaluate_task(task: str) -> list[EntryResult]:
    cfg = compose_task(task)
    return [evaluate_entry(entry, task, cfg) for entry in ENTRIES if task in entry.tasks]


def evaluate_all(tasks: tuple[str, ...] = MICRODUCK_TASKS) -> list[EntryResult]:
    return [result for task in tasks for result in evaluate_task(task)]

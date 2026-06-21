"""Dirty Go1 checkpoint replay in Drake.

This is an M2 bridge script for the Drake contribution plan. It deliberately
does not implement UniLab's backend interface yet; it only proves that a
trained Go1 actor can be loaded, observed from Drake state, and used to drive a
single Drake plant through Drake-native PD actuators.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    JointActuatorIndex,
    JointIndex,
    MeshcatVisualizer,
    Parser,
    Rgba,
    RigidTransform,
    Simulator,
    StartMeshcat,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation
from pydrake.multibody.tree import PdControllerGains

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "src" / "unilab" / "assets" / "robots" / "go1" / "scene_flat_drake.xml"
DEFAULT_CHECKPOINT = (
    ROOT / "logs" / "rsl_rl_ppo" / "Go1JoystickFlat" / "2026-06-05_02-36-19_mujoco" / "model_150.pt"
)

CTRL_DT = 0.02
SIM_DT = 0.01
ACTION_SCALE = 0.25
KP = 35.0
KD = 0.5
GAIT_FREQUENCY = 2.0
DEFAULT_ANGLES = np.array(
    [
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
        0.0,
        1.0,
        -1.8,
        0.0,
        1.0,
        -1.8,
    ],
    dtype=np.float64,
)
HOME_V = np.zeros(18, dtype=np.float64)
TORQUE_LIMITS = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float64)
CTRL_LIMITS = np.array(
    [
        [-0.863, 0.863],
        [-0.686, 4.501],
        [-2.818, -0.888],
    ]
    * 4,
    dtype=np.float64,
)


class Go1Actor(torch.nn.Module):
    """Minimal actor matching the checkpoint's `mlp.*` keys."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(49, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


@dataclass
class DrakeGo1:
    diagram: object
    plant: object
    scene_graph: object
    simulator: Simulator
    context: object
    plant_context: object
    model_instance: object
    trunk: object
    num_filtered_geometries: int
    meshcat: object | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0))
    parser.add_argument("--sim-dt", type=float, default=SIM_DT)
    parser.add_argument("--ctrl-dt", type=float, default=CTRL_DT)
    parser.add_argument("--base-z", type=float, default=0.27)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    parser.add_argument("--action-scale", type=float, default=ACTION_SCALE)
    parser.add_argument("--action-mode", choices=("policy", "zero"), default="policy")
    parser.add_argument(
        "--control-mode",
        choices=("native-pd", "external-torque-pd"),
        default="native-pd",
        help="Use Drake's in-plant PD actuator controller, or the old diagnostic torque-PD path.",
    )
    parser.add_argument("--realtime-rate", type=float, default=1.0)
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--meshcat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--meshcat-port",
        type=int,
        default=7000,
        help="Reserved for future use; Drake 1.53 StartMeshcat() chooses its own port.",
    )
    parser.add_argument("--hold", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--clip-action", type=float, default=None)
    parser.add_argument("--torque-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-drake-warnings", action="store_true")
    return parser.parse_args()


def make_home_q(base_z: float) -> np.ndarray:
    return np.concatenate(
        [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(base_z)], dtype=np.float64),
            DEFAULT_ANGLES,
        ]
    )


def load_actor(checkpoint: Path) -> Go1Actor:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor_state = ckpt.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError(f"{checkpoint} does not look like an RSL-RL actor checkpoint")

    actor = Go1Actor()
    mlp_state = {key: value for key, value in actor_state.items() if key.startswith("mlp.")}
    actor.load_state_dict(mlp_state, strict=True)
    actor.eval()
    return actor


def add_debug_floor(meshcat: object | None) -> None:
    if meshcat is None:
        return
    meshcat.SetObject("/debug/floor", Box(3.0, 3.0, 0.01), Rgba(0.35, 0.38, 0.40, 0.45))
    meshcat.SetTransform("/debug/floor", RigidTransform([0.0, 0.0, -0.005]))


def exclude_robot_self_collisions(
    plant: object, scene_graph: object, model_instance: object
) -> int:
    """Mirror MuJoCo's Go1 collision filtering: robot geoms should not collide with each other."""
    robot_geometries = GeometrySet()
    count = 0
    for body_index in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body_index)
        for geometry_id in plant.GetCollisionGeometriesForBody(body):
            robot_geometries.Add(geometry_id)
            count += 1

    if count:
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration().ExcludeWithin(robot_geometries)
        )
    return count


def build_drake_go1(scene: Path, args: argparse.Namespace) -> DrakeGo1:
    if not scene.exists():
        raise FileNotFoundError(f"Drake scene not found: {scene}")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=args.sim_dt)
    plant.set_contact_model(ContactModel.kPointContactOnly)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    model_instances = Parser(plant).AddModels(str(scene))
    if len(model_instances) != 1:
        raise RuntimeError(f"expected one model instance from {scene}, got {len(model_instances)}")
    model_instance = model_instances[0]
    for i, limit in enumerate(TORQUE_LIMITS):
        actuator = plant.get_joint_actuator(JointActuatorIndex(i))
        # Drake's MJCF parser maps Go1 position-actuator ctrlrange values to
        # effort limits. Restore the MuJoCo forcerange limits instead.
        actuator.set_effort_limit(float(limit))
        if args.control_mode == "native-pd":
            actuator.set_controller_gains(PdControllerGains(p=float(args.kp), d=float(args.kd)))
    num_filtered_geometries = exclude_robot_self_collisions(plant, scene_graph, model_instance)
    plant.Finalize()

    meshcat = None
    if args.meshcat:
        _ = args.meshcat_port
        meshcat = StartMeshcat()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        add_debug_floor(meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    if plant.num_positions() != 19 or plant.num_velocities() != 18 or plant.num_actuators() != 12:
        raise RuntimeError(
            "unexpected Go1 dimensions: "
            f"nq={plant.num_positions()} nv={plant.num_velocities()} nu={plant.num_actuators()}"
        )

    plant.SetPositions(plant_context, make_home_q(args.base_z))
    plant.SetVelocities(plant_context, HOME_V)
    if args.control_mode == "native-pd":
        plant.get_actuation_input_port(model_instance).FixValue(
            plant_context, np.zeros(12, dtype=np.float64)
        )
        set_native_pd_target(plant, plant_context, model_instance, DEFAULT_ANGLES)
    else:
        plant.get_actuation_input_port().FixValue(plant_context, np.zeros(12, dtype=np.float64))

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(max(0.0, float(args.realtime_rate)))
    simulator.Initialize()
    diagram.ForcedPublish(context)

    return DrakeGo1(
        diagram=diagram,
        plant=plant,
        scene_graph=scene_graph,
        simulator=simulator,
        context=context,
        plant_context=plant_context,
        model_instance=model_instance,
        trunk=plant.GetBodyByName("trunk"),
        num_filtered_geometries=num_filtered_geometries,
        meshcat=meshcat,
    )


def print_model_order(drake: DrakeGo1) -> None:
    print("Drake model dimensions:")
    print(
        f"  nq={drake.plant.num_positions()} "
        f"nv={drake.plant.num_velocities()} "
        f"nu={drake.plant.num_actuators()}"
    )
    print("Actuator order:")
    for i in range(drake.plant.num_actuators()):
        actuator = drake.plant.get_joint_actuator(JointActuatorIndex(i))
        print(f"  {i:2d}: {actuator.name()}")
    print("Joint order:")
    for i in range(drake.plant.num_joints()):
        joint = drake.plant.get_joint(JointIndex(i))
        print(
            f"  {i:2d}: {joint.name()} "
            f"q_start={joint.position_start()} v_start={joint.velocity_start()}"
        )


def feet_phase_from_phase(phase: float) -> np.ndarray:
    feet_phase = np.empty(4, dtype=np.float64)
    feet_phase[0] = phase
    feet_phase[3] = phase
    feet_phase[1] = (phase + 0.5) % 1.0
    feet_phase[2] = (phase + 0.5) % 1.0
    return feet_phase


def read_drake_state(
    drake: DrakeGo1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plant = drake.plant
    context = drake.plant_context
    q = np.asarray(plant.GetPositions(context), dtype=np.float64)
    v = np.asarray(plant.GetVelocities(context), dtype=np.float64)

    x_wb = plant.EvalBodyPoseInWorld(context, drake.trunk)
    r_wb = x_wb.rotation().matrix()
    v_wb = plant.EvalBodySpatialVelocityInWorld(context, drake.trunk)
    gyro_body = r_wb.T @ np.asarray(v_wb.rotational(), dtype=np.float64)
    linvel_body = r_wb.T @ np.asarray(v_wb.translational(), dtype=np.float64)
    upvector_world = r_wb[:, 2]

    dof_pos = q[7:]
    dof_vel = v[6:]
    return dof_pos, dof_vel, gyro_body, upvector_world, linvel_body


def make_actor_obs(
    *,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    gyro: np.ndarray,
    upvector: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    phase: float,
) -> np.ndarray:
    obs = np.concatenate(
        [
            gyro,
            -upvector,
            dof_pos - DEFAULT_ANGLES,
            dof_vel,
            last_action,
            command,
            feet_phase_from_phase(phase),
        ],
        dtype=np.float64,
    )
    if obs.shape != (49,):
        raise RuntimeError(f"bad actor obs shape: {obs.shape}")
    return obs.astype(np.float32, copy=False)


def policy_action(actor: Go1Actor, obs: np.ndarray, clip_action: float | None) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action = actor(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
    if clip_action is not None:
        action = np.clip(action, -float(clip_action), float(clip_action))
    return action


def pd_torque(
    action: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    *,
    action_scale: float,
    kp: float,
    kd: float,
    torque_clip: bool,
) -> np.ndarray:
    target = action_to_target_q(action, action_scale)
    tau = kp * (target - dof_pos) - kd * dof_vel
    if torque_clip:
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
    return tau


def action_to_target_q(action: np.ndarray, action_scale: float) -> np.ndarray:
    target_q = np.asarray(action, dtype=np.float64) * action_scale + DEFAULT_ANGLES
    return np.clip(target_q, CTRL_LIMITS[:, 0], CTRL_LIMITS[:, 1])


def set_native_pd_target(
    plant: object,
    plant_context: object,
    model_instance: object,
    target_q: np.ndarray,
) -> None:
    desired_state = np.concatenate(
        [np.asarray(target_q, dtype=np.float64), np.zeros(12, dtype=np.float64)]
    )
    plant.get_desired_state_input_port(model_instance).FixValue(plant_context, desired_state)


def read_net_actuation(drake: DrakeGo1) -> np.ndarray:
    return np.asarray(
        drake.plant.get_net_actuation_output_port(drake.model_instance).Eval(drake.plant_context),
        dtype=np.float64,
    )


def run_replay(args: argparse.Namespace) -> None:
    if not args.show_drake_warnings:
        logging.getLogger("drake").setLevel(logging.ERROR)

    command = np.asarray(args.command, dtype=np.float64)
    actor = load_actor(args.checkpoint)
    drake = build_drake_go1(args.scene, args)
    print_model_order(drake)
    if drake.meshcat is not None:
        print(f"Meshcat: {drake.meshcat.web_url()}")
    print(f"Robot self-collision filtering: {drake.num_filtered_geometries} collision geometries")
    if args.start_delay > 0:
        print(f"Starting replay in {args.start_delay:.1f}s...")
        time.sleep(args.start_delay)

    phase = 0.0
    last_action = np.zeros(12, dtype=np.float64)
    final_stats: dict[str, float] = {}

    for step in range(args.steps):
        dof_pos, dof_vel, gyro, upvector, _linvel_body = read_drake_state(drake)
        obs = make_actor_obs(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            gyro=gyro,
            upvector=upvector,
            last_action=last_action,
            command=command,
            phase=phase,
        )
        if not np.isfinite(obs).all():
            raise FloatingPointError(f"non-finite observation at step {step}")

        if args.action_mode == "zero":
            action = np.zeros(12, dtype=np.float64)
        else:
            action = policy_action(actor, obs, args.clip_action)
        target_q = action_to_target_q(action, args.action_scale)
        if args.control_mode == "native-pd":
            set_native_pd_target(drake.plant, drake.plant_context, drake.model_instance, target_q)
        else:
            tau = pd_torque(
                action,
                dof_pos,
                dof_vel,
                action_scale=args.action_scale,
                kp=args.kp,
                kd=args.kd,
                torque_clip=args.torque_clip,
            )
            drake.plant.get_actuation_input_port().FixValue(drake.plant_context, tau)
        drake.simulator.AdvanceTo((step + 1) * args.ctrl_dt)
        tau = (
            read_net_actuation(drake)
            if args.control_mode == "native-pd"
            else np.asarray(tau, dtype=np.float64)
        )
        _, _, _, _, linvel_body = read_drake_state(drake)

        last_action = action
        phase = (phase + args.ctrl_dt * GAIT_FREQUENCY) % 1.0

        base_z = float(
            drake.plant.EvalBodyPoseInWorld(drake.plant_context, drake.trunk).translation()[2]
        )
        final_stats = {
            "step": float(step + 1),
            "time": float((step + 1) * args.ctrl_dt),
            "base_z": base_z,
            "linvel_x": float(linvel_body[0]),
            "action_max": float(np.max(np.abs(action))),
            "tau_max": float(np.max(np.abs(tau))),
        }
        if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0):
            print(
                f"step={step + 1:04d} t={(step + 1) * args.ctrl_dt:6.2f} "
                f"base_z={base_z: .3f} linvel_x={linvel_body[0]: .3f} "
                f"|a|max={np.max(np.abs(action)): .3f} |tau|max={np.max(np.abs(tau)): .3f}"
            )

        if base_z < 0.05:
            print(f"Stopping early: base_z dropped to {base_z:.3f} at step {step + 1}.")
            break

    print("Replay finished:")
    for key, value in final_stats.items():
        if key == "step":
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.4f}")

    if drake.meshcat is not None and args.hold != 0:
        if args.hold < 0:
            input("Press Enter to exit Meshcat replay...")
        else:
            print(f"Holding Meshcat open for {args.hold:.1f}s...")
            time.sleep(args.hold)


def main() -> None:
    args = parse_args()
    run_replay(args)


if __name__ == "__main__":
    main()

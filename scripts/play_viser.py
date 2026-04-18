# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""Viser-based interactive viewer for trained MuJoCo policies.

Opens a web-based 3D viewer (powered by *viser*) so you can inspect a trained
policy from any browser — no local display or GLFW required.  Supports multiple
environments with an in-browser dropdown to switch between them.

Prerequisites::

    uv sync --extra viser

Usage::

    # Zero-action mode (no checkpoint needed)
    uv run python scripts/play_viser.py task=go2_joystick/mujoco interactive.action_mode=zero

    # With a trained policy
    uv run python scripts/play_viser.py task=go2_joystick/mujoco interactive.action_mode=policy

    # Multiple environments with env switching
    uv run python scripts/play_viser.py task=go2_joystick/mujoco algo.num_envs=4 viser.port=8080

    # Motion tracking task
    uv run python scripts/play_viser.py task=g1_motion_tracking/mujoco interactive.action_mode=policy

Camera controls (browser):
    Left-drag    - rotate
    Scroll       - zoom
    Right-drag   - pan
"""

import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import (
    ensure_registries,
    get_entrypoint_log_root,
)
from unilab.utils.render_many import get_grid_offsets
from unilab.utils.rsl_rl_compat import (
    convert_config_v3_to_v4,
    convert_config_v5,
    is_rsl_rl_v4,
    is_rsl_rl_v5,
)
from unilab.utils.rsl_rl_vec_env_wrapper import RslRlVecEnvWrapper
from unilab.utils.viser_scene import VISER_AVAILABLE, MujocoViserScene, build_visible_env_indices

ensure_registries()

from unilab.base import registry

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)

if not VISER_AVAILABLE:
    print("[play_viser] viser is not installed. Install with: uv sync --extra viser")
    sys.exit(1)

import viser  # noqa: E402
from play_interactive import (  # noqa: E402
    PlayInteractiveArgs,
    _available_backends_for_task,
    _backend_adapter,
    _infer_checkpoint_actor_input_dim,
    resolve_checkpoint,
)

# --------------------------------------------------------------------------- #
# Core viewer                                                                 #
# --------------------------------------------------------------------------- #


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Return the composed PPO algo config as a plain dict.

    Args:
        cfg: Hydra config for the current playback run.

    Returns:
        The resolved ``cfg.algo`` subtree as a mutable dict for rsl_rl.
    """
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


def _load_env_playback_model(env: Any, env_index: int) -> mujoco.MjModel:
    """Resolve the exact MuJoCo model for one playback env.

    Args:
        env: UniLab env exposing the playback-model contract.
        env_index: Selected vectorized environment index.

    Returns:
        The MuJoCo model assigned to the selected env.
    """
    model = env.get_playback_model(env_index)
    if not isinstance(model, mujoco.MjModel):
        raise TypeError(f"Expected mujoco.MjModel for playback, got {type(model)!r}")
    return model


def _scene_offset(offset_xy: np.ndarray) -> tuple[float, float, float]:
    """Convert a 2D grid offset into a 3D scene offset.

    Args:
        offset_xy: XY grid offset.

    Returns:
        XYZ offset tuple with zero Z displacement.
    """
    return (float(offset_xy[0]), float(offset_xy[1]), 0.0)


def _build_scene_entries(
    server: Any,
    env: Any,
    *,
    mode: str,
    selected_visible_idx: int,
    visible_env_indices: np.ndarray,
    spacing: float,
) -> list[dict[str, Any]]:
    """Construct viser scenes and MuJoCo data objects for active env views.

    Args:
        server: Active viser server.
        env: UniLab env exposing the playback-model contract.
        mode: Display mode, either ``single`` or ``all``.
        selected_visible_idx: Selected viewer slot in single-env mode.
        visible_env_indices: Runtime env indices exposed in the viewer.
        spacing: Grid spacing for all-env layout.

    Returns:
        Scene-entry dictionaries consumed by the playback loop.
    """
    entries: list[dict[str, Any]] = []
    visible_envs = int(len(visible_env_indices))
    if mode == "single":
        env_idx = int(visible_env_indices[selected_visible_idx])
        mj_model = _load_env_playback_model(env, env_idx)
        entries.append(
            {
                "slot_idx": selected_visible_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "scene": MujocoViserScene(server, mj_model, name_prefix="/mujoco/single"),
            }
        )
        return entries

    offsets = get_grid_offsets(visible_envs, spacing=spacing)
    for local_idx, env_idx in enumerate(visible_env_indices):
        env_idx = int(env_idx)
        mj_model = _load_env_playback_model(env, env_idx)
        entries.append(
            {
                "slot_idx": local_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "scene": MujocoViserScene(
                    server,
                    mj_model,
                    name_prefix=f"/mujoco/env_{local_idx}",
                    position_offset=_scene_offset(offsets[local_idx]),
                    render_plane=(local_idx == 0),
                ),
            }
        )
    return entries


def _close_scene_entries(entries: list[dict[str, Any]]) -> None:
    """Remove all active viser scenes owned by the playback loop.

    Args:
        entries: Scene-entry dictionaries created by ``_build_scene_entries``.

    Returns:
        None.
    """
    for entry in entries:
        entry["scene"].close()


def play_viser(args: PlayInteractiveArgs, cfg: DictConfig) -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[play_viser] Device: {device}")

    # --- Validate backend ---------------------------------------------------
    available_backends = _available_backends_for_task(args.task)
    if available_backends and "mujoco" not in available_backends:
        print(
            f"[play_viser] Task {args.task} does not support MuJoCo backend. "
            f"Available: {available_backends or ('<none>',)}"
        )
        return

    # --- Create environment --------------------------------------------------
    num_envs = int(OmegaConf.select(cfg, "viser.max_envs") or 1)

    if cfg is None:
        env = registry.make(args.task, num_envs=num_envs, sim_backend="mujoco")
    else:
        from unilab.training import create_env

        env_cfg_override = _backend_adapter(cfg).build_task_env_cfg_override()
        env = create_env(
            cfg,
            num_envs=num_envs,
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=args.task,
        )

    # --- Load policy ---------------------------------------------------------
    actor_obs_dim = int(env.obs_groups_spec.get("obs", sum(env.obs_groups_spec.values())))
    flat_obs_dim = int(sum(env.obs_groups_spec.values()))

    policy_obs_mode = args.policy_obs_mode
    algo_log_name = getattr(args, "algo_log_name", "rsl_rl_ppo")
    ckpt = None
    if args.action_mode == "policy":
        ckpt = resolve_checkpoint(
            args.task,
            args.load_run,
            getattr(args, "checkpoint", None),
            algo_log_name,
            getattr(args, "log_root", None),
        )
        if policy_obs_mode == "auto" and ckpt is not None:
            ckpt_dim = _infer_checkpoint_actor_input_dim(ckpt)
            if ckpt_dim == actor_obs_dim:
                policy_obs_mode = "actor"
            elif ckpt_dim == flat_obs_dim:
                policy_obs_mode = "flat"
            elif ckpt_dim is not None:
                raise RuntimeError(
                    f"Checkpoint actor input dim mismatch: "
                    f"ckpt={ckpt_dim}, actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim}."
                )
            else:
                policy_obs_mode = "flat"

    wrapped_env = RslRlVecEnvWrapper(env, device=device, policy_obs_mode=policy_obs_mode)

    train_cfg = _algo_config_dict(cfg)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
    if is_rsl_rl_v5():
        train_cfg = cast(dict[str, Any], convert_config_v5(train_cfg))
    elif is_rsl_rl_v4():
        train_cfg = convert_config_v3_to_v4(train_cfg)

    policy = None
    if args.action_mode == "policy":
        if ckpt is None:
            print("[play_viser] WARNING: no checkpoint found — falling back to zero actions.")
        else:
            log_dir = str(
                get_entrypoint_log_root(
                    ROOT_DIR,
                    algo_log_name=algo_log_name,
                    log_root=getattr(args, "log_root", None),
                )
                / args.task
                / "play_temp"
            )
            runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=device)
            runner.load(
                ckpt,
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=device)

    print(f"[play_viser] Action mode: {args.action_mode}")

    # --- GUI controls --------------------------------------------------------
    max_visible_envs = min(int(OmegaConf.select(cfg, "viser.max_envs") or num_envs), num_envs)
    env_options = tuple(f"env_{i}" for i in range(max_visible_envs))
    initial_env_idx = int(OmegaConf.select(cfg, "viser.env_idx") or 0)
    initial_env_idx = min(initial_env_idx, max_visible_envs - 1)
    initial_mode = str(OmegaConf.select(cfg, "viser.display_mode") or "all")
    if initial_mode not in {"single", "all"}:
        initial_mode = "all"
    visible_env_indices = build_visible_env_indices(num_envs, max_visible_envs)

    # --- Load MuJoCo model for visualization ---------------------------------
    if bool(getattr(args, "use_env_visual_model", True)):
        print(
            "[play_viser] Using backend playback models for visualization so per-env "
            "MuJoCo model variants (for example object scale) stay correct."
        )
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = env.cfg.ctrl_dt
    render_spacing = float(
        OmegaConf.select(cfg, "training.render_spacing") or getattr(env.cfg, "render_spacing", 1.0)
    )

    # --- Setup viser server --------------------------------------------------
    port = int(OmegaConf.select(cfg, "viser.port") or 8080)
    server = viser.ViserServer(port=port)

    with server.gui.add_folder("Controls"):
        display_dropdown = server.gui.add_dropdown(
            "Display",
            options=("all", "single"),
            initial_value=initial_mode,
        )
        env_dropdown = server.gui.add_dropdown(
            "Environment",
            options=env_options,
            initial_value=env_options[initial_env_idx],
        )
        pause_button = server.gui.add_button("Pause / Resume")
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )

    paused = {"value": False}
    env_idx = {"value": initial_env_idx}
    display_mode = {"value": initial_mode}
    visible_envs = {"value": max_visible_envs}
    scene_entries = {
        "value": _build_scene_entries(
            server,
            env,
            mode=display_mode["value"],
            selected_visible_idx=env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
    }

    def _rebuild_scenes() -> None:
        _close_scene_entries(scene_entries["value"])
        scene_entries["value"] = _build_scene_entries(
            server,
            env,
            mode=display_mode["value"],
            selected_visible_idx=env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
        if display_mode["value"] == "single":
            runtime_idx = int(visible_env_indices[env_idx["value"]])
            print(f"[play_viser] Showing env_{env_idx['value']} (runtime env {runtime_idx})")
        else:
            print(
                "[play_viser] Showing "
                f"{visible_envs['value']} env slots mapped to runtime envs "
                f"{visible_env_indices.tolist()}"
            )

    @pause_button.on_click
    def _on_pause_click(event: Any) -> None:
        del event
        paused["value"] = not paused["value"]
        status = "paused" if paused["value"] else "resumed"
        print(f"[play_viser] {status}")

    @display_dropdown.on_update
    def _on_display_mode_update(event: Any) -> None:
        del event
        display_mode["value"] = str(display_dropdown.value)
        env_dropdown.disabled = display_mode["value"] == "all"
        _rebuild_scenes()

    @env_dropdown.on_update
    def _on_env_switch(event: Any) -> None:
        del event
        selected = env_dropdown.value
        idx = int(selected.split("_")[1])
        env_idx["value"] = idx
        if display_mode["value"] == "single":
            _rebuild_scenes()
        else:
            print(f"[play_viser] Selected env_{idx} (runtime env {int(visible_env_indices[idx])})")

    env_dropdown.disabled = display_mode["value"] == "all"

    obs, _info = wrapped_env.reset()
    action_low = env.action_space.low
    action_high = env.action_space.high

    print(f"[play_viser] Server running at http://localhost:{port}")
    print(f"[play_viser] {num_envs} environment(s) loaded. Open browser to view.")
    if display_mode["value"] == "all":
        print(
            "[play_viser] Rendering "
            f"{visible_envs['value']} env slots simultaneously from runtime envs "
            f"{visible_env_indices.tolist()}."
        )
    print("[play_viser] Press Ctrl+C to quit.")

    # --- Main loop -----------------------------------------------------------
    try:
        with torch.inference_mode():
            while True:
                t0 = time.perf_counter()

                if not paused["value"]:
                    if args.action_mode == "policy" and policy is not None:
                        actions = policy(obs)
                    elif args.action_mode == "random":
                        actions = (
                            torch.from_numpy(
                                np.random.uniform(
                                    action_low,
                                    action_high,
                                    size=(num_envs, env.action_space.shape[0]),
                                )
                            )
                            .to(device)
                            .float()
                        )
                    else:  # zero
                        actions = torch.zeros(num_envs, env.action_space.shape[0], device=device)

                    obs, _, _, _ = wrapped_env.step(actions)

                physics_batch = env.get_physics_state_snapshot()
                for entry in scene_entries["value"]:
                    phys = physics_batch[int(entry["runtime_env_idx"])].astype(np.float64)
                    mujoco.mj_setState(entry["model"], entry["data"], phys, state_spec)
                    mujoco.mj_forward(entry["model"], entry["data"])
                    entry["scene"].update(entry["data"])

                # Real-time pacing
                speed = speed_slider.value
                target_dt = ctrl_dt / speed
                elapsed = time.perf_counter() - t0
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[play_viser] Shutting down.")
    finally:
        _close_scene_entries(scene_entries["value"])


# --------------------------------------------------------------------------- #
# Hydra entry point                                                           #
# --------------------------------------------------------------------------- #


def _build_play_args(cfg: DictConfig) -> PlayInteractiveArgs:
    return PlayInteractiveArgs(
        task=str(cfg.training.task_name),
        load_run=str(cfg.algo.load_run),
        checkpoint=(
            str(OmegaConf.select(cfg, "algo.checkpoint"))
            if OmegaConf.select(cfg, "algo.checkpoint") not in (None, -1, "-1")
            else None
        ),
        action_mode=str(cfg.interactive.action_mode),
        policy_obs_mode=str(cfg.interactive.policy_obs_mode),
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=(
            str(cfg.training.log_root)
            if OmegaConf.select(cfg, "training.log_root") is not None
            else None
        ),
        show_target_bodies=bool(cfg.interactive.show_target_bodies),
        show_reward_debug=bool(cfg.interactive.show_reward_debug),
        target_show_axes=bool(cfg.interactive.target_show_axes),
        target_body_names=str(cfg.interactive.target_body_names),
        target_max_bodies=int(cfg.interactive.target_max_bodies),
        target_marker_radius=float(cfg.interactive.target_marker_radius),
        target_axis_length=float(cfg.interactive.target_axis_length),
        target_marker_alpha=float(cfg.interactive.target_marker_alpha),
        reward_debug_show_velocity=bool(cfg.interactive.reward_debug_show_velocity),
        reward_debug_lin_vel_scale=float(cfg.interactive.reward_debug_lin_vel_scale),
        reward_debug_ang_vel_scale=float(cfg.interactive.reward_debug_ang_vel_scale),
        reward_debug_show_connectors=bool(cfg.interactive.reward_debug_show_connectors),
        reward_debug_show_global_anchor=bool(cfg.interactive.reward_debug_show_global_anchor),
        camera_follow_body=bool(cfg.interactive.camera_follow_body),
        camera_focus_body_name=str(cfg.interactive.camera_focus_body_name),
        camera_height_offset=float(cfg.interactive.camera_height_offset),
        camera_distance=(
            float(cfg.interactive.camera_distance)
            if OmegaConf.select(cfg, "interactive.camera_distance") is not None
            else None
        ),
        camera_elevation=(
            float(cfg.interactive.camera_elevation)
            if OmegaConf.select(cfg, "interactive.camera_elevation") is not None
            else None
        ),
        camera_azimuth=(
            float(cfg.interactive.camera_azimuth)
            if OmegaConf.select(cfg, "interactive.camera_azimuth") is not None
            else None
        ),
        use_env_visual_model=bool(cfg.interactive.use_env_visual_model),
    )


@hydra.main(version_base="1.3", config_path="../conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("play_viser.py only supports MuJoCo backend; use task=<task>/mujoco.")
    play_viser(_build_play_args(cfg), cfg)


if __name__ == "__main__":
    main()

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""Reusable viser (browser-based) playback loop for train and eval entrypoints.

The viser frontend renders through a MuJoCo playback shell: the physics owner
(any backend declaring ``supports_physics_state_playback``) produces
physics-state snapshots, and per-env MuJoCo playback models drive the rendered
scene.  Backends without the upstream unisim physics-state playback contract
fail closed in :func:`assert_physics_state_playback_supported`.

Both the standalone viewer (``unilab.scripts.play_viser``) and the post-training
playback path (``training.play_render_mode=viser`` in the train scripts) drive
this loop with their own playback sessions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
import torch
import viser
from omegaconf import DictConfig, OmegaConf
from unisim.visualization.render_many import get_grid_offsets

from unilab.visualization.interactive_playback import PlaybackControls, PlaybackSession
from unilab.visualization.playback_state import (
    PhysicsStateApplier,
    assert_physics_state_playback_supported,
)
from unilab.visualization.viser_scene import (
    MujocoViserBatchScene,
    MujocoViserScene,
    build_visible_env_indices,
)


def _load_env_playback_model(env: Any, env_index: int) -> mujoco.MjModel:
    """Resolve the MuJoCo playback model for one env.

    Backends either return an ``MjModel`` directly (mujoco) or a model file
    path (mjwarp per-env variants); both are accepted.
    """
    model = env.get_playback_model(env_index)
    if isinstance(model, mujoco.MjModel):
        return model
    if isinstance(model, str):
        if model.lower().endswith(".mjb"):
            return mujoco.MjModel.from_binary_path(model)
        return mujoco.MjModel.from_xml_path(model)
    raise TypeError(f"Expected mujoco.MjModel or model file path for playback, got {type(model)!r}")


def _scene_offset(offset_xy: np.ndarray) -> tuple[float, float, float]:
    """Convert a 2D grid offset into a 3D scene offset."""
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
    """Construct viser scenes and MuJoCo data objects for active env views."""
    entries: list[dict[str, Any]] = []
    if mode == "single":
        env_idx = int(visible_env_indices[selected_visible_idx])
        mj_model = _load_env_playback_model(env, env_idx)
        entries.append(
            {
                "slot_idx": selected_visible_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "applier": PhysicsStateApplier(env, mj_model, env_index=env_idx),
                "scene": MujocoViserScene(server, mj_model, name_prefix="/mujoco/single"),
            }
        )
        return entries

    offsets = get_grid_offsets(len(visible_env_indices), spacing=spacing)
    models: list[mujoco.MjModel] = []
    data: list[mujoco.MjData] = []
    appliers: list[PhysicsStateApplier] = []
    for env_idx in visible_env_indices:
        model = _load_env_playback_model(env, int(env_idx))
        models.append(model)
        data.append(mujoco.MjData(model))
        appliers.append(PhysicsStateApplier(env, model, env_index=int(env_idx)))

    # MuJoCo task instances normally share one model.  Viser can then render
    # each geom as a batched mesh, reducing per-frame messages from
    # O(environments * geoms) to O(geoms).  Keep the old scene-per-env path for
    # heterogeneous playback models, which cannot share batched geometry.
    matching_models = all(
        model.ngeom == models[0].ngeom
        and np.array_equal(model.geom_type, models[0].geom_type)
        and np.array_equal(model.geom_size, models[0].geom_size)
        and np.array_equal(model.geom_dataid, models[0].geom_dataid)
        and np.array_equal(model.geom_rgba, models[0].geom_rgba)
        for model in models[1:]
    )
    if matching_models:
        return [
            {
                "batch": True,
                "runtime_env_indices": visible_env_indices.copy(),
                "models": models,
                "model": models[0],
                "data": data,
                "appliers": appliers,
                "scene": MujocoViserBatchScene(
                    server,
                    models,
                    name_prefix="/mujoco/batch",
                    position_offsets=np.column_stack(
                        (np.asarray(offsets, dtype=np.float64), np.zeros(len(offsets)))
                    ),
                    render_plane=True,
                ),
            }
        ]

    for local_idx, env_idx in enumerate(visible_env_indices):
        env_idx = int(env_idx)
        mj_model = models[local_idx]
        entries.append(
            {
                "slot_idx": local_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "applier": appliers[local_idx],
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
    for entry in entries:
        entry["scene"].close()


def _advance_session(
    session: PlaybackSession,
    controls: PlaybackControls,
    *,
    pause_on_done: bool,
    completed: bool,
) -> bool:
    if completed:
        return True
    advanced = session.advance(controls)
    if not pause_on_done:
        return False
    if not advanced:
        return False
    state = session.env.state
    return bool((state.terminated | state.truncated).any().item())


def run_viser_playback(
    session: PlaybackSession,
    *,
    port: int = 8080,
    max_visible_envs: int = 16,
    env_idx: int = 0,
    display_mode: str = "all",
    render_spacing: float = 1.0,
    start_paused: bool = False,
    speed: float = 1.0,
    pause_on_done: bool = False,
    entrypoint: str = "viser playback",
    log: Callable[[str], None] = print,
) -> None:
    """Serve a playback session through a browser-based viser viewer.

    Blocks until Ctrl+C.  The session must satisfy the
    :class:`~unilab.visualization.interactive_playback.PlaybackSession`
    protocol over an env whose backend declares physics-state playback;
    unsupported backends fail closed at startup.
    """
    env = session.env
    assert_physics_state_playback_supported(env, entrypoint=entrypoint)

    num_envs = int(env.num_envs)
    max_visible_envs = min(int(max_visible_envs), num_envs)
    env_options = tuple(f"env_{i}" for i in range(max_visible_envs))
    initial_env_idx = min(int(env_idx), max_visible_envs - 1)
    initial_mode = display_mode if display_mode in {"single", "all"} else "all"
    visible_env_indices = build_visible_env_indices(num_envs, max_visible_envs)

    ctrl_dt = env.cfg.ctrl_dt

    # --- Setup viser server --------------------------------------------------
    server = viser.ViserServer(port=int(port))

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
        step_button = server.gui.add_button("Step")
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )

    controls = PlaybackControls(paused=bool(start_paused), speed=float(speed))
    speed_slider.value = controls.speed
    selected_env_idx = {"value": initial_env_idx}
    selected_mode = {"value": initial_mode}
    scene_entries = {
        "value": _build_scene_entries(
            server,
            env,
            mode=selected_mode["value"],
            selected_visible_idx=selected_env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
    }

    def _rebuild_scenes() -> None:
        _close_scene_entries(scene_entries["value"])
        scene_entries["value"] = _build_scene_entries(
            server,
            env,
            mode=selected_mode["value"],
            selected_visible_idx=selected_env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
        if selected_mode["value"] == "single":
            runtime_idx = int(visible_env_indices[selected_env_idx["value"]])
            log(f"Showing env_{selected_env_idx['value']} (runtime env {runtime_idx})")
        else:
            log(
                f"Showing {max_visible_envs} env slots mapped to runtime envs "
                f"{visible_env_indices.tolist()}"
            )

    @pause_button.on_click
    def _on_pause_click(event: Any) -> None:
        del event
        paused = controls.toggle_pause()
        status = "paused" if paused else "resumed"
        log(status)

    @step_button.on_click
    def _on_step_click(event: Any) -> None:
        del event
        if not controls.paused:
            controls.pause()
            log("paused for single-step mode")
        controls.request_single_step()
        log("single step requested")

    @display_dropdown.on_update
    def _on_display_mode_update(event: Any) -> None:
        del event
        selected_mode["value"] = str(display_dropdown.value)
        env_dropdown.disabled = selected_mode["value"] == "all"
        _rebuild_scenes()

    @env_dropdown.on_update
    def _on_env_switch(event: Any) -> None:
        del event
        selected = env_dropdown.value
        idx = int(selected.split("_")[1])
        selected_env_idx["value"] = idx
        if selected_mode["value"] == "single":
            _rebuild_scenes()
        else:
            log(f"Selected env_{idx} (runtime env {int(visible_env_indices[idx])})")

    env_dropdown.disabled = selected_mode["value"] == "all"

    session.reset()

    log(f"Server running at http://localhost:{server.get_port()}")
    log(f"{num_envs} environment(s) loaded. Open browser to view.")
    if selected_mode["value"] == "all":
        log(
            f"Rendering {max_visible_envs} env slots simultaneously from runtime envs "
            f"{visible_env_indices.tolist()}."
        )
    log("Press Ctrl+C to quit.")

    # --- Main loop -----------------------------------------------------------
    try:
        with torch.inference_mode():
            completed = False
            while True:
                t0 = time.perf_counter()

                controls.set_speed(float(speed_slider.value))
                was_completed = completed
                completed = _advance_session(
                    session,
                    controls,
                    pause_on_done=pause_on_done,
                    completed=completed,
                )
                if completed and not was_completed:
                    log("Playback completed; holding the final physics state.")

                physics_batch = session.physics_state()
                for entry in scene_entries["value"]:
                    if entry.get("batch", False):
                        for runtime_idx, applier, data in zip(
                            entry["runtime_env_indices"],
                            entry["appliers"],
                            entry["data"],
                            strict=True,
                        ):
                            applier.apply(physics_batch[int(runtime_idx)], data)
                        entry["scene"].update(entry["data"])
                        continue
                    entry["applier"].apply(
                        physics_batch[int(entry["runtime_env_idx"])], entry["data"]
                    )
                    entry["scene"].update(entry["data"])

                # Real-time pacing
                target_dt = controls.target_dt(ctrl_dt)
                elapsed = time.perf_counter() - t0
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[viser playback] Shutting down.")
    finally:
        _close_scene_entries(scene_entries["value"])


def run_viser_playback_from_cfg(
    session: PlaybackSession,
    cfg: DictConfig,
    *,
    entrypoint: str = "viser playback",
    start_paused: bool = False,
    speed: float = 1.0,
    log: Callable[[str], None] = print,
) -> None:
    """Run :func:`run_viser_playback` with viewer options from the Hydra config.

    Reads the shared ``viser.*`` config group (``port``, ``max_envs``,
    ``env_idx``, ``display_mode``) and ``training.render_spacing``.
    """
    env = session.env
    render_spacing = float(
        OmegaConf.select(cfg, "training.render_spacing") or getattr(env.cfg, "render_spacing", 1.0)
    )
    run_viser_playback(
        session,
        port=int(OmegaConf.select(cfg, "viser.port", default=8080) or 8080),
        max_visible_envs=int(OmegaConf.select(cfg, "viser.max_envs", default=16) or 16),
        env_idx=int(OmegaConf.select(cfg, "viser.env_idx", default=0) or 0),
        display_mode=str(OmegaConf.select(cfg, "viser.display_mode", default="all") or "all"),
        render_spacing=render_spacing,
        start_paused=start_paused,
        speed=speed,
        pause_on_done=bool(OmegaConf.select(cfg, "viser.pause_on_done", default=False)),
        entrypoint=entrypoint,
        log=log,
    )


__all__ = ["run_viser_playback", "run_viser_playback_from_cfg"]

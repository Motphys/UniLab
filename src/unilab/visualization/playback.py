"""Playback rendering helpers for interactive and offline visualization."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.scene import SceneCfg

ObsT = TypeVar("ObsT")


def _resolve_record_video(
    *,
    record_video: bool | None,
    output_video: str | Path | None,
) -> bool:
    if record_video is not None:
        return bool(record_video)
    return output_video is not None


def _resolve_headless(
    *,
    headless: bool | None,
    record_video: bool,
) -> bool:
    if headless is not None:
        return bool(headless)
    return record_video


def _env_cfg_value(env: Any, name: str, default: Any) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _configured_model_file(env: Any) -> str | None:
    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None) if cfg is not None else None
    if scene is None:
        return None
    if not isinstance(scene, SceneCfg):
        raise TypeError("env.cfg.scene must be a SceneCfg")
    return scene.model_file


def render_play_mode(
    env,
    *,
    sim_backend: str,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | Path | None = None,
    render_spacing: float | None = None,
    render_offset_mode: str | None = None,
    headless: bool | None = None,
    record_video: bool | None = None,
    frame_state_getter: Callable[[], np.ndarray] | None = None,
    camera_kwargs: dict[str, Any] | None = None,
) -> str | None:
    """Render play mode with explicit headless and video-recording controls."""
    should_record_video = _resolve_record_video(
        record_video=record_video,
        output_video=output_video,
    )
    should_run_headless = _resolve_headless(
        headless=headless,
        record_video=should_record_video,
    )

    if sim_backend == "motrix":
        if should_run_headless or should_record_video:
            if num_steps is None:
                raise ValueError("Motrix captured playback requires a finite num_steps value.")
            if should_record_video and output_video is None:
                raise ValueError("Motrix video recording requires an output_video path.")

            cam_kw = dict(camera_kwargs or {})
            effective_spacing = (
                float(render_spacing)
                if render_spacing is not None
                else float(_env_cfg_value(env, "render_spacing", 1.0))
            )
            env.init_play_renderer(
                render_spacing=effective_spacing,
                render_offset_mode=(
                    str(render_offset_mode) if render_offset_mode is not None else "grid"
                ),
                headless=should_run_headless,
                capture=True,
                width=1280,
                height=720,
                camera_kwargs=cam_kw,
            )

            obs = initialize()
            frames: list[np.ndarray] | None = [] if should_record_video else None
            for _ in range(num_steps):
                obs = step(obs)
                frame = np.asarray(env.capture_play_video_frame(), dtype=np.uint8)
                if frames is not None:
                    frames.append(frame.copy())

            if not should_record_video:
                return None

            assert output_video is not None
            assert frames is not None
            import mediapy as media

            ctrl_dt = float(_env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
            media.write_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
            return str(output_video)

        env.init_play_renderer(
            render_spacing=render_spacing,
            render_offset_mode=(
                str(render_offset_mode) if render_offset_mode is not None else "grid"
            ),
        )
        obs = initialize()
        last_render_time = time.perf_counter()
        render_dt = 1.0 / 60.0
        steps_run = 0

        while num_steps is None or steps_run < num_steps:
            obs = step(obs)
            current_time = time.perf_counter()
            elapsed = current_time - last_render_time
            if elapsed < render_dt:
                time.sleep(render_dt - elapsed)
            last_render_time = time.perf_counter()
            env.render_play_frame()
            steps_run += 1
        return None

    if not should_run_headless:
        raise NotImplementedError("MuJoCo play mode does not support interactive rendering here.")
    if not should_record_video:
        raise ValueError("MuJoCo play rendering requires record_video=true.")
    if num_steps is None:
        raise ValueError("MuJoCo play rendering requires a finite num_steps value.")
    if output_video is None:
        raise ValueError("MuJoCo play rendering requires an output_video path.")
    if frame_state_getter is None:
        frame_state_getter = env.get_physics_state_snapshot
    assert frame_state_getter is not None

    obs = initialize()
    state_list = []
    for _ in range(num_steps):
        obs = step(obs)
        state_list.append(np.asarray(frame_state_getter(), dtype=np.float32).copy())

    from unilab.visualization import render_many

    cam_kw = dict(camera_kwargs or {})
    use_tracking = bool(cam_kw.pop("cam_tracking", False))
    tracking_env_idx = int(cam_kw.pop("cam_tracking_env_idx", 0))
    tracking_extra_envs = int(cam_kw.pop("cam_tracking_extra_envs", 2))
    effective_spacing = (
        float(render_spacing)
        if render_spacing is not None
        else float(_env_cfg_value(env, "render_spacing", 1.0))
    )
    with tempfile.TemporaryDirectory(prefix="unilab-playback-models-") as tmp_dir:
        model_files = _resolve_render_play_model_files(
            env,
            num_envs=state_list[0].shape[0],
            tmp_dir=tmp_dir,
        )

        if use_tracking:
            frames = render_many.render_states_get_frames_tracking(
                state_list,
                model_files,
                width=1280,
                height=720,
                tracking_env_idx=tracking_env_idx,
                max_extra_envs=tracking_extra_envs,
                cam_distance=cam_kw.get("cam_distance", 2.0),
                cam_elevation=cam_kw.get("cam_elevation", -20),
                cam_azimuth=cam_kw.get("cam_azimuth", 90),
                render_spacing=effective_spacing,
            )
        else:
            frames = render_many.render_states_get_frames(
                state_list,
                model_files,
                width=1280,
                height=720,
                camera_id=-1,
                render_spacing=effective_spacing,
                **cam_kw,
            )

    import mediapy as media

    ctrl_dt = float(_env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
    media.write_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
    return str(output_video)


def _resolve_render_play_model_files(
    env: Any,
    *,
    num_envs: int,
    tmp_dir: str | Path,
) -> str | list[str]:
    """Resolve visual MuJoCo model files for offline play/video export."""
    configured_model_file = _configured_model_file(env)
    visual_model_file = str(configured_model_file) if configured_model_file else None
    if not hasattr(env, "get_playback_model"):
        if visual_model_file is None:
            raise ValueError("MuJoCo playback requires either cfg.scene or get_playback_model().")
        return visual_model_file

    first_model = env.get_playback_model(0)
    if isinstance(first_model, (str, Path)):
        return str(first_model)

    import mujoco as _mujoco

    mujoco: Any = _mujoco

    visual_base = (
        mujoco.MjModel.from_xml_path(visual_model_file) if visual_model_file is not None else None
    )
    tmp_root = Path(tmp_dir)
    path_by_model_id: dict[int, str] = {}
    model_files: list[str] = []
    for env_idx in range(num_envs):
        playback_model = env.get_playback_model(env_idx)
        if isinstance(playback_model, (str, Path)):
            model_files.append(str(playback_model))
            continue
        key = id(playback_model)
        saved = path_by_model_id.get(key)
        if saved is None:
            output_path = tmp_root / f"model_{len(path_by_model_id)}.mjb"
            if visual_model_file is None or visual_base is None:
                mujoco.mj_saveModel(playback_model, str(output_path))
                saved = str(output_path)
            else:
                saved = _materialize_visual_playback_model(
                    visual_model_file=visual_model_file,
                    visual_base_model=visual_base,
                    playback_model=playback_model,
                    output_path=output_path,
                )
            path_by_model_id[key] = saved
        model_files.append(saved)

    if len(set(model_files)) == 1:
        return model_files[0]
    return model_files


def _materialize_visual_playback_model(
    *,
    visual_model_file: str,
    visual_base_model: Any,
    playback_model: Any,
    output_path: str | Path,
) -> str:
    """Compile a visual MuJoCo model using geom sizes from a playback model."""
    import mujoco as _mujoco

    mujoco: Any = _mujoco

    spec = mujoco.MjSpec.from_file(visual_model_file)
    for geom_id in range(visual_base_model.ngeom):
        geom_name = mujoco.mj_id2name(visual_base_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not geom_name:
            continue
        playback_geom_id = mujoco.mj_name2id(playback_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if playback_geom_id < 0:
            continue
        geom = spec.geom(geom_name)
        if geom is None:
            continue
        geom.size = list(np.asarray(playback_model.geom_size[playback_geom_id], dtype=np.float64))

    visual_model = spec.compile()
    output = Path(output_path)
    mujoco.mj_saveModel(visual_model, str(output))
    return str(output)

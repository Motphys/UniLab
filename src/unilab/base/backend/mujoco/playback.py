"""MuJoCo-owned playback execution helpers."""

from __future__ import annotations

import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from unilab.base.backend.playback_common import env_cfg_value
from unilab.base.scene import SceneCfg

ObsT = TypeVar("ObsT")


def run_mujoco_playback(
    *,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    headless: bool,
    record_video: bool,
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
) -> str | None:
    if not headless:
        raise NotImplementedError("MuJoCo play mode does not support interactive rendering here.")
    if not record_video:
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
    marker_list: list[np.ndarray | None] = []
    for _ in range(num_steps):
        obs = step(obs)
        state_list.append(np.asarray(frame_state_getter(), dtype=np.float32).copy())
        if extra_data_getter is not None:
            marker = extra_data_getter()
            marker_list.append(
                np.asarray(marker, dtype=np.float32).copy() if marker is not None else None
            )
        else:
            marker_list.append(None)

    marker_positions_list = (
        marker_list if any(marker is not None for marker in marker_list) else None
    )

    from unilab.visualization import render_many

    cam_kw = dict(camera_kwargs or {})
    use_tracking = bool(cam_kw.pop("cam_tracking", False))
    tracking_env_idx = int(cam_kw.pop("cam_tracking_env_idx", 0))
    tracking_extra_envs = int(cam_kw.pop("cam_tracking_extra_envs", 2))
    effective_spacing = (
        float(render_spacing)
        if render_spacing is not None
        else float(env_cfg_value(env, "render_spacing", 1.0))
    )
    with tempfile.TemporaryDirectory(prefix="unilab-playback-models-") as tmp_dir:
        model_files = resolve_render_play_model_files(
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
                marker_positions_list=marker_positions_list,
            )
        else:
            frames = render_many.render_states_get_frames(
                state_list,
                model_files,
                width=1280,
                height=720,
                camera_id=-1,
                render_spacing=effective_spacing,
                marker_positions_list=marker_positions_list,
                **cam_kw,
            )

    import mediapy as media

    ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
    media.write_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
    return str(output_video)


def _configured_model_file(env: Any) -> str | None:
    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None) if cfg is not None else None
    if scene is None:
        return None
    if not isinstance(scene, SceneCfg):
        raise TypeError("env.cfg.scene must be a SceneCfg")
    return scene.model_file


def _visual_model_file(env: Any) -> str | None:
    backend = getattr(env, "_backend", None)
    backend_visual_model_file = getattr(backend, "scene_visual_model_file", None)
    if backend_visual_model_file:
        return str(backend_visual_model_file)
    return _configured_model_file(env)


def resolve_render_play_model_files(
    env: Any,
    *,
    num_envs: int,
    tmp_dir: str | Path,
) -> str | list[str]:
    """Resolve visual MuJoCo model files for offline play/video export."""
    visual_model_file = _visual_model_file(env)
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
                saved = materialize_visual_playback_model(
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


def materialize_visual_playback_model(
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

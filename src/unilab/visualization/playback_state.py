"""Contract-driven physics-state application for interactive render frontends.

Interactive entrypoints (``play_interactive``, ``play_viser``) render through a
MuJoCo playback shell: the physics owner produces physics-state snapshots and
the shell only renders.  This module owns the two contract touchpoints of that
path (unilabsim/unisim#291):

- :func:`assert_physics_state_playback_supported` fails at startup with an
  actionable message when the selected backend does not implement the
  playback contract, instead of surfacing a ``NotImplementedError`` from the
  first snapshot fetch deep inside the viewer loop.
- :class:`PhysicsStateApplier` splits snapshot rows through the unisim
  ``physics_state_layout`` contract (``split_state``) and replays mocap
  geometry through the contract ``get_playback_mocap_state`` entrypoint, so
  frontends never hardcode ``1 + nq + nv`` slices or
  ``mjSTATE_FULLPHYSICS``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def assert_physics_state_playback_supported(env: Any, *, entrypoint: str) -> None:
    """Fail at startup unless the env's backend supports physics-state playback.

    Raises:
        NotImplementedError: naming the backend and the rendering paths that
            remain available for it.
    """
    capabilities = env.play_capabilities
    if capabilities.supports_physics_state_playback:
        return
    backend_name = type(getattr(env, "_backend", env)).__name__
    if (
        capabilities.supports_native_interactive_renderer
        or capabilities.supports_native_video_capture
    ):
        hint = "use the backend's native interactive/video rendering entrypoints instead"
    else:
        hint = (
            "no unified rendering path exists for this backend yet "
            "(tracked by unilabsim/unisim#291)"
        )
    raise NotImplementedError(
        f"{entrypoint}: {backend_name} does not support physics-state playback; {hint}."
    )


class PhysicsStateApplier:
    """Apply contract physics-state rows to one MuJoCo playback ``MjData``.

    Cold-path construction validates the playback model against the backend
    physics-state layout; :meth:`apply` then only writes arrays and forwards
    the model.  Mocap geometry is replayed through the contract
    ``get_playback_mocap_state`` entrypoint when the backend declares
    ``supports_mocap_playback`` (the returned state is aligned with the
    playback model, which may differ from the physics model); otherwise the
    snapshot tail is consumed directly.
    """

    def __init__(self, env: Any, model: mujoco.MjModel, env_index: int = 0) -> None:
        self._model = model
        self._env = env
        self._env_index = int(env_index)
        self._mocap_via_contract = False

        layout = env.get_physics_state_layout()
        if model.nq != layout.nq or model.nv != layout.nv:
            raise ValueError(
                f"Playback model dimensions (nq={model.nq}, nv={model.nv}) do not match "
                f"the backend physics-state layout (nq={layout.nq}, nv={layout.nv}); "
                "the viewer model must share the physics joint structure."
            )
        self._layout = layout
        if model.nmocap == 0:
            return
        capabilities = env.play_capabilities
        if capabilities.supports_mocap_playback:
            self._mocap_via_contract = True
        elif layout.nmocap != model.nmocap:
            raise NotImplementedError(
                f"{type(getattr(env, '_backend', env)).__name__} snapshots do not carry the "
                f"{model.nmocap} mocap bodies of the playback model "
                f"(layout nmocap={layout.nmocap}) and the backend does not declare "
                "supports_mocap_playback; mocap geometry cannot be replayed."
            )

    def apply(self, state: np.ndarray, data: mujoco.MjData) -> None:
        """Write one snapshot row into ``data`` and forward the playback model."""
        parts = self._layout.split_state(np.asarray(state, dtype=np.float64))
        data.time = float(parts.time)
        data.qpos[:] = parts.qpos
        data.qvel[:] = parts.qvel
        if self._model.nmocap:
            if self._mocap_via_contract:
                mocap_pos, mocap_quat = self._env.get_playback_mocap_state(self._env_index)
            else:
                mocap_pos, mocap_quat = parts.mocap_pos, parts.mocap_quat
            data.mocap_pos[:] = np.asarray(mocap_pos, dtype=np.float64).reshape(
                self._model.nmocap, 3
            )
            data.mocap_quat[:] = np.asarray(mocap_quat, dtype=np.float64).reshape(
                self._model.nmocap, 4
            )
        mujoco.mj_forward(self._model, data)


__all__ = ["PhysicsStateApplier", "assert_physics_state_playback_supported"]

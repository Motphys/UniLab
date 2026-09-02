"""Drake physics + MuJoCo rendering integration contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unilab.base.np_env import NpEnv


def test_drake_env_auto_playback_is_mujoco_record_plan() -> None:
    class _Env(NpEnv):
        @property
        def action_space(self):
            raise NotImplementedError

        def apply_action(self, actions, state):
            raise NotImplementedError

        def update_state(self, state):
            raise NotImplementedError

    env = object.__new__(_Env)
    env._backend = SimpleNamespace(  # type: ignore[attr-defined]
        backend_type="drake",
        resolve_play_render_plan=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    plan = env.resolve_play_render_plan(
        play_render_mode="auto",
        play_steps=3,
        output_video=Path("play.mp4"),
    )
    assert plan.play_render_mode == "record"

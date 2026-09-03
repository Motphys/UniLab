from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    PlaybackControls,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
    create_appo_playback_session,
    create_hora_distill_playback_session,
    create_rsl_rl_playback_session,
    create_sac_playback_session,
    prepare_motion_overlay_selection,
)

_VEL_LIMIT = [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]


class _FakeWrappedEnv:
    def __init__(self, env: Any):
        self.env = env
        self.reset_calls = 0
        self.step_calls = 0
        self.last_actions = None

    def reset(self):
        self.reset_calls += 1
        return "obs", {}

    def step(self, actions):
        self.step_calls += 1
        self.last_actions = actions
        return f"obs_{self.step_calls}", 0.0, False, {}


def _fake_env(num_envs: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((num_envs, 4), dtype=np.float32),
        state=SimpleNamespace(info={"motion_data": object()}),
    )


def test_playback_controls_gate_single_step_and_speed() -> None:
    controls = PlaybackControls(paused=True, speed=2.0)

    assert controls.consume_step_permission() is False
    controls.request_single_step()
    assert controls.consume_step_permission() is True
    assert controls.consume_step_permission() is False

    controls.resume()
    assert controls.consume_step_permission() is True

    controls.set_speed(0.0)
    assert controls.speed > 0.0
    controls.set_speed(4.0)
    assert controls.target_dt(0.02) == pytest.approx(0.005)


def test_playback_session_advance_respects_pause_and_single_step() -> None:
    env = _fake_env()
    wrapped = _FakeWrappedEnv(env)
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped,
        device="cpu",
        action_mode="zero",
        policy=None,
        num_envs=1,
    )
    controls = PlaybackControls(paused=True)

    session.reset()
    assert session.advance(controls) is False
    assert wrapped.step_calls == 0

    controls.request_single_step()
    assert session.advance(controls) is True
    assert wrapped.step_calls == 1
    assert torch.equal(wrapped.last_actions, torch.zeros((1, 2)))
    assert session.advance(controls) is False


def test_create_rsl_rl_playback_session_loads_checkpoint_and_runner_log_dir() -> None:
    env = SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )
    captured: dict[str, Any] = {}

    class Wrapper:
        def __init__(self, wrapped_env, *, device, policy_obs_mode):
            captured["wrapper_env"] = wrapped_env
            captured["device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return "obs", {}

        def step(self, actions):
            return "obs", 0.0, False, {}

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            captured["runner_log_dir"] = log_dir
            captured["train_cfg"] = train_cfg
            captured["runner_device"] = device

        def load(self, checkpoint, load_cfg):
            captured["checkpoint"] = checkpoint
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, *, device):
            captured["policy_device"] = device
            return lambda obs: torch.ones((1, 2))

    session, policy_obs_mode, checkpoint = create_rsl_rl_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="MyTask",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="custom_ppo",
            log_root=None,
            num_envs=1,
        ),
        env_factory=lambda num_envs: env,
        algo_config={"runner": {"logger": "tensorboard"}},
        root_dir=Path("/repo"),
        device="cpu",
        checkpoint_resolver=lambda *args: "/tmp/model_10.pt",
        checkpoint_input_dim_reader=lambda path: 5,
        entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: (
            Path("/tmp") / algo_log_name
        ),
        wrapper_cls=Wrapper,
        runner_cls=Runner,
        policy_obs_dims_getter=lambda spec: (5, 7),
        train_cfg_normalizer=lambda cfg: cfg,
        log=lambda message: None,
    )

    assert session.env is env
    assert policy_obs_mode == "actor"
    assert checkpoint == "/tmp/model_10.pt"
    assert captured["runner_log_dir"].replace("\\", "/") == "/tmp/custom_ppo/MyTask/play_temp"
    assert captured["checkpoint"] == "/tmp/model_10.pt"
    assert captured["train_cfg"]["runner"]["logger"] == "none"


def test_create_rsl_rl_playback_session_rejects_missing_env() -> None:
    with pytest.raises(RuntimeError, match="Playback env factory"):
        create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task="MyTask",
                load_run="-1",
                checkpoint=None,
                action_mode="zero",
                policy_obs_mode="auto",
                algo_log_name="custom_ppo",
                log_root=None,
                num_envs=1,
            ),
            env_factory=lambda num_envs: None,
            algo_config={},
            root_dir=Path("/repo"),
            device="cpu",
            checkpoint_resolver=lambda *args: None,
            checkpoint_input_dim_reader=lambda path: None,
            entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp"),
            wrapper_cls=object,
            runner_cls=object,
            policy_obs_dims_getter=lambda spec: (0, 0),
            train_cfg_normalizer=lambda cfg: cfg,
            log=lambda message: None,
        )


def test_create_hora_distill_playback_session_loads_student_policy(tmp_path: Path) -> None:
    from tensordict import TensorDict

    checkpoint_path = tmp_path / "hora_stage2_last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    runtime_cfg = SimpleNamespace(training=SimpleNamespace(task_name="RuntimeTask"))
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="SharpaInhandRotation"),
        algo=SimpleNamespace(load_run=str(checkpoint_path)),
    )
    captured: dict[str, Any] = {}

    class FakeEnv:
        num_actions = 2
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        num_actions = 2

        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["wrapper_env"] = env
            captured["wrapper_device"] = device
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            obs = TensorDict(
                {
                    "actor": torch.ones((1, 3), dtype=torch.float32),
                    "proprio_hist": torch.ones((1, 2, 3), dtype=torch.float32),
                },
                batch_size=[1],
            )
            return obs, {}

        def step(self, actions):
            captured["actions"] = actions
            return self.reset()[0], torch.zeros((1,)), torch.zeros((1,), dtype=torch.bool), {}

    class FakeModule:
        def __init__(self, name: str):
            self.name = name
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    actor = FakeModule("actor")
    hist_normalizer = FakeModule("hist_normalizer")

    def fake_student_policy(actor_obj, hist_obj, obs, *, device):
        captured["student_policy"] = (actor_obj, hist_obj, obs, device)
        return torch.full((1, 2), 0.5, dtype=torch.float32)

    deps = {
        "resolve_stage2_checkpoint_path": lambda cfg_obj: (checkpoint_path, checkpoint_path.parent),
        "get_log_root": lambda root_dir, cfg_obj: tmp_path / "logs",
        "format_stage2_play_checkpoint_error": lambda *args, **kwargs: "missing",
        "checkpoint_reader": lambda path, *, map_location, weights_only: {
            "model_state_dict": {},
            "distill_runtime_cfg": {"algo": {"model": {"hidden_dims": [8]}}},
        },
        "apply_teacher_defaults": lambda cfg_obj: cfg_obj,
        "cfg_with_checkpoint_runtime": lambda cfg_obj, checkpoint: runtime_cfg,
        "build_play_env_cfg_override": lambda cfg_obj: {"env": "override"},
        "create_env": lambda cfg_obj, *, num_envs, env_cfg_override: FakeEnv(),
        "wrapper_cls": FakeWrapper,
        "build_student_actor_and_normalizer": lambda wrapped_env, cfg_obj, *, device: (
            actor,
            hist_normalizer,
        ),
        "load_distilled_checkpoint": lambda actor_obj, hist_obj, path, *, device: captured.update(
            {
                "loaded": (actor_obj, hist_obj, path, device),
            }
        ),
        "student_policy": fake_student_policy,
    }

    session, policy_obs_mode, checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="SharpaInhandRotation",
            load_run=str(checkpoint_path),
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=lambda message: None,
    )

    assert session.env is captured["wrapper_env"]
    assert policy_obs_mode == "actor"
    assert checkpoint == str(checkpoint_path)
    assert captured["policy_obs_mode"] == "actor"
    assert captured["loaded"][0] is actor
    assert captured["loaded"][1] is hist_normalizer
    assert captured["loaded"][2] == checkpoint_path
    assert actor.eval_calls == 1
    assert hist_normalizer.eval_calls == 1

    session.reset()
    session.step_once()

    assert torch.equal(captured["actions"], torch.full((1, 2), 0.5))
    assert captured["student_policy"][0] is actor
    assert captured["student_policy"][1] is hist_normalizer
    assert str(captured["student_policy"][3]) == "cpu"


def test_create_hora_distill_playback_session_missing_checkpoint_uses_zero_actions(
    tmp_path: Path,
) -> None:
    runtime_cfg = SimpleNamespace(training=SimpleNamespace(task_name="RuntimeTask"))
    cfg = SimpleNamespace(
        training=SimpleNamespace(task_name="SharpaInhandRotation"),
        algo=SimpleNamespace(load_run="missing"),
    )
    captured: dict[str, Any] = {}
    messages: list[str] = []

    class FakeEnv:
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    class FakeWrapper:
        def __init__(self, env: Any, *, device: str, policy_obs_mode: str):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return "obs", {}

        def step(self, actions):
            captured["actions"] = actions
            return "obs", torch.zeros((1,)), torch.zeros((1,), dtype=torch.bool), {}

    deps = {
        "resolve_stage2_checkpoint_path": lambda cfg_obj: (None, None),
        "get_log_root": lambda root_dir, cfg_obj: tmp_path / "logs",
        "format_stage2_play_checkpoint_error": lambda *args, **kwargs: "missing checkpoint",
        "apply_teacher_defaults": lambda cfg_obj: runtime_cfg,
        "build_play_env_cfg_override": lambda cfg_obj: {"env": "override"},
        "create_env": lambda cfg_obj, *, num_envs, env_cfg_override: FakeEnv(),
        "wrapper_cls": FakeWrapper,
    }

    session, policy_obs_mode, checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="SharpaInhandRotation",
            load_run="missing",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
            num_envs=1,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        deps=deps,
        log=messages.append,
    )

    assert policy_obs_mode == "actor"
    assert checkpoint is None
    assert captured["policy_obs_mode"] == "actor"
    assert messages == [
        "missing checkpoint",
        "WARNING: falling back to zero actions.",
        "Policy obs mode: actor",
        "Action mode: policy",
    ]

    session.reset()
    session.step_once()

    assert torch.equal(captured["actions"], torch.zeros((1, 2)))


def _rsl_rl_session_test_env() -> SimpleNamespace:
    return SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
    )


class _RslRlTestWrapper:
    def __init__(self, wrapped_env, *, device, policy_obs_mode):
        self.env = wrapped_env

    def reset(self):
        return "obs", {}

    def step(self, actions):
        return "obs", 0.0, False, {}


def _rsl_rl_session_kwargs(tmp_path: Path) -> dict[str, Any]:
    return dict(
        playback_cfg=RslRlPlaybackConfig(
            task="MyTask",
            load_run="-1",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="rsl_rl_ppo",
            log_root=None,
            num_envs=1,
        ),
        env_factory=lambda num_envs: _rsl_rl_session_test_env(),
        algo_config={},
        root_dir=tmp_path,
        device="cpu",
        checkpoint_resolver=lambda *args: None,
        checkpoint_input_dim_reader=lambda path: 5,
        entrypoint_log_root=lambda root_dir, *, algo_log_name, log_root=None: (
            tmp_path / algo_log_name
        ),
        wrapper_cls=_RslRlTestWrapper,
        policy_obs_dims_getter=lambda spec: (5, 7),
        train_cfg_normalizer=lambda cfg: cfg,
        log=lambda message: None,
    )


def test_create_rsl_rl_playback_session_runs_sim2sim_preflight(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    checkpoint = run_dir / "model_10.pt"
    torch.save({"actor": {}}, checkpoint)
    preflight_calls: list[str | None] = []

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            pass

        def load(self, checkpoint, load_cfg):
            pass

        def get_inference_policy(self, *, device):
            return lambda obs: torch.ones((1, 2))

    kwargs = _rsl_rl_session_kwargs(tmp_path)
    kwargs["checkpoint_resolver"] = lambda *args: str(checkpoint)
    kwargs["runner_cls"] = Runner
    kwargs["sim2sim_preflight"] = lambda run_dir: preflight_calls.append(run_dir)

    _session, _mode, resolved = create_rsl_rl_playback_session(**kwargs)

    assert resolved == str(checkpoint)
    assert preflight_calls == [str(run_dir)]


def test_create_rsl_rl_playback_session_wraps_load_with_dim_guard(tmp_path: Path) -> None:
    from unilab.utils.sim2sim import CrossBackendIncompatibleError

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    checkpoint = run_dir / "model_10.pt"
    torch.save({"actor": {}}, checkpoint)

    class MismatchRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            pass

        def load(self, checkpoint, load_cfg):
            raise RuntimeError("size mismatch for actor.mlp.0.weight: ...")

        def get_inference_policy(self, *, device):
            raise AssertionError("unreachable")

    kwargs = _rsl_rl_session_kwargs(tmp_path)
    kwargs["checkpoint_resolver"] = lambda *args: str(checkpoint)
    kwargs["runner_cls"] = MismatchRunner

    with pytest.raises(CrossBackendIncompatibleError):
        create_rsl_rl_playback_session(**kwargs)


def test_sac_playback_session_runs_sim2sim_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uni_rl.common.actor_factory as actor_factory
    from omegaconf import OmegaConf

    import unilab.utils.checkpoint as checkpoint_utils
    import unilab.visualization.interactive_playback as interactive_playback
    import unilab.visualization.interactive_playback as offpolicy_play

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {}}, checkpoint)
    preflight_calls: list[tuple[Any, str]] = []

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            pass

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 3, "critic": 5}
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "device": None},
            "algo": {
                "algo_log_name": "sac",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
            },
        }
    )

    def fake_preflight(source_run_dir, target_cfg, *, algo_name, strict):
        preflight_calls.append((source_run_dir, algo_name))
        return target_cfg

    monkeypatch.setattr(interactive_playback, "resolve_sim2sim_config", fake_preflight)
    monkeypatch.setattr(
        offpolicy_play,
        "default_device",
        lambda torch_module, preferred=None: "cpu",
    )
    monkeypatch.setattr(offpolicy_play, "resolve_play_obs_dims", lambda spec: (3, 5))
    monkeypatch.setattr(
        offpolicy_play,
        "resolve_play_actor_spec",
        lambda algo_name, cfg, *, obs_dim, critic_obs_dim: ("sac", {}),
    )
    monkeypatch.setattr(
        checkpoint_utils,
        "resolve_offpolicy_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )
    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    _session, _mode, resolved = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="sac",
            log_root=None,
        ),
        cfg=cfg,
        env_factory=lambda num_envs: FakeEnv(),
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: None,
    )

    assert resolved == str(checkpoint)
    assert preflight_calls == [(str(tmp_path), "sac")]


def test_keyboard_commander_nudges_stack_and_clamp_to_vel_limit() -> None:
    commander = KeyboardCommander.from_vel_limit(_VEL_LIMIT, step_lin=0.1, step_ang=0.2)
    assert commander.command.tolist() == [0.0, 0.0, 0.0]

    # Linear and angular axes stack independently.
    commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
    commander.nudge(KeyboardCommander.AXIS_VYAW, +1.0)
    assert commander.command == pytest.approx([0.1, 0.0, 0.2])

    # Repeated nudges saturate at the configured velocity limits.
    for _ in range(50):
        commander.nudge(KeyboardCommander.AXIS_VX, +1.0)
        commander.nudge(KeyboardCommander.AXIS_VY, -1.0)
    assert commander.command[0] == pytest.approx(1.0)
    assert commander.command[1] == pytest.approx(-0.4)

    commander.zero()
    assert commander.command.tolist() == [0.0, 0.0, 0.0]


def test_keyboard_commander_rejects_bad_vel_limit_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        KeyboardCommander.from_vel_limit([[0.0, 0.0], [1.0, 1.0]])


def test_prepare_motion_overlay_selection_filters_body_names() -> None:
    env = SimpleNamespace(
        motion_loader=object(),
        motion_sampler=object(),
        cfg=SimpleNamespace(body_names=("base", "left_foot", "right_foot")),
    )
    messages: list[str] = []

    selection = prepare_motion_overlay_selection(
        env,
        show_target_bodies=True,
        show_reward_debug=False,
        target_body_names="right_foot,missing,base",
        target_max_bodies=1,
        log=messages.append,
    )

    assert selection.enabled is True
    assert selection.selected_indices.tolist() == [2]
    assert messages == ["WARNING: body name not found in task body list: missing"]


def test_appo_hora_playback_session_uses_hora_wrapper_and_actor_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rsl_rl.utils as rsl_rl_utils
    import uni_rl.hora.models as hora_models
    import uni_rl.hora.rsl_rl as hora_rsl
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {"weight": torch.tensor(1.0)}}, checkpoint)
    captured: dict[str, Any] = {}

    class FakeHoraWrapper:
        def __init__(self, env, *, device, policy_obs_mode):
            captured["wrapper_cls"] = "hora"
            captured["policy_obs_mode"] = policy_obs_mode
            self.env = env
            self.device = device
            self.num_envs = env.num_envs

        def get_observations(self):
            return TensorDict(
                {
                    "actor": torch.zeros((1, 3)),
                    "priv_info": torch.zeros((1, 2)),
                    "proprio_hist": torch.zeros((1, 4, 3)),
                },
                batch_size=1,
            )

        def reset(self):
            return self.get_observations(), {}

        def step(self, actions):
            captured["step_actions"] = actions
            return self.get_observations(), torch.zeros(1), torch.zeros(1).bool(), {}

    class FakeActor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["actor_kwargs"] = kwargs

        def load_state_dict(self, state_dict, strict=True):
            captured["loaded_actor"] = state_dict
            return None

        def forward(self, obs):
            captured["policy_obs"] = obs
            return torch.ones((1, 2), dtype=torch.float32)

    fake_env = SimpleNamespace(
        num_envs=1,
        obs_groups_spec={"obs": 3, "critic": 5},
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
        state=SimpleNamespace(info={}),
    )
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "play_env_num": 1, "log_root": None},
            "algo": {
                "algo_log_name": "hora_appo",
                "load_run": str(tmp_path),
                "checkpoint": str(checkpoint),
                "runtime_impl": "hora_appo",
                "obs_groups": {"actor": {"actor": 0, "priv_info": 0}},
                "actor": {"class_name": "fake.Actor"},
                "critic": {},
            },
        }
    )

    monkeypatch.setattr(hora_rsl, "HoraRslRlVecEnvWrapper", FakeHoraWrapper)
    monkeypatch.setattr(rsl_rl_utils, "resolve_callable", lambda path: FakeActor)
    monkeypatch.setattr(
        hora_models,
        "build_hora_shared_actor_critic",
        lambda **kwargs: torch.nn.Identity(),
    )

    session, policy_obs_mode, resolved_checkpoint = create_appo_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="auto",
            algo_log_name="hora_appo",
            log_root=None,
        ),
        cfg=cfg,
        rl_cfg=OmegaConf.to_container(cfg.algo, resolve=True),
        env_factory=lambda num_envs: fake_env,
        root_dir=tmp_path,
        device="cpu",
        wrapper_cls=object,
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert isinstance(session.actor, FakeActor)
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["wrapper_cls"] == "hora"
    assert captured["policy_obs_mode"] == "actor"
    torch.testing.assert_close(captured["loaded_actor"]["weight"], torch.tensor(1.0))
    assert isinstance(captured["policy_obs"], TensorDict)
    torch.testing.assert_close(captured["step_actions"], torch.ones((1, 2)))


def test_sac_hora_playback_session_updates_priv_info_after_reset_and_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uni_rl.common.actor_factory as actor_factory
    from omegaconf import OmegaConf

    import unilab.utils.checkpoint as checkpoint_utils
    import unilab.visualization.interactive_playback as offpolicy_play

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor": {}}, checkpoint)
    reset_priv = np.array([[4.0, 5.0]], dtype=np.float32)
    step_priv = np.array([[8.0, 9.0]], dtype=np.float32)
    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_actor"] = state_dict

        def explore(self, obs, priv_info, deterministic=True):
            captured["reset_priv_info"] = priv_info.detach().cpu().numpy()
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 3, "critic": 5}
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = None

        def init_state(self):
            self.state = SimpleNamespace(info={})

        def reset(self, env_indices):
            del env_indices
            return (
                {
                    "obs": np.zeros((1, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.zeros((1, 3), dtype=np.float32), reset_priv],
                        axis=1,
                    ),
                },
                {"critic_info": reset_priv},
            )

        def step(self, actions):
            captured["actions"] = actions
            self.state = SimpleNamespace(
                obs={
                    "obs": np.ones((1, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.ones((1, 3), dtype=np.float32), step_priv],
                        axis=1,
                    ),
                },
                info={"critic_info": step_priv},
            )
            return self.state

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "device": None},
            "algo": {
                "algo_log_name": "hora_sac",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
                "runtime_impl": "hora_sac",
            },
        }
    )

    monkeypatch.setattr(
        offpolicy_play,
        "default_device",
        lambda torch_module, preferred=None: "cpu",
    )
    monkeypatch.setattr(offpolicy_play, "resolve_play_obs_dims", lambda spec: (3, 5))
    monkeypatch.setattr(
        offpolicy_play,
        "resolve_play_actor_spec",
        lambda algo_name, cfg, *, obs_dim, critic_obs_dim: (
            "hora_sac",
            {"priv_info_dim": 2},
        ),
    )
    monkeypatch.setattr(
        checkpoint_utils,
        "resolve_offpolicy_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )
    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    session, policy_obs_mode, resolved_checkpoint = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_sac",
            log_root=None,
        ),
        cfg=cfg,
        env_factory=lambda num_envs: FakeEnv(),
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["loaded_actor"] == {}
    np.testing.assert_allclose(captured["reset_priv_info"], reset_priv)
    np.testing.assert_allclose(session.current_priv_info, step_priv)
    assert captured["deterministic"] is True


def test_hora_distill_playback_session_loads_stage2_checkpoint_and_student_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uni_rl.hora.distill as distill
    import uni_rl.hora.rsl_rl as hora_rsl
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    import unilab.base.config_adapter as config_adapter
    import unilab.training as training

    checkpoint = tmp_path / "hora_stage2_last.pt"
    torch.save({"model_state_dict": {}, "distill_runtime_cfg": {}}, checkpoint)
    captured: dict[str, Any] = {}

    class FakeWrapper:
        def __init__(self, env, *, device, policy_obs_mode):
            self.env = env
            self.device = device
            self.num_envs = env.num_envs
            self.num_actions = 2
            captured["policy_obs_mode"] = policy_obs_mode

        def get_observations(self):
            return TensorDict(
                {
                    "actor": torch.zeros((1, 3)),
                    "priv_info": torch.zeros((1, 2)),
                    "proprio_hist": torch.zeros((1, 4, 3)),
                },
                batch_size=1,
            )

        def reset(self):
            return self.get_observations(), {}

        def step(self, actions):
            captured["actions"] = actions
            return self.get_observations(), torch.zeros(1), torch.zeros(1).bool(), {}

    fake_env = SimpleNamespace(
        num_envs=1,
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        ),
        get_physics_state_snapshot=lambda: np.zeros((1, 4), dtype=np.float32),
        state=SimpleNamespace(info={}),
    )
    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "sim_backend": "mujoco", "log_root": None},
            "algo": {"algo_log_name": "hora_distill", "load_run": "run", "checkpoint": -1},
        }
    )

    monkeypatch.setattr(
        training,
        "resolve_hora_stage2_checkpoint_path",
        lambda cfg, *, root_dir: (checkpoint, tmp_path),
    )

    def fake_cfg_with_checkpoint_runtime(cfg, checkpoint_payload):
        captured["runtime_helper_checkpoint"] = checkpoint_payload
        return cfg

    monkeypatch.setattr(
        distill,
        "cfg_with_checkpoint_runtime",
        fake_cfg_with_checkpoint_runtime,
    )

    class FakeBackendAdapter:
        def __init__(self, *args, **kwargs):
            pass

        def build_play_env_cfg_override(self):
            return {}

    monkeypatch.setattr(config_adapter, "BackendAdapter", FakeBackendAdapter)
    monkeypatch.setattr(
        distill,
        "student_policy",
        lambda actor, hist_normalizer, obs, *, device: torch.ones((1, 2)),
    )
    monkeypatch.setattr(config_adapter, "create_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(hora_rsl, "HoraRslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(
        distill,
        "build_student_actor_and_normalizer",
        lambda wrapped_env, cfg, *, device: (
            SimpleNamespace(eval=lambda: None),
            SimpleNamespace(eval=lambda: None),
        ),
    )

    def fake_load_distilled_checkpoint(actor, hist_normalizer, checkpoint_path, *, device):
        captured["loaded_checkpoint"] = checkpoint_path
        return {}

    monkeypatch.setattr(distill, "load_distilled_checkpoint", fake_load_distilled_checkpoint)

    session, policy_obs_mode, resolved_checkpoint = create_hora_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="hora_distill",
            log_root=None,
        ),
        cfg=cfg,
        root_dir=tmp_path,
        device="cpu",
        log=lambda message: None,
    )

    session.reset()
    assert session.advance(PlaybackControls()) is True
    assert policy_obs_mode == "actor"
    assert resolved_checkpoint == str(checkpoint)
    assert captured["runtime_helper_checkpoint"] == {
        "model_state_dict": {},
        "distill_runtime_cfg": {},
    }
    assert captured["policy_obs_mode"] == "actor"
    assert captured["loaded_checkpoint"] == checkpoint
    torch.testing.assert_close(captured["actions"], torch.ones((1, 2)))


def test_infer_checkpoint_actor_input_dim_mlp_key(tmp_path: Path) -> None:
    from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros((8, 42))}}, checkpoint)

    assert infer_checkpoint_actor_input_dim(str(checkpoint)) == 42


def test_infer_checkpoint_actor_input_dim_actor_prefixed_key(tmp_path: Path) -> None:
    from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

    checkpoint = tmp_path / "model_10.pt"
    torch.save(
        {"actor_state_dict": {"actor.mlp.0.weight": torch.zeros((8, 17))}},
        checkpoint,
    )

    assert infer_checkpoint_actor_input_dim(str(checkpoint)) == 17


def test_infer_checkpoint_actor_input_dim_generic_first_layer_key(tmp_path: Path) -> None:
    from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

    checkpoint = tmp_path / "model_10.pt"
    torch.save(
        {"actor_state_dict": {"encoder.0.weight": torch.zeros((4, 23))}},
        checkpoint,
    )

    assert infer_checkpoint_actor_input_dim(str(checkpoint)) == 23


def test_infer_checkpoint_actor_input_dim_returns_none_when_undetectable(
    tmp_path: Path,
) -> None:
    from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

    not_a_dict = tmp_path / "model_list.pt"
    torch.save({"actor_state_dict": ["not", "a", "dict"]}, not_a_dict)
    missing = tmp_path / "model_missing.pt"
    torch.save({"model_state_dict": {}}, missing)
    no_matching_key = tmp_path / "model_other.pt"
    torch.save(
        {"actor_state_dict": {"mlp.2.weight": torch.zeros((8, 8))}},
        no_matching_key,
    )

    assert infer_checkpoint_actor_input_dim(str(not_a_dict)) is None
    assert infer_checkpoint_actor_input_dim(str(missing)) is None
    assert infer_checkpoint_actor_input_dim(str(no_matching_key)) is None


@pytest.mark.parametrize("value", [None, "", "-1", "None", "null"])
def test_normalize_checkpoint_value_maps_sentinels_to_none(value: object) -> None:
    from unilab.visualization.interactive_playback import normalize_checkpoint_value

    assert normalize_checkpoint_value(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("12", "12"), (12, "12"), ("run_2024", "run_2024"), ("/abs/model_5.pt", "/abs/model_5.pt")],
)
def test_normalize_checkpoint_value_keeps_real_values(value: object, expected: str) -> None:
    from unilab.visualization.interactive_playback import normalize_checkpoint_value

    assert normalize_checkpoint_value(value) == expected


def _play_interactive_args(**overrides: Any) -> Any:
    from unilab.visualization.interactive_playback import PlayInteractiveArgs

    defaults: dict[str, Any] = {
        "task": "MyTask",
        "load_run": "-1",
        "checkpoint": None,
        "action_mode": "policy",
        "policy_obs_mode": "auto",
        "algo_log_name": "rsl_rl_ppo",
        "log_root": None,
        "show_target_bodies": False,
        "show_reward_debug": False,
        "target_show_axes": False,
        "target_body_names": "",
        "target_max_bodies": 32,
        "target_marker_radius": 0.05,
        "target_axis_length": 0.2,
        "target_marker_alpha": 0.7,
        "reward_debug_show_velocity": False,
        "reward_debug_lin_vel_scale": 1.0,
        "reward_debug_ang_vel_scale": 1.0,
        "reward_debug_show_connectors": False,
        "reward_debug_show_global_anchor": False,
        "camera_follow_body": False,
        "camera_focus_body_name": "",
        "camera_height_offset": 0.0,
        "camera_distance": None,
        "camera_elevation": None,
        "camera_azimuth": None,
        "use_env_visual_model": False,
        "speed": 1.0,
        "start_paused": False,
    }
    defaults.update(overrides)
    return PlayInteractiveArgs(**defaults)


def test_build_playback_config_maps_play_interactive_args() -> None:
    from unilab.visualization.interactive_playback import build_playback_config

    args = _play_interactive_args(
        task="OtherTask",
        load_run="run_1",
        checkpoint="12",
        action_mode="random",
        policy_obs_mode="flat",
        algo_log_name="custom_ppo",
        log_root="/tmp/logs",
        speed=2.5,
        start_paused=True,
    )

    playback_cfg = build_playback_config(args, num_envs=3)

    assert playback_cfg == RslRlPlaybackConfig(
        task="OtherTask",
        load_run="run_1",
        checkpoint="12",
        action_mode="random",
        policy_obs_mode="flat",
        algo_log_name="custom_ppo",
        log_root="/tmp/logs",
        num_envs=3,
        speed=2.5,
        start_paused=True,
    )


def test_available_backends_for_task_reads_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from unilab.base import registry
    from unilab.visualization.interactive_playback import available_backends_for_task

    monkeypatch.setattr(
        registry,
        "list_registered_envs",
        lambda: {
            "KnownTask": {"available_backends": ["mujoco", "motrix"]},
            "BadTask": {"available_backends": "mujoco"},
        },
    )

    assert available_backends_for_task("KnownTask") == ("mujoco", "motrix")
    assert available_backends_for_task("UnknownTask") == ()
    assert available_backends_for_task("BadTask") == ()


def test_build_play_backend_adapter_injects_root_dir_and_materializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unilab.base.config_adapter as config_adapter
    from unilab.visualization.interactive_playback import build_play_backend_adapter

    captured: dict[str, Any] = {}
    sentinel_materializer = object()

    class FakeBackendAdapter:
        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            captured["cfg"] = cfg
            captured.update(kwargs)

    monkeypatch.setattr(config_adapter, "BackendAdapter", FakeBackendAdapter)
    monkeypatch.setattr(config_adapter, "materialize_scene_visual_override", sentinel_materializer)

    cfg = SimpleNamespace(training=SimpleNamespace(task_name="Task"))
    adapter = build_play_backend_adapter(cfg, root_dir="/repo", algo_name="appo")

    assert isinstance(adapter, FakeBackendAdapter)
    assert captured == {
        "cfg": cfg,
        "root_dir": "/repo",
        "algo_name": "appo",
        "scene_materializer": sentinel_materializer,
    }


def test_create_rsl_rl_playback_session_uses_runner_loader_and_exposes_runner(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    checkpoint = run_dir / "model_10.pt"
    torch.save({"actor_state_dict": {}}, checkpoint)
    captured: dict[str, Any] = {}

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            pass

        def load(self, checkpoint, load_cfg):
            raise AssertionError("runner.load must be bypassed when runner_loader is injected")

        def get_inference_policy(self, *, device):
            return lambda obs: torch.ones((1, 2))

    def runner_loader(runner, path):
        captured["loader_runner"] = runner
        captured["loader_path"] = path

    kwargs = _rsl_rl_session_kwargs(tmp_path)
    kwargs["checkpoint_resolver"] = lambda *args: str(checkpoint)
    kwargs["runner_cls"] = Runner
    kwargs["runner_loader"] = runner_loader

    session, _mode, resolved = create_rsl_rl_playback_session(**kwargs)

    assert resolved == str(checkpoint)
    assert isinstance(captured["loader_runner"], Runner)
    assert captured["loader_path"] == str(checkpoint)
    assert session.runner is captured["loader_runner"]


def test_create_rsl_rl_playback_session_forwards_guard_algo_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import unilab.visualization.interactive_playback as interactive_playback

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    checkpoint = run_dir / "model_10.pt"
    torch.save({"actor_state_dict": {}}, checkpoint)
    captured: dict[str, Any] = {}

    class Runner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            pass

        def load(self, checkpoint, load_cfg):
            pass

        def get_inference_policy(self, *, device):
            return lambda obs: torch.ones((1, 2))

    import contextlib

    @contextlib.contextmanager
    def fake_dim_guard(**kwargs):
        captured["dim_guard"] = kwargs
        yield

    monkeypatch.setattr(interactive_playback, "policy_load_dim_guard", fake_dim_guard)

    kwargs = _rsl_rl_session_kwargs(tmp_path)
    kwargs["checkpoint_resolver"] = lambda *args: str(checkpoint)
    kwargs["runner_cls"] = Runner
    kwargs["guard_algo_name"] = "him_ppo"

    create_rsl_rl_playback_session(**kwargs)

    assert captured["dim_guard"] == {
        "env_obs_dim": 5,
        "env_action_dim": 2,
        "algo_name": "him_ppo",
    }


def test_create_sac_playback_session_td3_load_filters_noise_scales(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uni_rl.common.actor_factory as actor_factory
    from omegaconf import OmegaConf

    import unilab.utils.checkpoint as checkpoint_utils

    checkpoint = tmp_path / "model_10.pt"
    torch.save(
        {"actor": {"weight": torch.ones(1), "noise_scales": torch.zeros(1)}},
        checkpoint,
    )
    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            captured["actor_load"] = (state_dict, strict)

    class FakeEnv:
        num_envs = 1
        obs_groups_spec = {"obs": 3, "critic": 5}
        action_space = SimpleNamespace(
            shape=(2,),
            low=np.full((2,), -1.0),
            high=np.full((2,), 1.0),
        )
        state = SimpleNamespace(info={})

        def get_physics_state_snapshot(self):
            return np.zeros((1, 4), dtype=np.float32)

    cfg = OmegaConf.create(
        {
            "training": {"task_name": "Task", "device": None},
            "algo": {
                "algo_log_name": "td3",
                "load_run": "run",
                "actor_hidden_dim": 16,
                "use_layer_norm": False,
            },
        }
    )

    def fake_build_actor(algo_type, *args, **kwargs):
        captured["build_actor_algo_type"] = algo_type
        return FakeActor()

    monkeypatch.setattr(actor_factory, "build_actor", fake_build_actor)
    monkeypatch.setattr(
        checkpoint_utils,
        "resolve_offpolicy_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(tmp_path)),
    )

    session, policy_obs_mode, resolved = create_sac_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="Task",
            load_run="run",
            checkpoint=None,
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="td3",
            log_root=None,
        ),
        cfg=cfg,
        env_factory=lambda num_envs: FakeEnv(),
        root_dir=tmp_path,
        device="cpu",
        algo_name="td3",
        log=lambda message: None,
    )

    assert resolved == str(checkpoint)
    assert policy_obs_mode == "actor"
    assert captured["build_actor_algo_type"] == "td3"
    assert isinstance(session.actor, FakeActor)
    assert session.actor_algo_type == "td3"
    loaded_state, strict = captured["actor_load"]
    assert set(loaded_state.keys()) == {"weight"}
    assert strict is False

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
    import uni_rl.algos.common.actor_factory as actor_factory
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


def test_playback_explicitly_skips_training_progress_for_stateful_runner(tmp_path):
    from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner

    captured = {}

    class Runner(TrainingStateOnPolicyRunner):
        def __init__(self, *args, **kwargs):
            pass

        def load(self, path, **kwargs):
            captured.update(kwargs)

        def get_inference_policy(self, **kwargs):
            return lambda obs: torch.ones((1, 2))

    kwargs = _rsl_rl_session_kwargs(tmp_path)
    kwargs["runner_cls"] = Runner
    kwargs["checkpoint_resolver"] = lambda *args: str(tmp_path / "model.pt")
    create_rsl_rl_playback_session(**kwargs)
    assert captured["restore_training_state"] is False
    assert captured["load_cfg"]["actor"] is True
    assert not any(value for key, value in captured["load_cfg"].items() if key != "actor")


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
    kwargs["guard_algo_name"] = "custom_algo"

    create_rsl_rl_playback_session(**kwargs)

    assert captured["dim_guard"] == {
        "env_obs_dim": 5,
        "env_action_dim": 2,
        "algo_name": "custom_algo",
    }


def test_create_sac_playback_session_td3_load_filters_noise_scales(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import uni_rl.algos.common.actor_factory as actor_factory
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

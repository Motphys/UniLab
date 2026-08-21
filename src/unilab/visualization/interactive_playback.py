"""Shared core for interactive policy playback entrypoints."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import torch

from unilab.training.sim2sim import policy_load_dim_guard, resolve_sim2sim_config

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class RslRlPlaybackConfig:
    """Configuration needed to bootstrap an RSL-RL interactive playback session."""

    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    num_envs: int = 1
    speed: float = 1.0
    start_paused: bool = False


@dataclass
class PlayInteractiveArgs:
    """Scalar play arguments shared by interactive playback entrypoints."""

    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    show_target_bodies: bool
    show_reward_debug: bool
    target_show_axes: bool
    target_body_names: str
    target_max_bodies: int
    target_marker_radius: float
    target_axis_length: float
    target_marker_alpha: float
    reward_debug_show_velocity: bool
    reward_debug_lin_vel_scale: float
    reward_debug_ang_vel_scale: float
    reward_debug_show_connectors: bool
    reward_debug_show_global_anchor: bool
    camera_follow_body: bool
    camera_focus_body_name: str
    camera_height_offset: float
    camera_distance: float | None
    camera_elevation: float | None
    camera_azimuth: float | None
    use_env_visual_model: bool
    speed: float
    start_paused: bool
    keyboard: bool = False
    keyboard_step_lin: float = 0.1
    keyboard_step_ang: float = 0.2
    require_keyboard_command_obs: bool = True
    algo: str = "ppo"


def build_playback_config(args: Any, *, num_envs: int = 1) -> RslRlPlaybackConfig:
    """Build an :class:`RslRlPlaybackConfig` from a play-args-like object."""
    return RslRlPlaybackConfig(
        task=str(args.task),
        load_run=str(args.load_run),
        checkpoint=getattr(args, "checkpoint", None),
        action_mode=str(args.action_mode),
        policy_obs_mode=str(args.policy_obs_mode),
        algo_log_name=str(getattr(args, "algo_log_name", "rsl_rl_ppo")),
        log_root=getattr(args, "log_root", None),
        num_envs=num_envs,
        speed=float(getattr(args, "speed", 1.0)),
        start_paused=bool(getattr(args, "start_paused", False)),
    )


@dataclass
class PlaybackControls:
    """Viewer-independent playback control state."""

    paused: bool = False
    speed: float = 1.0
    _single_step_requests: int = field(default=0, init=False, repr=False)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def request_single_step(self, count: int = 1) -> None:
        self._single_step_requests += max(int(count), 0)

    def set_speed(self, value: float) -> None:
        self.speed = max(float(value), 1e-6)

    def consume_step_permission(self) -> bool:
        if self.paused:
            if self._single_step_requests <= 0:
                return False
            self._single_step_requests -= 1
            return True
        if self._single_step_requests > 0:
            self._single_step_requests -= 1
        return True

    def target_dt(self, ctrl_dt: float) -> float:
        return float(ctrl_dt) / max(float(self.speed), 1e-6)


@dataclass
class KeyboardCommander:
    """Mutable ``[vx, vy, vyaw]`` velocity command driven by keyboard nudges.

    Per-axis nudges stack and are clamped to the task's ``commands.vel_limit``.
    """

    low: np.ndarray
    high: np.ndarray
    step_lin: float = 0.1
    step_ang: float = 0.2
    command: np.ndarray = field(init=False)

    AXIS_VX: ClassVar[int] = 0
    AXIS_VY: ClassVar[int] = 1
    AXIS_VYAW: ClassVar[int] = 2

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float64).reshape(3)
        self.high = np.asarray(self.high, dtype=np.float64).reshape(3)
        self.command = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_vel_limit(
        cls, vel_limit: Any, *, step_lin: float = 0.1, step_ang: float = 0.2
    ) -> "KeyboardCommander":
        limit = np.asarray(vel_limit, dtype=np.float64)
        if limit.shape != (2, 3):
            raise ValueError(f"commands.vel_limit must have shape (2, 3), got {limit.shape}")
        return cls(low=limit[0], high=limit[1], step_lin=float(step_lin), step_ang=float(step_ang))

    def nudge(self, axis: int, sign: float) -> None:
        base = self.step_lin if axis in (self.AXIS_VX, self.AXIS_VY) else self.step_ang
        delta = base * (1.0 if sign >= 0 else -1.0)
        self.command[axis] = float(
            np.clip(self.command[axis] + delta, self.low[axis], self.high[axis])
        )

    def zero(self) -> None:
        self.command[:] = 0.0

    def describe(self) -> str:
        return (
            f"cmd vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} vyaw={self.command[2]:+.2f}"
        )


@dataclass(frozen=True)
class MotionOverlaySelection:
    """Cold-path selection of task bodies used by playback overlays."""

    enabled: bool
    selected_indices: np.ndarray


class PlaybackSession(Protocol):
    """Viewer-facing session contract shared by all policy families."""

    env: Any

    def reset(self) -> Any: ...

    def advance(self, controls: PlaybackControls) -> bool: ...

    def physics_state(self) -> np.ndarray: ...

    @property
    def info(self) -> dict[str, Any]: ...


class RslRlPlaybackSession:
    """Policy/action stepping core shared by native and web viewers."""

    def __init__(
        self,
        *,
        env: Any,
        wrapped_env: Any,
        device: str,
        action_mode: str,
        policy: Callable[[Any], Any] | None,
        num_envs: int,
        runner: Any | None = None,
        actor: Any | None = None,
    ) -> None:
        self.env = env
        self.wrapped_env = wrapped_env
        self.device = device
        self.action_mode = action_mode
        self.policy = policy
        self.num_envs = int(num_envs)
        self.runner = runner
        self.actor = actor
        self.obs: Any | None = None
        self.step_count = 0

    def reset(self) -> Any:
        self.obs, _info = self.wrapped_env.reset()
        self.step_count = 0
        return self.obs

    def step_once(self) -> Any:
        actions = self._build_actions()
        self.obs, _reward, _done, _info = self.wrapped_env.step(actions)
        self.step_count += 1
        return self.obs

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _build_actions(self) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.policy is not None:
            return self.policy(self.obs)
        if self.action_mode == "random":
            actions = np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            )
            return torch.from_numpy(actions).to(self.device).float()
        return torch.zeros(self.num_envs, action_dim, device=self.device)


class OffPolicyPlaybackSession:
    """Direct env stepping session for SAC-style off-policy actors."""

    def __init__(
        self,
        *,
        env: Any,
        device: str,
        action_mode: str,
        actor: Any | None,
        actor_algo_type: str,
        normalizer: Any | None,
        num_envs: int,
        obs_extractor: Callable[[dict[str, np.ndarray]], np.ndarray],
        priv_info_resolver: Callable[..., np.ndarray | None],
    ) -> None:
        self.env = env
        self.device = device
        self.action_mode = action_mode
        self.actor = actor
        self.actor_algo_type = str(actor_algo_type)
        self.normalizer = normalizer
        self.num_envs = int(num_envs)
        self.obs_extractor = obs_extractor
        self.priv_info_resolver = priv_info_resolver
        self.obs: np.ndarray | None = None
        self.current_priv_info: np.ndarray | None = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        if self.env.state is None:
            self.env.init_state()
        env_indices = np.arange(self.num_envs, dtype=np.int32)
        reset_result = self.env.reset(env_indices)
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise ValueError(f"Unexpected env.reset return format: {type(reset_result)!r}")
        obs_out, info_out = reset_result
        self.obs = np.asarray(self.obs_extractor(obs_out), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(obs_out, info_out)
        self.step_count = 0
        return self.obs

    def step_once(self) -> np.ndarray:
        actions = self._build_actions()
        state = self.env.step(actions)
        self.obs = np.asarray(self.obs_extractor(state.obs), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(state.obs, state.info)
        self.step_count += 1
        return self.obs

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _resolve_priv_info(
        self,
        obs_dict: dict[str, np.ndarray],
        info: dict[str, Any] | None,
    ) -> np.ndarray | None:
        if self.actor_algo_type != "hora_sac":
            return None
        if self.action_mode != "policy" or self.actor is None:
            return None
        from unilab.base.observations import split_obs_dict

        actor_obs_np, critic_np = split_obs_dict(obs_dict)
        priv_info = self.priv_info_resolver(
            algo_type=self.actor_algo_type,
            obs_np=np.asarray(actor_obs_np, dtype=np.float32),
            critic_np=np.asarray(critic_np, dtype=np.float32),
            info=info,
        )
        if priv_info is None:
            raise ValueError("HORA-SAC interactive play step is missing privileged info.")
        return np.asarray(priv_info, dtype=np.float32)

    def _build_actions(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.actor is not None:
            obs_torch = torch.from_numpy(self.obs).to(self.device)
            if self.normalizer is not None:
                obs_torch = self.normalizer(obs_torch, update=False)
            if self.actor_algo_type == "hora_sac":
                if self.current_priv_info is None:
                    raise ValueError("HORA-SAC interactive play step is missing privileged info.")
                priv_info_torch = torch.from_numpy(self.current_priv_info).to(self.device)
                actions = self.actor.explore(
                    obs_torch,
                    priv_info_torch,
                    deterministic=True,
                )
            else:
                actions = self.actor.explore(obs_torch, deterministic=True)
            return actions.detach().cpu().numpy().astype(np.float32)
        if self.action_mode == "random":
            return np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            ).astype(np.float32)
        return np.zeros((self.num_envs, action_dim), dtype=np.float32)


_HORA_DISTILL_CHECKPOINT_UNAVAILABLE = "hora_distill_checkpoint_unavailable"


def make_sim2sim_preflight(
    cfg: Any,
    *,
    algo_name: str,
) -> Callable[[str | None], Any] | None:
    """Build a sim2sim contract preflight closure for interactive playback entrypoints.

    Returns ``None`` when no Hydra config is available (legacy non-Hydra path).
    Old runs without a contract snapshot keep the fallback + warning semantics of
    :func:`resolve_sim2sim_config`.
    """
    if cfg is None:
        return None

    def _preflight(source_run_dir: str | None) -> Any:
        return resolve_sim2sim_config(
            source_run_dir,
            cfg,
            algo_name=algo_name,
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )

    return _preflight


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_checkpoint_actor_input_dim(ckpt_path: str) -> int | None:
    """Infer the actor MLP input dim from an rsl_rl checkpoint, if detectable."""
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = loaded.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        return None

    # Common rsl-rl naming: "mlp.0.weight" or nested prefixes ending with ".0.weight".
    for key in ("mlp.0.weight", "actor.mlp.0.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])

    for key, w in state_dict.items():
        if key.endswith(".0.weight") and isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])
    return None


def _scene_visual_materializer() -> Any:
    from unilab.base.backend import materialize_scene_visual_override

    return materialize_scene_visual_override


def build_play_backend_adapter(cfg: Any, *, root_dir: str | Path, algo_name: str = "ppo") -> Any:
    """Build the BackendAdapter used by play entrypoints to derive env cfg overrides."""
    from unilab.training import BackendAdapter

    return BackendAdapter(
        cfg,
        root_dir=root_dir,
        algo_name=algo_name,
        scene_materializer=_scene_visual_materializer(),
    )


def available_backends_for_task(task_name: str) -> tuple[str, ...]:
    """Return the registered backends for a task, or ``()`` when unknown."""
    from unilab.base import registry

    envs = registry.list_registered_envs()
    task_meta = envs.get(task_name, {})
    backends = task_meta.get("available_backends", ())
    if not isinstance(backends, list):
        return ()
    return tuple(str(backend) for backend in backends)


def create_rsl_rl_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    env_factory: Callable[[int], Any],
    algo_config: dict[str, Any],
    root_dir: str | Path,
    device: str | None,
    checkpoint_resolver: Callable[[str, str, str | None, str, str | None], str | None],
    checkpoint_input_dim_reader: Callable[[str], int | None],
    entrypoint_log_root: Callable[..., Path],
    wrapper_cls: Any,
    runner_cls: Any,
    policy_obs_dims_getter: Callable[[Any], tuple[int, int]],
    train_cfg_normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    sim2sim_preflight: Callable[[str | None], Any] | None = None,
    runner_loader: Callable[[Any, str], None] | None = None,
    guard_algo_name: str | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session and load the selected policy checkpoint."""

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")
    actor_obs_dim, flat_obs_dim = policy_obs_dims_getter(env.obs_groups_spec)

    policy_obs_mode = playback_cfg.policy_obs_mode
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path = checkpoint_resolver(
            playback_cfg.task,
            playback_cfg.load_run,
            playback_cfg.checkpoint,
            playback_cfg.algo_log_name,
            playback_cfg.log_root,
        )
        if policy_obs_mode == "auto" and checkpoint_path is not None:
            ckpt_dim = checkpoint_input_dim_reader(checkpoint_path)
            if ckpt_dim == actor_obs_dim:
                policy_obs_mode = "actor"
            elif ckpt_dim == flat_obs_dim:
                policy_obs_mode = "flat"
            elif ckpt_dim is not None:
                raise RuntimeError(
                    "Checkpoint actor input dim mismatch: "
                    f"ckpt={ckpt_dim}, actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim}. "
                    "Please pass --policy_obs_mode actor|flat explicitly if needed."
                )
            else:
                policy_obs_mode = "flat"

    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    log(f"Policy obs mode: {policy_obs_mode} (actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim})")

    train_cfg = train_cfg_normalizer(copy.deepcopy(algo_config))
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    policy = None
    runner = None
    if playback_cfg.action_mode == "policy":
        if checkpoint_path is None:
            log("WARNING: no checkpoint found - falling back to zero actions.")
        else:
            if sim2sim_preflight is not None:
                sim2sim_preflight(str(Path(checkpoint_path).parent))
            log_dir = str(
                entrypoint_log_root(
                    Path(root_dir),
                    algo_log_name=playback_cfg.algo_log_name,
                    log_root=playback_cfg.log_root,
                )
                / playback_cfg.task
                / "play_temp"
            )
            runner = runner_cls(wrapped_env, train_cfg, log_dir=log_dir, device=device_name)
            policy_obs_dim = actor_obs_dim if policy_obs_mode == "actor" else flat_obs_dim
            action_shape = env.action_space.shape
            policy_action_dim = int(action_shape[0]) if action_shape is not None else None
            with policy_load_dim_guard(
                env_obs_dim=policy_obs_dim,
                env_action_dim=policy_action_dim,
                algo_name=guard_algo_name or playback_cfg.algo_log_name,
            ):
                if runner_loader is not None:
                    runner_loader(runner, checkpoint_path)
                else:
                    runner.load(
                        checkpoint_path,
                        load_cfg={
                            "actor": True,
                            "critic": False,
                            "optimizer": False,
                            "iteration": False,
                            "rnd": False,
                        },
                    )
            policy = runner.get_inference_policy(device=device_name)

    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
        runner=runner,
    )
    return session, policy_obs_mode, checkpoint_path


def normalize_checkpoint_value(value: object) -> str | None:
    """Normalize a raw checkpoint selector value; sentinel values map to ``None``."""
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "-1", "None", "null"} else text


_normalize_checkpoint_value = normalize_checkpoint_value


def _cfg_checkpoint_value(cfg: Any) -> str | None:
    from omegaconf import OmegaConf

    return _normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint", default=None))


def _resolve_appo_checkpoint_from_cfg(
    cfg: Any,
    *,
    root_dir: str | Path,
) -> tuple[str | None, str | None]:
    from unilab.training import (
        get_log_root,
        resolve_appo_checkpoint_path,
        resolve_task_checkpoint_path,
    )

    selected_checkpoint = _cfg_checkpoint_value(cfg)
    if selected_checkpoint is not None:
        selected_path, selected_dir = resolve_task_checkpoint_path(
            root_dir,
            task_name=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            algo_log_name=str(cfg.algo.algo_log_name),
            checkpoint=selected_checkpoint,
            log_root=getattr(cfg.training, "log_root", None),
        )
        return (
            str(selected_path) if selected_path is not None else None,
            str(selected_dir) if selected_dir is not None else None,
        )

    base_log_dir = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    checkpoint_path, checkpoint_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _build_appo_actor(
    *,
    env: Any,
    wrapped_env: Any,
    cfg: Any,
    rl_cfg: dict[str, Any],
    device: str,
    is_hora: bool,
) -> tuple[Any, int, int]:
    """Build the APPO actor and return it with ``(obs_dim, action_dim)``."""
    from copy import deepcopy

    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.base.observations import get_obs_dims

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    rl_cfg_dict = deepcopy(rl_cfg)

    if is_hora:
        from unilab.algos.hora.appo import _update_hora_obs_groups
        from unilab.algos.hora.models import build_hora_shared_actor_critic
        from unilab.algos.hora.rsl_rl_compat import (
            convert_config_v3_to_v4,
            is_rsl_rl_v4,
            is_rsl_rl_v5,
        )

        obs_td = wrapped_env.get_observations()
        num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
        obs_dim = int(obs_td["actor"].shape[-1])
        priv_info_dim = int(obs_td["priv_info"].shape[-1])
        if priv_info_dim <= 0:
            raise ValueError("HORA APPO interactive play requires privileged info.")
        _update_hora_obs_groups(rl_cfg_dict, obs_dim=obs_dim, priv_info_dim=priv_info_dim)
        if is_rsl_rl_v5():
            pass
        elif is_rsl_rl_v4():
            rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

        actor_cfg = deepcopy(rl_cfg_dict["actor"])
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        critic_cfg = deepcopy(rl_cfg_dict.get("critic") or rl_cfg_dict.get("actor") or {})
        critic_cfg.pop("class_name", None)
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)
        shared_model = build_hora_shared_actor_critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            priv_info_dim=priv_info_dim,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
        ).to(device)
        td_example = TensorDict(
            {
                "actor": torch.zeros((num_envs, obs_dim), device=device),
                "priv_info": torch.zeros(
                    (num_envs, priv_info_dim),
                    device=device,
                ),
            },
            batch_size=num_envs,
        )
        actor = actor_cls(
            td_example,
            rl_cfg_dict["obs_groups"],
            "actor",
            action_dim,
            shared_model=shared_model,
            **actor_cfg,
        )
        return actor.to(device).eval(), obs_dim, action_dim

    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
    obs_groups = rl_cfg_dict.setdefault("obs_groups", {})
    if "obs_groups" not in rl_cfg_dict or not isinstance(obs_groups, dict):
        obs_groups = {}
        rl_cfg_dict["obs_groups"] = obs_groups
    actor_group = obs_groups.get("actor", obs_groups.get("policy", {}))
    if isinstance(actor_group, dict) and "policy" in actor_group:
        actor_group["policy"] = obs_dim
        obs_groups["actor"] = actor_group
    else:
        obs_groups["actor"] = {"policy": obs_dim}
    critic_group = obs_groups.get("critic")
    if critic_group is None:
        obs_groups["critic"] = {"policy": critic_dim if critic_dim > 0 else obs_dim}
    elif isinstance(critic_group, dict) and "policy" in critic_group:
        critic_group["policy"] = critic_dim if critic_dim > 0 else obs_dim

    obs_example = torch.zeros((num_envs, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)
    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    return actor.to(device).eval(), obs_dim, action_dim


def create_appo_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    rl_cfg: dict[str, Any],
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    wrapper_cls: Any,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an APPO interactive playback session."""

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    from unilab.algos.hora.runtime import is_hora_appo_runtime

    is_hora = is_hora_appo_runtime(rl_cfg)
    selected_wrapper_cls = wrapper_cls
    policy_obs_mode = playback_cfg.policy_obs_mode
    if is_hora:
        from unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper

        selected_wrapper_cls = HoraRslRlVecEnvWrapper
        policy_obs_mode = "actor"

    wrapped_env = selected_wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy = None
    actor = None
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path, checkpoint_dir = _resolve_appo_checkpoint_from_cfg(cfg, root_dir=root_dir)
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            log(
                "WARNING: no APPO checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
        else:
            resolve_sim2sim_config(
                checkpoint_dir,
                cfg,
                algo_name="appo",
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            actor, actor_obs_dim, action_dim = _build_appo_actor(
                env=env,
                wrapped_env=wrapped_env,
                cfg=cfg,
                rl_cfg=rl_cfg,
                device=device_name,
                is_hora=is_hora,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=actor_obs_dim,
                env_action_dim=action_dim,
                algo_name="appo",
            ):
                actor.load_state_dict(checkpoint["actor"])
            policy = actor
            log(f"Loading APPO checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        RslRlPlaybackSession(
            env=env,
            wrapped_env=wrapped_env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            policy=policy,
            num_envs=playback_cfg.num_envs,
            actor=actor,
        ),
        policy_obs_mode,
        checkpoint_path,
    )


def create_sac_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    algo_name: str = "sac",
    log: LogFn = print,
) -> tuple[OffPolicyPlaybackSession, str, str | None]:
    """Create an interactive playback session for off-policy actors."""

    import os

    from unilab.algos.common.actor_factory import build_actor
    from unilab.algos.offpolicy.worker import resolve_offpolicy_actor_priv_info
    from unilab.training.offpolicy import (
        default_device,
        extract_play_obs,
        load_play_actor,
        resolve_play_actor_spec,
        resolve_play_obs_dims,
    )
    from unilab.training.run import resolve_offpolicy_checkpoint_path

    device_name = default_device(torch, str(device) if device is not None else None)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        algo_name,
        cfg,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    if algo_name == "flashsac":
        actor_kwargs.update(
            {
                "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
                "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
                "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
            }
        )

    actor = None
    checkpoint_path: str | None = None
    normalizer = None
    if bool(getattr(cfg.algo, "obs_normalization", False)):
        from unilab.algos.common.normalization import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=obs_dim, device=device_name)
    if playback_cfg.action_mode == "policy":
        actor = build_actor(
            actor_algo_type,
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device_name,
            **actor_kwargs,
        )
        actor.eval()
        checkpoint_path, checkpoint_dir = resolve_offpolicy_checkpoint_path(
            Path(root_dir),
            cfg.algo.algo_log_name,
            cfg.training.task_name,
            cfg.algo.load_run,
        )
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log(
                f"WARNING: no {algo_name} checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
            actor = None
        else:
            resolve_sim2sim_config(
                checkpoint_dir,
                cfg,
                algo_name=algo_name,
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=True)
            with policy_load_dim_guard(
                env_obs_dim=obs_dim,
                env_action_dim=action_dim,
                algo_name=algo_name,
            ):
                load_play_actor(algo_name, actor, normalizer, checkpoint)
            log(f"Loading {algo_name} checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        OffPolicyPlaybackSession(
            env=env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            actor=actor,
            actor_algo_type=actor_algo_type,
            normalizer=normalizer,
            num_envs=playback_cfg.num_envs,
            obs_extractor=extract_play_obs,
            priv_info_resolver=resolve_offpolicy_actor_priv_info,
        ),
        "actor",
        checkpoint_path,
    )


def _default_hora_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    from unilab.algos.hora.distill import (
        build_student_actor_and_normalizer,
        cfg_with_checkpoint_runtime,
        load_distilled_checkpoint,
        student_policy,
    )
    from unilab.algos.hora.distill_config import apply_teacher_defaults
    from unilab.algos.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import (
        BackendAdapter,
        create_env,
        format_hora_stage2_checkpoint_error,
        get_log_root,
        resolve_hora_stage2_checkpoint_path,
    )

    return {
        "apply_teacher_defaults": apply_teacher_defaults,
        "build_play_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="hora_distill",
            scene_materializer=_scene_visual_materializer(),
        ).build_play_env_cfg_override(),
        "build_student_actor_and_normalizer": build_student_actor_and_normalizer,
        "cfg_with_checkpoint_runtime": cfg_with_checkpoint_runtime,
        "create_env": create_env,
        "format_stage2_play_checkpoint_error": format_hora_stage2_checkpoint_error,
        "get_log_root": get_log_root,
        "load_distilled_checkpoint": load_distilled_checkpoint,
        "resolve_stage2_checkpoint_path": lambda cfg: resolve_hora_stage2_checkpoint_path(
            cfg,
            root_dir=root_dir,
        ),
        "student_policy": student_policy,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
        "checkpoint_reader": torch.load,
    }


def create_hora_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an interactive playback session for HORA stage-2 student checkpoints."""

    resolved_deps = dict(_default_hora_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    load_path, load_path_dir = resolved_deps["resolve_stage2_checkpoint_path"](cfg)
    checkpoint_path = str(load_path) if load_path is not None else None
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if load_path is None or load_path_dir is None or not Path(load_path).exists():
            task_log_root = resolved_deps["get_log_root"](Path(root_dir), cfg) / str(
                cfg.training.task_name
            )
            log(
                resolved_deps["format_stage2_play_checkpoint_error"](
                    cfg,
                    task_log_root=task_log_root,
                    load_path=load_path,
                    load_path_dir=load_path_dir,
                )
            )
            log("WARNING: falling back to zero actions.")
            runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)
        else:
            resolve_sim2sim_config(
                load_path_dir,
                cfg,
                algo_name="hora_distill",
                strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
            )
            log(f"Loading distilled checkpoint: {load_path}")
            checkpoint = resolved_deps["checkpoint_reader"](
                load_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
                    f"(found keys: {set(checkpoint.keys())})."
                )
            runtime_cfg = resolved_deps["cfg_with_checkpoint_runtime"](cfg, checkpoint)
    else:
        runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)

    env_cfg_override = resolved_deps["build_play_env_cfg_override"](runtime_cfg)
    create_env = resolved_deps["create_env"]
    try:
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=str(runtime_cfg.training.task_name),
        )
    except TypeError:
        if deps is None:
            raise
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
        )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    torch_device = torch.device(device_name)

    if playback_cfg.action_mode == "policy" and load_path is not None and Path(load_path).exists():
        actor, hist_normalizer = resolved_deps["build_student_actor_and_normalizer"](
            wrapped_env,
            runtime_cfg,
            device=torch_device,
        )
        with policy_load_dim_guard(algo_name="hora_distill"):
            resolved_deps["load_distilled_checkpoint"](
                actor,
                hist_normalizer,
                load_path,
                device=torch_device,
            )
        actor.eval()
        hist_normalizer.eval()
        student_policy = resolved_deps["student_policy"]

        def policy(obs: Any) -> Any:
            return student_policy(actor, hist_normalizer, obs, device=torch_device)

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def prepare_motion_overlay_selection(
    env: Any,
    *,
    show_target_bodies: bool,
    show_reward_debug: bool,
    target_body_names: str,
    target_max_bodies: int,
    log: LogFn = print,
) -> MotionOverlaySelection:
    """Resolve body indices used by motion-target and reward-debug overlays."""

    if not (show_target_bodies or show_reward_debug):
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    if not (hasattr(env, "motion_loader") and hasattr(env, "motion_sampler")):
        log("WARNING: target/reward visualization only works for motion-tracking tasks.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    names = tuple(getattr(env.cfg, "body_names", ()))
    if len(names) == 0:
        log("WARNING: task has no body_names; cannot visualize targets.")
        return MotionOverlaySelection(
            enabled=False,
            selected_indices=np.zeros((0,), dtype=np.int32),
        )

    name_to_idx = {name: i for i, name in enumerate(names)}
    if target_body_names.strip():
        chosen = []
        for name in [n.strip() for n in target_body_names.split(",") if n.strip()]:
            if name in name_to_idx:
                chosen.append(name_to_idx[name])
            else:
                log(f"WARNING: body name not found in task body list: {name}")
        selected_indices = np.array(chosen, dtype=np.int32)
    else:
        selected_indices = np.arange(len(names), dtype=np.int32)

    if target_max_bodies > 0:
        selected_indices = selected_indices[:target_max_bodies]

    return MotionOverlaySelection(
        enabled=selected_indices.size > 0,
        selected_indices=selected_indices,
    )


__all__ = [
    "KeyboardCommander",
    "MotionOverlaySelection",
    "OffPolicyPlaybackSession",
    "PlayInteractiveArgs",
    "PlaybackControls",
    "PlaybackSession",
    "RslRlPlaybackConfig",
    "RslRlPlaybackSession",
    "available_backends_for_task",
    "build_play_backend_adapter",
    "build_playback_config",
    "create_appo_playback_session",
    "create_hora_distill_playback_session",
    "create_rsl_rl_playback_session",
    "create_sac_playback_session",
    "infer_checkpoint_actor_input_dim",
    "make_sim2sim_preflight",
    "normalize_checkpoint_value",
    "prepare_motion_overlay_selection",
    "select_torch_device",
]

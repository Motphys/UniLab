"""Shared core for interactive policy playback entrypoints."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

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
    ) -> None:
        self.env = env
        self.wrapped_env = wrapped_env
        self.device = device
        self.action_mode = action_mode
        self.policy = policy
        self.num_envs = int(num_envs)
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


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    if playback_cfg.action_mode == "policy":
        if checkpoint_path is None:
            log("WARNING: no checkpoint found - falling back to zero actions.")
        else:
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
    "PlaybackControls",
    "RslRlPlaybackConfig",
    "RslRlPlaybackSession",
    "create_rsl_rl_playback_session",
    "prepare_motion_overlay_selection",
    "select_torch_device",
]

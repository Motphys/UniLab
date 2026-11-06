"""Shared experiment tracking utilities for local files and W&B."""

from __future__ import annotations

import dataclasses
import getpass
import importlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from unilab.utils.device import get_device_info_dict
from unilab.utils.sim2sim import extract_contract_snapshot


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _plain_dict(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _load_wandb() -> Any | None:
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def _tb_proto_modules() -> tuple[Any, Any]:
    """Import tensorboard's protobuf modules as ``Any``.

    The generated modules define their messages through reflection, which
    static checkers cannot see; returning them as ``Any`` keeps attribute
    access clean without per-line suppressions.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2

    return event_pb2, summary_pb2


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _copy_plain_training_state(state: Any) -> dict[str, Any]:
    """Validate and detach versioned training progress at a checkpoint boundary.

    This intentionally accepts only plain JSON-shaped values. It rejects tensors
    and other owner objects before they can enter a checkpoint payload.
    """

    def copy_value(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Training state numbers must be finite")
            return value
        if type(value) is not dict and type(value) is not list:
            raise TypeError(
                "Training state must contain only plain JSON dictionaries/lists/scalars"
            )
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("Training state dictionary keys must be strings")
            return {key: copy_value(item) for key, item in value.items()}
        return [copy_value(item) for item in value]

    if not isinstance(state, dict):
        raise TypeError("Training state export must return a dictionary")
    result = copy_value(state)
    if not isinstance(result, dict):
        raise TypeError("Training state export must return a dictionary")
    return result


def get_git_info(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir)

    def _run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--short")

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def _write_json(path: Path, payload: Any, *, trailing_newline: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def build_run_config_payload(
    *,
    run_metadata: dict[str, Any],
    full_cfg: Any,
    contract_snapshot: Any | None = None,
) -> dict[str, Any]:
    """Assemble the canonical ``{"run", "config", ...}`` run-config payload."""
    payload: dict[str, Any] = {
        "run": _json_safe(run_metadata),
        "config": _json_safe(_plain_dict(full_cfg)),
    }
    if contract_snapshot is not None:
        payload["contract_snapshot"] = _json_safe(contract_snapshot)
    return payload


def write_run_config_snapshot(
    log_dir: str | Path,
    *,
    run_metadata: dict[str, Any],
    full_cfg: Any,
    filename: str = "run_config.json",
    contract_snapshot: Any | None = None,
    trailing_newline: bool = False,
) -> dict[str, Any]:
    """Write the run-config snapshot into ``log_dir`` and return the payload."""
    payload = build_run_config_payload(
        run_metadata=run_metadata,
        full_cfg=full_cfg,
        contract_snapshot=contract_snapshot,
    )
    _write_json(Path(log_dir) / filename, payload, trailing_newline=trailing_newline)
    return payload


def build_wandb_run_name(algo_name: str, task_name: str, log_dir: str | Path | None) -> str:
    if log_dir is None:
        return f"{algo_name}__{task_name}"
    run_dir = Path(log_dir)
    return f"{algo_name}__{task_name}__{run_dir.name}"


def build_wandb_settings(
    training_cfg: Any,
    *,
    algo_name: str,
    task_name: str,
    sim_backend: str,
    log_dir: str | Path | None,
) -> dict[str, Any]:
    name = _cfg_get(training_cfg, "wandb_name")
    if not name:
        name = build_wandb_run_name(algo_name, task_name, log_dir)

    group = _cfg_get(training_cfg, "wandb_group")
    if not group:
        group = task_name

    job_type = _cfg_get(training_cfg, "wandb_job_type")
    if not job_type:
        job_type = algo_name

    tags = [str(tag) for tag in (_cfg_get(training_cfg, "wandb_tags", []) or [])]
    auto_tags = [algo_name, task_name, sim_backend, f"user-{getpass.getuser()}"]
    for tag in auto_tags:
        if tag not in tags:
            tags.append(tag)

    return {
        "project": _cfg_get(training_cfg, "wandb_project", "unilab"),
        "entity": _cfg_get(training_cfg, "wandb_entity"),
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": tags,
        "notes": _cfg_get(training_cfg, "wandb_notes"),
        "mode": _cfg_get(training_cfg, "wandb_mode"),
    }


class ExperimentTracker:
    """Tracks experiment metadata locally and optionally in Weights & Biases."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        log_dir: str | Path,
        algo_name: str,
        task_name: str,
        sim_backend: str,
        training_cfg: Any,
        full_cfg: Any,
        device: str | None = None,
        configured_device: str | None = None,
        collector_device: str | None = None,
        device_resolution: Mapping[str, Mapping[str, str]] | None = None,
        seed_info: Any | None = None,
        tf32_matmul_policy: Any | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.log_dir = Path(log_dir)
        self.algo_name = algo_name
        self.task_name = task_name
        self.sim_backend = sim_backend
        self.training_cfg = training_cfg
        self.full_cfg = full_cfg
        self.device = device
        self.configured_device = configured_device
        self.collector_device = collector_device
        self.device_resolution = dict(device_resolution) if device_resolution is not None else None
        self.seed_info = seed_info
        self.tf32_matmul_policy = tf32_matmul_policy
        self.enabled = str(_cfg_get(training_cfg, "logger", "tensorboard")).lower() == "wandb"

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        self._run = None
        self._owns_run = False
        self._started = False
        self._start_monotonic = 0.0
        self._start_utc = ""
        self._summary: dict[str, Any] = {}

    @property
    def run(self) -> Any | None:
        return self._run

    @property
    def run_url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    @property
    def wandb_settings(self) -> dict[str, Any]:
        return build_wandb_settings(
            self.training_cfg,
            algo_name=self.algo_name,
            task_name=self.task_name,
            sim_backend=self.sim_backend,
            log_dir=self.log_dir,
        )

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._start_monotonic = time.perf_counter()
        self._start_utc = datetime.now(timezone.utc).isoformat()

        metadata = {
            "algo": self.algo_name,
            "task": self.task_name,
            "sim_backend": self.sim_backend,
            "device": self.device,
            "configured_device": self.configured_device,
            "collector_device": self.collector_device,
            "resolved_devices": self.device_resolution,
            "log_dir": str(self.log_dir),
            "start_time_utc": self._start_utc,
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "git": get_git_info(self.root_dir),
            "hardware": get_device_info_dict(),
            "wandb": self.wandb_settings,
        }
        if self.seed_info is not None:
            if hasattr(self.seed_info, "to_dict"):
                seed_payload = self.seed_info.to_dict()
            elif isinstance(self.seed_info, dict):
                seed_payload = dict(self.seed_info)
            else:
                seed_payload = {"effective_seed": self.seed_info}
            metadata.update(seed_payload)
        if self.tf32_matmul_policy is not None and hasattr(self.tf32_matmul_policy, "manifest"):
            metadata["tf32_matmul"] = self.tf32_matmul_policy.manifest()

        payload = write_run_config_snapshot(
            self.log_dir,
            run_metadata=metadata,
            full_cfg=self.full_cfg,
            contract_snapshot=extract_contract_snapshot(self.full_cfg),
        )
        if self.device_resolution:
            mapping = ", ".join(
                f"{name} {entry['configured']} -> {entry['resolved']}"
                for name, entry in self.device_resolution.items()
            )
            print(f"[experiment_tracking] Device resolution: {mapping}")

        if not self.enabled:
            return

        self._wandb = _load_wandb()
        if self._wandb is None:
            print("[experiment_tracking] wandb not installed, skipping W&B experiment tracking.")
            return

        self._run = self._wandb.run
        if self._run is None:
            kwargs = {
                "project": self.wandb_settings["project"],
                "name": self.wandb_settings["name"],
                "config": payload,
                "dir": str(self.log_dir),
                "reinit": True,
            }
            for key in ("entity", "group", "job_type", "tags", "notes", "mode"):
                value = self.wandb_settings.get(key)
                if value not in (None, "", []):
                    kwargs[key] = value
            self._run = self._wandb.init(**kwargs)
            self._owns_run = True
        else:
            self._run.config.update(payload, allow_val_change=True)

        if self._run is not None:
            self._run.summary["algo"] = self.algo_name
            self._run.summary["task"] = self.task_name
            self._run.summary["sim_backend"] = self.sim_backend
            if self.device:
                self._run.summary["device"] = self.device
            if self.collector_device:
                self._run.summary["collector_device"] = self.collector_device
            self._run.summary["log_dir"] = str(self.log_dir)

    def update_summary(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            self._summary.update(summary)

        if not self._started:
            return

        wall_time_sec = time.perf_counter() - self._start_monotonic
        payload = {
            **self._summary,
            "algo": self.algo_name,
            "task": self.task_name,
            "sim_backend": self.sim_backend,
            "log_dir": str(self.log_dir),
            "start_time_utc": self._start_utc,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_sec": wall_time_sec,
            "wandb_run_url": self.run_url,
        }
        if self.seed_info is not None:
            if hasattr(self.seed_info, "to_dict"):
                payload.update(self.seed_info.to_dict())
            elif isinstance(self.seed_info, dict):
                payload.update(self.seed_info)
            else:
                payload["effective_seed"] = self.seed_info
        if self.tf32_matmul_policy is not None and hasattr(self.tf32_matmul_policy, "manifest"):
            payload["tf32_matmul"] = self.tf32_matmul_policy.manifest()
        _write_json(self.log_dir / "run_summary.json", _json_safe(payload))

        if self._run is not None:
            for key, value in payload.items():
                self._run.summary[key] = _json_safe(value)

    def log_video(self, video_path: str | Path | None, key: str = "media/play_video") -> None:
        if video_path is None:
            return

        video = Path(video_path)
        if not video.exists():
            return

        self._summary["play_video_path"] = str(video)
        if self._run is not None and self._wandb is not None:
            self._wandb.log({key: self._wandb.Video(str(video), format="mp4")})

    def finish(self) -> None:
        if not self._started:
            return

        self.update_summary()
        if self._run is not None and self._wandb is not None and self._owns_run:
            self._wandb.finish()
        self._run = None
        self._wandb = None


def patch_rsl_rl_action_std_logging(runner: Any) -> None:
    """Log the current distribution's public standard deviation, including state dependence."""
    original_log = runner.logger.log

    def _safe_log(self: Any, *args: Any, **kwargs: Any) -> Any:
        distribution = runner.alg.get_policy().distribution
        kwargs["action_std"] = distribution.std.detach().clone()
        return original_log(*args, **kwargs)

    runner.logger.log = _safe_log.__get__(runner.logger, type(runner.logger))


class _BatchedScalarWriter:
    """Buffer rsl-rl's per-tag ``add_scalar`` calls and emit one event per step.

    TensorBoard's writer thread performs one open/write/close per record, which
    saturates its async queue (depth 10) and blocks the training loop when the
    log directory lives on a network filesystem. Batching one iteration's
    scalars into a single ``Event`` keeps it to one record per step group.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._pending: list[tuple[str, float]] = []
        self._pending_step: int | None = None

    def add_scalar(self, tag: str, value: Any, step: Any = None, *args: Any, **kwargs: Any) -> None:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            self._flush()
            self._writer.add_scalar(tag, value, step, *args, **kwargs)
            return
        step_key = int(step) if step is not None else 0
        if self._pending and step_key != self._pending_step:
            self._flush()
        self._pending_step = step_key
        self._pending.append((tag, scalar))

    def _flush(self) -> None:
        if not self._pending:
            return
        event_pb2, summary_pb2 = _tb_proto_modules()
        event = event_pb2.Event(
            wall_time=time.time(),
            step=self._pending_step or 0,
            summary=summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(tag=tag, simple_value=value)
                    for tag, value in self._pending
                ]
            ),
        )
        self._writer.file_writer.add_event(event)
        self._pending = []
        self._pending_step = None

    def flush(self) -> None:
        self._flush()
        self._writer.flush()

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


class _SwallowScalarWriter:
    """Drop ``add_scalar`` writes while staying truthy for rsl-rl's console log."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


def patch_rsl_rl_tensorboard_logging(runner: Any, log_interval: int = 1) -> None:
    """Batch and optionally throttle rsl-rl's TensorBoard scalar writes.

    Wraps ``runner.logger.writer`` so each step's ~15-30 ``add_scalar`` calls
    become a single event record, and gates backend writes to every
    ``log_interval`` iterations (console output and episode bookkeeping are
    unaffected; the final iteration is always logged). No-op for non-TB
    backends and when no writer exists.
    """
    logger = getattr(runner, "logger", None)
    writer = getattr(logger, "writer", None)
    if logger is None or writer is None:
        return
    if getattr(logger, "logger_type", "tensorboard") != "tensorboard":
        return
    if isinstance(writer, _BatchedScalarWriter):
        return

    batched = _BatchedScalarWriter(writer)
    logger.writer = batched

    interval = max(1, int(log_interval))
    if interval <= 1:
        return

    swallow = _SwallowScalarWriter(batched)
    original_log = logger.log

    def _gated_log(self: Any, it: int, *args: Any, **kwargs: Any) -> Any:
        total_it = args[1] if len(args) > 1 else int(kwargs.get("total_it", 0))
        due = it % interval == 0 or it >= total_it
        logger.writer = batched if due else swallow
        try:
            return original_log(it, *args, **kwargs)
        finally:
            logger.writer = batched

    logger.log = _gated_log.__get__(logger, type(logger))


def patch_rsl_rl_wandb_writer() -> None:
    """Patch rsl-rl W&B writer so it can reuse an already-open run."""
    try:
        import rsl_rl.utils.wandb_utils as wandb_utils
    except ImportError:
        return

    if getattr(wandb_utils, "_UNILAB_PATCHED", False):
        return

    wandb = _load_wandb()
    if wandb is None:
        return
    wandb_mod = wandb

    from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

    class PatchedWandbSummaryWriter(TensorboardSummaryWriter):
        def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
            super().__init__(log_dir, flush_secs=flush_secs)

            run_name = os.path.split(log_dir)[-1]
            project = cfg.get("wandb_project", "unilab")
            entity = cfg.get("wandb_entity") or os.environ.get("WANDB_USERNAME")
            group = cfg.get("wandb_group")
            job_type = cfg.get("wandb_job_type")
            tags = cfg.get("wandb_tags")
            notes = cfg.get("wandb_notes")
            mode = cfg.get("wandb_mode")

            self.logged_videos: set[str] = set()
            self._owns_run = wandb_mod.run is None
            if self._owns_run:
                kwargs = {
                    "project": project,
                    "name": run_name,
                    "config": {"log_dir": log_dir},
                    "settings": wandb_mod.Settings(start_method="thread"),
                }
                if entity:
                    kwargs["entity"] = entity
                if group:
                    kwargs["group"] = group
                if job_type:
                    kwargs["job_type"] = job_type
                if tags:
                    kwargs["tags"] = tags
                if notes:
                    kwargs["notes"] = notes
                if mode:
                    kwargs["mode"] = mode
                wandb_mod.init(**kwargs)
            else:
                wandb_mod.config.update({"log_dir": log_dir}, allow_val_change=True)

        def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
            wandb_mod.config.update({"train_cfg": train_cfg}, allow_val_change=True)
            env_payload: Any
            if isinstance(env_cfg, dict):
                env_payload = env_cfg
            elif dataclasses.is_dataclass(env_cfg) and not isinstance(env_cfg, type):
                env_payload = dataclasses.asdict(env_cfg)
            elif hasattr(env_cfg, "to_dict"):
                env_payload = env_cfg.to_dict()  # type: ignore[union-attr]
            else:
                env_payload = str(env_cfg)
            wandb_mod.config.update({"env_cfg": env_payload}, allow_val_change=True)

        def add_scalar(
            self,
            tag: Any,
            scalar_value: Any,
            global_step: Any = None,
            walltime: Any = None,
            new_style: Any = False,
            double_precision: Any = False,
        ) -> None:
            super().add_scalar(
                tag,
                scalar_value,
                global_step=global_step,
                walltime=walltime,
                new_style=new_style,
                double_precision=double_precision,
            )
            wandb_mod.log({tag: scalar_value}, step=global_step)

        def stop(self) -> None:
            if self._owns_run:
                wandb_mod.finish()

        def save_model(self, model_path: str, it: int) -> None:
            wandb_mod.save(model_path, base_path=os.path.dirname(model_path))

        def save_file(self, path: str) -> None:
            wandb_mod.save(path, base_path=os.path.dirname(path))

        def save_video(self, video: Path, it: int) -> None:
            if video.name not in self.logged_videos:
                wandb_mod.log({"video": wandb_mod.Video(str(video), format="mp4")}, step=it)
                self.logged_videos.add(video.name)

    wandb_utils.WandbSummaryWriter = PatchedWandbSummaryWriter
    setattr(wandb_utils, "_UNILAB_PATCHED", True)


def _require_unilab_logger_state(loaded_dict: Any, path: str) -> tuple[float, int]:
    """Validate the required ``unilab_logger_state`` checkpoint entry.

    Pre-patch rsl-rl checkpoints are unsupported: there is no compatibility
    window, offline migration, or silent restart from zero progress (issue
    #2416). The returned ``(tot_time, tot_timesteps)`` pair is only produced
    for a payload in the minimum supported format.
    """
    minimum_format_hint = (
        "The minimum supported checkpoint format is written by UniLab's patched "
        "OnPolicyRunner.save (resume-state patch, issue #441, shipped since v1.0.0) "
        "and carries 'unilab_logger_state' = {'tot_time': float, 'tot_timesteps': int} "
        "plus, when the environment publishes training state, a versioned "
        "'unilab_env_training_state'. Resume from a checkpoint saved by a supported "
        "UniLab version or retrain with one; pre-patch checkpoints cannot be migrated."
    )
    state = loaded_dict.get("unilab_logger_state") if isinstance(loaded_dict, dict) else None
    if state is None:
        raise ValueError(
            f"Checkpoint {path} does not contain 'unilab_logger_state'; pre-patch "
            f"rsl-rl checkpoints are not supported. {minimum_format_hint}"
        )
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint {path} has an invalid 'unilab_logger_state' "
            f"(expected a dict, got {type(state).__name__}). {minimum_format_hint}"
        )
    tot_time_raw: Any = state.get("tot_time")
    if type(tot_time_raw) not in (int, float):
        raise ValueError(
            f"Checkpoint {path} has an invalid 'unilab_logger_state.tot_time' "
            f"(expected a finite non-negative number, got {tot_time_raw!r}). "
            f"{minimum_format_hint}"
        )
    tot_time = float(tot_time_raw)
    if tot_time != tot_time or tot_time in (float("inf"), float("-inf")) or tot_time < 0.0:
        raise ValueError(
            f"Checkpoint {path} has an invalid 'unilab_logger_state.tot_time' "
            f"(expected a finite non-negative number, got {tot_time_raw!r}). "
            f"{minimum_format_hint}"
        )
    tot_timesteps = state.get("tot_timesteps")
    if type(tot_timesteps) is not int:
        raise ValueError(
            f"Checkpoint {path} has an invalid 'unilab_logger_state.tot_timesteps' "
            f"(expected a non-negative integer, got {tot_timesteps!r}). "
            f"{minimum_format_hint}"
        )
    if tot_timesteps < 0:
        raise ValueError(
            f"Checkpoint {path} has an invalid 'unilab_logger_state.tot_timesteps' "
            f"(expected a non-negative integer, got {tot_timesteps!r}). "
            f"{minimum_format_hint}"
        )
    return tot_time, tot_timesteps


def patch_rsl_rl_resume_state() -> None:
    """Persist and restore UniLab progress through rsl-rl checkpoints.

    Without this patch, rsl-rl's ``Logger.__init__`` writes ``tot_time = 0`` and
    ``tot_timesteps = 0`` and ``OnPolicyRunner.load`` never refreshes them, so the
    ``Train/mean_reward/time`` and ``Train/mean_episode_length/time`` TensorBoard
    scalars (which use ``int(self.tot_time)`` as their step) restart from 0 on
    every resumed run and visually overlap the original segment. See issue #441.

    The patch wraps ``OnPolicyRunner.save`` / ``OnPolicyRunner.load`` to round-trip
    plain progress in the saved dict:

    - ``unilab_logger_state``: ``Logger.tot_time`` / ``tot_timesteps`` (required);
    - ``unilab_env_training_state``: the wrapped TorchEnv's versioned training
      state, written only when the environment publishes one.

    Loading fails closed (issue #2416). A checkpoint must carry a valid
    ``unilab_logger_state``; pre-patch rsl-rl checkpoints without it are
    rejected with a message describing the minimum supported format instead of
    silently restarting progress from 0. When the active environment publishes
    training state, the checkpoint must also carry its versioned
    ``unilab_env_training_state``; environments without exportable state
    neither write nor require the key, and no placeholder value is fabricated.

    Checkpoints are immutable load inputs. The loader never repairs, rewrites,
    upgrades, or migrates a checkpoint file. Environment state is plain
    versioned JSON data; no Torch tensor, backend, environment, or manager
    object is serialized through this path.
    """
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except ImportError:
        return

    if getattr(OnPolicyRunner, "_UNILAB_RESUME_PATCHED", False):
        return

    import torch

    def _patched_save(self: Any, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["unilab_logger_state"] = {
            "tot_time": float(getattr(self.logger, "tot_time", 0.0)),
            "tot_timesteps": int(getattr(self.logger, "tot_timesteps", 0)),
        }
        env = getattr(self, "env", None)
        export_env_training_state = getattr(env, "export_training_state", None)
        if callable(export_env_training_state):
            env_state = export_env_training_state()
            # An environment without exportable training state publishes None;
            # the key stays absent instead of carrying a fabricated placeholder.
            if env_state is not None:
                saved_dict["unilab_env_training_state"] = _copy_plain_training_state(env_state)
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def _patched_load(
        self: Any,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> Any:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Validate the checkpoint contract before mutating any live state so a
        # rejected checkpoint cannot half-load the algorithm or the logger.
        tot_time, tot_timesteps = _require_unilab_logger_state(loaded_dict, path)
        env = getattr(self, "env", None)
        export_env_training_state = getattr(env, "export_training_state", None)
        env_publishes_state = (
            callable(export_env_training_state) and export_env_training_state() is not None
        )
        env_state = loaded_dict.get("unilab_env_training_state")
        import_training_state: Any = None
        if env_state is None:
            if env_publishes_state:
                raise ValueError(
                    f"Checkpoint {path} does not contain 'unilab_env_training_state', "
                    "but the active environment publishes versioned training state; "
                    "the checkpoint is either pre-patch or belongs to a different "
                    "environment. Resume from a checkpoint saved by a supported "
                    "UniLab version for this task or retrain."
                )
        else:
            # Reject non-plain payloads before touching the algorithm.
            env_state = _copy_plain_training_state(env_state)
            import_training_state = getattr(env, "import_training_state", None)
            if not callable(import_training_state):
                raise TypeError(
                    "Checkpoint contains unilab_env_training_state, but the active "
                    "environment does not implement import_training_state"
                )
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        self.logger.tot_time = tot_time
        self.logger.tot_timesteps = tot_timesteps
        if env_state is not None:
            import_training_state(env_state)
        return loaded_dict["infos"]

    OnPolicyRunner.save = _patched_save  # type: ignore[assignment]
    OnPolicyRunner.load = _patched_load  # type: ignore[assignment]
    OnPolicyRunner._UNILAB_RESUME_PATCHED = True  # type: ignore[attr-defined]

"""Shared experiment tracking utilities for local files and W&B."""

from __future__ import annotations

import dataclasses
import getpass
import importlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


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


def _fallback_device_info_dict() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "chip": platform.processor() or "unknown",
        "cpu_total_cores": str(os.cpu_count() or "unknown"),
        "gpu_name": "unknown",
        "memory": "unknown",
    }


def _benchmark_device_info_path() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "benchmark" / "core" / "device_info.py"
        if candidate.is_file():
            return candidate
    return None


def get_device_info_dict() -> dict[str, str]:
    try:
        module_path = _benchmark_device_info_path()
        if module_path is None:
            return _fallback_device_info_dict()
        spec = importlib.util.spec_from_file_location(
            "unilab_benchmark_device_info",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load device info module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        getter = getattr(module, "get_device_info_dict")
        return dict(getter())
    except Exception:
        return _fallback_device_info_dict()


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
        collector_device: str | None = None,
        seed_info: Any | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.log_dir = Path(log_dir)
        self.algo_name = algo_name
        self.task_name = task_name
        self.sim_backend = sim_backend
        self.training_cfg = training_cfg
        self.full_cfg = full_cfg
        self.device = device
        self.collector_device = collector_device
        self.seed_info = seed_info
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
            "collector_device": self.collector_device,
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

        payload = {
            "run": _json_safe(metadata),
            "config": _json_safe(_plain_dict(self.full_cfg)),
        }
        self._write_json(self.log_dir / "run_config.json", payload)

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
        self._write_json(self.log_dir / "run_summary.json", _json_safe(payload))

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

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def patch_rsl_rl_wandb_writer() -> None:
    """Patch rsl-rl W&B writer so it can reuse an already-open run."""
    try:
        import rsl_rl.utils.wandb_utils as wandb_utils
    except Exception:
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

"""Shared real-CUDA harness for Issue 705 device runtime acceptance tests."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, cast
from unittest.mock import patch

import pytest
import torch
from tests.manager.test_g1_reference_differential import _cfg

from unilab.base.backend import (
    BufferPlacement,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    create_backend,
    env_backend_kwargs,
)
from unilab.base.backend.base import SimBackend
from unilab.base.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unilab.envs.locomotion.g1.managed_device import create_g1_managed_device_runtime
from unilab.manager import DeviceManagedRuntime, DeviceTransition


@dataclass
class DeviceRuntimeHarness:
    backend: SimBackend
    runtime: DeviceManagedRuntime
    placement: BufferPlacement
    device: torch.device
    producer_stream: torch.cuda.Stream
    action: torch.Tensor
    action_lease: DeviceBufferLease
    action_event: torch.cuda.Event
    transition: DeviceTransition

    def step(
        self, value: float = 0.0, *, after: DeviceCompletion | None = None
    ) -> DeviceTransition:
        """Publish one stable-address policy action through a fresh lease epoch."""

        self.action_lease.invalidate()
        with torch.cuda.stream(self.producer_stream):
            (self.transition.completion if after is None else after).wait(self.producer_stream)
            self.action.fill_(value)
            completion = DeviceCompletion.record(
                placement=self.placement,
                owner_id=self.action_lease.owner_id,
                epoch=self.action_lease.epoch,
                stream=self.producer_stream,
                event=self.action_event,
            )
        view = DeviceTensorView(
            tensor_handle=self.action,
            contract=self.runtime.bound_plan.control.buffer,
            lease=self.action_lease,
            completion=completion,
        )
        self.transition = self.runtime.step(view)
        return self.transition

    def wait(self) -> None:
        """Explicit low-frequency metrics barrier outside measured rollout scopes."""

        self.transition.completion.event.synchronize()


def require_cuda() -> None:
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("Issue 705 device acceptance requires an active CUDA Warp device")


@contextmanager
def runtime_harness(
    *,
    num_envs: int,
    seed: int,
    max_episode_steps: int,
    record_lifecycle: bool = False,
    minimal_rewards: bool = False,
    mjwarp_nconmax: int | None = None,
    mjwarp_njmax: int | None = None,
) -> Iterator[DeviceRuntimeHarness]:
    require_cuda()
    cfg = _cfg(
        max_episode_seconds=None,
        observation_noise_level=0.0,
        observation_noise_seed=None,
    )
    if minimal_rewards:
        assert cfg.reward_config is not None
        cfg.reward_config.scales = {
            "tracking_lin_vel": 2.0,
            "alive": 0.1,
        }
    cfg.mjwarp_nconmax = mjwarp_nconmax
    cfg.mjwarp_njmax = mjwarp_njmax
    assert cfg.scene is not None
    backend = create_backend(
        "mjwarp",
        deepcopy(cfg.scene),
        num_envs,
        cfg.sim_dt,
        base_name=cfg.asset.base_name,
        push_body_name=cfg.domain_rand.push_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        runtime = create_g1_managed_device_runtime(
            backend=backend,
            cfg=cfg,
            reset_seed=seed,
            max_episode_steps=max_episode_steps,
            record_lifecycle=record_lifecycle,
            enable_stability_diagnostics=True,
        )
        placement = runtime.bound_plan.control.buffer.placement
        assert placement.device_index is not None
        device = torch.device(f"cuda:{placement.device_index}")
        producer_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=device))
        transition = runtime.reset()
        action = torch.zeros(
            (num_envs, *runtime.bound_plan.control.buffer.row_shape),
            dtype=torch.float32,
            device=device,
        )
        yield DeviceRuntimeHarness(
            backend=backend,
            runtime=runtime,
            placement=placement,
            device=device,
            producer_stream=producer_stream,
            action=action,
            action_lease=DeviceBufferLease("issue705-stable-policy-action"),
            action_event=cast(torch.cuda.Event, torch.cuda.Event(enable_timing=False)),
            transition=transition,
        )
    finally:
        backend.cleanup_scene_assets()


@contextmanager
def forbid_host_roundtrip(backend: SimBackend) -> Iterator[None]:
    """Fault-inject every forbidden host/device fallback during a rollout."""

    with ExitStack() as stack:
        for method in (
            "get_actuator_names",
            "get_body_ids",
            "get_joint_dof_pos_indices",
            "get_joint_dof_vel_indices",
            "get_sensor_ids",
            "get_keyframe_qpos",
            "get_init_qvel",
            "get_base_pos",
            "get_base_quat",
            "get_base_lin_vel",
            "get_base_ang_vel",
            "get_dof_pos",
            "get_dof_vel",
            "get_sensor_data",
            "get_scene_model_file",
            "set_state",
        ):
            stack.enter_context(
                patch.object(backend, method, side_effect=AssertionError(f"hot-path {method}"))
            )
        for method in ("_upload", "_download", "_synchronize", "_refresh_host_cache"):
            stack.enter_context(
                patch.object(
                    backend, method, side_effect=AssertionError(f"device fallback {method}")
                )
            )
        stack.enter_context(
            patch.object(
                backend,
                "_resolve_mjwarp_typed_mutation_selector",
                side_effect=AssertionError("hot-path selector resolution"),
            )
        )
        stack.enter_context(
            patch("torch.cuda.synchronize", side_effect=AssertionError("global synchronize"))
        )
        for method in ("cpu", "numpy", "item", "tolist"):
            stack.enter_context(
                patch.object(
                    torch.Tensor,
                    method,
                    side_effect=AssertionError(f"host tensor extraction: {method}"),
                )
            )
        stack.enter_context(
            patch("torch.nonzero", side_effect=AssertionError("host done-index extraction"))
        )
        stack.enter_context(
            patch.object(Path, "read_text", side_effect=AssertionError("asset read"))
        )
        stack.enter_context(
            patch.object(Path, "read_bytes", side_effect=AssertionError("asset read"))
        )
        yield

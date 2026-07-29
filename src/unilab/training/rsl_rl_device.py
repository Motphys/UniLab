"""Device-resident RSL-RL adapters over the managed runtime contract."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from unilab.base.backend import (
    BufferContract,
    BufferOwner,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    ExecutionProfile,
)
from unilab.manager import DeviceManagedRuntime, DeviceTransition


class DeviceRslRlContractError(RuntimeError):
    """Raised when the device VecEnv or runner contract is violated."""


@dataclass(frozen=True)
class DeviceRslRlTrafficDiagnostics:
    """Public counters for runner-owned device traffic and host boundaries."""

    action_publications: int = 0
    action_device_to_device_bytes: int = 0
    finite_metric_materializations: int = 0
    finite_metric_device_to_host_bytes: int = 0


@dataclass(frozen=True)
class DeviceRslRlLoggingDiagnostics:
    """Low-frequency episode-metric materialization counters."""

    rollout_steps: int = 0
    metric_materializations: int = 0
    metric_device_to_host_bytes: int = 0


class DeviceRslRlVecEnvWrapper:
    """RSL-RL VecEnv backed only by ``DeviceManagedRuntime`` CUDA views."""

    def __init__(
        self,
        env: Any,
        device: str | torch.device = "cuda:0",
        policy_obs_mode: str = "flat",
        *,
        reset_seed: int = 0,
    ) -> None:
        if policy_obs_mode == "auto":
            policy_obs_mode = "flat"
        if policy_obs_mode not in {"actor", "flat"}:
            raise DeviceRslRlContractError(
                f"unsupported device policy_obs_mode={policy_obs_mode!r}"
            )
        if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
            raise DeviceRslRlContractError("device reset_seed must be a non-negative integer")
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise DeviceRslRlContractError(
                "device RSL-RL adapter requires an available CUDA device"
            )
        requested_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        active_index = torch.cuda.current_device()
        if requested_index != active_index:
            raise DeviceRslRlContractError(
                "mjwarp device RSL-RL profile requires the active CUDA device "
                f"cuda:{active_index}, got cuda:{requested_index}"
            )
        requested_device = torch.device(f"cuda:{requested_index}")

        try:
            runtime = env.create_device_managed_runtime(reset_seed=reset_seed)
        except AttributeError as exc:
            raise DeviceRslRlContractError(
                "environment does not declare the device-managed runtime factory contract"
            ) from exc
        if not isinstance(runtime, DeviceManagedRuntime):
            raise DeviceRslRlContractError(
                "environment device runtime factory returned an incompatible object"
            )
        if runtime.device != requested_device:
            raise DeviceRslRlContractError(
                f"device runtime is on {runtime.device}, requested {requested_device}"
            )
        if runtime.plan.backend_io.execution_profile is not ExecutionProfile.DEVICE_RESIDENT:
            raise DeviceRslRlContractError("device wrapper requires a device_resident task plan")
        if runtime.bound_plan.backend_type != "mjwarp":
            raise DeviceRslRlContractError(
                "the verified RSL-RL device profile requires the independent mjwarp backend"
            )

        self.env = env
        self.cfg = env.cfg
        self.device = requested_device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = runtime.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.runtime = runtime

        group_widths = {
            group.key: group.width for group in runtime.plan.policy_abi.observation_groups
        }
        try:
            self.num_obs = int(group_widths["obs"])
            self.num_privileged_obs = int(group_widths["critic"])
        except KeyError as exc:
            raise DeviceRslRlContractError(
                "device RSL-RL profile requires obs and critic observation groups"
            ) from exc
        self.num_actions = int(runtime.plan.policy_abi.action_dim)
        if runtime.control_contract.row_shape != (self.num_actions,):
            raise DeviceRslRlContractError(
                "device policy ABI action width differs from the runtime control contract"
            )
        max_episode_steps = getattr(self.cfg, "max_episode_steps", None)
        if max_episode_steps is None:
            raise DeviceRslRlContractError(
                "device RSL-RL profile requires a finite max_episode_steps"
            )
        self.max_episode_length = int(max_episode_steps)

        self._action_contract: BufferContract = runtime.control_contract
        if (
            self._action_contract.placement.device_type != "cuda"
            or self._action_contract.placement.device_index != requested_index
            or self._action_contract.owner is not BufferOwner.RUNNER
            or self._action_contract.dtype != "float32"
        ):
            raise DeviceRslRlContractError(
                "device runtime control contract is incompatible with RSL-RL action staging"
            )
        self._action_buffer = torch.empty(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self._action_lease = DeviceBufferLease(f"rsl-rl-device-action:{id(self):x}")
        self._action_event = cast(
            torch.cuda.Event,
            torch.cuda.Event(enable_timing=False),
        )
        self._dones = torch.empty((self.num_envs,), dtype=torch.bool, device=self.device)
        self._finite_keys = ("obs", "critic", "reward", "action")
        finite_widths = (self.num_obs, self.num_privileged_obs, 1, self.num_actions)
        finite_offsets = (
            0,
            finite_widths[0],
            finite_widths[0] + finite_widths[1],
            finite_widths[0] + finite_widths[1] + finite_widths[2],
        )
        self._finite_slices = tuple(
            slice(offset, offset + width)
            for offset, width in zip(finite_offsets, finite_widths, strict=True)
        )
        total_finite_width = sum(finite_widths)
        self._finite_accumulator = torch.ones(
            (self.num_envs, total_finite_width),
            dtype=torch.bool,
            device=self.device,
        )
        self._finite_scratch = torch.empty_like(self._finite_accumulator)
        self._finite_value_scratch = torch.empty(
            (self.num_envs, total_finite_width),
            dtype=torch.float32,
            device=self.device,
        )
        self._traffic = DeviceRslRlTrafficDiagnostics()
        self._last_transition = self.runtime.reset()
        self._last_observations, _, _, _ = self._consume_transition(self._last_transition)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.runtime.episode_length_buffer

    @episode_length_buf.setter
    def episode_length_buf(self, values: torch.Tensor) -> None:
        self.runtime.set_episode_length_buffer(values)

    @property
    def traffic_diagnostics(self) -> DeviceRslRlTrafficDiagnostics:
        return self._traffic

    @property
    def managed_policy_abi_snapshot(self) -> dict[str, Any]:
        return self.runtime.policy_abi_snapshot

    @property
    def last_transition(self) -> DeviceTransition:
        return self._last_transition

    def _publish_actions(self, actions: torch.Tensor) -> DeviceTensorView:
        expected = (self.num_envs, self.num_actions)
        if (
            not isinstance(actions, torch.Tensor)
            or actions.device != self.device
            or actions.dtype is not torch.float32
            or tuple(actions.shape) != expected
            or not actions.is_contiguous()
        ):
            raise DeviceRslRlContractError(
                f"device policy actions must be contiguous float32 {expected} on {self.device}"
            )
        stream = torch.cuda.current_stream(self.device)
        self._action_lease.invalidate()
        self._accumulate_finite_value(actions, index=3)
        # No host predicate is allowed here.  Keep physics state valid while
        # the preallocated device finite guard records the fault; the rollout
        # boundary reports it before a learner update can consume the batch.
        torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0, out=self._action_buffer)
        completion = DeviceCompletion.record(
            placement=self._action_contract.placement,
            owner_id=self._action_lease.owner_id,
            epoch=self._action_lease.epoch,
            stream=stream,
            event=self._action_event,
        )
        self._traffic = DeviceRslRlTrafficDiagnostics(
            action_publications=self._traffic.action_publications + 1,
            action_device_to_device_bytes=(
                self._traffic.action_device_to_device_bytes
                + self._action_buffer.numel() * self._action_buffer.element_size()
            ),
            finite_metric_materializations=self._traffic.finite_metric_materializations,
            finite_metric_device_to_host_bytes=(self._traffic.finite_metric_device_to_host_bytes),
        )
        return DeviceTensorView(
            tensor_handle=self._action_buffer,
            contract=self._action_contract,
            lease=self._action_lease,
            completion=completion,
        )

    @staticmethod
    def _transition_tensors(
        transition: DeviceTransition,
        *,
        final: bool = False,
    ) -> dict[str, torch.Tensor]:
        buffers = transition.final_observations if final else transition.observations
        return {buffer.key: buffer.view.torch() for buffer in buffers}

    def _as_tensordict(
        self,
        transition: DeviceTransition,
        *,
        final: bool = False,
    ) -> TensorDict:
        groups = self._transition_tensors(transition, final=final)
        try:
            actor = groups["obs"]
            critic = groups["critic"]
        except KeyError as exc:
            raise DeviceRslRlContractError(
                "device transition lacks obs or critic observation group"
            ) from exc
        return TensorDict(
            {"actor": actor, "policy": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _accumulate_finite_value(self, value: torch.Tensor, *, index: int) -> None:
        field_slice = self._finite_slices[index]
        scratch = self._finite_scratch[:, field_slice]
        accumulator = self._finite_accumulator[:, field_slice]
        value_scratch = self._finite_value_scratch[:, field_slice]
        if value.ndim == 1:
            scratch = scratch[:, 0]
            accumulator = accumulator[:, 0]
            value_scratch = value_scratch[:, 0]
        torch.abs(value, out=value_scratch)
        torch.le(value_scratch, torch.finfo(value.dtype).max, out=scratch)
        torch.logical_and(accumulator, scratch, out=accumulator)

    def _accumulate_finite_checks(
        self,
        *,
        observations: TensorDict,
        final_observations: TensorDict,
        rewards: torch.Tensor,
    ) -> None:
        for index, value in enumerate((observations["actor"], observations["critic"], rewards)):
            self._accumulate_finite_value(value, index=index)
        # Timeout bootstrapping feeds final/terminal observations to the
        # critic.  Fold them into the same preallocated actor/critic guards so
        # a terminal-only fault cannot reach the learner unnoticed.  This is
        # still fully device-side and deliberately has no done-row branch.
        self._accumulate_finite_value(final_observations["actor"], index=0)
        self._accumulate_finite_value(final_observations["critic"], index=1)

    def _consume_transition(
        self,
        transition: DeviceTransition,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not isinstance(transition, DeviceTransition):
            raise DeviceRslRlContractError("device runtime returned an invalid transition")
        stream = torch.cuda.current_stream(self.device)
        transition.completion.wait(stream)
        observations = self._as_tensordict(transition)
        final_observations = self._as_tensordict(transition, final=True)
        rewards = transition.reward.torch()
        terminated = transition.terminated.torch()
        truncated = transition.truncated.torch()
        torch.logical_or(terminated, truncated, out=self._dones)
        self._accumulate_finite_checks(
            observations=observations,
            final_observations=final_observations,
            rewards=rewards,
        )
        extras: dict[str, Any] = {
            "time_outs": truncated,
            "time_out_bootstrap_obs": final_observations,
        }
        return observations, rewards, self._dones, extras

    def finish_rollout(self) -> None:
        """Validate asynchronous finite guards at one iteration boundary."""

        host_flags = self._finite_accumulator.to(device="cpu", non_blocking=False)
        byte_count = host_flags.numel() * host_flags.element_size()
        self._traffic = DeviceRslRlTrafficDiagnostics(
            action_publications=self._traffic.action_publications,
            action_device_to_device_bytes=self._traffic.action_device_to_device_bytes,
            finite_metric_materializations=self._traffic.finite_metric_materializations + 1,
            finite_metric_device_to_host_bytes=(
                self._traffic.finite_metric_device_to_host_bytes + byte_count
            ),
        )
        self._finite_accumulator.fill_(True)
        failed = tuple(
            key
            for key, field_slice in zip(
                self._finite_keys,
                self._finite_slices,
                strict=True,
            )
            if not bool(torch.all(host_flags[:, field_slice]))
        )
        if failed:
            raise DeviceRslRlContractError(
                "device rollout contains non-finite values in: " + ", ".join(failed)
            )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_view = self._publish_actions(actions)
        self._last_transition = self.runtime.step(action_view)
        result = self._consume_transition(self._last_transition)
        self._last_observations = result[0]
        return result

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        self._last_transition = self.runtime.reset()
        observations, _, _, _ = self._consume_transition(self._last_transition)
        self._last_observations = observations
        return observations, {}

    def get_observations(self) -> TensorDict:
        self._last_transition.completion.wait(torch.cuda.current_stream(self.device))
        return self._last_observations

    def get_privileged_observations(self) -> torch.Tensor:
        return cast(torch.Tensor, self.get_observations()["critic"])

    def close(self) -> None:
        self.env.close()


class DeviceRolloutLogger(Logger):
    """RSL-RL logger with one episode-metric D2H boundary per iteration."""

    def __init__(self, *args: Any, rollout_steps: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            isinstance(rollout_steps, bool)
            or not isinstance(rollout_steps, int)
            or rollout_steps <= 0
        ):
            raise DeviceRslRlContractError("device logger rollout_steps must be positive")
        self._rollout_steps = rollout_steps
        raw_device = torch.device(self.device)
        self._torch_device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if raw_device.type == "cuda" and raw_device.index is None
            else raw_device
        )
        self._metric_channels = 5 if self.cfg["algorithm"]["rnd_cfg"] else 3
        self._rollout_metrics = torch.empty(
            (self._metric_channels, rollout_steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self._rollout_cursor = 0
        self._device_diagnostics = DeviceRslRlLoggingDiagnostics()

    @property
    def device_diagnostics(self) -> DeviceRslRlLoggingDiagnostics:
        return self._device_diagnostics

    def _validate_step_tensor(
        self,
        value: torch.Tensor,
        *,
        name: str,
        dtype: torch.dtype,
    ) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.device != self._torch_device
            or value.dtype is not dtype
            or tuple(value.shape) != (self.num_envs,)
            or not value.is_contiguous()
        ):
            raise DeviceRslRlContractError(
                f"device logger {name} must be a contiguous {dtype} "
                f"({self.num_envs},) tensor on {self._torch_device}"
            )

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        if self.writer is None:
            return
        if "episode" in extras or "log" in extras:
            raise DeviceRslRlContractError(
                "device logger requires typed pre-aggregated task metrics; arbitrary per-step extras are unsupported"
            )
        if self._rollout_cursor >= self._rollout_steps:
            raise DeviceRslRlContractError("device episode metric staging overflow")
        self._validate_step_tensor(rewards, name="rewards", dtype=torch.float32)
        self._validate_step_tensor(dones, name="dones", dtype=torch.bool)
        if intrinsic_rewards is not None:
            self._validate_step_tensor(
                intrinsic_rewards,
                name="intrinsic_rewards",
                dtype=torch.float32,
            )

        if intrinsic_rewards is not None:
            self.cur_ereward_sum.add_(rewards)
            self.cur_ireward_sum.add_(intrinsic_rewards)
            self.cur_reward_sum.add_(rewards).add_(intrinsic_rewards)
        else:
            self.cur_reward_sum.add_(rewards)
        self.cur_episode_length.add_(1)

        step = self._rollout_cursor
        self._rollout_metrics[0, step].copy_(self.cur_reward_sum, non_blocking=True)
        self._rollout_metrics[1, step].copy_(self.cur_episode_length, non_blocking=True)
        self._rollout_metrics[2, step].copy_(dones, non_blocking=True)
        if intrinsic_rewards is not None:
            self._rollout_metrics[3, step].copy_(self.cur_ereward_sum, non_blocking=True)
            self._rollout_metrics[4, step].copy_(self.cur_ireward_sum, non_blocking=True)

        self.cur_reward_sum.masked_fill_(dones, 0.0)
        self.cur_episode_length.masked_fill_(dones, 0.0)
        if intrinsic_rewards is not None:
            self.cur_ereward_sum.masked_fill_(dones, 0.0)
            self.cur_ireward_sum.masked_fill_(dones, 0.0)
        self._rollout_cursor += 1

    def _materialize_episode_metrics(self) -> None:
        if self.writer is None:
            self._rollout_cursor = 0
            return
        if self._rollout_cursor != self._rollout_steps:
            raise DeviceRslRlContractError(
                "device logger metric boundary does not match the configured rollout length"
            )
        staged = self._rollout_metrics[:, : self._rollout_cursor]
        host = staged.to(device="cpu", non_blocking=False)
        for step in range(self._rollout_cursor):
            mask = host[2, step] > 0.0
            self.rewbuffer.extend(host[0, step, mask].tolist())
            self.lenbuffer.extend(host[1, step, mask].tolist())
            if self._metric_channels == 5:
                self.erewbuffer.extend(host[3, step, mask].tolist())
                self.irewbuffer.extend(host[4, step, mask].tolist())
        self._device_diagnostics = DeviceRslRlLoggingDiagnostics(
            rollout_steps=self._device_diagnostics.rollout_steps + self._rollout_cursor,
            metric_materializations=self._device_diagnostics.metric_materializations + 1,
            metric_device_to_host_bytes=(
                self._device_diagnostics.metric_device_to_host_bytes
                + staged.numel() * staged.element_size()
            ),
        )
        self._rollout_cursor = 0

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._materialize_episode_metrics()
        super().log(*args, **kwargs)


class DeviceOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner variant with explicit iteration-level device boundaries."""

    env: DeviceRslRlVecEnvWrapper
    logger: DeviceRolloutLogger
    current_learning_iteration: int

    def __init__(
        self,
        env: DeviceRslRlVecEnvWrapper,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        if not isinstance(env, DeviceRslRlVecEnvWrapper):
            raise DeviceRslRlContractError("DeviceOnPolicyRunner requires DeviceRslRlVecEnvWrapper")
        if train_cfg.get("check_for_nan") is not False:
            raise DeviceRslRlContractError(
                "device owner must set algo.check_for_nan=false; deferred finite guards run at rollout boundaries"
            )
        super().__init__(cast(Any, env), train_cfg, log_dir=log_dir, device=device)
        self.logger = DeviceRolloutLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            rollout_steps=int(self.cfg["num_steps_per_env"]),
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    )
                    self.logger.process_env_step(
                        rewards,
                        dones,
                        extras,
                        intrinsic_rewards,
                    )

                self.env.finish_rollout()
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=(
                    self.alg.rnd.weight
                    if self.cfg["algorithm"]["rnd_cfg"] and self.alg.rnd is not None
                    else None
                ),
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                if self.logger.log_dir is None:
                    raise DeviceRslRlContractError("active device logger has no log_dir")
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            if self.logger.log_dir is None:
                raise DeviceRslRlContractError("active device logger has no log_dir")
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()


@dataclass(frozen=True)
class DeviceRslRlPPORuntime:
    """Owner-selected classes and cold-path wrapper arguments."""

    wrapper_cls: type[DeviceRslRlVecEnvWrapper]
    runner_cls: type[DeviceOnPolicyRunner]
    wrapper_kwargs: dict[str, Any]
    required_backend: str = "mjwarp"
    required_execution_profile: str = "device_resident"


def resolve_mjwarp_device_ppo_runtime(
    rl_cfg: dict[str, Any],
) -> DeviceRslRlPPORuntime:
    """Resolve the explicitly selected mjwarp device PPO implementation."""

    if rl_cfg.get("runtime_impl") != "mjwarp_device_v1":
        raise DeviceRslRlContractError(
            "mjwarp device runtime resolver requires algo.runtime_impl=mjwarp_device_v1"
        )
    seed = rl_cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DeviceRslRlContractError("mjwarp device PPO requires a non-negative algo.seed")
    return DeviceRslRlPPORuntime(
        wrapper_cls=DeviceRslRlVecEnvWrapper,
        runner_cls=DeviceOnPolicyRunner,
        wrapper_kwargs={"reset_seed": seed},
    )


__all__ = [
    "DeviceOnPolicyRunner",
    "DeviceRolloutLogger",
    "DeviceRslRlContractError",
    "DeviceRslRlLoggingDiagnostics",
    "DeviceRslRlPPORuntime",
    "DeviceRslRlTrafficDiagnostics",
    "DeviceRslRlVecEnvWrapper",
    "resolve_mjwarp_device_ppo_runtime",
]

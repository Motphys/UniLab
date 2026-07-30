"""Order-independent random streams for compiled DR and Event terms.

The sampler uses semantic counters instead of a mutable global generator.  A
sample is identified by ``run seed + term key/version + environment id +
trigger count + component``.  Registration order, selected-row ordering, and
unrelated terms therefore cannot perturb an existing term's sequence.

This module deliberately has no backend dependency.  Device streams operate
directly on preallocated Torch buffers; the manager can wrap their borrowed
output in its normal typed mutation envelope and completion event.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

KEYED_RNG_ALGORITHM = "splitmix32-v1"
_UINT32_MASK = 0xFFFFFFFF
_ENV_MULTIPLIER = 0x9E3779B1
_TRIGGER_MULTIPLIER = 0x85EBCA77
_COMPONENT_MULTIPLIER = 0xC2B2AE3D
_MIX_MULTIPLIER = 0x045D9F3B
_NORMAL_SALT = 0xA511E9B3
_UINT32_SCALE = 1.0 / float(1 << 32)


class KeyedRandomContractError(ValueError):
    """Raised when a random stream specification or active mask is invalid."""


class StaleKeyedRandomBatchError(RuntimeError):
    """Raised when a borrowed sample is read after its stream advanced."""


class RandomDistribution(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"


class RandomCorrelation(str, Enum):
    PER_ELEMENT = "per_element"
    PER_ENTITY = "per_entity"
    PER_ENV = "per_env"
    GLOBAL = "global"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KeyedRandomContractError(f"{name} must be a non-empty string")
    return value.strip()


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise KeyedRandomContractError("random row_shape must be a tuple")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
        raise KeyedRandomContractError("random row_shape dimensions must be positive integers")
    return value


def _finite_pair(value: object, *, distribution: RandomDistribution) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise KeyedRandomContractError("random parameters must contain exactly two numbers")
    first, second = (float(value[0]), float(value[1]))
    if not math.isfinite(first) or not math.isfinite(second):
        raise KeyedRandomContractError("random parameters must be finite")
    if distribution is RandomDistribution.UNIFORM and first > second:
        raise KeyedRandomContractError("uniform lower bound cannot exceed its upper bound")
    if distribution is RandomDistribution.NORMAL and second < 0.0:
        raise KeyedRandomContractError("normal standard deviation cannot be negative")
    return first, second


def _stable_u32(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


@dataclass(frozen=True)
class KeyedRandomSpec:
    """Immutable identity and shape of one compiled random stream.

    ``PER_ENTITY`` treats the first row-shape axis as the entity axis and
    shares one sample across all remaining components of each entity.
    ``PER_ENV`` shares one scalar across the complete row shape. ``GLOBAL``
    additionally removes the environment id from the key; worlds at the same
    trigger count receive the same scalar.
    """

    term_key: str
    term_version: str
    row_shape: tuple[int, ...]
    distribution: RandomDistribution
    correlation: RandomCorrelation
    parameters: tuple[float, float]
    algorithm: str = KEYED_RNG_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_key", _non_empty(self.term_key, "term_key"))
        object.__setattr__(self, "term_version", _non_empty(self.term_version, "term_version"))
        object.__setattr__(self, "row_shape", _shape(self.row_shape))
        if not isinstance(self.distribution, RandomDistribution):
            raise KeyedRandomContractError("distribution must be a RandomDistribution")
        if not isinstance(self.correlation, RandomCorrelation):
            raise KeyedRandomContractError("correlation must be a RandomCorrelation")
        object.__setattr__(
            self,
            "parameters",
            _finite_pair(self.parameters, distribution=self.distribution),
        )
        if self.correlation is RandomCorrelation.PER_ENTITY and not self.row_shape:
            raise KeyedRandomContractError("per-entity correlation requires an entity axis")
        if self.algorithm != KEYED_RNG_ALGORITHM:
            raise KeyedRandomContractError(f"unsupported keyed RNG algorithm {self.algorithm!r}")

    @property
    def width(self) -> int:
        return math.prod(self.row_shape)

    @property
    def fingerprint(self) -> str:
        payload = {
            "algorithm": self.algorithm,
            "correlation": self.correlation.value,
            "distribution": self.distribution.value,
            "parameters": self.parameters,
            "row_shape": self.row_shape,
            "term_key": self.term_key,
            "term_version": self.term_version,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"{self.algorithm}:{digest}"


def _component_keys(spec: KeyedRandomSpec) -> np.ndarray:
    if spec.correlation in {RandomCorrelation.PER_ENV, RandomCorrelation.GLOBAL}:
        components = np.zeros((spec.width,), dtype=np.int64)
    elif spec.correlation is RandomCorrelation.PER_ENTITY:
        component_width = math.prod(spec.row_shape[1:])
        components = np.repeat(
            np.arange(spec.row_shape[0], dtype=np.int64),
            component_width,
        )
    else:
        components = np.arange(spec.width, dtype=np.int64)
    return np.bitwise_and(components * _COMPONENT_MULTIPLIER, _UINT32_MASK)


def _seed_term_key(run_seed: int, spec: KeyedRandomSpec) -> int:
    seed_key = _stable_u32(f"seed:{run_seed}")
    term_key = _stable_u32(f"term:{spec.term_key}:{spec.term_version}:{spec.algorithm}")
    return (seed_key + term_key) & _UINT32_MASK


def _mix_numpy(value: np.ndarray) -> np.ndarray:
    value = np.bitwise_and(value, _UINT32_MASK)
    for _ in range(2):
        value = np.bitwise_xor(value, np.right_shift(value, 16))
        value = np.bitwise_and(value * _MIX_MULTIPLIER, _UINT32_MASK)
    return np.bitwise_xor(value, np.right_shift(value, 16))


def _uniform_numpy(key: np.ndarray) -> np.ndarray:
    return (_mix_numpy(key).astype(np.float64) + 0.5) * _UINT32_SCALE


def keyed_random_reference(
    spec: KeyedRandomSpec,
    *,
    run_seed: int,
    env_ids: np.ndarray,
    trigger_counts: np.ndarray,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Independent NumPy reference for explicit oracle/cold-path use."""

    if not isinstance(spec, KeyedRandomSpec):
        raise KeyedRandomContractError("reference sampler requires a KeyedRandomSpec")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise KeyedRandomContractError("run_seed must be an integer")
    env_ids = np.asarray(env_ids)
    trigger_counts = np.asarray(trigger_counts)
    if env_ids.ndim != 1 or trigger_counts.shape != env_ids.shape:
        raise KeyedRandomContractError("env_ids and trigger_counts must be equal-length vectors")
    if env_ids.dtype.kind not in "iu" or trigger_counts.dtype.kind not in "iu":
        raise KeyedRandomContractError("env_ids and trigger_counts must use integer dtypes")
    if np.any(env_ids < 0) or np.any(trigger_counts < 0):
        raise KeyedRandomContractError("env_ids and trigger_counts must be non-negative")
    int64_max = np.iinfo(np.int64).max
    if np.any(env_ids > int64_max) or np.any(trigger_counts > int64_max):
        raise KeyedRandomContractError("env_ids and trigger_counts must fit signed int64")

    env_key = np.zeros_like(env_ids, dtype=np.int64)
    if spec.correlation is not RandomCorrelation.GLOBAL:
        env_key[:] = env_ids.astype(np.int64, copy=False)
    row_key = (
        env_key[:, None] * _ENV_MULTIPLIER
        + trigger_counts.astype(np.int64, copy=False)[:, None] * _TRIGGER_MULTIPLIER
        + _seed_term_key(run_seed, spec)
    )
    key = np.bitwise_and(row_key + _component_keys(spec)[None, :], _UINT32_MASK)
    first, second = spec.parameters
    if spec.distribution is RandomDistribution.UNIFORM:
        values = _uniform_numpy(key) * (second - first) + first
    else:
        u1 = np.maximum(_uniform_numpy(key), np.finfo(np.float64).tiny)
        u2 = _uniform_numpy(np.bitwise_and(key + _NORMAL_SALT, _UINT32_MASK))
        values = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
        values = values * second + first
    return np.ascontiguousarray(values.reshape(len(env_ids), *spec.row_shape), dtype=dtype)


@dataclass(frozen=True)
class KeyedRandomTrafficDiagnostics:
    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    global_synchronizations: int = 0
    sample_allocations: int = 0


@dataclass(frozen=True)
class KeyedRandomBatch:
    """Borrowed stable-address output valid until the next sample call."""

    _stream: KeyedRandomStream
    _epoch: int

    def _live_stream(self) -> KeyedRandomStream:
        stream = self._stream
        if stream.epoch != self._epoch:
            raise StaleKeyedRandomBatchError(
                "keyed random batch is stale because its stream advanced"
            )
        return stream

    @property
    def values(self) -> torch.Tensor:
        return self._live_stream()._values

    @property
    def active_mask(self) -> torch.Tensor:
        return self._live_stream()._active_mask

    @property
    def fingerprint(self) -> str:
        return self._live_stream().spec.fingerprint


class KeyedRandomStream:
    """Preallocated Torch implementation of one counter-keyed random stream."""

    def __init__(
        self,
        spec: KeyedRandomSpec,
        *,
        run_seed: int,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(spec, KeyedRandomSpec):
            raise KeyedRandomContractError("stream requires a KeyedRandomSpec")
        if isinstance(run_seed, bool) or not isinstance(run_seed, int):
            raise KeyedRandomContractError("run_seed must be an integer")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise KeyedRandomContractError("num_envs must be a positive integer")
        if dtype not in {torch.float32, torch.float64}:
            raise KeyedRandomContractError("keyed random streams require float32 or float64")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        if resolved_device.type not in {"cpu", "cuda"}:
            raise KeyedRandomContractError("keyed random streams require a CPU or CUDA device")

        self.spec = spec
        self.run_seed = run_seed
        self.num_envs = num_envs
        self.device = resolved_device
        self.dtype = dtype
        self._epoch = 0
        self._traffic = KeyedRandomTrafficDiagnostics()
        shape = (num_envs, spec.width)
        row_shape = (num_envs, 1)
        self._trigger_counts = torch.zeros(num_envs, dtype=torch.int64, device=resolved_device)
        self._active_mask = torch.empty(num_envs, dtype=torch.bool, device=resolved_device)
        self._env_keys = torch.arange(num_envs, dtype=torch.int64, device=resolved_device).view(
            row_shape
        )
        if spec.correlation is RandomCorrelation.GLOBAL:
            self._env_keys.zero_()
        component_keys = _component_keys(spec)
        self._component_keys = torch.as_tensor(
            component_keys,
            dtype=torch.int64,
            device=resolved_device,
        ).view(1, spec.width)
        self._row_key = torch.empty(row_shape, dtype=torch.int64, device=resolved_device)
        self._row_tmp = torch.empty(row_shape, dtype=torch.int64, device=resolved_device)
        self._key = torch.empty(shape, dtype=torch.int64, device=resolved_device)
        self._key_tmp = torch.empty(shape, dtype=torch.int64, device=resolved_device)
        self._float_a = torch.empty(shape, dtype=dtype, device=resolved_device)
        self._float_b = torch.empty(shape, dtype=dtype, device=resolved_device)
        self._candidate = torch.empty(shape, dtype=dtype, device=resolved_device)
        self._values = torch.zeros(
            (num_envs, *spec.row_shape),
            dtype=dtype,
            device=resolved_device,
        )
        self._seed_term = _seed_term_key(run_seed, spec)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def traffic_diagnostics(self) -> KeyedRandomTrafficDiagnostics:
        return self._traffic

    @property
    def output_address(self) -> int:
        return int(self._values.data_ptr())

    def _fill_key(self, *, salt: int = 0) -> None:
        torch.mul(self._env_keys, _ENV_MULTIPLIER, out=self._row_key)
        torch.mul(
            self._trigger_counts.view(self.num_envs, 1),
            _TRIGGER_MULTIPLIER,
            out=self._row_tmp,
        )
        self._row_key.add_(self._row_tmp)
        self._row_key.add_((self._seed_term + salt) & _UINT32_MASK)
        self._row_key.bitwise_and_(_UINT32_MASK)
        self._key.copy_(self._row_key)
        self._key.add_(self._component_keys)
        self._key.bitwise_and_(_UINT32_MASK)
        for _ in range(2):
            torch.bitwise_right_shift(self._key, 16, out=self._key_tmp)
            torch.bitwise_xor(self._key, self._key_tmp, out=self._key)
            torch.mul(self._key, _MIX_MULTIPLIER, out=self._key)
            self._key.bitwise_and_(_UINT32_MASK)
        torch.bitwise_right_shift(self._key, 16, out=self._key_tmp)
        torch.bitwise_xor(self._key, self._key_tmp, out=self._key)

    def _fill_uniform(self, output: torch.Tensor, *, salt: int = 0) -> None:
        self._fill_key(salt=salt)
        torch.add(self._key, 0.5, out=output)
        output.mul_(_UINT32_SCALE)

    def _sample_candidate(self) -> None:
        first, second = self.spec.parameters
        if self.spec.distribution is RandomDistribution.UNIFORM:
            self._fill_uniform(self._candidate)
            self._candidate.mul_(second - first).add_(first)
            return
        self._fill_uniform(self._float_a)
        self._fill_uniform(self._float_b, salt=_NORMAL_SALT)
        self._float_a.clamp_min_(torch.finfo(self.dtype).tiny)
        torch.log(self._float_a, out=self._candidate)
        self._candidate.mul_(-2.0).sqrt_()
        self._float_b.mul_(2.0 * math.pi)
        torch.cos(self._float_b, out=self._float_b)
        self._candidate.mul_(self._float_b).mul_(second).add_(first)

    def sample(self, active_mask: torch.Tensor) -> KeyedRandomBatch:
        """Advance selected worlds and return one borrowed all-world buffer.

        The mask is already device-resident manager state.  The call performs
        no host predicate, transfer, allocation, or global synchronization.
        Inactive rows retain their previous output and trigger count.
        """

        if not isinstance(active_mask, torch.Tensor):
            raise KeyedRandomContractError("active_mask must be a torch.Tensor")
        if (
            active_mask.device != self.device
            or active_mask.dtype is not torch.bool
            or tuple(active_mask.shape) != (self.num_envs,)
            or not active_mask.is_contiguous()
        ):
            raise KeyedRandomContractError(
                "active_mask must be a contiguous bool vector on the stream device"
            )
        self._active_mask.copy_(active_mask)
        self._sample_candidate()
        candidate = self._candidate.view(self.num_envs, *self.spec.row_shape)
        mask = self._active_mask.view(self.num_envs, *((1,) * len(self.spec.row_shape)))
        torch.where(mask, candidate, self._values, out=self._values)
        self._trigger_counts.add_(self._active_mask)
        self._epoch += 1
        return KeyedRandomBatch(self, self._epoch)

    def capture_trigger_counts(self) -> np.ndarray:
        """Materialize counters at an explicit diagnostics/oracle boundary."""

        return self._trigger_counts.detach().cpu().numpy().copy()


__all__ = [
    "KEYED_RNG_ALGORITHM",
    "KeyedRandomBatch",
    "KeyedRandomContractError",
    "KeyedRandomSpec",
    "KeyedRandomStream",
    "KeyedRandomTrafficDiagnostics",
    "RandomCorrelation",
    "RandomDistribution",
    "StaleKeyedRandomBatchError",
    "keyed_random_reference",
]

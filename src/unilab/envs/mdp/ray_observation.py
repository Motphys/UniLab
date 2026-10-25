"""Tensor-native ray-pattern observation terms for the Manager runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch
from unisim.backend.base import torch_device_matches

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(frozen=True)
class RayPatternCfg:
    """Declare one observation-owned ray pattern.

    ``caster`` is a public UniSim ray-caster factory name, not an imported
    plugin class. ``body_names`` is entity-local and selects both the backend
    ray-scene projection and the pose-only packed read block.
    """

    entity_name: str
    body_names: Sequence[str]
    num_rays: int
    pattern_start: float
    pattern_stop: float
    max_distance: float
    caster: str = "uni_ray"
    pattern_center_height: float = 0.0
    caster_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise TypeError("RayPatternCfg entity_name must be a non-empty string")
        if isinstance(self.body_names, (str, bytes)):
            raise TypeError("RayPatternCfg body_names must be a sequence of strings")
        names = tuple(self.body_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise TypeError("RayPatternCfg body_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"RayPatternCfg body_names must be unique: {names}")
        if isinstance(self.num_rays, bool) or not isinstance(self.num_rays, int):
            raise TypeError("RayPatternCfg num_rays must be an integer")
        if self.num_rays <= 0:
            raise ValueError("RayPatternCfg num_rays must be positive")
        for name, value in (
            ("pattern_start", self.pattern_start),
            ("pattern_stop", self.pattern_stop),
            ("max_distance", self.max_distance),
            ("pattern_center_height", self.pattern_center_height),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"RayPatternCfg {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"RayPatternCfg {name} must be finite")
        if self.max_distance <= 0.0:
            raise ValueError("RayPatternCfg max_distance must be positive")
        if not isinstance(self.caster, str) or not self.caster:
            raise TypeError("RayPatternCfg caster must be a non-empty factory name")
        if not isinstance(self.caster_options, dict):
            raise TypeError("RayPatternCfg caster_options must be a mapping")
        object.__setattr__(self, "body_names", names)


class RayScan(ManagerTermBase):
    """Bind one tensor caster and publish its distance observation carrier."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        pattern = cfg.params.get("ray_pattern")
        if not isinstance(pattern, RayPatternCfg):
            raise TypeError(
                "RayScan requires params['ray_pattern'] to be a RayPatternCfg; "
                f"got {type(pattern).__name__}"
            )
        self.ray_pattern = pattern
        self.tensor_body_names = tuple(pattern.body_names)
        self.observation_width = pattern.num_rays
        self._device = torch.device(env.device)
        self._entity_name = pattern.entity_name
        self._body_names = tuple(pattern.body_names)
        self._caster: Any = None
        self._bound = False
        self._closed = False
        self._distance: torch.Tensor | None = None
        self._origins: torch.Tensor | None = None
        self._directions: torch.Tensor | None = None

    def bind_read_phase(self) -> None:
        """Materialize and bind the caster after packed pose reads compile."""
        if self._bound or self._closed:
            return
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None:
            raise NotImplementedError("RayScan requires the Manager packed Torch read plan")
        try:
            scene = self._env.backend.get_ray_scene_description(self._body_names)
            caster = _create_ray_caster(
                self.ray_pattern.caster,
                num_envs=self._env.num_envs,
                num_rays=self.ray_pattern.num_rays,
                device=self._device,
                **self.ray_pattern.caster_options,
            )
            capabilities = caster.get_ray_capabilities()
            if not (
                capabilities.supports_tensor_pose_sync
                and capabilities.supports_tensor_rays
                and capabilities.supports_device_output
                and capabilities.stream_event_ownership == "caller"
                and torch_device_matches(capabilities.torch_devices, self._device)
            ):
                raise NotImplementedError(
                    f"RayScan caster '{self.ray_pattern.caster}' does not support the "
                    f"tensor pose/ray/output profile on {self._device}"
                )
            caster.materialize(scene)
            caster.bind_pose_layout(tuple(scene.body_names))
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"RayScan caster '{self.ray_pattern.caster}' read-phase binding failed: {exc}"
            ) from exc
        self._caster = caster
        self._distance = torch.empty(
            (self._env.num_envs, self.ray_pattern.num_rays),
            dtype=torch.float32,
            device=self._device,
        )
        angles = torch.linspace(
            self.ray_pattern.pattern_start,
            self.ray_pattern.pattern_stop,
            self.ray_pattern.num_rays,
            dtype=torch.float32,
            device=self._device,
        )
        self._directions = torch.stack(
            (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
        ).contiguous()
        self._origins = torch.zeros(
            (self._env.num_envs, self.ray_pattern.num_rays, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._origins[:, :, 2] = self.ray_pattern.pattern_center_height
        self._origins = self._origins.contiguous()
        self._bound = True

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if not self._bound or self._caster is None or self._distance is None:
            raise NotImplementedError("RayScan must bind its read phase before compute")
        read_plan = getattr(env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            raise RuntimeError("RayScan requires a refreshed packed read phase")
        pose = read_plan.ray_pose_view(self._entity_name, self._body_names)
        try:
            self._caster.update_pose_tensor(pose)
            result = self._caster.trace_tensor(
                self._origins,
                self._directions,
                self.ray_pattern.max_distance,
            )
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(f"RayScan tensor trace failed: {exc}") from exc
        self._distance.copy_(result.distance)
        return self._distance

    def close(self) -> None:
        """Release caster resources before the scene read plan closes."""
        self._closed = True
        caster, self._caster = self._caster, None
        if caster is not None:
            caster.close()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Retain caster state; observations refresh from the next read phase."""
        del env_ids


__all__ = ["RayPatternCfg", "RayScan"]


def _create_ray_caster(caster: str, **options: Any) -> Any:
    """Lazily call the installed UniSim public caster factory."""
    import unisim

    factory = getattr(unisim, "create_ray_caster", None)
    if not callable(factory):
        raise NotImplementedError(
            "installed unisim-core does not expose create_ray_caster; use the "
            "ray-local optional dependency profile"
        )
    return factory(caster, **options)

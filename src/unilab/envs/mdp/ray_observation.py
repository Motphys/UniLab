"""Tensor-native ray-pattern observation terms for the Manager runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch
from unisim.backend.base import torch_device_matches
from unisim.ray_query import (
    RayGeometryTensorView,
    RayPoseTensorView,
    RayTraceFilter,
    RayTraceOutputs,
)

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


def _real_sequence(value: object, name: str, width: int) -> tuple[float, ...]:
    """Validate and normalize a finite real sequence used by ray patterns."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of {width} real numbers")
    values = tuple(value)
    if len(values) != width or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in values
    ):
        raise TypeError(f"{name} must be a sequence of {width} real numbers")
    if any(not math.isfinite(float(item)) for item in values):
        raise ValueError(f"{name} must be finite")
    return tuple(float(item) for item in values)


@dataclass(frozen=True)
class RayRingPatternCfg:
    """Rays on a body-local yaw arc with one shared pitch."""

    num_rays: int
    start: float
    stop: float
    pitch: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.num_rays, bool) or not isinstance(self.num_rays, int):
            raise TypeError("RayRingPatternCfg num_rays must be an integer")
        if self.num_rays <= 0:
            raise ValueError("RayRingPatternCfg num_rays must be positive")
        for name, value in (("start", self.start), ("stop", self.stop), ("pitch", self.pitch)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"RayRingPatternCfg {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"RayRingPatternCfg {name} must be finite")

    @property
    def ray_count(self) -> int:
        return self.num_rays

    def origins(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((self.ray_count, 3), dtype=torch.float32, device=device)

    def directions(self, device: torch.device) -> torch.Tensor:
        angles = torch.linspace(
            self.start, self.stop, self.num_rays, dtype=torch.float32, device=device
        )
        pitch = float(self.pitch)
        return torch.stack(
            (
                torch.cos(angles) * math.cos(pitch),
                torch.sin(angles) * math.cos(pitch),
                torch.full_like(angles, math.sin(pitch)),
            ),
            dim=-1,
        ).contiguous()


@dataclass(frozen=True)
class RayGridPatternCfg:
    """Parallel rays on an inclusive two-dimensional local grid."""

    size: Sequence[float]
    resolution: float | Sequence[float]
    offset: Sequence[float] = (0.0, 0.0)
    direction: Sequence[float] = (0.0, 0.0, -1.0)

    def __post_init__(self) -> None:
        size = _real_sequence(self.size, "RayGridPatternCfg size", 2)
        if any(value <= 0.0 for value in size):
            raise ValueError("RayGridPatternCfg size components must be positive")
        object.__setattr__(self, "size", size)

        if isinstance(self.resolution, bool) or not isinstance(
            self.resolution, (int, float, Sequence)
        ):
            raise TypeError(
                "RayGridPatternCfg resolution must be a real number or two real numbers"
            )
        if isinstance(self.resolution, (str, bytes)):
            raise TypeError(
                "RayGridPatternCfg resolution must be a real number or two real numbers"
            )
        if isinstance(self.resolution, Sequence):
            resolution = _real_sequence(self.resolution, "RayGridPatternCfg resolution", 2)
        else:
            value = float(self.resolution)
            if not math.isfinite(value):
                raise ValueError("RayGridPatternCfg resolution must be finite")
            resolution = (value, value)
        if any(value <= 0.0 for value in resolution):
            raise ValueError("RayGridPatternCfg resolution components must be positive")
        object.__setattr__(self, "resolution", resolution)

        object.__setattr__(
            self,
            "offset",
            _real_sequence(self.offset, "RayGridPatternCfg offset", 2),
        )
        direction = _real_sequence(self.direction, "RayGridPatternCfg direction", 3)
        if torch.linalg.vector_norm(torch.tensor(direction)).item() == 0.0:
            raise ValueError("RayGridPatternCfg direction must be nonzero")
        object.__setattr__(self, "direction", direction)

    @property
    def _counts(self) -> tuple[int, int]:
        resolution = self.resolution
        if not isinstance(resolution, Sequence):
            resolution = (resolution, resolution)
        return tuple(  # type: ignore[return-value]
            int(math.floor(axis / step + 1.0e-9)) + 1
            for axis, step in zip(self.size, resolution, strict=True)
        )

    @property
    def ray_count(self) -> int:
        count_x, count_y = self._counts
        return count_x * count_y

    def origins(self, device: torch.device) -> torch.Tensor:
        count_x, count_y = self._counts
        x = torch.linspace(-self.size[0] / 2.0, self.size[0] / 2.0, count_x, device=device)
        y = torch.linspace(-self.size[1] / 2.0, self.size[1] / 2.0, count_y, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        origins = torch.zeros((self.ray_count, 3), dtype=torch.float32, device=device)
        origins[:, 0] = (grid_x.flatten() + self.offset[0]).float()
        origins[:, 1] = (grid_y.flatten() + self.offset[1]).float()
        return origins.contiguous()

    def directions(self, device: torch.device) -> torch.Tensor:
        direction = torch.tensor(self.direction, dtype=torch.float32, device=device)
        direction = direction / torch.linalg.vector_norm(direction)
        return direction.expand(self.ray_count, 3).clone().contiguous()


@dataclass(frozen=True)
class RayCameraPatternCfg:
    """Unit ray directions for a pinhole camera in an explicit local frame."""

    width: int
    height: int
    fovy: float
    forward: Sequence[float] = (1.0, 0.0, 0.0)
    up: Sequence[float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("RayCameraPatternCfg width must be an integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("RayCameraPatternCfg height must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RayCameraPatternCfg width and height must be positive")
        if isinstance(self.fovy, bool) or not isinstance(self.fovy, (int, float)):
            raise TypeError("RayCameraPatternCfg fovy must be a real number")
        if not math.isfinite(float(self.fovy)) or not 0.0 < float(self.fovy) < 180.0:
            raise ValueError("RayCameraPatternCfg fovy must be in (0, 180) degrees")
        forward = _real_sequence(self.forward, "RayCameraPatternCfg forward", 3)
        up = _real_sequence(self.up, "RayCameraPatternCfg up", 3)
        forward_tensor = torch.tensor(forward, dtype=torch.float64)
        up_tensor = torch.tensor(up, dtype=torch.float64)
        if (
            torch.linalg.vector_norm(forward_tensor).item() == 0.0
            or torch.linalg.vector_norm(up_tensor).item() == 0.0
            or abs(float(torch.dot(forward_tensor, up_tensor).item())) > 1.0e-6
        ):
            raise ValueError("RayCameraPatternCfg forward and up must be nonzero and orthogonal")
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "up", up)

    @property
    def ray_count(self) -> int:
        return self.width * self.height

    def origins(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((self.ray_count, 3), dtype=torch.float32, device=device)

    def directions(self, device: torch.device) -> torch.Tensor:
        forward = torch.tensor(self.forward, dtype=torch.float32, device=device)
        up = torch.tensor(self.up, dtype=torch.float32, device=device)
        right = torch.linalg.cross(forward, up)
        right = right / torch.linalg.vector_norm(right)
        focal = (self.height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)
        x = ((torch.arange(self.width, device=device) + 0.5) - self.width / 2.0) / focal
        y = ((torch.arange(self.height, device=device) + 0.5) - self.height / 2.0) / focal
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        directions = (
            forward[None, None, :]
            + grid_x[..., None] * right[None, None, :]
            - grid_y[..., None] * up[None, None, :]
        )
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        return directions.reshape(-1, 3).contiguous()


RayPattern = RayRingPatternCfg | RayGridPatternCfg | RayCameraPatternCfg
RayAlignment = Literal["body", "yaw", "world"]


@dataclass(frozen=True)
class RayPatternCfg:
    """Declare one observation-owned ray pattern and its attachment frame.

    ``caster`` is a public UniSim ray-caster factory name, not an imported
    plugin class. ``attachment_body_name`` is a public full-scene body identity
    from ``RayCollisionDescription.scene.body_names``. The local pattern and
    sensor offset are transformed by the selected attachment alignment.
    """

    entity_name: str
    pattern: RayPattern
    max_distance: float
    attachment_body_name: str | None = None
    attachment_site_name: str | None = None
    alignment: RayAlignment = "body"
    caster: str = "uni_ray"
    sensor_offset: Sequence[float] = (0.0, 0.0, 0.0)
    min_distance: float = 0.0
    include_normal: bool = False
    include_hit_point: bool = False
    sync_geometry: bool = False
    allowed_geom_names: Sequence[str] | None = None
    allowed_tags: Sequence[str] | None = None
    excluded_body_names: Sequence[str] | None = None
    excluded_ancestor_body_names: Sequence[str] | None = None
    caster_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise TypeError("RayPatternCfg entity_name must be a non-empty string")
        if not isinstance(
            self.pattern,
            (RayRingPatternCfg, RayGridPatternCfg, RayCameraPatternCfg),
        ):
            raise TypeError(
                "RayPatternCfg pattern must be RayRingPatternCfg, RayGridPatternCfg, "
                "or RayCameraPatternCfg"
            )
        if self.alignment not in ("body", "yaw", "world"):
            raise ValueError("RayPatternCfg alignment must be one of 'body', 'yaw', or 'world'")
        for name, value in (
            ("max_distance", self.max_distance),
            ("min_distance", self.min_distance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"RayPatternCfg {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"RayPatternCfg {name} must be finite")
        if self.max_distance <= 0.0:
            raise ValueError("RayPatternCfg max_distance must be positive")
        if self.min_distance < 0.0 or self.min_distance >= self.max_distance:
            raise ValueError(
                "RayPatternCfg min_distance must satisfy 0 <= min_distance < max_distance"
            )
        if (self.attachment_body_name is None) == (self.attachment_site_name is None):
            raise ValueError(
                "RayPatternCfg requires exactly one of attachment_body_name or attachment_site_name"
            )
        for attachment_value in (
            self.attachment_body_name,
            self.attachment_site_name,
        ):
            if attachment_value is not None and (
                not isinstance(attachment_value, str) or not attachment_value
            ):
                raise TypeError("RayPatternCfg attachment names must be non-empty strings")
        if isinstance(self.sensor_offset, (str, bytes)) or not isinstance(
            self.sensor_offset, Sequence
        ):
            raise TypeError("RayPatternCfg sensor_offset must be a sequence of three real numbers")
        object.__setattr__(
            self,
            "sensor_offset",
            _real_sequence(self.sensor_offset, "RayPatternCfg sensor_offset", 3),
        )
        if not isinstance(self.include_normal, bool):
            raise TypeError("RayPatternCfg include_normal must be a boolean")
        if not isinstance(self.include_hit_point, bool):
            raise TypeError("RayPatternCfg include_hit_point must be a boolean")
        if not isinstance(self.sync_geometry, bool):
            raise TypeError("RayPatternCfg sync_geometry must be a boolean")
        if not isinstance(self.caster, str) or not self.caster:
            raise TypeError("RayPatternCfg caster must be a non-empty factory name")
        if not isinstance(self.caster_options, dict):
            raise TypeError("RayPatternCfg caster_options must be a mapping")
        if "collision" in self.caster_options:
            raise ValueError(
                "RayPatternCfg caster_options cannot override the public collision descriptor"
            )
        for name in (
            "allowed_geom_names",
            "allowed_tags",
            "excluded_body_names",
            "excluded_ancestor_body_names",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"RayPatternCfg {name} must be a string sequence or None")
            values = tuple(value)
            if any(not isinstance(item, str) or not item for item in values):
                raise TypeError(f"RayPatternCfg {name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"RayPatternCfg {name} must be unique: {values}")
            object.__setattr__(self, name, values)


class RayScan(ManagerTermBase):
    """Bind one tensor caster and publish a ray observation carrier."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        pattern = cfg.params.get("ray_pattern")
        if not isinstance(pattern, RayPatternCfg):
            raise TypeError(
                "RayScan requires params['ray_pattern'] to be a RayPatternCfg; "
                f"got {type(pattern).__name__}"
            )
        self.ray_pattern = pattern
        output_width = 1
        if pattern.include_normal:
            output_width += 3
        if pattern.include_hit_point:
            output_width += 3
        self.observation_width = pattern.pattern.ray_count * output_width
        self._device = torch.device(env.device)
        self._entity_name = pattern.entity_name
        self._caster: Any = None
        self._bound = False
        self._closed = False
        self._distance: torch.Tensor | None = None
        self._normal: torch.Tensor | None = None
        self._hit_point: torch.Tensor | None = None
        self._observation: torch.Tensor | None = None
        self._ray_filter: RayTraceFilter | None = None
        self._outputs = RayTraceOutputs()
        self._attachment_body_index: int | None = None
        self._attachment_body_name: str | None = None
        self._attachment_site_local_pos: torch.Tensor | None = None
        self._attachment_site_local_quat: torch.Tensor | None = None
        self._local_origins: torch.Tensor | None = None
        self._local_directions: torch.Tensor | None = None
        self._ray_origins: torch.Tensor | None = None
        self._ray_directions: torch.Tensor | None = None
        self._pose_sources: dict[str, tuple[tuple[str, ...], torch.Tensor]] = {}
        self._pose_body_names: tuple[str, ...] = ()
        self._pose_pos: torch.Tensor | None = None
        self._pose_quat: torch.Tensor | None = None

    def bind_read_phase(self) -> None:
        """Materialize and bind the caster after packed pose reads compile."""
        if self._bound or self._closed:
            return
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None:
            raise NotImplementedError("RayScan requires the Manager packed Torch read plan")
        try:
            collision = self._env.backend.get_ray_collision_description()
            scene = collision.scene
            attachment_body_name = self.ray_pattern.attachment_body_name
            if self.ray_pattern.attachment_site_name is not None:
                (site_frame,) = self._env.backend.get_ray_site_frames(
                    (self.ray_pattern.attachment_site_name,)
                )
                attachment_body_name = site_frame.body_name
                self._attachment_site_local_pos = torch.tensor(
                    site_frame.pos, dtype=torch.float32, device=self._device
                )
                self._attachment_site_local_quat = torch.tensor(
                    site_frame.quat, dtype=torch.float32, device=self._device
                )
            assert attachment_body_name is not None
            self._attachment_body_name = attachment_body_name
            if attachment_body_name not in scene.body_names:
                raise KeyError(
                    f"ray pattern attachment body {attachment_body_name!r} is not in "
                    f"{scene.body_names}"
                )
            self._attachment_body_index = tuple(scene.body_names).index(attachment_body_name)
            self._pose_sources = self._resolve_pose_sources(collision)
            self._validate_pose_coverage(collision)
            self._pose_body_names = tuple(scene.body_names)
            self._ray_filter = self._resolve_ray_filter(collision)
            caster = _create_ray_caster(
                self.ray_pattern.caster,
                num_envs=self._env.num_envs,
                num_rays=self.ray_pattern.pattern.ray_count,
                device=self._device,
                collision=collision,
                **self.ray_pattern.caster_options,
            )
            try:
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
                if self.ray_pattern.min_distance != 0.0 and not capabilities.supports_min_distance:
                    raise NotImplementedError(
                        f"RayScan caster '{self.ray_pattern.caster}' does not support min_distance"
                    )
                if self._ray_filter.has_constraints() and not capabilities.supports_static_filter:
                    raise NotImplementedError(
                        f"RayScan caster '{self.ray_pattern.caster}' does not support "
                        "static ray filters"
                    )
                if self.ray_pattern.include_normal and not capabilities.supports_tensor_normal:
                    raise NotImplementedError(
                        f"RayScan caster '{self.ray_pattern.caster}' does not support tensor normals"
                    )
                if self.ray_pattern.include_hit_point and not capabilities.supports_hit_point:
                    raise NotImplementedError(
                        f"RayScan caster '{self.ray_pattern.caster}' does not support "
                        "tensor hit points"
                    )
                if self.ray_pattern.sync_geometry and not (
                    capabilities.supports_geometry_sync
                    and capabilities.supports_tensor_geometry_sync
                ):
                    raise NotImplementedError(
                        f"RayScan caster '{self.ray_pattern.caster}' does not support "
                        "tensor geometry sync"
                    )
                caster.materialize(scene)
                caster.bind_pose_layout(tuple(scene.body_names))
            except Exception:
                caster.close()
                raise
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(
                f"RayScan caster '{self.ray_pattern.caster}' read-phase binding failed: {exc}"
            ) from exc
        self._caster = caster
        self._outputs = RayTraceOutputs(
            normal=self.ray_pattern.include_normal,
            hit_point=self.ray_pattern.include_hit_point,
        )
        self._distance = torch.empty(
            (self._env.num_envs, self.ray_pattern.pattern.ray_count),
            dtype=torch.float32,
            device=self._device,
        )
        if self.ray_pattern.include_normal or self.ray_pattern.include_hit_point:
            output_width = 1
            if self.ray_pattern.include_normal:
                output_width += 3
                self._normal = torch.empty(
                    (self._env.num_envs, self.ray_pattern.pattern.ray_count, 3),
                    dtype=torch.float32,
                    device=self._device,
                )
            if self.ray_pattern.include_hit_point:
                output_width += 3
                self._hit_point = torch.empty(
                    (self._env.num_envs, self.ray_pattern.pattern.ray_count, 3),
                    dtype=torch.float32,
                    device=self._device,
                )
            self._observation = torch.empty(
                (self._env.num_envs, self.ray_pattern.pattern.ray_count, output_width),
                dtype=torch.float32,
                device=self._device,
            )
        self._local_origins = self.ray_pattern.pattern.origins(self._device)
        self._local_origins = self._local_origins + torch.tensor(
            self.ray_pattern.sensor_offset,
            dtype=torch.float32,
            device=self._device,
        )
        self._local_directions = self.ray_pattern.pattern.directions(self._device)
        self._ray_origins = torch.empty(
            (self._env.num_envs, self.ray_pattern.pattern.ray_count, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._ray_directions = torch.empty_like(self._ray_origins)
        self._pose_pos = torch.zeros(
            (self._env.num_envs, len(scene.body_names), 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._pose_quat = torch.zeros(
            (self._env.num_envs, len(scene.body_names), 4),
            dtype=torch.float32,
            device=self._device,
        )
        self._pose_quat[..., 0] = 1.0
        if self.ray_pattern.sync_geometry:
            self._sync_geometry()
        self._bound = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        ray_pattern: RayPatternCfg | None = None,
    ) -> torch.Tensor:
        if ray_pattern is not None and ray_pattern is not self.ray_pattern:
            raise ValueError(
                "RayScan was bound to a different ray_pattern; manager term parameters "
                "must remain immutable"
            )
        if not self._bound or self._caster is None or self._distance is None:
            raise NotImplementedError("RayScan must bind its read phase before compute")
        read_plan = getattr(env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            raise RuntimeError("RayScan requires a refreshed packed read phase")
        pose = self._refresh_full_pose(read_plan)
        try:
            self._refresh_attachment_rays(pose)
            self._caster.update_pose_tensor(pose)
            result = self._caster.trace_tensor(
                self._ray_origins,
                self._ray_directions,
                self.ray_pattern.max_distance,
                outputs=self._outputs,
                min_distance=self.ray_pattern.min_distance,
                ray_filter=self._ray_filter,
            )
        except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise type(exc)(f"RayScan tensor trace failed: {exc}") from exc
        if self._observation is None:
            self._distance.copy_(result.distance)
            return self._distance
        output_slice = 1
        self._observation[..., 0].copy_(result.distance)
        if self._normal is not None:
            self._normal.copy_(result.normal)
            self._observation[..., output_slice : output_slice + 3].copy_(self._normal)
            output_slice += 3
        if self._hit_point is not None:
            self._hit_point.copy_(result.hit_point)
            self._observation[..., output_slice : output_slice + 3].copy_(self._hit_point)
        return self._observation.reshape(self._env.num_envs, -1)

    def _refresh_attachment_rays(self, pose: Any) -> None:
        """Transform the local ray pattern by the selected attachment frame."""

        assert self._attachment_body_index is not None
        assert self._local_origins is not None
        assert self._local_directions is not None
        assert self._ray_origins is not None
        assert self._ray_directions is not None
        index = self._attachment_body_index
        position = pose.pos_w[:, index, :].unsqueeze(1)
        quaternion = pose.quat_w[:, index, :]
        if self._attachment_site_local_pos is not None:
            assert self._attachment_site_local_quat is not None
            quaternion = _quat_multiply_wxyz(
                quaternion, self._attachment_site_local_quat.expand_as(quaternion)
            )
            position = position + _quat_rotate_wxyz(
                pose.quat_w[:, index, :], self._attachment_site_local_pos.expand(1, 3)
            )[:, 0, :].unsqueeze(1)
        rotation = _attachment_rotation_wxyz(quaternion, self.ray_pattern.alignment)
        self._ray_origins.copy_(
            position + torch.einsum("bij,rj->bri", rotation, self._local_origins)
        )
        self._ray_directions.copy_(torch.einsum("bij,rj->bri", rotation, self._local_directions))

    def _resolve_pose_sources(
        self, collision: Any
    ) -> dict[str, tuple[tuple[str, ...], torch.Tensor]]:
        """Map public full-scene body identities to entity-local pose reads."""

        scene = collision.scene
        body_names = tuple(scene.body_names)
        body_ids = self._env.backend.get_body_ids(body_names)
        collision_index_by_id = {
            int(body_id): index for index, body_id in enumerate(body_ids.tolist())
        }
        sources: dict[str, tuple[tuple[str, ...], torch.Tensor]] = {}
        entities = getattr(self._env.scene, "entities", {})
        for entity_name, entity in entities.items():
            try:
                local_names = tuple(entity.body_names)
            except NotImplementedError:
                continue
            if not local_names:
                continue
            local_ids = self._env.backend.get_body_ids(local_names)
            selected_names: list[str] = []
            destinations: list[int] = []
            for local_name, local_id in zip(local_names, local_ids.tolist(), strict=True):
                collision_index = collision_index_by_id.get(int(local_id))
                if collision_index is None:
                    continue
                selected_names.append(local_name)
                destinations.append(collision_index)
            if not selected_names:
                continue
            sources[entity_name] = (
                tuple(selected_names),
                torch.tensor(destinations, dtype=torch.long, device=self._device),
            )
        return sources

    def _validate_pose_coverage(self, collision: Any) -> None:
        """Require entity reads for every non-root full-scene body."""

        scene = collision.scene
        source = self._pose_sources.get(self._entity_name)
        attachment_name = self._attachment_body_name
        assert attachment_name is not None
        if source is None or attachment_name not in source[0]:
            raise KeyError(
                f"ray pattern attachment body {attachment_name!r} is not owned by "
                f"entity {self._entity_name!r}"
            )
        covered = {
            body_name for local_names, _ in self._pose_sources.values() for body_name in local_names
        }
        uncovered = set(scene.body_names) - covered
        root_name = scene.body_names[0]
        if uncovered and (
            uncovered != {root_name}
            or (collision.body_parent_ids is not None and int(collision.body_parent_ids[0]) != -1)
        ):
            raise NotImplementedError(
                "RayScan full-scene pose aggregation requires every non-root body to "
                "belong to a scene entity with a packed body read; uncovered bodies: "
                f"{sorted(uncovered)}"
            )

    def _refresh_full_pose(self, read_plan: Any) -> RayPoseTensorView:
        """Assemble the full public body-axis pose from entity-local read views."""

        assert self._pose_body_names
        assert self._pose_pos is not None
        assert self._pose_quat is not None
        self._pose_pos.zero_()
        self._pose_quat.zero_()
        self._pose_quat[..., 0] = 1.0
        for entity_name, (local_names, destinations) in self._pose_sources.items():
            source = read_plan.ray_pose_view(entity_name, local_names)
            self._pose_pos[:, destinations, :] = source.pos_w
            self._pose_quat[:, destinations, :] = source.quat_w
        return RayPoseTensorView(
            body_names=self._pose_body_names,
            pos_w=self._pose_pos,
            quat_w=self._pose_quat,
        )

    def _resolve_ray_filter(self, collision: Any) -> RayTraceFilter:
        """Resolve public names/tags to the compact static filter on the cold path."""

        scene = collision.scene
        geom_names = tuple(collision.geom_names)
        body_names = tuple(scene.body_names)
        allowed_geom_ids = None
        if self.ray_pattern.allowed_geom_names is not None:
            unknown = set(self.ray_pattern.allowed_geom_names) - set(geom_names)
            if unknown:
                raise KeyError(f"ray pattern geom names are missing: {sorted(unknown)}")
            allowed_geom_ids = tuple(
                geom_names.index(name) for name in self.ray_pattern.allowed_geom_names
            )
        excluded_body_ids = None
        if self.ray_pattern.excluded_body_names is not None:
            unknown = set(self.ray_pattern.excluded_body_names) - set(body_names)
            if unknown:
                raise KeyError(f"ray pattern body names are missing: {sorted(unknown)}")
            excluded_body_ids = tuple(
                body_names.index(name) for name in self.ray_pattern.excluded_body_names
            )
        excluded_ancestor_body_ids = None
        if self.ray_pattern.excluded_ancestor_body_names is not None:
            unknown = set(self.ray_pattern.excluded_ancestor_body_names) - set(body_names)
            if unknown:
                raise KeyError(f"ray pattern ancestor body names are missing: {sorted(unknown)}")
            excluded_ancestor_body_ids = tuple(
                body_names.index(name) for name in self.ray_pattern.excluded_ancestor_body_names
            )
        ray_filter = RayTraceFilter(
            allowed_geom_ids=allowed_geom_ids,
            allowed_tags=self.ray_pattern.allowed_tags,
            excluded_body_ids=excluded_body_ids,
            excluded_ancestor_body_ids=excluded_ancestor_body_ids,
        )
        ray_filter.geom_mask(collision)
        return ray_filter

    def close(self) -> None:
        """Release caster resources before the scene read plan closes."""
        self._closed = True
        caster, self._caster = self._caster, None
        if caster is not None:
            caster.close()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Retain caster state; observations refresh from the next read phase."""
        del env_ids
        if self._bound and self.ray_pattern.sync_geometry:
            self._sync_geometry()

    def _sync_geometry(self) -> None:
        """Publish backend-owned per-row geometry to the caster on reset paths."""

        if self._caster is None:
            return
        state = self._env.backend.get_ray_geometry_state()
        self._caster.update_geometry_tensor(
            RayGeometryTensorView(
                geom_sizes=torch.tensor(state.geom_sizes, dtype=torch.float32, device=self._device),
                geom_local_pos=torch.tensor(
                    state.geom_local_pos, dtype=torch.float32, device=self._device
                ),
                geom_local_quat=torch.tensor(
                    state.geom_local_quat, dtype=torch.float32, device=self._device
                ),
                geom_enabled=torch.tensor(state.geom_enabled, device=self._device),
                geom_mesh_ids=torch.tensor(
                    state.geom_mesh_ids, dtype=torch.int32, device=self._device
                ),
            )
        )


__all__ = [
    "RayAlignment",
    "RayCameraPatternCfg",
    "RayGridPatternCfg",
    "RayPatternCfg",
    "RayRingPatternCfg",
    "RayScan",
]


def _attachment_rotation_wxyz(quaternion: torch.Tensor, alignment: RayAlignment) -> torch.Tensor:
    """Return one ``[batch, 3, 3]`` attachment rotation matrix."""

    batch = quaternion.shape[0]
    identity = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    if alignment == "world":
        return identity.unsqueeze(0).expand(batch, 3, 3)

    basis = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    rotated_basis = _quat_rotate_wxyz(quaternion, basis)
    if alignment == "body":
        return rotated_basis.transpose(1, 2)

    forward = rotated_basis[:, 0, :].clone()
    forward[:, 2] = 0.0
    forward_norm = torch.linalg.vector_norm(forward, dim=1)
    singular = forward_norm < 1.0e-3
    left = rotated_basis[:, 1, :].clone()
    left[:, 2] = 0.0
    left_norm = torch.linalg.vector_norm(left, dim=1).clamp(min=1.0e-6)
    left = left / left_norm.unsqueeze(1)
    fallback = torch.stack((left[:, 1], -left[:, 0], torch.zeros_like(left[:, 0])), dim=1)
    forward = torch.where(
        singular.unsqueeze(1),
        fallback,
        forward / forward_norm.clamp(min=1.0e-6).unsqueeze(1),
    )
    up = torch.tensor((0.0, 0.0, 1.0), dtype=quaternion.dtype, device=quaternion.device).expand(
        batch, 3
    )
    left = torch.stack((-forward[:, 1], forward[:, 0], torch.zeros_like(forward[:, 0])), dim=1)
    return torch.stack((forward, left, up), dim=2)


def _quat_multiply_wxyz(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose batched ``wxyz`` quaternions with matching leading dimensions."""

    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate shared local vectors by full-batch public ``wxyz`` quaternions."""

    qvec = quaternion[:, None, 1:]
    scalar = quaternion[:, None, 0:1]
    cross = 2.0 * torch.linalg.cross(qvec, vector[None, :, :])
    return vector[None, :, :] + scalar * cross + torch.linalg.cross(qvec, cross)


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

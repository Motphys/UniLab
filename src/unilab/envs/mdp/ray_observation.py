"""Tensor-native ray-pattern observation terms for the Manager runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch
from unisim.backend.base import torch_device_matches
from unisim.ray_query import RayPoseTensorView, RayTraceFilter, RayTraceOutputs

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(frozen=True)
class RayPatternCfg:
    """Declare one observation-owned ray pattern.

    ``caster`` is a public UniSim ray-caster factory name, not an imported
    plugin class. ``attachment_body_name`` is a public full-scene body identity
    from ``RayCollisionDescription.scene.body_names``; the local pattern and
    sensor offset are transformed by that body pose.
    """

    entity_name: str
    attachment_body_name: str
    num_rays: int
    pattern_start: float
    pattern_stop: float
    max_distance: float
    pattern_pitch: float = 0.0
    caster: str = "uni_ray"
    sensor_offset: Sequence[float] = (0.0, 0.0, 0.0)
    min_distance: float = 0.0
    include_normal: bool = False
    allowed_geom_names: Sequence[str] | None = None
    allowed_tags: Sequence[str] | None = None
    excluded_body_names: Sequence[str] | None = None
    excluded_ancestor_body_names: Sequence[str] | None = None
    caster_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise TypeError("RayPatternCfg entity_name must be a non-empty string")
        if isinstance(self.num_rays, bool) or not isinstance(self.num_rays, int):
            raise TypeError("RayPatternCfg num_rays must be an integer")
        if self.num_rays <= 0:
            raise ValueError("RayPatternCfg num_rays must be positive")
        for name, value in (
            ("pattern_start", self.pattern_start),
            ("pattern_stop", self.pattern_stop),
            ("pattern_pitch", self.pattern_pitch),
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
        if not isinstance(self.attachment_body_name, str) or not self.attachment_body_name:
            raise TypeError("RayPatternCfg attachment_body_name must be a non-empty string")
        if isinstance(self.sensor_offset, (str, bytes)) or not isinstance(
            self.sensor_offset, Sequence
        ):
            raise TypeError("RayPatternCfg sensor_offset must be a sequence of three real numbers")
        offset = tuple(self.sensor_offset)
        if len(offset) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in offset
        ):
            raise TypeError("RayPatternCfg sensor_offset must be a sequence of three real numbers")
        if any(not math.isfinite(float(value)) for value in offset):
            raise ValueError("RayPatternCfg sensor_offset must be finite")
        object.__setattr__(self, "sensor_offset", tuple(float(value) for value in offset))
        if not isinstance(self.include_normal, bool):
            raise TypeError("RayPatternCfg include_normal must be a boolean")
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
    """Bind one tensor caster and publish a distance/normal observation carrier."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        pattern = cfg.params.get("ray_pattern")
        if not isinstance(pattern, RayPatternCfg):
            raise TypeError(
                "RayScan requires params['ray_pattern'] to be a RayPatternCfg; "
                f"got {type(pattern).__name__}"
            )
        self.ray_pattern = pattern
        self.observation_width = pattern.num_rays * (4 if pattern.include_normal else 1)
        self._device = torch.device(env.device)
        self._entity_name = pattern.entity_name
        self._caster: Any = None
        self._bound = False
        self._closed = False
        self._distance: torch.Tensor | None = None
        self._normal: torch.Tensor | None = None
        self._observation: torch.Tensor | None = None
        self._ray_filter: RayTraceFilter | None = None
        self._outputs = RayTraceOutputs()
        self._attachment_body_index: int | None = None
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
            attachment_name = self.ray_pattern.attachment_body_name
            if attachment_name not in scene.body_names:
                raise KeyError(
                    f"ray pattern attachment body {attachment_name!r} is not in {scene.body_names}"
                )
            self._attachment_body_index = tuple(scene.body_names).index(attachment_name)
            self._pose_sources = self._resolve_pose_sources(collision)
            self._validate_pose_coverage(collision)
            self._pose_body_names = tuple(scene.body_names)
            self._ray_filter = self._resolve_ray_filter(collision)
            caster = _create_ray_caster(
                self.ray_pattern.caster,
                num_envs=self._env.num_envs,
                num_rays=self.ray_pattern.num_rays,
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
        self._outputs = RayTraceOutputs(normal=self.ray_pattern.include_normal)
        self._distance = torch.empty(
            (self._env.num_envs, self.ray_pattern.num_rays),
            dtype=torch.float32,
            device=self._device,
        )
        if self.ray_pattern.include_normal:
            self._normal = torch.empty(
                (self._env.num_envs, self.ray_pattern.num_rays, 3),
                dtype=torch.float32,
                device=self._device,
            )
            self._observation = torch.empty(
                (self._env.num_envs, self.ray_pattern.num_rays, 4),
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
        pitch = float(self.ray_pattern.pattern_pitch)
        self._local_directions = torch.stack(
            (
                torch.cos(angles) * math.cos(pitch),
                torch.sin(angles) * math.cos(pitch),
                torch.full_like(angles, math.sin(pitch)),
            ),
            dim=-1,
        ).contiguous()
        self._local_origins = torch.zeros(
            (self.ray_pattern.num_rays, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._local_origins[:] = torch.tensor(
            self.ray_pattern.sensor_offset,
            dtype=torch.float32,
            device=self._device,
        )
        self._ray_origins = torch.empty(
            (self._env.num_envs, self.ray_pattern.num_rays, 3),
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
        assert self._normal is not None
        self._observation[..., 0].copy_(result.distance)
        self._normal.copy_(result.normal)
        self._observation[..., 1:4].copy_(self._normal)
        return self._observation.reshape(self._env.num_envs, -1)

    def _refresh_attachment_rays(self, pose: Any) -> None:
        """Transform the local ray pattern by the body-only attachment pose."""

        assert self._attachment_body_index is not None
        assert self._local_origins is not None
        assert self._local_directions is not None
        assert self._ray_origins is not None
        assert self._ray_directions is not None
        index = self._attachment_body_index
        position = pose.pos_w[:, index, :].unsqueeze(1)
        quaternion = pose.quat_w[:, index, :]
        self._ray_origins.copy_(position + _quat_rotate_wxyz(quaternion, self._local_origins))
        self._ray_directions.copy_(_quat_rotate_wxyz(quaternion, self._local_directions))

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
        attachment_name = self.ray_pattern.attachment_body_name
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


__all__ = ["RayPatternCfg", "RayScan"]


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

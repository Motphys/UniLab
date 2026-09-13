"""Deterministic generated fixture for same-layout SimTool tool variants.

The generated sources are deliberately internal: they stand in for the private
600-tool catalog without turning a synthetic workload into a production task
registration or a public asset promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.base.variants import (
    FixedModelVariantCatalogCfg,
    FixedModelVariantCfg,
)
from unilab.envs import ManagerBasedRlEnvCfg, mdp
from unilab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from unilab.managers.scene_entity_config import SceneEntityCfg

_TETRAHEDRON_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


@dataclass(frozen=True)
class _VariantSpec:
    """Parameters that vary while preserving the representative public layout."""

    name: str
    mass_kg: float
    mesh_scale: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class RepresentativeSimToolRealSourceSet:
    """Materialized absolute sources ready for UniSim backend consumption."""

    model_files: tuple[Path, ...]


def write_representative_simtool_real_sources(
    output_dir: str | Path,
    *,
    variant_count: int = 3,
) -> RepresentativeSimToolRealSourceSet:
    """Write deterministic same-layout MJCF variants and their shared mesh."""

    if isinstance(variant_count, bool) or not isinstance(variant_count, int):
        raise TypeError("variant_count must be an integer")
    if variant_count < 2:
        raise ValueError("variant_count must be at least 2")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    mesh_file = root / "simtool_handle.obj"
    mesh_file.write_text(_TETRAHEDRON_OBJ, encoding="utf-8")

    variants: list[_VariantSpec] = []
    model_files: list[Path] = []
    for index in range(variant_count):
        name = f"tool_{index:04d}"
        variant = _VariantSpec(
            name=name,
            mass_kg=0.4 + 0.25 * index,
            mesh_scale=(1.0 + 0.08 * index, 0.9 + 0.06 * index, 0.8 + 0.05 * index),
            rgba=(0.1 + 0.2 * index % 1.0, 0.8 - 0.15 * index, 0.2 + 0.25 * index, 1.0),
        )
        model_file = root / f"{name}.xml"
        model_file.write_text(_variant_xml(variant, mesh_file), encoding="utf-8")
        variants.append(variant)
        model_files.append(model_file)

    return RepresentativeSimToolRealSourceSet(model_files=tuple(model_files))


def build_representative_simtool_real_env_cfg(
    sources: RepresentativeSimToolRealSourceSet,
) -> ManagerBasedRlEnvCfg:
    """Build a direct Manager-Based config without registering a synthetic task."""

    tool_joints = SceneEntityCfg("tool", joint_names=("tool_pitch",))
    tool_body = SceneEntityCfg("tool", body_names=("tool",))
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file=str(sources.model_files[0]),
            entities={
                "tool": EntityCfg(
                    root_body_name="tool",
                    joint_names=("tool_pitch",),
                    body_names=("tool",),
                    geom_names=("floor", "handle"),
                    actuator_names=("tool_motor",),
                )
            },
        ),
        fixed_model_variants=FixedModelVariantCatalogCfg(
            variants=tuple(
                FixedModelVariantCfg(path.stem, str(path)) for path in sources.model_files
            ),
        ),
        sim_dt=0.002,
        ctrl_dt=0.01,
        max_episode_seconds=1.0,
        seed=7,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=mdp.joint_pos_rel,
                        params={"asset_cfg": tool_joints},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=mdp.joint_vel_rel,
                        params={"asset_cfg": tool_joints},
                    ),
                }
            )
        },
        actions={
            "effort": mdp.JointEffortActionCfg(
                entity_name="tool",
                actuator_names=("tool_pitch",),
                scale=0.5,
            )
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=mdp.reset_scene_to_default,
                mode="reset",
            ),
            "randomize_body_mass_inertia": EventTermCfg(
                func=mdp.randomize_body_mass_inertia,
                mode="reset",
                params={
                    "asset_cfg": tool_body,
                    "scale_range": (0.95, 1.05),
                },
            ),
        },
        rewards={"alive": RewardTermCfg(func=mdp.is_alive, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        policy_observation_group="policy",
    )


def _variant_xml(
    variant: _VariantSpec,
    mesh_file: Path,
) -> str:
    scale = " ".join(f"{value:.6f}" for value in variant.mesh_scale)
    rgba = " ".join(f"{value:.6f}" for value in variant.rgba)
    mass = f"{variant.mass_kg:.6f}"
    return f"""<mujoco model="SimToolRealRepresentative">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="tool_collision" file="{mesh_file.name}" scale="{scale}"/>
    <material name="tool_paint" rgba="{rgba}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="tool" pos="0 0 0.3">
      <joint name="tool_pitch" type="hinge" axis="0 1 0" damping="0.02" armature="0.01"/>
      <geom name="handle" type="mesh" mesh="tool_collision" mass="{mass}"
            material="tool_paint"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="tool_motor" joint="tool_pitch" ctrllimited="true" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

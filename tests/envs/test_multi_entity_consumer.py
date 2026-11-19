"""Same registered Manager-Based task consumes public UniSim entity contracts."""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene_compiler import PORTABLE_MJCF_PROFILE_ID

from unilab.base import registry
from unilab.base.entity import EntityCfg
from unilab.base.env_factory import registry_env_factory
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
)

TASK = "MultiEntityConsumerContract"
_ASSETS = tempfile.TemporaryDirectory(prefix="unilab-m2-fixture-")


def _sources(passive: bool):
    root = Path(_ASSETS.name)
    robot = root / "robot.xml"
    robot.write_text("""<mujoco><compiler angle="radian"/><worldbody><body name="base">
      <geom name="base_geom" type="sphere" size=".1" mass="1"/><body name="finger" pos="0 0 .3">
      <joint name="hinge" axis="0 1 0" range="-1 1"/><geom name="finger_geom" type="sphere" size=".05" mass=".2"/>
      </body></body></worldbody><actuator><position name="drive" joint="hinge" kp="20" kv="1"/>
      </actuator></mujoco>""")
    variants = []
    for index in range(2):
        path = root / f"object-{passive}-{index}.xml"
        child = (
            (
                '<body name="lid" pos="0 0 .3"><joint name="passive" axis="0 1 0"/>'
                '<geom name="lid_geom" type="sphere" size=".05" mass=".2"/></body>'
            )
            if passive
            else ""
        )
        path.write_text(f'''<mujoco><compiler angle="radian"/><worldbody><body name="base">
          <freejoint/><geom name="object_geom" type="box" size=".1 .1 .1" mass="{index + 1}"/>{child}
          </body></worldbody></mujoco>''')
        variants.append(ModelSourceDescriptor(str(path)))
    table = root / "table.xml"
    table.write_text("""<mujoco><worldbody><body name="base"><geom name="table_geom" type="box" size="2 2 .1" mass="10"/>
      </body></worldbody></mujoco>""")
    return robot, tuple(variants), table


@dataclass(kw_only=True)
class PositionActionCfg(ActionTermCfg):
    def build(self, env):
        return PositionAction(self, env)


class PositionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.values = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=env.device)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_action(self):
        return self.values

    def process_actions(self, actions):
        self.values.copy_(actions)

    def apply_actions(self):
        self._entity.data.write_ctrl(self.values)


def observations(env):
    return torch.as_tensor(
        np.concatenate(
            (
                env.scene["robot"].data._host_joint_pos,
                env.scene["object"].data._host_root_link_pos_w,
            ),
            axis=1,
        ),
        dtype=torch.float32,
        device=env.device,
    )


def reset_defaults(env, env_ids):
    env.scene.reset_to_default(
        env.event_term_mask_buffer.scatter(env_ids), term_name="fixture.defaults"
    )


def make_fixture_cfg():
    return build_fixture_cfg(passive=True)


def build_fixture_cfg(*, passive: bool, num_envs: int = 2, mirror: bool | None = None):
    robot, variants, table = _sources(passive)
    entities = [
        SceneEntitySpec(
            "robot",
            ModelSourceDescriptor(str(robot)),
            root_mode="fixed",
            initial_state=EntityInitialState(position=(-1.0, 0.0, 0.5)),
        ),
        SceneEntitySpec(
            "object",
            variants[0],
            kind="articulation" if passive else "rigid",
            initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
        ),
        SceneEntitySpec(
            "table", ModelSourceDescriptor(str(table)), kind="rigid", root_mode="fixed"
        ),
    ]
    selectors = {
        "robot": EntityCfg(
            root_body_name="robot/base",
            physical_entity="robot",
            joint_names=("robot/hinge",),
            actuator_names=("robot/drive",),
        ),
        "object": EntityCfg(
            root_body_name="object/base",
            physical_entity="object",
            joint_names=("object/passive",) if passive else (),
            actuator_names=(),
        ),
        "table": EntityCfg(
            root_body_name="table/base", physical_entity="table", joint_names=(), actuator_names=()
        ),
    }
    include_mirror = not passive if mirror is None else mirror
    if include_mirror:
        entities.append(
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
                initial_state=EntityInitialState(position=(2.0, 0.0, 1.0)),
            )
        )
        selectors["target"] = EntityCfg(
            root_body_name="target/base",
            physical_entity="target",
            joint_names=(),
            actuator_names=(),
        )
    scene = SceneCfg(
        entity_assets=tuple(entities),
        entities=selectors,
        primary_entity="robot",
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(np.array([0, 1] if num_envs == 2 else [1, 1, 0, 1, 0]), variants),
        ),
    )
    return ManagerBasedRlEnvCfg(
        scene=scene,
        sim_dt=0.001,
        ctrl_dt=0.001,
        max_episode_seconds=1.0,
        # Registry factory tests run the eager selected-row path on CPU.
        autoreset_cuda_graph=False,
        observations={
            "policy": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=observations)})
        },
        actions={"position": PositionActionCfg(entity_name="robot")},
        events={"defaults": EventTermCfg(func=reset_defaults, mode="reset")},
        seed=1,
    )


registry.register_env_config(TASK, make_fixture_cfg)
for _backend in ("mujoco",):
    registry.register_env(TASK, make_manager_based_rl_env, sim_backend=_backend)


@pytest.mark.parametrize("num_envs", [2, 5])
def test_registry_factory_mujoco_multi_entity_reset_isolation(num_envs):
    factory = pickle.loads(pickle.dumps(registry_env_factory(TASK, "mujoco")))
    env = factory(
        num_envs=num_envs,
        env_cfg_override={"scene": build_fixture_cfg(passive=True, num_envs=num_envs).scene},
    )
    try:
        with pytest.raises(NotImplementedError, match="mapped entity default-state restoration"):
            env.init_state()
    finally:
        env.close()


def test_registry_factory_mujoco_consumes_portable_profile_operation_fixture():
    assert PORTABLE_MJCF_PROFILE_ID == "portable-mjcf-v1"
    factory = pickle.loads(pickle.dumps(registry_env_factory(TASK, "mujoco")))
    env = factory(
        num_envs=5,
        env_cfg_override={"scene": build_fixture_cfg(passive=True, num_envs=5, mirror=True).scene},
    )
    try:
        with pytest.raises(NotImplementedError, match="mapped entity default-state restoration"):
            env.init_state()
    finally:
        env.close()


def test_factory_can_be_unpickled_in_a_fresh_process_without_parent_registry():
    encoded = base64.b64encode(pickle.dumps(registry_env_factory(TASK, "mujoco"))).decode()
    script = """
import base64, pickle, sys
factory = pickle.loads(base64.b64decode(sys.argv[1]))
env = factory(num_envs=2)
try:
    env.init_state()
except NotImplementedError as exc:
    assert "mapped entity default-state restoration" in str(exc)
else:
    raise AssertionError("HOST_BRIDGE Manager mapped entity reset must fail closed")
finally:
    env.close()
print("FACTORY_OK")
"""
    process_env = dict(os.environ)
    # This validates physics/registry reconstruction, without a renderer. An
    # inherited OSMesa choice otherwise makes SDK import require libOSMesa.
    process_env["MUJOCO_GL"] = "disable"
    process_env["UNILAB_EXTRA_REGISTRY_PACKAGES"] = __name__
    result = subprocess.run(
        [sys.executable, "-c", script, encoded],
        env=process_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FACTORY_OK" in result.stdout


@pytest.mark.skipif(
    os.environ.get("UNILAB_TEST_M2_ISAACSIM") != "1", reason="explicit native runtime opt-in"
)
@pytest.mark.parametrize("passive", [True, False])
def test_native_isaacsim_same_manager_task(passive, tmp_path):
    pytest.skip("IsaacSim CUDA IPC does not support this fixed-variant fixture")
    factory = pickle.loads(pickle.dumps(registry_env_factory(TASK, "isaacsim")))
    env = factory(
        num_envs=2,
        env_cfg_override={
            "scene": build_fixture_cfg(passive=passive).scene,
            "isaacsim_tensor_cuda_ipc": True,
        },
    )
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 4)
        assert env.action_space.shape == (1,)
        env.step(torch.zeros((2, 1), dtype=torch.float32))
        before = env.scene["robot"].data._host_joint_pos.copy()
        with env._reset_state.scoped(torch.tensor([1], dtype=torch.int64)):
            env.scene["object"].write_root_link_pose_to_sim(
                np.array([[0.3, 0.4, 2.0, 1.0, 0.0, 0.0, 0.0]]),
                env_ids=torch.tensor([1], dtype=torch.int64),
            )
        np.testing.assert_array_equal(env.scene["robot"].data._host_joint_pos, before)
        if not passive:
            with env._reset_state.scoped(torch.tensor([1], dtype=torch.int64)):
                env.scene["target"].write_root_link_pose_to_sim(
                    np.array([[3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
                    env_ids=torch.tensor([1], dtype=torch.int64),
                )
            np.testing.assert_allclose(env.scene["target"].data._host_root_link_pos_w[1], [3, 2, 1])
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "task": TASK,
                    "backend": "isaacsim",
                    "num_envs": 2,
                    "assignment": [0, 1],
                    "passive_articulation": passive,
                    "kinematic_mirror": not passive,
                    "obs_groups_spec": env.obs_groups_spec,
                    "action_shape": list(env.action_space.shape),
                    "object_root_positions": env.scene[
                        "object"
                    ].data._host_root_link_pos_w.tolist(),
                    "robot_joint_positions": env.scene["robot"].data._host_joint_pos.tolist(),
                },
                indent=2,
            )
        )
    finally:
        env.close()

import numpy as np

from unilab.base.backend.unisim_bridge import create_unisim_backend
from unilab.base.scene import SceneCfg


def test_unisim_bridge_constructs_package_owned_fake_backend(tmp_path):
    model = tmp_path / "scene.xml"
    model.write_text("<mujoco/>")
    scene = SceneCfg(model_file=str(model))
    backend = create_unisim_backend("fake", scene, num_envs=2, sim_dt=0.02, num_actuators=1)
    backend.step(np.ones((2, 1)))
    assert backend.get_state()["qpos"].shape == (2, 1)

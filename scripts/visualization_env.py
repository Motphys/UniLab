import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import time
import unilab.envs.locomotion.g1
import unilab.envs.locomotion.go1
import unilab.envs.locomotion.go2
from unilab.base import registry

parser = argparse.ArgumentParser(description="Visualize robot tasks")
parser.add_argument("--task", type=str, default="Go1JoystickFlatTerrain", help="Task name")
parser.add_argument("--backend", type=str, choices=["motrix", "mujoco"], default="mujoco", help="Physics backend")
args = parser.parse_args()

np.set_printoptions(precision=3, suppress=True, linewidth=500)

# Get model file from registered task
env_meta = registry._envs[args.task]
cfg = env_meta.env_cfg_cls()

if args.backend == "motrix":
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    model = mtx.load_model(cfg.model_file)
    model.options.timestep = 0.01

    # Create single env
    data = mtx.SceneData(model, batch=[1])

    # Get init state from keyframe
    kf = model.keyframes[0]
    kf.apply(data)

    # Launch renderer
    with RenderApp() as render:
        settings = RenderSettings.performance()
        settings.enable_shadow = True
        render.launch(model, batch=1, render_settings=settings)

        print("Press ESC to exit")

        first = True
        def physics_step():
            global first
            if first:
                print(f"step state - dof_pos[:7] : {data.dof_pos[0, :7]}")
                print(f"step state - dof_pos[7:] : {data.dof_pos[0, 7:]}")
                first = False

            model.step(data)

        def render_step():
            render.sync(data)

        mtx.run.render_loop(
            model.options.timestep,
            60.0,
            physics_step,
            render_step,
        )

elif args.backend == "mujoco":
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(cfg.model_file)
    model.opt.timestep = 0.01
    
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC to exit")
        first = True
        while viewer.is_running():
            step_start = time.time()

            if first:
                print(f"step state - qpos[:7] : {data.qpos[:7]}")
                print(f"step state - qpos[7:] : {data.qpos[7:]}")
                first = False

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

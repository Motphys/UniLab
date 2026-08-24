"""Interactive Manager-Based MuJoCo playback for A2Arm position-force CSE-PPO."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.cse_ppo import CSEOnPolicyRunner
from unilab.algos.rsl_rl import RslRlVecEnvWrapper
from unilab.base.backend import materialize_scene_visual_override
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.locomotion.a2arm.state import A2ArmPosForceState
from unilab.training import ensure_registries, parse_checkpoint_path
from unilab.visualization.a2arm_pos_force import (
    KEY_BACKSPACE,
    KEY_SPACE,
    clear_teleop_override,
    draw_markers,
    install_teleop_override,
    make_key_callback,
    make_teleop_from_state,
    print_legend,
)


def _select_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "training.device")
    if configured:
        return str(configured)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    return cast(dict[str, Any], resolved)


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="ppo_cse",
        scene_materializer=materialize_scene_visual_override,
    )


def _load_checkpoint(cfg: DictConfig) -> Path | None:
    path, _directory = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if path is None or not path.is_file():
        print(
            "[play] checkpoint not found; pass algo.load_run=<run directory or checkpoint> "
            "and optionally algo.checkpoint=<iteration or filename>"
        )
        return None
    keys = set(torch.load(path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in keys:
        print(f"[play] {path} is not a CSE-PPO checkpoint (keys={sorted(keys)}).")
        return None
    return path


def _print_force_estimate(runner: CSEOnPolicyRunner, env: Any, obs: torch.Tensor) -> None:
    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    pred = runner.actor_critic.estimator.predict(obs).detach().cpu().numpy()[0]
    # CSE target layout is [base velocity, EE sphere, EE force, base force].
    ee_est = pred[6:9] / 0.01
    base_est = pred[9:12] / 0.01
    print(
        f"[force] EE est={ee_est.round(1)} true={state.force_ee_world[0].round(1)} | "
        f"base est={base_est.round(1)} true={state.force_base_world[0].round(1)}"
    )


def play_interactive(cfg: DictConfig, device: str) -> None:
    import mujoco
    import mujoco.viewer

    checkpoint = _load_checkpoint(cfg)
    if checkpoint is None:
        return
    override = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=override,
        sim_backend="mujoco",
    )
    wrapped = RslRlVecEnvWrapper(env, device=device)
    runner = CSEOnPolicyRunner(wrapped, _algo_config_dict(cfg), log_dir=None, device=device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)
    env.set_autoreset(False)

    state: A2ArmPosForceState = env.command_manager.get_term("task_state")
    teleop = make_teleop_from_state(state)
    paused = {"value": False}
    reset_requested = {"value": False}
    show_range = {"value": True}

    def toggle_pause() -> None:
        paused["value"] = not paused["value"]
        print(f"[play] {'paused' if paused['value'] else 'resumed'}")

    def request_reset() -> None:
        reset_requested["value"] = True

    def toggle_range() -> None:
        show_range["value"] = not show_range["value"]
        print(f"[play] sample range {'on' if show_range['value'] else 'off'}")

    callback = make_key_callback(
        teleop,
        on_pause=toggle_pause,
        on_reset=request_reset,
        on_toggle_range=toggle_range,
    )

    obs_td, _info = wrapped.reset()
    obs = obs_td["actor"]
    install_teleop_override(env, teleop)
    model = env.get_playback_model(0)
    data = mujoco.MjData(model)
    print_legend()
    print(f"[play] loading {checkpoint}; close the viewer or press Escape to quit")
    diagnostics = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
        viewer.cam.distance = 2.5
        with torch.inference_mode():
            while viewer.is_running():
                started = time.perf_counter()
                if reset_requested["value"]:
                    reset_requested["value"] = False
                    obs = wrapped.reset()[0]["actor"]
                    teleop.reset()
                    install_teleop_override(env, teleop)
                    print("[play] reset")
                if not paused["value"]:
                    teleop.advance_forces()
                    install_teleop_override(env, teleop)
                    obs = wrapped.step(policy(obs))[0]["actor"]
                    diagnostics += 1
                    if diagnostics % 25 == 0:
                        _print_force_estimate(runner, env, obs)

                physics = env.get_physics_state_snapshot()[0]
                mujoco.mj_setState(model, data, physics, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(model, data)
                draw_markers(viewer, env, show_range=show_range["value"])
                viewer.sync()
                remaining = float(env.step_dt) - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    clear_teleop_override(env)
    env.close()


@hydra.main(version_base="1.3", config_path="../conf/ppo_cse", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    play_interactive(cfg, _select_device(cfg))


if __name__ == "__main__":
    main()

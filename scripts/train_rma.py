"""Run from project root; requires the original UniLab runtime and robot assets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rsl_rl"))
import argparse

import torch
from unilab_rma import ActorCriticRMA, AdaptationModule, RMAConfig
from unilab_rma.env_adapter import UniLabRMAEnv
from unilab_rma.evaluation_diagnostics import DEFAULT_TASK, RMA_TASKS, command_overrides
from unilab_rma.runner import train_student, train_teacher


def build_environment(num_envs, cfg, device, seed=42, *, task=DEFAULT_TASK, command=None):
    from hydra import compose, initialize_config_dir
    from unilab_rma.scene_contract import prepare_scene

    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.base.registry import ensure_registries

    if task not in RMA_TASKS:
        raise ValueError(f"Unsupported RMA task: {task}")
    overrides = [f"task={task}"]
    if command is not None:
        overrides.extend(command_overrides(command))
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base=None):
        composed = compose(config_name="config", overrides=overrides)
    ensure_registries()
    adapter = BackendAdapter(composed, root_dir=ROOT, algo_name="ppo")
    override = adapter.build_task_env_cfg_override()
    prepare_scene(override, ROOT)
    override["auto_reset"] = False
    override["seed"] = seed
    raw = create_env(composed, num_envs=num_envs, env_cfg_override=override)
    try:
        return raw, UniLabRMAEnv(raw, cfg, device)
    except Exception:
        raw.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--task",
        choices=RMA_TASKS,
        default=DEFAULT_TASK,
        help="Original RMA task or an isolated forward-walking experiment",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Parent for unique CSV/TensorBoard runs; default: <output_stem>_logs",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("iterations and num-envs must be positive")
    if args.phase == "student" and args.checkpoint is None:
        parser.error("Student training requires a trained teacher checkpoint")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new --output to keep it.")
    torch.manual_seed(42)
    cfg = RMAConfig()
    model, adaptation = ActorCriticRMA(cfg).to(args.device), AdaptationModule(cfg).to(args.device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if saved.get("format") != "unilab-rma-v1":
            raise ValueError("Not an RMA checkpoint")
        if saved["config"] != model.metadata():
            raise ValueError("Checkpoint dimension/config mismatch")
        model.load_state_dict(saved["teacher"], strict=True)
        adaptation.load_state_dict(saved["adaptation"], strict=True)
    raw, env = build_environment(args.num_envs, cfg, args.device, task=args.task)
    try:
        fn = train_teacher if args.phase == "teacher" else train_student
        from unilab_rma.training_log import TrainingLog

        metadata = dict(
            arguments=vars(args),
            model_config=model.metadata(),
            horizon=24,
            seed=42,
            torch_version=torch.__version__,
            requested_policy_device=args.device,
            environment_device=str(raw.device),
            sim_backend="mujoco",
        )
        with TrainingLog(args.output, args.phase, metadata, args.log_dir) as logger:
            print(f"Training logs: {logger.directory}", flush=True)
            fn(env, model, adaptation, args.iterations, args.output, logger=logger)
    finally:
        raw.close()


if __name__ == "__main__":
    main()

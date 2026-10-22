"""Evaluate RMA policies; opt into interactive playback with --render."""

import argparse
import json
import math
from pathlib import Path

import torch
from train_rma import build_environment
from unilab_rma import ActorCriticRMA, AdaptationModule, HistoryBuffer, RMAConfig
from unilab_rma.evaluation_diagnostics import DEFAULT_TASK, RMA_TASKS, EvaluationDiagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--policy", choices=["teacher", "student"], default="student")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--task", choices=RMA_TASKS, default=DEFAULT_TASK)
    p.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "WZ"),
        help="Fix body-frame vx, vy (m/s), yaw rate (rad/s), including after reset",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Report commanded/actual velocity and weighted reward terms",
    )
    p.add_argument(
        "--trace-dir",
        type=Path,
        help="Opt in to per-transition CSV and JSON in a unique subdirectory of this parent",
    )
    p.add_argument(
        "--trace-warmup-steps",
        type=int,
        default=100,
        help="Exclude this many initial steps per episode from the additional trace window summary",
    )
    p.add_argument(
        "--backend",
        choices=["pytorch", "tensorrt"],
        default="pytorch",
        help="Inference backend; TensorRT uses a persistent C++ worker",
    )
    p.add_argument(
        "--engine", type=Path, help="TensorRT engine exported from this student checkpoint"
    )
    p.add_argument(
        "--tensorrt-worker",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build/deploy_trt/policy_worker",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Show interactive playback; omitted by default for fast evaluation",
    )
    p.add_argument(
        "--render-speed",
        type=float,
        default=1.0,
        help="Playback speed with --render: 1.0 is real time, 0.5 is half speed",
    )
    args = p.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        p.error("steps and num-envs must be positive")
    if args.trace_warmup_steps < 0:
        p.error("trace-warmup-steps must be nonnegative")
    if not math.isfinite(args.render_speed) or args.render_speed <= 0:
        p.error("render-speed must be finite and positive")
    if args.command is not None and not all(math.isfinite(x) for x in args.command):
        p.error("--command requires finite vx, vy, and yaw rate")
    if args.backend == "tensorrt" and (args.engine is None or args.policy != "student"):
        p.error("TensorRT requires --policy student and --engine matching the checkpoint")
    if args.backend != "tensorrt" and args.engine is not None:
        p.error("--engine requires --backend tensorrt")
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if saved.get("format") != "unilab-rma-v1":
        raise ValueError("Not an RMA checkpoint")
    if args.policy == "student" and saved.get("phase") != "student":
        raise ValueError("Student evaluation requires an adaptation-trained checkpoint")
    torch.manual_seed(args.seed)
    cfg = RMAConfig(**saved["config"])
    model = ActorCriticRMA(cfg).to(args.device).eval()
    adaptation = AdaptationModule(cfg).to(args.device).eval()
    model.load_state_dict(saved["teacher"])
    adaptation.load_state_dict(saved["adaptation"])
    environment_options = {}
    if args.task != DEFAULT_TASK:
        environment_options["task"] = args.task
    if args.command is not None:
        environment_options["command"] = args.command
    raw, env = build_environment(args.num_envs, cfg, args.device, args.seed, **environment_options)
    returns = torch.zeros(args.num_envs, device=args.device)
    completed, completed_lengths, tracking = [], [], []
    partial_returns = torch.zeros(args.num_envs, device=args.device)
    partial_lengths = torch.zeros(args.num_envs, device=args.device)
    termination_count, step_rewards = 0, []
    latent_errors, action_errors = [], []
    steps = range(args.steps)
    playback = None
    runtime = None
    trace = None
    backend_action_errors, backend_max_abs_error = [], 0.0
    diagnostics = EvaluationDiagnostics() if args.diagnostics else None
    try:
        if args.backend == "tensorrt":
            from unilab_rma.tensorrt_runtime import TensorRTWorker

            runtime = TensorRTWorker(args.tensorrt_worker, args.engine, cfg)
        obs, critic, priv = env.reset()
        if args.trace_dir is not None:
            from unilab_rma.evaluation_trace import EvaluationTrace

            trace = EvaluationTrace(
                args.trace_dir,
                raw,
                vars(args),
                args.checkpoint,
                warmup_steps=args.trace_warmup_steps,
            )
        history = HistoryBuffer(args.num_envs, cfg, args.device)
        history.push(obs)
        # The default path retains the original range iterator: no renderer,
        # rendering imports, per-step clocks, or playback sleeps.
        if args.render:
            from unilab_rma.rendering import rendered_steps

            playback = rendered_steps(raw, args.steps, speed=args.render_speed)
            steps = playback
        with torch.no_grad():
            for _ in steps:
                if trace is not None:
                    trace_before = critic
                if diagnostics is not None:
                    diagnostics.record_state(critic)
                tracking.append((critic[:, 49:51] - obs[:, 42:44]).square().sum(-1).mean().item())
                teacher_z = model.encoder(priv)
                student_z = adaptation(history.data)
                if args.policy == "student":
                    latent_errors.append((student_z - teacher_z).square().mean().item())
                z = teacher_z if args.policy == "teacher" else student_z
                teacher_action = model.actor(obs, teacher_z).tanh()
                action = model.actor(obs, z).tanh()
                if runtime is not None:
                    deployed_action = runtime.infer(obs, history.data)
                    difference = deployed_action - action
                    backend_action_errors.append(difference.square().mean().item())
                    backend_max_abs_error = max(
                        backend_max_abs_error, difference.abs().max().item()
                    )
                    action = deployed_action
                if args.policy == "student":
                    action_errors.append((action - teacher_action).square().mean().item())
                (obs, critic, priv), reward, term, trunc, terminal = env.step(action)
                if trace is not None:
                    trace.record(
                        trace_before, action, terminal[1], reward, term, trunc, raw.reward_manager
                    )
                if diagnostics is not None:
                    diagnostics.record_rewards(raw.reward_manager)
                returns += reward
                partial_returns += reward
                partial_lengths += 1
                done = term | trunc
                completed.extend(returns[done].tolist())
                completed_lengths.extend(partial_lengths[done].tolist())
                returns[done] = 0
                partial_returns[done] = 0
                partial_lengths[done] = 0
                termination_count += int(term.sum())
                step_rewards.append(reward.mean().item())
                history.push(obs, done)
        report = {
            "policy": args.policy,
            "seed": args.seed,
            "completed_episodes": len(completed),
            "mean_completed_episode_return": sum(completed) / len(completed) if completed else None,
            "mean_completed_episode_length_steps": sum(completed_lengths) / len(completed_lengths)
            if completed_lengths
            else None,
            "mean_step_reward": sum(step_rewards) / len(step_rewards) if step_rewards else None,
            "planar_velocity_rmse_m_per_s": (sum(tracking) / len(tracking)) ** 0.5
            if tracking
            else None,
            "unfinished_episodes": int((partial_lengths > 0).sum()),
            "mean_unfinished_episode_return": (
                partial_returns[partial_lengths > 0].mean().item()
                if (partial_lengths > 0).any()
                else None
            ),
            "student_latent_mse_vs_teacher": (
                sum(latent_errors) / len(latent_errors) if latent_errors else None
            ),
            "student_action_rmse_vs_teacher": (
                (sum(action_errors) / len(action_errors)) ** 0.5 if action_errors else None
            ),
            "terminations": termination_count,
            "total_transitions": len(step_rewards) * args.num_envs,
        }
        if args.command is not None:
            report["command_override"] = args.command
        if args.task != DEFAULT_TASK:
            report["task"] = args.task
        if diagnostics is not None:
            report["diagnostics"] = diagnostics.report()
        if runtime is not None:
            report.update(
                {
                    "inference_backend": "tensorrt_cpp",
                    "tensorrt_action_rmse_vs_pytorch": (
                        (sum(backend_action_errors) / len(backend_action_errors)) ** 0.5
                        if backend_action_errors
                        else None
                    ),
                    "tensorrt_action_max_abs_error_vs_pytorch": (
                        backend_max_abs_error if backend_action_errors else None
                    ),
                }
            )
        if trace is not None:
            report = trace.finish(report)
        print(json.dumps(report, indent=2))
    finally:
        try:
            if playback is not None:
                playback.close()  # Release the viewer even if policy inference fails.
        finally:
            try:
                if runtime is not None:
                    runtime.close()
            finally:
                try:
                    if trace is not None:
                        trace.close()
                finally:
                    raw.close()


if __name__ == "__main__":
    main()

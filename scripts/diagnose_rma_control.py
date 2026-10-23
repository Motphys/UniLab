"""Short single-world teacher control diagnosis; never updates model weights."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from train_rma import build_environment
from unilab_rma import ActorCriticRMA, AdaptationModule, RMAConfig
from unilab_rma.control_diagnostics import ControlDiagnostics
from unilab_rma.evaluation_diagnostics import RMA_TASKS, EvaluationDiagnostics
from unilab_rma.evaluation_trace import EvaluationTrace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=RMA_TASKS, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--command", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    p.add_argument("--output-dir", type=Path, default=Path("runs/rma/control_diagnostics"))
    args = p.parse_args()
    if args.steps < 1 or args.steps > 5000:
        p.error("steps must be between 1 and 5000 for this short diagnostic")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "unilab-rma-v1" or saved.get("phase") != "teacher":
        p.error("This diagnostic requires a teacher checkpoint")
    torch.manual_seed(args.seed)
    cfg = RMAConfig(**saved["config"])
    model = ActorCriticRMA(cfg).eval()
    # The reference evaluator initializes this even in teacher mode. Preserve
    # its RNG consumption so environment observations have the same sequence.
    AdaptationModule(cfg).eval()
    model.load_state_dict(saved["teacher"])
    raw, env = build_environment(1, cfg, "cpu", args.seed, task=args.task, command=args.command)
    trace = None
    try:
        obs, critic, priv = env.reset()
        trace = EvaluationTrace(args.output_dir, raw, dict(vars(args), num_envs=1), args.checkpoint)
        recorder = ControlDiagnostics(raw, trace.directory)
        diagnostics = EvaluationDiagnostics()
        with torch.no_grad():
            for _ in range(args.steps):
                before = recorder.before()
                diagnostics.record_state(critic)
                action = model.actor(obs, model.encoder(priv)).tanh()
                # Read final controls/physics BEFORE any explicit episode reset.
                state = raw.step(action.to(raw.device))
                terminal = env.read(state.obs)
                term, trunc = bool(state.terminated[0]), bool(state.truncated[0])
                recorder.record(before, action, term, trunc)
                trace.record(
                    critic,
                    action,
                    terminal[1],
                    state.reward,
                    state.terminated,
                    state.truncated,
                    raw.reward_manager,
                )
                diagnostics.record_rewards(raw.reward_manager)
                if term or trunc:
                    raw.reset(torch.tensor([0], device=raw.device))
                # Match UniLabRMAEnv.step, including observation refresh/noise.
                obs, critic, priv = env.read(raw.get_observations())
        control = recorder.finish()
        report = trace.finish(
            dict(
                policy="teacher",
                seed=args.seed,
                task=args.task,
                total_transitions=args.steps,
                diagnostics=diagnostics.report(),
                control=control,
            )
        )
        source = Path(__file__)
        report["diagnostic_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        (trace.directory / "diagnostic_summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(dict(directory=str(trace.directory), **control), indent=2))
    finally:
        if trace is not None:
            trace.close()
        raw.close()


if __name__ == "__main__":
    main()

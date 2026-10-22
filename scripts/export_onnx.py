"""Export a STUDENT checkpoint, then require ONNX Runtime numerical validation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
import argparse
import json

import numpy as np
import torch
from unilab_rma import RMAConfig
from unilab_rma.deployment import DeploymentPolicy, go2_control_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "unilab-rma-v1" or saved.get("phase") != "student":
        raise ValueError("Export requires an adaptation-trained student checkpoint")
    cfg = RMAConfig(**saved["config"])
    module = DeploymentPolicy(cfg).eval()
    actor_state = {
        k[len("actor.") :]: v for k, v in saved["teacher"].items() if k.startswith("actor.")
    }
    module.actor.load_state_dict(actor_state, strict=True)
    module.adaptation.load_state_dict(saved["adaptation"], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    obs = torch.randn(1, cfg.obs_dim)
    hist = torch.randn(1, cfg.history_length, cfg.obs_dim)
    hist[:, -1] = obs
    # Static batch=1; fixed history=50; FP32 IO, FP16 internal TensorRT build later.
    torch.onnx.export(
        module,
        (obs, hist),
        str(args.output),
        opset_version=18,
        input_names=["observation", "history"],
        output_names=["action"],
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    names = [initializer.name for initializer in graph.graph.initializer]
    if any("critic" in name or "encoder" in name for name in names):
        raise AssertionError("Privileged teacher leaked into deployment graph")
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    maximum = 0.0
    for i in range(12):
        obs = torch.randn(1, cfg.obs_dim)
        hist = torch.randn(1, cfg.history_length, cfg.obs_dim)
        if i % 3 == 0:
            hist.zero_()  # episode-start padding
        hist[:, -1] = obs
        with torch.no_grad():
            reference = module(obs, hist).numpy()
        result = session.run(None, {"observation": obs.numpy(), "history": hist.numpy()})[0]
        np.testing.assert_allclose(reference, result, rtol=1e-4, atol=1e-5)
        maximum = max(maximum, float(np.max(np.abs(reference - result))))
    metadata = {
        "format": "unilab-rma-deployment-v1",
        "obs_dim": cfg.obs_dim,
        "history_length": cfg.history_length,
        "history_order": "oldest_to_newest_including_current",
        "history_reset": "zero_pad_then_append_reset_observation",
        "action_dim": cfg.action_dim,
        "latent_dim": cfg.latent_dim,
        "policy_period_s": 0.02,
        "action_scale": 0.25,
        "normalization": "raw UniLab policy observations; no running normalization",
        "observation_fields": [
            ["gyro", 3],
            ["projected_gravity", 3],
            ["joint_pos_rel", 12],
            ["joint_vel_rel", 12],
            ["last_action", 12],
            ["command", 3],
            ["gait_phase", 4],
        ],
        "joint_order": ["FL", "FR", "RL", "RR"],
        "action_actuator_order": ["FL", "FR", "RL", "RR"],
        "action_actuator_order_note": "Legacy alias for action output order, NOT scene storage order",
        **go2_control_layout(),
        "per_leg_order": ["hip", "thigh", "calf"],
        "onnx_max_abs_error": maximum,
        "default_angles": "must read matching scene home keyframe in actuator order; not supplied by source-only copy",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ONNX verified: {args.output}; max_abs_error={maximum:.8g}")


if __name__ == "__main__":
    main()

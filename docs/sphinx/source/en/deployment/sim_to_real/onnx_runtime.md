# ONNX Runtime

UniLab exports ONNX policies from the existing training playback paths. Use the
same algorithm family and task owner that produced the checkpoint; the playback
code loads the checkpoint, exports `policy.onnx`, and verifies the exported
graph when that path implements ONNX Runtime checking.

## Export Paths

| Algorithm path | Entry script | Export behavior in repo |
| --- | --- | --- |
| PPO (torch) | `scripts/train_rsl_rl.py` | `EXPORT_POLICY=True` in the script entrypoint; playback calls `runner.export_policy_to_onnx(...)` and `runner.export_policy_to_jit(...)`. |
| HIM-PPO | `scripts/train_him_ppo.py` | Same script-level export pattern as PPO. |
| APPO | `scripts/train_appo.py` | Playback writes `policy.onnx` and verifies ONNX Runtime output against PyTorch. |
| SAC / TD3 / FlashSAC | `scripts/train_offpolicy.py` | Playback writes `policy.onnx`; SAC and FlashSAC use `actor.as_export_module()` before export. |
| MLX PPO | `scripts/train_mlx_ppo.py` | Playback converts the MLX actor weights into a PyTorch module, writes `policy.onnx`, and verifies ONNX Runtime output. |

## Commands

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1

uv run scripts/train_appo.py task=g1_motion_tracking/motrix \
  training.play_only=true \
  algo.load_run=-1

uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco \
  training.play_only=true \
  algo.load_run=-1
```

The exported file is written into the selected run directory. For deployment
prototypes, keep the exported `policy.onnx` together with the deploy-side
configuration and motion assets used by the runtime.

## G1 Deployment Prototype

The committed G1 WBT deployment helpers use these artifacts:

| Artifact | Producer |
| --- | --- |
| `policy.onnx` | Training playback export above. |
| `deploy_config.yaml` | `scripts/deploy/export_deploy_config.py`. |
| `dance1.bin` or another motion binary | `scripts/deploy/export_motion_bin.py`. |

Example validation run:

```bash
uv run scripts/deploy/export_deploy_config.py \
  --output logs/deploy/deploy_config.yaml

uv run scripts/deploy/export_motion_bin.py \
  --output logs/deploy/dance1.bin

uv run scripts/deploy/sim_prototype.py \
  --onnx runs/<run>/policy.onnx \
  --config logs/deploy/deploy_config.yaml \
  --motion logs/deploy/dance1.bin
```

`scripts/deploy/sim_prototype.py` checks that the ONNX input width matches the
`obs_dim` in `deploy_config.yaml` and then drives the policy in MuJoCo with the
same observation layout the deployment side expects.

## ANE / Core ML Notes

The repository contains experimental Core ML / Apple Neural Engine helpers under
`src/unilab/algos/torch/common/ane_actor.py`,
`src/unilab/algos/torch/common/ane_wrapper.py`, and
`src/unilab/algos/torch/common/ane_inference.py`. The documented deployment
path above stays on the committed ONNX export behavior in the training scripts.

## See Also

- {doc}`latency_budget`
- {doc}`safety_layers`
- `unilab.algos.torch.common.ane_actor.ANEActor`

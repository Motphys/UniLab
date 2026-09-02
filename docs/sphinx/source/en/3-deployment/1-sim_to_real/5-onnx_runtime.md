# ONNX Runtime

UniLab exports ONNX policies from the existing training playback paths. Use the
same algorithm family and task owner that produced the checkpoint; the playback
code loads the checkpoint, exports `policy.onnx`, and verifies the exported
graph when that path implements ONNX Runtime checking.

## Export Paths

| Algorithm path | Entry script | Export behavior in repo |
| --- | --- | --- |
| PPO (torch) | `src/unilab/scripts/train_rsl_rl.py` | `EXPORT_POLICY=True` in the script entrypoint; playback calls `runner.export_policy_to_onnx(...)` and `runner.export_policy_to_jit(...)`. |
| HIM-PPO | `scripts/train_him_ppo.py` | Same script-level export pattern as PPO. |
| APPO | `src/unilab/scripts/train_appo.py` | Playback writes `policy.onnx` and verifies ONNX Runtime output against PyTorch. |
| SAC / TD3 / FlashSAC | `src/unilab/scripts/train_sac.py` / `src/unilab/scripts/train_td3.py` / `src/unilab/scripts/train_flashsac.py` | Playback writes `policy.onnx`; SAC and FlashSAC use `actor.as_export_module()` before export. |

## Commands

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1

uv run eval --algo appo --task g1_motion_tracking --sim motrix --load-run -1

uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` sets playback mode and maps `--load-run` to the checkpoint
selector used by the routed training script. The exported file is written into
the selected run directory. For deployment, keep the exported `policy.onnx`
together with the task owner YAML it was trained from — that YAML is the
authority on the observation and action contract the runtime must reproduce.

## Verifying the Exported Graph

The playback path validates the exported graph against PyTorch before writing
it, so a successful export already establishes numerical parity. What it does
**not** establish is that your hardware-side loop assembles the same input
vector. Before hardware bring-up:

- Read the actor obs width off the composed config (not off a doc table) and
  confirm it matches the ONNX input width.
- Confirm your term order and per-term history ordering against the owner's
  `env.observations.actor.terms`. For G1 whole-body tracking, see
  {doc}`2-g1_whole_body`.

## See Also

- {doc}`8-latency_budget`
- {doc}`7-safety_layers`

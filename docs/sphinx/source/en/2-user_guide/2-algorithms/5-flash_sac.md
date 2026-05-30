# FlashSAC

FlashSAC is the third algorithm on the shared off-policy entrypoint. Select it
with `--algo flashsac`; defaults live in
`conf/offpolicy/algo/flashsac.yaml`, and the implementation lives under
`src/unilab/algos/torch/flash_sac/`.

## Quick Start

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## Key Fields

For the off-policy playback path (`scripts/train_offpolicy.py` / CLI `--algo flashsac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

`scripts/train_offpolicy.py` rejects `training.num_gpus > 1` for FlashSAC, so
keep the default single-GPU path unless the implementation changes.

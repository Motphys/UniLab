# FlashSAC

FlashSAC is the third algorithm on the shared off-policy entrypoint. Select it
with `--algo flashsac`; defaults live in
`conf/offpolicy/algo/flashsac.yaml`, and the implementation lives under
`src/unilab/algos/torch/flash_sac/`.

It shares the off-policy training script with SAC and TD3, but does not use the
same default networks: the actor uses a block-based structure and the critic
uses a distributional (categorical) Q variant.

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
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

FlashSAC supports the shared off-policy multi-GPU runner. Enable it with:

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco \
  training.num_gpus=2 \
  training.multi_gpu_sync_mode=local_sgd
```

Multi-GPU FlashSAC requires CUDA and synchronized collection. The learner owns
its distributed synchronization hooks: gradients are averaged in `sync_sgd`,
parameters and persistent normalization buffers are averaged in `local_sgd`, and
reward normalizer state is updated on rank 0 then broadcast to the other ranks.

The log root is `logs/flash_sac/<task>/`.

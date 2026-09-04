# FlashSAC

FlashSAC runs through `src/unilab/scripts/train_flashsac.py` in its own config tree.
Select it with `--algo flashsac`; defaults are inlined in
`src/unilab/conf/flashsac/config.yaml`, and the implementation lives under
`uni_rl.algos.flash_sac` (unilab-rl repo).

It shares the off-policy runner design with SAC and TD3, but does not use the
same default networks: the actor uses a block-based structure and the critic
uses a distributional (categorical) Q variant.

## Quick Start

```bash
uv run train --algo flashsac --task g1_walk_flat --sim mujoco
uv run train --algo flashsac --task go2_joystick_flat --sim mujoco training.no_play=true
```

## Key Fields

For the off-policy playback path (`src/unilab/scripts/train_flashsac.py` / CLI `--algo flashsac`),
set `training.export_onnx=false` to skip `policy.onnx` export while still recording
playback video. See {doc}`/en/1-getting_started/3-evaluation_and_playback`.

- `algo.algo_log_name=flash_sac`
- `algo.num_envs=1024`
- `algo.max_iterations=5000`
- `algo.tau=0.01`
- `algo.save_interval=1000`
- `algo.algo_params.actor_num_blocks=2`
- `algo.algo_params.critic_num_blocks=2`

FlashSAC requires synchronized collection and the same sole replay path as SAC
and TD3: bounded host ingress plus one complete replay ring on a CUDA or Apple
MPS learner device. CPU and XPU training are unsupported.

The log root is `logs/flash_sac/<task>/`.

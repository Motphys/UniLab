# BFM

BFM (Body-Frame Motion) learns humanoid locomotion from motion capture data via
future-prediction auxiliary critic (FB-CPR-Aux). The training script is
`scripts/train_bfm.py`, with main config at `conf/bfm/config.yaml` and
algorithm defaults in the same directory. The current log name is `bfm`.

## Runtime Model

BFM uses off-policy learning with prioritized trajectory replay. The trajectory
buffer stores complete episodes (seq_length=8) with body-frame observations.
Expert motion data (LAFAN1) loads from HuggingFace on first run. The agent
architecture includes a discriminator for motion imitation, a future-prediction
auxiliary critic, and an actor with body-frame encoding.

## Quick Start

```bash
# CUDA (MuJoCo backend)
uv run train --algo bfm --task g1_bfm --sim mujoco

# CUDA (Motrix backend)
uv run train --algo bfm --task g1_bfm --sim motrix

# ROCm
make sync-rocm
HIP_VISIBLE_DEVICES=0 uv run train --algo bfm --task g1_bfm --sim mujoco
```

## Key Fields

- `algo.algo_log_name=bfm`
- `algo.num_envs=1024`
- `algo.num_env_steps=1536000000` (1.536B total steps)
- `algo.buffer_size=5120000` (5.12M transitions)
- `algo.buffer_device=cuda` (or `cpu` to save GPU memory)
- `algo.use_trajectory_buffer=true` (trajectory-based replay)
- `algo.model.seq_length=8` (trajectory sequence length)

Override parameters:

```bash
uv run train --algo bfm --task g1_bfm --sim mujoco \
  algo.num_envs=2048 \
  algo.buffer_size=10000000 \
  algo.num_env_steps=500000000
```

## Backends

- **MuJoCo**: CPU physics, faster for <2048 envs
- **Motrix**: GPU-accelerated physics, scales to 4096+ envs

## Troubleshooting

### ROCm: "Found no NVIDIA driver"

CUDA torch installed instead of ROCm. Fix:

```bash
make sync-rocm
```

### "libmujoco.so.3.X.Y: cannot open shared object file"

Native extension compiled against wrong mujoco version. Rebuild:

```bash
uv pip install --python .venv/bin/python --no-build-isolation \
    --no-binary mujoco-uni-runtime --force-reinstall --no-cache \
    mujoco-uni-runtime==0.3.1
```

### "No trajectories with length >= 9"

`num_seed_steps` too small for trajectory buffer. Increase:

```bash
uv run train --algo bfm --task g1_bfm --sim mujoco algo.num_seed_steps=4096
```

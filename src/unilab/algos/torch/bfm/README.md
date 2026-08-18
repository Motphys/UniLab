# BFM Training

Body-Frame Motion (BFM) algorithm for humanoid locomotion with motion reference.

## Setup

Install dependencies:
```bash
uv sync --extra mujoco --extra motrix
```

Motion data is auto-downloaded from HuggingFace on first run.

## Training

**MuJoCo backend:**
```bash
uv run train --algo bfm --task g1_bfm --sim mujoco
```

**Motrix backend:**
```bash
uv run train --algo bfm --task g1_bfm --sim motrix
```

Checkpoints save to `logs/bfm/G1Bfm/`.

## Evaluation

Eval runs automatically every 12M steps during training.

**Manual eval from checkpoint:**
```bash
uv run train --algo bfm --task g1_bfm --sim mujoco training.play_only=true +training.resume=logs/bfm/G1Bfm/<run>
```

Replace `<run>` with your checkpoint directory name.

## Configuration

Default config: `conf/bfm/config.yaml`
- 4096 parallel envs
- 5.12M replay buffer
- 1.536B total steps

Override any parameter:
```bash
uv run train --algo bfm --task g1_bfm --sim mujoco algo.num_envs=2048 algo.seed=42
```

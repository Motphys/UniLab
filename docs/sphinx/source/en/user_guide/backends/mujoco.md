# MuJoCo Backend

MuJoCo is the default backend path in the committed owner configs. The Python
dependency is `mujoco-uni==3.8.0rc2` in `pyproject.toml`, and the adapter lives
under `src/unilab/base/backend/mujoco/`.

## When To Use It

- You want the default training route for PPO, APPO, off-policy SAC/TD3, or
  FlashSAC.
- The task owner exists only as `conf/.../<task>/mujoco.yaml`.
- You need MuJoCo-specific tooling such as `scripts/play_viser.py` or scene
  export from a MuJoCo XML/MJB model.

## Commands

```bash
uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco
uv run scripts/train_appo.py task=go1_joystick_flat/mujoco training.no_play=true
uv run scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

Playback mode is resolved by the backend contract in
`src/unilab/base/backend/base.py`. MuJoCo reports physics-state playback support
in `src/unilab/base/backend/mujoco/backend.py`; `auto` playback records video
rather than opening the Motrix native interactive renderer.

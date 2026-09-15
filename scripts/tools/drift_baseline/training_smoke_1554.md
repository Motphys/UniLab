# Training smoke — mjbatch executor validation (#1554)

Short rsl_rl PPO runs on the mujoco backend (RTX 4090, local integration
branches: mjbatch fork + unisim adapter), exercising the retained plain
position-action path:

- `Go2JoystickFlat` — plain position-action path.

The run used `algo.num_envs=32 algo.max_iterations=100 algo.seed=42`, with
zero NaN/Inf and playback video rendered after training.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco -- \
    algo.num_envs=32 algo.max_iterations=100 algo.seed=42
```

(Executed with `UV_NO_SYNC=1` and the local mjbatch/unisim checkouts
installed editable; `UNILAB_LOCAL_UNISIM` set for the dependency-source
sentinel.)

## Go2JoystickFlat (logs/rsl_rl_ppo/Go2JoystickFlat/2026-09-12_01-44-50_mujoco)

- `Train/mean_reward`: 0.4275 (iter 0) → 17.96 (iter 99), monotone-ish, no NaN.
- 100 iterations, ~0.07 s/iteration, playback video rendered.

# Training smoke — mjbatch executor validation (#1554)

Short rsl_rl PPO runs on the mujoco backend (RTX 4090, local integration
branches: mjbatch fork + unisim adapter), exercising both action paths:

- `Go2JoystickFlat` — plain position-action path.
- `Go2WJoystickFlat` — wheeled-leg task; its mixed action runs through
  `SimBackend.set_pre_step_control`, i.e. mjbatch's native per-substep
  callback (`callback_sensordata=False`).

Both runs: `algo.num_envs=32 algo.max_iterations=100 algo.seed=42`, zero
NaN/Inf, playback video rendered after training.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco -- \
    algo.num_envs=32 algo.max_iterations=100 algo.seed=42
uv run train --algo ppo --task go2w_joystick_flat --sim mujoco -- \
    algo.num_envs=32 algo.max_iterations=100 algo.seed=42
```

(Executed with `UV_NO_SYNC=1` and the local mjbatch/unisim checkouts
installed editable; `UNILAB_LOCAL_UNISIM` set for the dependency-source
sentinel.)

## Go2JoystickFlat (logs/rsl_rl_ppo/Go2JoystickFlat/2026-09-12_01-44-50_mujoco)

- `Train/mean_reward`: 0.4275 (iter 0) → 17.96 (iter 99), monotone-ish, no NaN.
- 100 iterations, ~0.07 s/iteration, playback video rendered.

## Go2WJoystickFlat (logs/rsl_rl_ppo/Go2WJoystickFlat/2026-09-12_01-45-37_mujoco)

- `Train/mean_reward`: 1.3125 (iter 0) → 42.99 (iter 99), monotone-ish, no NaN.
- run_summary: 99 completed iterations, 76,800 env steps,
  ~8,662 env-steps/s, final_mean_reward 42.99, mean_episode_length ~780
  (ctrl steps; no early termination mass), playback video rendered.

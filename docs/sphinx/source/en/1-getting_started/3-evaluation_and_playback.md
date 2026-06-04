# Evaluation and Playback

```bash
# Latest run
uv run eval --algo ppo --task go2_joystick_flat --sim motrix --load-run -1

# Headless video export
uv run eval --algo ppo --task go2_joystick_flat --sim motrix \
    --load-run -1 --render-mode record

# Off-policy playback can skip ONNX export and still record MP4
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1 \
    --render-mode record training.export_onnx=false

# Demo (downloads checkpoint from HF on first run)
uv run demo dance
```

Render modes:

- `interactive` — open viewer window (default on macOS Motrix).
- `record` — write MP4 to `runs/<run>/playback/`.
- `none` — skip rendering, just compute metrics.

`training.export_onnx=false` currently applies only to the off-policy playback path
(`scripts/train_offpolicy.py` and CLI runs with `--algo sac|td3|flashsac`). It skips
`policy.onnx` export and verification but still runs playback and video recording.

## MuJoCo Viewer Scripts

Use `uv run eval` for regular evaluation and video export. When you need a live
`mujoco.viewer` window for policy debugging, use the low-level
`scripts/play_interactive.py` script.

`scripts/play_interactive.py` is the general MuJoCo viewer entrypoint for PPO,
APPO, SAC, FlashSAC, and HORA distill policies. It uses `--algo / --task /
--sim` to select the algorithm and owner config. The viewer is always
`mujoco.viewer`; `--sim` only selects which config to read.

```bash
# Use the owner config's interactive.action_mode; the global default is zero action
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco

# Random actions
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=random

# Policy actions
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    algo.load_run=-1 interactive.action_mode=policy

uv run scripts/play_interactive.py --algo flashsac --task g1_walk_flat --sim motrix \
    algo.load_run=-1 interactive.action_mode=policy

uv run scripts/play_interactive.py --algo ppo --task go2_joystick_flat --sim mujoco \
    interactive.action_mode=policy interactive.keyboard=true
```

Select the action source with `interactive.action_mode=zero|random|policy`. When
omitted, the script uses the owner config setting. The global default is `zero`,
and some task YAMLs override it to `policy`.
Enable keyboard control with Hydra overrides: `interactive.action_mode=policy
interactive.keyboard=true`. When keyboard control is enabled, the script checks
that the policy obs contains the velocity command and exits if it does not.
In `policy` mode, locomotion velocity-command tasks whose policy obs contains
the velocity command automatically show green target-velocity and blue
current-velocity arrows. `zero` and `random` modes do not show velocity arrows.

See `unilab.visualization.playback` for the underlying API.

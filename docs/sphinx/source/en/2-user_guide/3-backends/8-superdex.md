# SuperDex Backend

SuperDex is an optional CPU physics adapter owned by `unisim.backend.superdex`.
The initial UniLab owner is the fixed-base `FR3JointTarget` task:
`src/unilab/conf/ppo/task/fr3_joint_target/superdex.yaml`. It uses seven torque
actions, 21 observation values, joint-state resets and the standard NumPy
manager runtime. Its support level is **Configured**; bounded rollout or short
training checks do not establish full-training performance or platform support.
The implementation is tracked in [#1534](https://github.com/Motphys/UniLab/issues/1534)
under [roadmap #1533](https://github.com/Motphys/UniLab/issues/1533).

## Installation

SuperDex Physics/Robotics 1.0.0 is published as Python wheels and is an
optional UniLab extra; no native source build is required. The wheels carry
the native batch executor and support CPython 3.12/3.13 on Linux x86_64 only;
on other platforms the extra is empty and the CLI reports a targeted runtime
diagnostic. CPU physics does not require CUDA. The FR3 owner has no record
(video) playback — the `.superdex_bot` asset carries no MJCF visual model — so
play defaults to the native interactive viewer (`play_render_mode=interactive`
with `play_env_num=1`); use `training.play_render_mode=none` for headless runs.

```bash
# Source checkout (default Python 3.13; wheels support CPython 3.12/3.13):
uv sync --extra superdex

# From PyPI:
pip install "unilab[superdex]"
```

The extra delegates version pins to UniSim through `unisim-core[superdex]`.
UniSim's superdex extra already carries the plain `mujoco` package (used for
MJCF conversion and the offline playback renderer); `mujoco-uni-runtime` is
not required — only the MuJoCo physics backend needs it. The current wheels
are a temporary unilabsim build (`superdex-physics-uni` /
`superdex-robotics-uni`); once the upstream project_superdex release publishes
the `superdex-physics` / `superdex-robotics` wheels, UniSim switches the
package names and UniLab needs no change.

Native FR3 assets are hosted on Hugging Face
([unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots)),
like the other robot mesh assets; the wheels do not carry robot binaries. The
asset hub registers `bots/arms/fr3_v2/fr3_v2.superdex_bot` and downloads the
snapshot into `src/unilab/assets/` on first use, checking its collision SDF,
render files, `LICENSE` and `NOTICE` before constructing physics. To pre-fetch
the assets (e.g. for CI or offline prep):

```bash
uv run unilab-pull-assets --robot fr3_v2
```

To audit a local `project_superdex` checkout instead, set
`SUPERDEX_ASSETS_PATH=/absolute/path/to/project_superdex/assets`, or set
`env.superdex_assets_root=/absolute/path/to/project_superdex/assets` to
override it for a specific owner invocation. An explicit root takes precedence
over the Hugging Face download and never downloads.

A local source build is only needed when changing the SuperDex engine itself:
`bash scripts/tools/setup_superdex_env.sh` clones the integration branch,
builds the native extensions and links the local UniSim/UniLab checkouts
editable. Regular use does not need it.

## Run the FR3 Task

```bash
uv run --no-sync train --algo ppo --task fr3_joint_target --sim superdex \
  algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_learning_epochs=1
```

The target joint positions, rewards, reset ranges and action scales live in the
task's `base.yaml`. The torque bounds `[20,20,20,20,5,5,5]` Nm are an explicit
research profile, not rated hardware limits. `superdex_effort_limits` declares
the same bounds at the native backend boundary. The SDK remains single-threaded;
the native scene executor below owns all supported CPU parallelism.

## Run the Go2 Task

The `go2_joystick_flat/superdex` owner trains and evaluates the Go2 quadruped
on SuperDex. It inherits the MuJoCo owner's policy I/O (49 actor observations,
52 critic observations, 12 position-target actions) and control timing,
declares its own command ranges and reward weights, and explicitly opts into
contact approximation; see "Validation and Ownership" below for the contract
details.

Train directly on SuperDex (CPU physics, automatic native workers):

```bash
uv run train --algo ppo --task go2_joystick_flat --sim superdex
```

The owner defaults to 1024 environments and 400 iterations. Logs and
checkpoints land in `logs/rsl_rl_ppo/Go2JoystickFlat/<timestamp>_superdex/`.

Evaluate a trained run with `--load-run`. Record playback is the default: it
steps 200 frames across 16 environments and writes `play_video.mp4` into the
run directory through the offline MuJoCo renderer while SuperDex remains the
physics backend:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim superdex \
  --load-run 2026-09-10_10-50-00_superdex
```

For the native SuperDex (Polyscope) viewer instead of a video, add
`--render-mode interactive`; the CLI forces `training.play_env_num=1` and the
owner layer switches the env to the serial executor for that run.

A checkpoint trained on MuJoCo can be evaluated on SuperDex directly
(sim2sim): pass its run directory to `--load-run`. The play entrypoint
validates the source `run_config.json` against the sim2sim contract before
constructing the environment and rejects incompatible policy I/O.

## Default CPU Environment Parallelism

The backend uses the `SceneBatchExecutor` shipped in the SuperDex wheel, a persistent C++
thread pool that owns the barrier across independent scenes. Each substep writes
batched generalized forces, advances scenes, and returns articulated/link state,
contact sensors and solver status without per-environment Python binding calls.
Asset materialization, reset and cache-frame transforms remain owned by the
UniSim adapter. This is CPU thread parallelism, not GPU physics, and it does
not change the PPO/APPO collector, learner or policy contracts. The decision is
recorded in {doc}`/adr/ADR-0009-superdex-persistent-cpu-workers` and
[unisim#41](https://github.com/unilabsim/unisim/issues/41).

Both task owners select automatic workers by default:

| Owner option | Meaning |
| --- | --- |
| `env.superdex_num_workers=0` | Automatic: `min(available physical CPU cores, num_envs)` |
| `env.superdex_num_workers=1` | One native C++ scene worker |
| `env.superdex_num_workers=K` | Explicit C++ worker count, capped at `num_envs` |

A native scene's `DebugDraw` state is thread-affine, so the batched mode is
incompatible with an attached native SuperDex debugger: the adapter fails
closed with an actionable error when a debugger client connects. To debug with
the native debugger, run `env.superdex_execution_mode=serial`, which steps
every scene on the environment thread and never builds the native worker pool
([unisim#55](https://github.com/unilabsim/unisim/issues/55)). Serial mode is a
debugging profile, not a performance configuration.

`eval --sim superdex --render-mode interactive` renders through the native
SuperDex (Polyscope) viewer instead of the MuJoCo viewer path. The owner layer
automatically switches the env to `superdex_execution_mode=serial`, and the CLI
forces `training.play_env_num=1` because the viewer draws exactly one scene;
an explicit `training.play_env_num=...` override is preserved.

With 1024 environments on a 16-core/32-thread host, automatic selection yields
16 workers. Concurrent multi-rank collectors are assigned whole physical-core
groups, including their logical siblings, so ranks do not split an SMT core.
`training.dp_collector_cpu_ids` may instead provide one explicit CPU-id list
per rank. The selected block is applied before SuperDex materializes its native
worker pool.

For every physics substep, the host runs the pre-step control callback, enters
the native batch barrier, then publishes the refreshed batch before the next
callback. Selected reset preserves caller row order and leaves unselected
environments unchanged. A native worker failure closes the executor and reports
the failure; it does not silently return stale state or switch to serial.
Closing an environment joins its C++ workers before scenes are destroyed.

Worker count alone is not evidence of speedup. Throughput comparisons must use
the same model, control sequence, batch size and substeps, and report complete
backend/env time, startup, RSS, CPU use and actual worker count. Count environment
control steps, not physics substeps. Existing contact approximations are unchanged.

The fixed root still has a named entity and readable body state. Reset terms
write joint state; they do not request a floating-root layout. The task does
not require contact sensors, cameras, site Jacobians or runtime material DR.
SuperDex has no native renderer. Its default record playback uses the offline
MuJoCo renderer with the authored MJCF visual model while SuperDex remains the
physics backend. `.superdex_bot` scenes must provide `visual_model_file` for
this path. Selecting playback mode `none` skips playback entirely; it is not
evidence that a checkpoint has executed a rollout.

`superdex_allow_contact_approximation` defaults to `false`. It is reserved for
explicitly audited MJCF conversion profiles: enabling it accepts a warning about
contact/material approximation, including missing torsional/rolling friction
equivalence. It does not establish arbitrary MJCF task compatibility.
Contact queries report the last completed solver step. Reset clears this state;
it does not provide a fresh geometric overlap test until a positive physics step
has completed. Kinematic body/joint getters are refreshed immediately at reset.

## Validation and Ownership

The `go2_joystick_flat/superdex` PPO owner is a research sim2sim profile. It
inherits the MuJoCo owner, preserving 49 actor observations, 52 critic
observations, 12 position-target actions, normalization, network dimensions and
control timing. It disables runtime PD gain randomization and explicitly opts
into contact approximation. The adapter's cold-path MJCF conversion is restricted;
this owner does not imply arbitrary scene support or equivalent walking behavior.

Create a small source checkpoint with the MuJoCo owner, then pass its path to the
optional checkpoint test:

```bash
uv run --no-sync train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.num_envs=2 algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 \
  training.device=cpu training.no_play=true
export UNILAB_SUPERDEX_GO2_CHECKPOINT=/absolute/path/to/source/run/model_1.pt
uv run --no-sync pytest tests/envs/test_go2_superdex.py -q
```

This validates the source `run_config.json` before environment construction,
checks rejection of changed policy action semantics, loads the actual policy
through the production playback session and executes 64 SuperDex control steps
without a renderer. It checks finite values and interface compatibility; a
two-iteration checkpoint is not expected to walk reliably.

```bash
uv run --no-sync pytest tests/assets/test_superdex_assets.py \
  tests/envs/test_fr3_superdex.py tests/test_cli_runtime_requirements.py -q
```

The optional native tests require the SDK and the FR3 assets (downloaded from
Hugging Face on demand, or provided through `SUPERDEX_ASSETS_PATH`); they cover
finite rollout data, selected reset isolation, immediate observation refresh
and a spawned `EnvFactory`. Missing runtime/assets produce an explicit skip;
such a run is not native validation. The base asset/config tests need no native
asset checkout.

Engine conversion and physics live in UniSim; asset registration, Hydra and task
terms remain in UniLab. See
{doc}`/adr/ADR-0007-unisim-extraction-boundary`,
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime` and
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`.

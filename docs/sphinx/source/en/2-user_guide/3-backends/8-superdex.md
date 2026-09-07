# SuperDex Backend

SuperDex is an optional CPU physics adapter owned by `unisim.backend.superdex`.
The initial UniLab owner is the fixed-base `FR3JointTarget` task:
`src/unilab/conf/ppo/task/fr3_joint_target/superdex.yaml`. It uses seven torque
actions, 21 observation values, joint-state resets and the standard NumPy
manager runtime. Its support level is **Configured**; bounded rollout or short
training checks do not establish full-training performance or platform support.
The implementation is tracked in [#1534](https://github.com/Motphys/UniLab/issues/1534)
under [roadmap #1533](https://github.com/Motphys/UniLab/issues/1533).

## Local Development Setup

This development profile uses locally linked UniSim and UniLab checkouts; it
does not require a new published version. SuperDex Physics/Robotics 1.0.0 requires
Python 3.12. CPU physics does not require CUDA. Native rendering and video are
not part of the FR3 owner; training defaults to `no_play=true`.

In the UniLab checkout, use an existing Python 3.12 virtual environment or create
one, then install the local packages:

```bash
uv venv --python 3.12
export UNILAB_LOCAL_UNISIM=/absolute/path/to/unisim
uv pip install -e "${UNILAB_LOCAL_UNISIM}[superdex,mujoco]" -e . --group pyproject.toml:dev
export UV_NO_SYNC=1
export SUPERDEX_ASSETS_PATH=/absolute/path/to/project_superdex/assets
```

`UNILAB_LOCAL_UNISIM` enables the repository's local dependency validation:
tests require an editable installation and check that its metadata and imported
module point to exactly that checkout. Without this variable, the normal
indexed-release requirement remains in force. `UV_NO_SYNC=1` preserves the local
links when running existing Make targets; `uv sync` would re-resolve the locked
release profile.

Native FR3 assets remain in the upstream SuperDex checkout. The asset hub
registers `bots/arms/fr3_v2/fr3_v2.superdex_bot`, checking its collision SDF,
render files, `LICENSE` and `NOTICE` before constructing physics. No native
robot binaries are bundled or downloaded by UniLab. Set
`env.superdex_assets_root=/absolute/path/to/project_superdex/assets` to override
`SUPERDEX_ASSETS_PATH` for a specific owner invocation.

## Run the FR3 Task

```bash
uv run --no-sync train --algo ppo --task fr3_joint_target --sim superdex \
  algo.max_iterations=2 algo.num_steps_per_env=16 \
  algo.algorithm.num_learning_epochs=1
```

The target joint positions, rewards, reset ranges and action scales live in the
task's `base.yaml`. The torque bounds `[20,20,20,20,5,5,5]` Nm are an explicit
research profile, not rated hardware limits. `superdex_effort_limits` declares
the same bounds at the native backend boundary. `superdex_num_threads=0` uses
serial physics; thread count is process-wide, so simultaneous backend instances
must agree. Existing spawned collectors own process parallelism.

The fixed root still has a named entity and readable body state. Reset terms
write joint state; they do not request a floating-root layout. The task does
not require contact sensors, cameras, site Jacobians or runtime material DR.
Selecting playback mode `none` skips playback entirely; it is not evidence that
a checkpoint has executed a rollout.

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

The optional native tests require the SDK and `SUPERDEX_ASSETS_PATH`; they cover
finite rollout data, selected reset isolation, immediate observation refresh
and a spawned `EnvFactory`. Missing runtime/assets produce an explicit skip;
such a run is not native validation. The base asset/config tests need no native
asset checkout.

Engine conversion and physics live in UniSim; asset registration, Hydra and task
terms remain in UniLab. See
{doc}`/adr/ADR-0007-unisim-extraction-boundary`,
{doc}`/adr/ADR-0006-community-manager-api-on-numpy-runtime` and
{doc}`/adr/ADR-0002-backend-capability-boundary-for-play-and-snapshot`.

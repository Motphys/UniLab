# DrakeUni C++ Batch Pool Plan

## Purpose

Build the next DrakeUni version in stages, ending with a pybind/C++ batch pool that gives UniLab the same high-level execution shape as MuJoCoUni:

```text
state0[N, state_dim] + control[N, T, control_dim]
  -> native batch rollout
  -> state_out[N, state_dim] + sensor_out[N, sensor_dim]
```

The first supported target is intentionally narrow:

- Go1 joystick flat task only.
- Drake model file: `src/unilab/assets/robots/go1/scene_flat_drake.xml`.
- Position-control only: control means absolute joint target positions in radians.
- No policy action tensor inside DrakeUni. UniLab converts action to control before calling the backend.
- No generic Drake-system support yet.

## Current Evidence

- UniLab already routes `backend_type == "drake"` through `create_backend` in `src/unilab/base/backend/__init__.py:83`.
- The current Drake backend builds one Drake `Diagram` from the model file, then caches `plant`, `scene_graph`, body handles, and actuator metadata in `src/unilab/base/backend/drake/backend.py:495`.
- The current Drake backend creates one `_DrakeRuntime` per env in `src/unilab/base/backend/drake/backend.py:593`.
- A current runtime contains `context`, `plant_context`, and `simulator` handles; the runtime is built in `src/unilab/base/backend/drake/backend.py:624`.
- The current Python step loop iterates over every env and calls `runtime.simulator.AdvanceTo(...)` in `src/unilab/base/backend/drake/backend.py:703`.
- MuJoCoUni already exposes the target shape: UniLab builds `control_traj`, calls `BatchEnvPool.step`, then receives `state_np, sensor_np` in `src/unilab/base/backend/mujoco/backend.py:777`.
- UniLab uses `uv_build` today and does not yet have a repo-native C++ extension build section in `pyproject.toml:1`.

## Mental Model

MuJoCo gives this split directly:

```text
mjModel = static model
mjData  = runtime workspace/state
```

Drake scatters the same idea across objects:

```text
Diagram = shared static graph
Context = one runtime state for that graph
Simulator = execution driver for one context
Plant / SceneGraph / PlantContext = cached handles for fast access
```

So DrakeUni should not invent a second static-world abstraction. The pool should be plain:

```text
DrakeEnvPool
  shared:
    diagram_
    plant_ pointer
    scene_graph_ pointer
    model/body/actuator indices

  runtime:
    envs_[env_id].context
    envs_[env_id].plant_context
    envs_[env_id].simulator
```

Cached pointers are allowed because they avoid repeated lookup. They must not duplicate heavy Drake data.

## Array Contract

### State

Use the current UniLab/Drake playback state format:

```text
state[:, 0] = time
state[:, 1 : 1+nq] = qpos in UniLab/MuJoCo-compatible order
state[:, 1+nq :] = qvel in UniLab/MuJoCo-compatible order
```

The current implementation already exposes this via `get_physics_state` in `src/unilab/base/backend/drake/backend.py:1169`.

For Go1:

```text
nq = 19
nv = 18
state_dim = 1 + 19 + 18 = 38
```

### Control

Use backend-level control, not policy action:

```text
control[N, T, 12] = absolute desired Go1 joint positions
```

UniLab already performs this action-to-control conversion before backend stepping in `src/unilab/base/np_env.py:115` and `src/unilab/envs/locomotion/common/base.py:91`.

For Go1 joystick:

```text
action[N, 12] -> action * action_scale + default_angles -> control[N, 12]
```

DrakeUni receives only `control`.

### Sensor Output

MuJoCoUni returns raw MuJoCo `sensordata`. Drake has no identical raw sensor array, so DrakeUni should return a packed Drake sensor packet plus a Python accessor layer.

Minimum Go1 sensor packet:

```text
gyro[N, 3]
local_linvel[N, 3]
upvector[N, 3]
base_pos[N, 3]
dof_pos[N, 12]
dof_vel[N, 12]
feet_pos[N, 4, 3]
feet_contact_force[N, 4, 3]  # initially zero, later real contact estimate
```

This is enough for the current Go1 joystick observation/reward path, which reads gyro, local linear velocity, upvector, dof position/velocity, foot contacts, foot positions, and base height in `src/unilab/envs/locomotion/go1/joystick.py:150`.

The Python backend can keep the public methods unchanged:

```text
get_sensor_data("gyro")
get_sensor_data("local_linvel")
get_sensor_data("upvector")
get_dof_pos()
get_dof_vel()
get_base_pos()
```

Internally those methods should read from the latest C++ sensor packet instead of looping over Drake objects in Python.

## Stage 1 - Contract And Python Oracle

Goal: make the DrakeUni contract explicit and freeze the current Python backend as the correctness oracle.

Status: implemented in the Python oracle path. `DrakeEnvPool` now lives in
`src/unilab/base/backend/drake/pool.py`, `DrakeBackend` calls the pool facade
for step/reset, and the backend owns cached `_physics_state` plus a Go1
`_sensor_packet` as the UniLab-facing truth. The Go1 tests now cover pool
buffer output, cached accessors, state layout, sensor packet shapes, per-env
reset independence, and control shape rejection.

Work:

1. Define the public pool contract in `src/unilab/base/backend/drake/pool.py`:

   ```python
   class DrakeEnvPool:
       def __init__(self, model_file, nbatch, sim_dt, *, nthread=0, kp=35.0, kd=0.5): ...
       def step(self, state0, *, nstep, control, push_force=None, return_sensor=True): ...
       def reset(self, env_ids, initial_state): ...
   ```

2. Keep the first `DrakeEnvPool` implementation in Python and reuse the current runtime idea:

   ```text
   one shared Diagram
   env_id -> Context / PlantContext / Simulator
   ```

3. Make `DrakeBackend` call this pool facade instead of embedding the pool concept directly in the backend.

4. Expand `tests/base/backend/test_drake_go1_pool.py` so it checks:

   ```text
   state shape and time advancement
   control shape rejection
   per-env independence
   get_physics_state shape
   Go1 sensor packet shapes
   replay-critical state output
   ```

Acceptance:

```text
uv run pytest tests/base/backend/test_drake_go1_pool.py -q
```

passes, and current Drake Go1 training/replay behavior is unchanged.

## Stage 2 - Native C++ Handle Pool

Goal: move the current Python env loop into a pybind/C++ pool while keeping the same array contract.

Status: implemented as an optional local native extension path. The C++ source
is `src/unilab/base/backend/drake/native/drake_env_pool.cc`, with a local build
helper at `scripts/build_drake_native.py`. The extension builds against
`/Users/huanghaochen/solver/drake/install`, exposes `NativeDrakeEnvPool`, loads
the Go1 Drake scene, keeps one simulator/context per env, releases the GIL
during native step/reset work, supports serial or chunked threaded stepping via
`nthread`, and returns the Go1 state/sensor buffer contract. Python fallback
remains the default until Stage 3 decides how to route `DrakeBackend` safely.

Important integration note: this machine has two Drake shared libraries:
pydrake's bundled library under `.venv/.../pydrake/lib/libdrake.so` and the
local C++ SDK under `/Users/huanghaochen/solver/drake/install/lib/libdrake.so`.
They are ABI-incompatible in one process. The package now lazy-loads
`DrakeBackend` from `unilab.base.backend.drake.__init__` so a clean native-only
process can import the extension. Stage 3 must choose one Drake runtime path
per process instead of mixing pydrake and the local native SDK.

Likely files:

```text
src/unilab/base/backend/drake/pool.py
src/unilab/base/backend/drake/native/drake_env_pool.cc
src/unilab/base/backend/drake/native/bindings.cc
```

Build decision:

- Prefer a small separate/local optional `drake-uni` style extension first, because `pyproject.toml:1` currently uses `uv_build` and UniLab has no native extension build path.
- Keep a Python fallback path when the native extension is missing.

C++ shape:

```cpp
class DrakeEnvPool {
 private:
  // Shared static owner and cached access handles.
  std::unique_ptr<systems::Diagram<double>> diagram_;
  const multibody::MultibodyPlant<double>* plant_;
  const geometry::SceneGraph<double>* scene_graph_;
  multibody::ModelInstanceIndex robot_;

  // Runtime table.
  std::vector<EnvRuntime> envs_;
};

struct EnvRuntime {
  std::unique_ptr<systems::Context<double>> context;
  systems::Context<double>* plant_context;
  std::unique_ptr<systems::Simulator<double>> simulator;
};
```

Native `step` routine:

```text
input:  state0[N, 1+nq+nv], control[N, T, nu]
output: state_out[N, 1+nq+nv], sensor_out[N, sensor_dim]

for env_id in assigned chunk:
  load state0[env_id] into envs_[env_id].plant_context
  for substep in T:
    set desired joint target from control[env_id, substep]
    set optional push force
    simulator.AdvanceTo(current_time + sim_dt)
  write state_out[env_id]
  write sensor_out[env_id]
```

Threading:

- Release the Python GIL during native `step`.
- Start serial, then add chunked C++ threading.
- Keep one context/simulator per env in this stage, because it is the safest bridge from the current Python backend.

Acceptance:

- Native C++ step matches the Stage 1 Python oracle on deterministic Go1 smoke cases.
- `nthread=0` or `1` runs serially.
- `nthread>1` produces finite arrays and no repeated-test data races.
- `DrakeBackend` can use the native pool without changing trainer code.
- A short Drake training run completes:

  ```text
  uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.no_play=true algo.max_iterations=2
  ```

## Stage 3 - Complete MuJoCoUni-Style pybind Contract

Goal: finish the user-facing DrakeUni contract so it has the same output role as MuJoCoUni: native batch rollout through pybind, returning state and sensor arrays to Python.

Status: implemented through a native-only backend path. Same-process mixing of
`pydrake` and `/Users/huanghaochen/solver/drake/install/lib/libdrake.so` remains
invalid, so `create_backend(..., drake_backend_mode="native")` now loads
`NativeDrakeBackend` before importing the pydrake backend. The native backend is
Go1-only, uses the pybind `NativeDrakeEnvPool` for step/reset, keeps cached
state/sensor arrays, expands the Go1 foot sensor names expected by the task, and
records replay videos through the existing MuJoCo playback helper. The pydrake
backend remains available by selecting `drake_backend_mode="pydrake"` or leaving
auto mode without `UNILAB_DRAKE_BACKEND=native`.

Evidence:

- `uv run python scripts/build_drake_native.py --drake-home /Users/huanghaochen/solver/drake/install`
- `uv run pytest tests/base/backend/test_drake_go1_pool.py tests/base/backend/test_drake_native_pool.py tests/scripts/test_train_scripts.py::test_ppo_go1_drake_native_config_matches_current_contact_support -q`
  passed 15 tests.
- `uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.no_play=true algo.max_iterations=1 algo.num_envs=8 algo.num_steps_per_env=4 algo.save_interval=1`
  completed a native Drake training smoke at
  `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-12_18-41-11_drake`.
- `uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.play_only=true training.play_render_mode=record training.play_steps=20 training.play_env_num=1 algo.load_run=2026-06-12_18-41-11_drake`
  generated
  `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-12_18-41-11_drake/play_video.mp4`.
- `ffprobe` reports the replay artifact as 1280x720, 20 frames, 0.4 seconds.

Final API:

```python
pool = DrakeEnvPool(model_file, nbatch=1024, nthread=8, sim_dt=0.01)

state_out, sensor_out = pool.step(
    state0,
    nstep=2,
    control=control_traj,
    return_sensor=True,
)

state_reset, sensor_reset = pool.reset(
    env_ids,
    initial_state,
)
```

Completion work:

1. Implement native `reset(env_ids, initial_state)`.
2. Implement native sensor packet output for all Go1 joystick needs:

   ```text
   gyro
   local_linvel
   upvector
   base_pos
   dof_pos
   dof_vel
   feet_pos
   feet_contact_force, initially zero or later real contact estimate
   ```

3. Make `DrakeBackend` thin in the same spirit as `MuJoCoBackend`:

   ```text
   owns cached state/sensor arrays
   builds control_traj
   calls pool.step(...)
   answers getter methods from cached arrays
   ```

4. Add optional memory optimization after correctness:

   ```text
   workers_[thread_id] -> temporary Context / Simulator
   state arrays -> persistent env state
   ```

   This is the MuJoCoUni-like worker model. It should only replace the handle pool if it matches the Stage 2 output and gives a real memory/throughput win.

Acceptance:

- `DrakeBackend` calls the C++ pool for step/reset/sensor refresh.
- Python fallback remains available when native extension is missing.
- Go1 Drake training runs end-to-end.
- Drake replay video can be generated through the existing playback path.
- Benchmark report records:

  ```text
  MuJoCoUni training command and result
  DrakeUni C++ training command and result
  step_ms / env_steps_per_sec / memory RSS
  replay artifact path
  ```

## Verification Matrix

Unit tests:

```text
uv run pytest tests/base/backend/test_drake_go1_pool.py -q
```

Native extension smoke:

```text
python -c "from unilab.base.backend.drake.pool import DrakeEnvPool; print(DrakeEnvPool)"
```

Backend integration:

```text
uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.no_play=true algo.max_iterations=2
```

Replay:

```text
uv run python scripts/train_rsl_rl.py task=go1_joystick_flat/drake training.play_only=true training.play_render_mode=record algo.load_run=<run_id>
```

Benchmark:

```text
N in {1, 16, 128, 1024}
nthread in {0, 1, 4, 8}
measure: step_ms, env_steps/sec, memory RSS
```

## Risks

- Drake's runtime state is not as flat as MuJoCo's `mjData`; flattening too early may silently drop runtime/cache/input state.
- Drake contact results may differ when contexts are recreated every call versus persisted per env.
- The repo currently uses `uv_build`; adding native extension packaging directly may be more invasive than a separate `drake-uni` package.
- Sensor parity is not one-to-one because MuJoCo has raw `sensordata`, while Drake must synthesize a sensor packet.
- Contact forces are currently stubbed as zero in the Python Drake backend, so real contact-force support is a later milestone.

## Stop Rules

- Do not generalize beyond Go1 joystick until C++ Go1 training and replay work.
- Do not move action conversion into DrakeUni.
- Do not remove the Python fallback until native packaging works on both macOS arm64 and Linux x86_64.
- Do not optimize into worker-context V2 until handle-pool V1 matches the Python oracle.

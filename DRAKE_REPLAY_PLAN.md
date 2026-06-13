# Drake Replay Plan

Status: active working plan; minimal Drake backend, Meshcat playback, and MP4 checkpoint recording achieved  
Last updated: 2026-06-08  
Owner intent: replay a trained UniLab checkpoint through UniLab using a Drake backend.

## Goal

Replay the existing `Go1JoystickFlat` PPO checkpoint through UniLab with `sim_backend=drake`.

The target is replay/evaluation, not training. The policy remains a PyTorch/RSL-RL checkpoint. UniLab should run the policy, send actions to Drake, step Drake physics, rebuild the UniLab observation contract from Drake state/sensors, and visualize or record the replay.

Target command shape:

```bash
uv run --no-sync eval \
  --algo ppo \
  --task go1_joystick_flat \
  --sim drake \
  --load-run 2026-06-05_02-36-19_mujoco \
  --render-mode record \
  training.play_env_num=1
```

## Current Facts

- Target task is Go1, not G1.
- Target env is `Go1JoystickFlat`.
- Existing trained checkpoint:
  - `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-05_02-36-19_mujoco/model_150.pt`
- Drake is installed in the UniLab uv environment:
  - `drake==1.53.0`
  - Python: `.venv/bin/python3`, Python 3.13.12
  - `pydrake` imports successfully.
- Drake plant/simulator smoke test passes.
- Drake Meshcat smoke test passes.
- MuJoCo Go1 replay already works and can record a single-env MP4.
- Drake cannot parse the current Go1 MJCF scene as-is because the MJCF references `.stl` visual meshes.
- In the Go1 MJCF, the STL meshes are visual meshes. Collision geometry is mostly primitive box/cylinder/capsule/sphere geometry.
- A dirty Drake replay loop now loads the existing actor checkpoint, rebuilds the 49D Go1 actor observation from Drake state, and sends joint position targets to Drake's native in-plant PD controller.
- The old external torque-PD approximation collapses quickly in Drake. The Drake-native PD path keeps Go1 upright in zero-action and policy smoke tests.
- The MuJoCo-trained policy now walks forward in the dirty Drake replay after actuator-target clamping and point-contact SAP contact settings. Treat the remaining velocity difference as a sim/contact parity gap, not as backend-contract completion.
- A direct MuJoCo dirty replay using the same actor, same hard-coded `[0.5, 0.0, 0.0]` command, same observation layout, same `Kp=35/Kd=0.5`, and same action-to-target mapping walks forward. This narrows the Drake issue to physics/sensor/contact/actuator semantic parity.
- A minimal `DrakeBackend` is now registered for `Go1JoystickFlat`.
- The normal RSL-RL script path can load the MuJoCo-trained checkpoint and run finite headless Drake playback with `training.play_render_mode=auto`.
- The normal RSL-RL script path can record finite Drake MP4 playback with `training.play_render_mode=record`.
- Drake replay metadata is now parsed from the Drake model files instead of Python-side robot constants.

## Resolved So Far

This is the current milestone boundary.

- Drake is installed and usable in the UniLab uv environment.
- Go1 can be imported into Drake through generated Drake-compatible MJCF copies and converted OBJ visual meshes.
- Drake reports the expected Go1 dimensions: `nq=19`, `nv=18`, `nu=12`.
- The MuJoCo-trained `Go1JoystickFlat` actor checkpoint loads outside UniLab.
- The dirty Drake replay reconstructs the 49D actor observation from Drake state.
- The dirty Drake replay maps actor actions to joint position targets using UniLab's contract:

```text
target_q = action * action_scale + default_angles
```

- Drake-native actuator PD is now the control direction.
- Policy-derived joint targets are clamped to MuJoCo actuator `ctrlrange`, matching MuJoCo position-actuator semantics.
- Robot self-collisions are filtered in Drake, matching the intended MuJoCo collision topology more closely.
- Drake point-contact SAP is selected for the dirty replay, producing much closer Go1 locomotion than the parser default.
- The dirty Drake replay visibly walks forward in Meshcat:
  - Drake dirty replay after fixes: `linvel_x=0.5923` at 6 seconds.
  - Direct MuJoCo dirty baseline: `linvel_x=0.670` at 6 seconds.
- `src/unilab/base/backend/drake/backend.py` implements the first backend contract for single-env Go1 replay.
- DrakeBackend parses actuator limits, effort limits, foot site sensors, contact sensor names, and the home keyframe from `scene_flat_drake.xml` plus included robot XML.
- `Go1JoystickFlat` is registered with `sim_backend="drake"`.
- `conf/ppo/task/go1_joystick_flat/drake.yaml` selects the Drake-compatible scene, disables unsupported domain randomization/pushes, and fixes the Go1 command to `[0.5, 0.0, 0.0]`.
- Headless checkpoint replay through UniLab's normal RSL-RL entrypoint runs to completion with this command:

```bash
uv run python scripts/train_rsl_rl.py \
  task=go1_joystick_flat/drake \
  training.play_only=true \
  algo.load_run=2026-06-05_02-36-19_mujoco \
  algo.checkpoint=150 \
  training.play_env_num=1 \
  training.play_steps=120 \
  training.play_render_mode=auto
```

- Meshcat playback through UniLab's playback contract works for Drake.
- MP4 playback through UniLab's playback contract works for Drake with `training.play_render_mode=record`.

## Still Open

- The Drake backend is intentionally narrow: Go1 only, single env only, replay only.
- Drake sensor reconstruction is implemented for the Go1 joystick replay contract, not yet as a generic Drake sensor/site abstraction.
- Contact/friction parity is improved but not exact; Drake still cannot reproduce MuJoCo `condim=6`, torsional friction, rolling friction, or joint `frictionloss` exactly through MJCF parsing.
- Multi-env rollout, training, MPI, and cluster-scale data generation remain out of scope for this milestone.

## Architecture Target

```mermaid
flowchart LR
  A["RSL-RL checkpoint"] --> B["UniLab eval runner"]
  B --> C["Go1JoystickFlat env"]
  C --> D["DrakeBackend"]
  D --> E["Drake MultibodyPlant"]
  E --> F["Drake state and sensors"]
  F --> G["UniLab observation contract"]
  G --> B
  E --> H["Meshcat or MP4 replay"]
```

## Success Criteria

Minimum success:

- `--sim drake` is accepted by the UniLab CLI and registry.
- `Go1JoystickFlat` can be instantiated with the Drake backend for `training.play_env_num=1`.
- The trained checkpoint loads.
- The policy loop runs for finite replay steps without crashing.
- Drake receives the policy action and steps the plant.
- The env returns actor observations with the expected shape `(1, 49)`.
- A visualization exists:
  - first acceptable: Meshcat interactive replay,
  - final target: `--render-mode record` produces an MP4.

Non-goals for the first milestone:

- Drake training.
- Multi-env Drake rollout.
- MPI.
- Domain randomization parity.
- Reward/contact parity.
- Perfect MuJoCo-to-Drake behavioral match.

## Contract To Restore

### Model Contract

For Go1 replay, Drake must expose a model equivalent enough to the MuJoCo Go1 model:

- floating base,
- 12 actuated joints,
- Go1 body and joint names recoverable,
- foot frames/sites recoverable or reconstructable,
- floor contact sufficient for replay dynamics.

Expected model dimensions from MuJoCo:

- `nq = 19`
- `nv = 18`
- `nu = 12`

### Controller Contract

The checkpoint outputs 12 normalized actions. UniLab maps these actions to joint position targets:

```text
target_q = action * action_scale + default_angles
```

For Drake, use Drake's native discrete-plant actuator PD path:

```text
desired_state = [target_q, target_v=0]
feedforward_u = 0
```

The desired state is written to `MultibodyPlant.get_desired_state_input_port(model_instance)`.
Feedforward actuation is fixed to zero through `get_actuation_input_port(model_instance)`.
Policy-derived joint targets are clamped to the MuJoCo actuator `ctrlrange` before being sent to Drake, because MuJoCo position actuators enforce that range during stepping.
The external torque-PD formula remains useful as a diagnostic path only:

```text
tau = kp * (target_q - q) - kd * qdot
```

Initial Go1 gains from UniLab:

- `kp = 35.0`
- `kd = 0.5`
- `action_scale = 0.25`
- actuator `ctrlrange`:
  - hip: `[-0.863, 0.863]`
  - thigh: `[-0.686, 4.501]`
  - calf: `[-2.818, -0.888]`

### Observation Contract

The Go1 actor observation is 49 values:

```text
gyro(3)
-gravity_or_upvector(3)
joint_position_error(12)
joint_velocity(12)
last_action(12)
command(3)
feet_phase(4)
```

For the first replay, the actor observation is more important than critic/reward observation. The critic/reward path can be minimal as long as eval playback does not crash.

Required Drake-derived quantities:

- base angular velocity in body frame for `gyro`,
- up vector or gravity direction in body frame,
- 12 joint positions in UniLab action order,
- 12 joint velocities in UniLab action order,
- foot positions for reward/diagnostics if replay path touches them,
- contact-like foot force values can be zero or approximate for dirty replay, then improved later.

### Reset Contract

First milestone reset behavior:

- single env only,
- use Go1 `home` qpos equivalent,
- support randomized command sampling from UniLab,
- disable unsupported domain randomization and pushes in the Drake config.

### Visualization Contract

Two-stage plan:

- Stage A: Meshcat interactive debug, because it is easiest to prove Drake state is moving.
- Stage B: MP4 recording, ideally wired through UniLab `--render-mode record`.

## Milestones

### M0: Environment And Baseline

Status: done

- Trained Go1 MuJoCo checkpoint.
- Recorded MuJoCo single-env replay video.
- Installed and verified Drake in the UniLab uv environment.
- Confirmed Drake parsing blocker for current Go1 MJCF: STL visual meshes.

Evidence:

- `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-05_02-36-19_mujoco/model_150.pt`
- `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-05_02-36-19_mujoco/play_video_env1_replay.mp4`
- `uv pip show drake`

### M1: Import Go1 Into Drake

Status: done

Tasks:

- Created a Drake-compatible Go1 scene copy.
- Converted visual meshes from STL to OBJ.
- Kept original MuJoCo assets untouched.
- Verified Drake parser builds a plant.
- Verified model dimensions.

Gate:

```text
Drake parses Go1 scene successfully and reports nq=19, nv=18, actuated joints=12.
```

Evidence:

- Generated script: `scripts/prepare_go1_drake_assets.py`
- Generated scene: `src/unilab/assets/robots/go1/scene_flat_drake.xml`
- Generated robot XML: `src/unilab/assets/robots/go1/go1_drake.xml`
- Generated visual meshes: `src/unilab/assets/robots/go1/assets_drake/*.obj`
- Drake parse check reported:
  - `num_positions: 19`
  - `num_velocities: 18`
  - `num_actuated_dofs: 12`
  - `num_bodies: 14`
  - `num_joints: 13`
- Visual inspection:
  - Loaded the generated Drake scene in Meshcat at `http://localhost:7000`.
  - User confirmed the imported Go1 model was visible.

Notes:

- Drake still warns that several MuJoCo tags/attributes are ignored or approximated.
- Drake ignores MJCF `site`, `sensor`, and `keyframe` tags, so later backend work must reconstruct sites/sensors/keyframe state through UniLab-side code or XML parsing.
- Drake ignores MuJoCo collision filter groups and changes unsupported `condim` values to `condim=3`, so contact behavior will not be MuJoCo-identical.
- For Go1 foot contacts specifically, Drake downgrades `condim=6` to `condim=3` and ignores torsional/rolling friction. It also ignores joint `frictionloss`. These are prime suspects for the policy moving the legs without producing MuJoCo-like forward ground impulse.

### M2: Dirty Drake Replay Loop

Status: done; dirty Drake replay walks forward

Purpose: prove the policy can drive Drake before full backend cleanup.

Tasks:

- Loaded the trained RSL-RL actor manually from `actor_state_dict`.
- Built one Drake plant/context for Go1.
- Reset to Go1 home state.
- Produced the 49D actor observation from Drake state.
- Stepped policy -> action -> Drake-native PD desired joint target -> Drake.
- Clamped policy-derived position targets to MuJoCo actuator `ctrlrange`.
- Filtered robot self-collisions in Drake.
- Switched the dirty replay contact model to point-contact SAP.
- Published motion in Meshcat.

Gate:

```text
One Go1 policy replay loop runs in Drake for N steps without crashing and moves forward.
```

Evidence:

- Added dirty script: `scripts/drake_go1_dirty_replay.py`
- Headless smoke:

```bash
uv run --no-sync python scripts/drake_go1_dirty_replay.py \
  --no-meshcat --steps 60 --realtime-rate 0 --print-every 10 --hold 0
```

- Meshcat smoke:

```bash
uv run --no-sync python scripts/drake_go1_dirty_replay.py \
  --meshcat --steps 40 --realtime-rate 0 --print-every 10 --hold 0
```

- Meshcat URL reported by Drake: `http://localhost:7000`
- Verified actor checkpoint shape:
  - actor input: 49
  - actor output: 12
  - hidden dims: 512, 256, 128
- Verified Drake model order:
  - actuator order matches Go1 action order: FR, FL, RR, RL with hip/thigh/calf in each leg.
  - joint position starts are q[7:] and joint velocity starts are v[6:].
- Native PD zero-action smoke:
  - command: `uv run --no-sync python -u scripts/drake_go1_dirty_replay.py --no-meshcat --steps 100 --action-mode zero --realtime-rate 0 --print-every 50 --hold 0`
  - result after 2 seconds: `base_z=0.2542`, `linvel_x=-0.0002`, `tau_max=5.5066`
- Native PD policy smoke:
  - command: `uv run --no-sync python -u scripts/drake_go1_dirty_replay.py --no-meshcat --steps 150 --realtime-rate 0 --print-every 75 --hold 0`
  - result after 3 seconds: `base_z=0.2734`, `linvel_x=0.0552`, `action_max=2.6602`, `tau_max=16.5909`
- Native PD plus MuJoCo `ctrlrange` clamp plus point-contact SAP smoke:
  - command: `uv run --no-sync python -u scripts/drake_go1_dirty_replay.py --no-meshcat --steps 300 --realtime-rate 0 --print-every 50 --hold 0`
  - result after 6 seconds: `base_z=0.3107`, `linvel_x=0.5923`, `action_max=2.3590`, `tau_max=8.2632`
- Direct MuJoCo dirty A/B replay:
  - same actor, command, observation layout, default angles, control timestep, and `Kp=35/Kd=0.5`
  - result after 6 seconds: `base_z=0.301`, `linvel_x=0.670`, `action_max=2.376`
  - conclusion: dirty policy/control reconstruction is good enough to walk in MuJoCo; Drake's remaining failure is parity-specific.

M2 caveats:

- Drake parses Go1 MJCF position-actuator `ctrlrange` values as actuator effort limits. The dirty script overrides actuator effort limits to the MuJoCo `forcerange` values before finalizing the plant.
- The external torque-PD approximation still collapses to base contact and should be treated as a diagnostic mode, not the backend direction.
- Drake-native in-plant PD fixes the immediate collapse in zero-action and policy replay smoke tests.
- Forward velocity is now close to the MuJoCo dirty baseline but still lower, so contact/friction/damping parity remains open before judging policy quality.
- The current strongest remaining suspect is Drake physical parity, especially foot contact/friction/solver behavior and MJCF features Drake ignores or approximates. Sensor frame/sign parity should still be audited, but the same observation layout works in MuJoCo.
- No Drake MP4 recording path exists yet; Meshcat is the current visualization path.

### M3: Minimal `DrakeBackend`

Status: done

Tasks:

- Added `src/unilab/base/backend/drake/`.
- Implemented only the methods needed by `Go1JoystickFlat` replay.
- Added backend dispatch in `src/unilab/base/backend/__init__.py`.
- Allowed `drake` in `src/unilab/base/registry.py`.
- Registered `Go1JoystickFlat` with `sim_backend="drake"`.
- Added `conf/ppo/task/go1_joystick_flat/drake.yaml`.
- Preserved the M2 physics/control contracts:
  - Drake-native actuator PD,
  - MuJoCo `ctrlrange` target clamping,
  - restored Go1 actuator effort limits,
  - robot self-collision filtering,
  - point-contact SAP,
  - MuJoCo-shaped qpos/qvel at the UniLab boundary,
  - reconstructed Go1 joystick sensors.

Gate:

```text
UniLab can instantiate Go1JoystickFlat with sim_backend=drake and play_env_num=1.
```

Evidence:

- Registry smoke reports `Go1JoystickFlat` available backends include `drake`.
- Direct env smoke produced `obs=(1, 49)`, `critic=(1, 52)`, and one successful Drake step.
- Ruff check passed for the new/changed backend registration files.

### M4: UniLab Checkpoint Replay Through Drake

Status: done for headless replay; visualization still belongs to M5

Tasks:

- Ran UniLab's RSL-RL script path with `task=go1_joystick_flat/drake`.
- Loaded the MuJoCo-trained checkpoint.
- Executed headless playback for finite steps.
- Confirmed observation/action contract through the wrapper and no termination in a direct 120-step policy-loop smoke.

Gate:

```bash
uv run python scripts/train_rsl_rl.py \
  task=go1_joystick_flat/drake \
  training.play_only=true \
  algo.load_run=2026-06-05_02-36-19_mujoco \
  algo.checkpoint=150 \
  training.play_env_num=1 \
  training.play_steps=120 \
  training.play_render_mode=auto
```

Evidence:

- Script output:
  - loaded `model_150.pt`,
  - initialized Drake Go1 backend,
  - filtered `42` robot collision geometries,
  - resolved actor input `49` and critic input `52`,
  - printed `Done.` after headless playback frames.
- Direct wrapper smoke over 120 policy steps:
  - base moved by `delta_xy=[1.53087284, -0.14706941]`,
  - `terminated=[False]`,
  - obs shapes remained `{'obs': (1, 49), 'critic': (1, 52)}`.

### M5: Drake Visualization And Recording

Status: done for first-pass Meshcat playback and MP4 recording

Tasks:

- Added Drake Meshcat playback integration inside `DrakeBackend`.
- Supported `training.play_render_mode=interactive` for Drake through UniLab's playback contract.
- Kept `training.play_render_mode=auto` as finite headless playback.
- Added Drake native RGB recording with `RgbdSensor`, VTK rendering, and `mediapy.write_video`.
- Supported `training.play_render_mode=record` for finite single-env Drake replay.
- Updated generated OBJ assets to include face normals so Drake VTK can render them reliably.

Interactive gate:

```bash
uv run python scripts/train_rsl_rl.py \
  task=go1_joystick_flat/drake \
  training.play_only=true \
  algo.load_run=2026-06-05_02-36-19_mujoco \
  algo.checkpoint=150 \
  training.play_env_num=1 \
  training.play_steps=1800 \
  training.play_render_mode=interactive
```

Recording gate:

```bash
uv run python scripts/train_rsl_rl.py \
  task=go1_joystick_flat/drake \
  training.play_only=true \
  algo.load_run=2026-06-05_02-36-19_mujoco \
  algo.checkpoint=150 \
  training.play_env_num=1 \
  training.play_steps=120 \
  training.play_render_mode=record
```

Evidence:

- Interactive replay printed `Drake Meshcat: http://localhost:7000`.
- Long interactive smoke ran through UniLab's normal RSL-RL path and exited with `Done.`.
- The backend reported the same Drake Go1 setup as headless playback: `42` filtered robot collision geometries and actor/critic dimensions `49`/`52`.
- Recording replay wrote:
  - `logs/rsl_rl_ppo/Go1JoystickFlat/2026-06-05_02-36-19_mujoco/play_video.mp4`
- Video readback via `mediapy.read_video` reported:
  - shape `(120, 360, 640, 3)`
  - dtype `uint8`
- Extracted sample frame:
  - `/tmp/drake_record_frame30.png`
- Validation:
  - `uv run ruff check src/unilab/base/backend/drake scripts/prepare_go1_drake_assets.py`
  - `uv run python -m compileall src/unilab/base/backend/drake scripts/prepare_go1_drake_assets.py`

M5 caveats:

- The first recording camera is fixed and wide. It is good enough to verify motion, but later work should add camera presets or follow-camera behavior.
- Lighting is basic VTK directional lighting.
- Recording is still single-env replay only, matching the current Drake backend scope.

## Risks And Decisions

### Known Risks

- Drake MJCF support is not identical to MuJoCo MJCF support.
- Joint ordering may differ. We must map by joint names, not assume array order.
- Quaternion and angular velocity conventions must be checked carefully.
- Foot contact sensors will not be identical to MuJoCo contact sensors.
- A MuJoCo-trained checkpoint may behave poorly in Drake even if the backend contract is correct.
- Drake's MJCF parser maps Go1 position-actuator `ctrlrange` metadata to effort limits; Drake backend construction must restore Go1 `forcerange` effort limits explicitly or through a Drake-specific XML rewrite.
- Drake-native PD is the current control direction. External torque PD remains a debug comparison path.

### Current Decisions

- Use Go1, not G1, for the first checkpoint replay.
- First target is replay only, not training.
- First target is single env only.
- Disable domain randomization and pushes for Drake replay config.
- Start with Meshcat for visual debugging.
- Keep original MuJoCo assets unchanged.
- Use Drake-native actuator PD for the first backend implementation.
- Clamp Drake actuator position targets to MuJoCo `ctrlrange`.
- Use Drake point-contact SAP for the first Go1 replay backend attempt unless a stronger parity test says otherwise.
- Filter robot self-collisions during Drake model construction while preserving robot-floor contact.

## Dynamic Update Log

Use this section to record plan changes as we learn.

- 2026-06-06: Created initial plan. Scope is Go1 checkpoint replay through UniLab using Drake backend, starting with model import and dirty replay before full backend cleanup.
- 2026-06-06: Completed M1. Added a reproducible Go1 Drake asset preparation script, generated OBJ visual meshes and Drake XML copies, and verified Drake parses the generated scene with expected Go1 dimensions.
- 2026-06-06: Accepted M1 after Meshcat visual inspection of the imported Drake Go1 model.
- 2026-06-06: Completed M2 gate. Added `scripts/drake_go1_dirty_replay.py`, loaded the MuJoCo-trained Go1 actor, reconstructed the 49D observation from Drake state, stepped policy actions through a PD torque loop, and smoke-tested Meshcat playback. Caveat: current Drake approximation collapses quickly, so control/contact parity remains open.
- 2026-06-07: Switched dirty replay from external torque PD to Drake-native in-plant actuator PD. Zero-action replay now stands stably around `base_z=0.254` after 2 seconds, and policy replay stays upright through smoke tests. Remaining caveat is locomotion parity, not immediate collapse.
- 2026-06-07: Ran a direct MuJoCo dirty A/B replay with the same actor/command/observation/action mapping. MuJoCo reaches `linvel_x=0.670` by 6 seconds, while Drake stays near zero. This points to Drake parity, not a broken actor loop.
- 2026-06-07: Rechecked Drake parser warnings. Prime parity suspects are ignored collision filter groups, foot `condim=6` downgraded to `condim=3`, ignored torsional/rolling friction, and ignored joint `frictionloss`.
- 2026-06-07: Patched dirty Drake replay to always filter robot self-collisions, clamp policy-derived position targets to MuJoCo actuator `ctrlrange`, and use Drake point-contact SAP. Drake forward replay improved to `linvel_x=0.592` by 6 seconds, close to the direct MuJoCo dirty baseline `0.670`.
- 2026-06-07: User visually confirmed the improved Meshcat replay. Marked M2 as the first successful Drake replay milestone: the checkpoint-driven dirty Drake Go1 replay walks forward.
- 2026-06-07: Completed M3. Added a minimal single-env Go1 `DrakeBackend`, backend dispatch, registry support, Go1 task registration, and a Drake-specific PPO task config.
- 2026-06-07: Completed headless M4 smoke. UniLab's normal RSL-RL script loads `model_150.pt`, constructs `DrakeBackend`, and runs 120 headless playback frames to `Done.` with `training.play_render_mode=auto`.
- 2026-06-08: Completed the Meshcat half of M5. `DrakeBackend` now rebuilds its single-env Drake diagram with `MeshcatVisualizer` only when `training.play_render_mode=interactive` is requested. A long UniLab RSL-RL replay printed `Drake Meshcat: http://localhost:7000` and completed with `Done.`.
- 2026-06-08: Completed the first MP4 half of M5. `DrakeBackend` now supports `training.play_render_mode=record` by rebuilding with a Drake RGBD camera and VTK renderer, recording frames, and writing `play_video.mp4` with `mediapy`. The generated OBJ assets now include face normals for VTK rendering.
- 2026-06-10: Removed Python-side robot constants from `DrakeBackend`. Replay metadata now comes directly from the Drake model files: position actuator `ctrlrange`, `forcerange`, joint ranges, foot `framepos` site sensors, contact sensor names, and the `home` keyframe.

## Next Action

Decide the next contribution milestone after M5. Reasonable candidates are camera quality/follow-camera polish, deeper Drake-MuJoCo parity audits, or expanding the Drake backend contract beyond this single-env replay path.

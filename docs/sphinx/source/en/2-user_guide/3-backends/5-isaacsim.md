# IsaacSim Backend

UniLab's `isaacsim` backend runs IsaacSim 5.1.0 and IsaacLab v2.3.0 in a
dedicated Python 3.11 worker process. The host process keeps the regular
`SimBackend` NumPy contract; pipe messages carry lifecycle commands and shared
memory carries batched state. The current support boundary is headless physics
for the registered G1 flat task owners. The support matrix intentionally marks
the PPO and SAC owners as `Configured`, not `Tested`.

## Runtime boundary

IsaacSim 5.1.0 is installed in a separate Python 3.11 environment because the
main UniLab environment supports Python 3.10--3.13. The setup entry point is
`scripts/tools/setup_isaacsim_env.sh`; it installs under
`$UNILAB_ISAACSIM_HOME` (default `$HOME/.unilab/isaacsim`) and accepts the Kit
EULA through `OMNI_KIT_ACCEPT_EULA=1` for non-interactive worker startup.

The backend resolves these optional variables without importing Kit in the host
process:

- `UNILAB_ISAACSIM_HOME` selects the runtime root.
- `UNILAB_ISAACSIM_PYTHON` overrides the worker interpreter path.
- `OMNI_KIT_ACCEPT_EULA=1` keeps worker startup non-interactive.

The expected runtime layout is
`$UNILAB_ISAACSIM_HOME/venv/bin/python`, the Python 3.11 site-packages and
library directories under that venv, and an IsaacLab v2.3.0 source checkout at
`$UNILAB_ISAACSIM_HOME/IsaacLab`.

The first headless Kit launch may spend minutes warming extension and shader
caches. On the validated 595.x NVIDIA driver family, the standalone/full-app
path can crash in `librtx.scenedb.plugin.so`; this backend therefore does not
claim GUI, camera capture, or native playback support. The worker always starts
IsaacLab's headless `AppLauncher` path.

The current worker supports MJCF materialization, batched articulation state,
position-target stepping, and masked root/joint resets. Contact-force sensors,
reset or interval domain randomization, host pre-step callbacks, GUI rendering,
camera capture, and native playback are unsupported and fail closed.

Use the top-level CLI to select the backend and owner:

```bash
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
uv run eval --algo sac --task g1_walk_flat --sim isaacsim --render-mode none
```

These commands require the external runtime and an NVIDIA CUDA device. The
repository does not claim that either command has completed full training or
playback validation; those claims require a maintainer validation entry.

## Inspecting The Contract

```bash
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
OMNI_KIT_ACCEPT_EULA=1 \
uv run --active --no-project \
  scripts/tools/probe_isaacsim_contract.py \
  --model-file src/unilab/assets/robots/g1/scene_flat.xml \
  --num-envs 2 --steps 2 --device cuda:0 \
  --output /tmp/isaacsim-contract.json
```

The command is a bounded developer probe. It only touches the XML/importer
during cold-path materialization and is useful for checking a newly installed
runtime; it is not a training or playback validation.

## Contract matrix

| UniLab contract | IsaacSim/IsaacLab operation | Observed result | Production constraint |
|---|---|---|---|
| MJCF scene materialization | `isaaclab.sim.converters.MjcfConverter` | G1 MJCF converts to USD successfully | Enable `isaacsim.asset.importer.mjcf` explicitly in headless workers |
| Batched articulation | `Articulation` + `ArticulationCfg` | 2 environments, 29 joints, 30 bodies | Resolve names at materialization; importer order is not the MJCF order |
| Quaternion layout | `robot.data.root_quat_w` / `body_quat_w` | `wxyz` | Keep `wxyz` at the shared-memory boundary |
| Base angular velocity | `robot.data.root_ang_vel_w` | World frame | Public getter remains world-frame; reset qvel conversion is a cold-path contract operation |
| Partial reset | `write_root_pose_to_sim`, `write_root_velocity_to_sim`, `write_joint_state_to_sim`, `reset(env_ids)` | Selected row changes; other row deltas are zero | Use masked batched writes; reject duplicate/out-of-range ids |
| Position control | `set_joint_position_target`, `write_data_to_sim`, `SimulationContext.step` | Target moves the first joint over bounded steps | `step(ctrl)` carries position targets; gains/limits are materialized explicitly |
| State getter boundary | `Articulation.data.*` tensors | All getters are batched with expected leading dimension | Worker copies tensors to host-owned shared-memory slots; hot getters do not parse assets |
| Rendering | IsaacLab headless `AppLauncher` | Not claimed by this probe | Keep GUI/native rendering fail-closed until a separate driver-tested slice |
| Domain randomization | IsaacLab manager/event APIs | Not exercised | Non-empty unsupported plans must fail closed |

The importer returns a different joint/body ordering (for example, left/right
branches are interleaved). The worker builds name-to-index maps and reorders
every state/control array; positional assumptions would violate the
`SimBackend` index contract. The full owner and capability status is maintained
in {doc}`../../5-reference/5-support_matrix`.

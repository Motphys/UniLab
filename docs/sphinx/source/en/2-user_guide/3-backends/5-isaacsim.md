# IsaacSim Backend Feasibility

This page records the evidence collected for the IsaacSim 5.1.0 / IsaacLab
v2.3.0 runtime used by the planned UniLab backend. It is a feasibility record,
not a support declaration: the runtime is currently exercised only by the
bounded probe in `scripts/tools/probe_isaacsim_contract.py`.

## Runtime boundary

IsaacSim 5.1.0 is installed in a separate Python 3.11 environment because the
main UniLab environment supports Python 3.10--3.13. The setup entry point is
`scripts/tools/setup_isaacsim_env.sh`; it installs under
`$UNILAB_ISAACSIM_HOME` (default `$HOME/.unilab/isaacsim`) and accepts the Kit
EULA through `OMNI_KIT_ACCEPT_EULA=1` for non-interactive worker startup.

The first headless Kit launch may spend minutes warming extension and shader
caches. The probe deliberately uses the IsaacLab headless `AppLauncher` path.
Standalone GUI/full-app playback is not part of this feasibility claim: on the
validated 595.x NVIDIA driver family, the standalone path can crash in
`librtx.scenedb.plugin.so` (see the setup script's runtime note).

## Reproducing the probe

```bash
VIRTUAL_ENV="$HOME/.unilab/isaacsim/venv" \
OMNI_KIT_ACCEPT_EULA=1 \
uv run --active --no-project \
  scripts/tools/probe_isaacsim_contract.py \
  --model-file src/unilab/assets/robots/g1/scene_flat.xml \
  --num-envs 2 --steps 2 --device cuda:0 \
  --output /tmp/isaacsim-contract.json
```

The command is bounded and only touches the XML/importer during cold-path
materialization. Its output is the evidence source for the matrix below.

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
branches are interleaved). A worker must therefore build name-to-index maps and
reorder every state/control array; positional assumptions would violate the
`SimBackend` index contract.

## Current scope

This page and the probe do not register `isaacsim`, add a task owner, or claim
training/play support. Those changes belong to the implementation slices under
issue #1369 and must add their own conformance and runtime evidence.

# DrakeUni Architecture

This note is the shared mental model for DrakeUni. It is intentionally short:
implementation stages, test plans, and migration checklists belong in `.omx`
planning artifacts, not here.

## Goal

DrakeUni should let UniLab train and replay the Go1 joystick task with Drake as
the physics backend.

The target is not to fork Drake. The target is a thin DrakeUni layer over an
official Drake installation:

```text
UniLab task / MDP
    -> UniLab Drake backend
    -> DrakeUni
    -> official Drake
```

UniLab owns task semantics. DrakeUni owns Drake batch execution.

## Current Shape

The current DrakeUni prototype works, but it keeps heavy Drake runtime objects
per env:

```text
shared:
    Diagram
    MultibodyPlant
    SceneGraph
    static handles

per env:
    Simulator
    Context
    plant_context
```

This proves the Go1 path can work, but it is not the final large-batch shape.

## Target Shape

The target shape follows the same idea as MuJoCoUni:

```text
shared static world
    +
compact per-env state rows
    +
reusable per-thread runtime workspaces
```

In one control step:

```text
compact state[env]
    -> load into thread workspace
    -> apply control
    -> advance physics to next control dt
    -> write compact state_out[env]
    -> write sensor/observation output[env]
```

The important memory target is:

```text
O(num_static_worlds * static Drake world)
+ O(num_envs * compact state)
+ O(num_threads * Drake runtime workspace)
```

not:

```text
O(num_envs * full Drake Simulator/Context)
```

## Static Ownership

Static data describes the world:

- robot and scene description
- Drake `Diagram`
- Drake `MultibodyPlant`
- Drake `SceneGraph`
- model, body, joint, actuator, geometry, sensor, and port handles
- static control limits and gains

Static data should have one authoritative owner.

If all envs use the same world, all envs point to the same static world. If
randomization creates static variants, those variants should be stored once and
envs should point to them by id:

```text
env_to_world[env] -> static_world_id
```

The batch pool should not secretly duplicate static worlds unless a concrete
Drake lifetime or thread-safety constraint proves it necessary.

## Runtime Ownership

Runtime data changes while physics advances. The target per-env runtime is a
compact state row, not a full Drake `Context`.

For Go1, the first compact state contract is provisional:

```text
state = [time, q, v]
```

Control remains separate:

```text
control = target joint position
```

If parity tests prove Drake needs more persistent state, add only the minimum
extra field and document why.

Do not store derived data in the compact row if Drake can recompute it:

- body poses
- contacts
- sensor outputs
- constraint matrices
- solver scratch
- cache entries

## Worker Workspace

A worker workspace is the temporary Drake runtime object used to perform
physics stepping. It is not another owner of env state.

Target rule:

```text
one ThreadWorkspace per worker thread
```

Each workspace owns its own mutable runtime objects, such as a root `Context` or
`Simulator`, a `plant_context` pointer, and reusable control/force buffers.

Workspaces are thread-local:

```text
worker 0 owns workspace 0
worker 1 owns workspace 1
...
```

No workspace is shared by two threads during stepping.

## Batch Pool Role

The batch pool is an execution engine. It should:

- schedule envs across worker threads
- load compact state into a worker workspace
- apply already-resolved control and forces
- advance physics
- write compact state and sensors back to output arrays

The batch pool should not:

- own task semantics
- sample domain randomization
- rewrite rewards
- parse assets in the hot path
- expose Drake internals to UniLab env code

## Randomization

Randomization policy belongs outside the batch pool.

Correct flow:

```text
UniLab task/domain-randomization logic
    -> resolved low-level randomization values
    -> static-world update or env_to_world update
    -> compact runtime update if needed
    -> batch pool executes
```

The pool may mechanically apply already-resolved low-level values, but it should
not decide what randomization means.

## Boundary

The first supported boundary is still Go1 joystick.

Do not generalize DrakeUni before the Go1 compact-runtime contract is stable.

The public UniLab-facing shape should remain simple:

```text
reset/set state
step(control, nsteps)
read compact state and sensors
record/replay through existing UniLab paths
```

That is the architecture we are aiming for: compact runtime, explicit static
ownership, one workspace per worker thread, and a thin DrakeUni layer over
official Drake.

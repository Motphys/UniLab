# Motrix Contact Sensor Notes

## Background

The tactile observation for the Sharpa in-hand rotation task depends on
contact-sensor data (fingertip contact force against the object). The code
works under the MuJoCo backend, but Motrix returns contact-sensor data in a
different format, so the read path needs to account for both shapes.

## Contact Sensor Configuration

The fingertip contact sensors are defined in the robot scene XML
`src/unilab/assets/robots/sharpa_wave/right_sharpa_wave.xml`:

```xml
<contact name="contact_right_thumb_elastomer_force" geom1="right_thumb_elastomer" geom2="object" num="1" data="force" reduce="netforce"/>
<!-- one sensor per fingertip, same parameters -->
```

Parameters:

- `num="1"` — report at most one contact point.
- `data="force"` — return only the force data.
- `reduce="netforce"` — reduce to a single net contact force in the global
  frame.

## Return-Format Difference Between Backends

### MuJoCo

Returns a `(num_envs, 3)` force vector in the global frame:

```
[[fx, fy, fz],    # env 0
 [fx, fy, fz],    # env 1
 ...]
```

### Motrix

Returns a `(num_envs, 1 + num * stride)` flat array whose **first element is
the actual contact count**:

```
[[count, fx, fy, fz],    # env 0, shape = (4,) because num=1, stride=3
 [count, fx, fy, fz],    # env 1
 ...]
```

For `reduce="netforce"` + `data="force"` + `num=1`:

- shape = `(num_envs, 4)`
- `[0]` = contact count (0 or 1)
- `[1:4]` = net force vector in the global frame (`netforce` uses global
  coordinates)

### General multi-contact layout (Motrix)

With `num=4, data="force pos normal tangent"` and no reduce:

```
shape = (num_envs, 1 + 4 * 12) = (num_envs, 49)

[count,
 f1_normal, f1_tangent0, f1_tangent1,   # contact 1 force (contact frame)
 f1_x, f1_y, f1_z,                       # contact 1 position
 f1_nx, f1_ny, f1_nz,                    # contact 1 normal
 f1_tx, f1_ty, f1_tz,                    # contact 1 tangent
 f2_normal, f2_tangent0, f2_tangent1,    # contact 2 ...
 ...
 padding_zeros]                          # zero-padded up to num contacts
```

Note: without a reduce mode the force is expressed in the **contact frame**
(one normal scalar plus two tangent scalars), not a global `xyz` vector. Only
`reduce="netforce"` returns a global-frame force vector.

## Motrix `reduce` Mode Reference

| Reduce mode | Frame | Returns |
| --- | --- | --- |
| `netforce` | global | single reduced net force vector |
| `maxforce` | contact | the contact point with the largest force |
| `mindist` | contact | the contact point with the shallowest penetration |
| none | contact | the first `num` contact points |

Only `netforce` returns a global-frame force vector; the other modes return
contact-frame data (one normal scalar plus two tangent scalars).

## Why a Single Norm Branch Is Not Enough

The env reads tactile force through `_read_tactile_force()` →
`_extract_sensor_scalar()` in
`src/unilab/envs/manipulation/sharpa_inhand/base.py`. That helper currently
collapses any `(N, >=3)` array with `np.linalg.norm(data[:, :3], axis=1)`.

If the env still routes both backend shapes through that one branch, the
MuJoCo `(N, 3)` case is correct (`norm` of the real force vector), but the
Motrix `(N, 4)` case would be wrong: `data[:, :3]` would pick up
`[count, fx, fy]` — treating the contact count as a force component and
dropping `fz`. The fix is not to special-case shapes inside the env, but to
move the per-backend knowledge behind a backend method.

## Recommended Contract: a Backend Method for Force Magnitude

Add `get_contact_force_magnitude(sensor_name) -> np.ndarray` to the
`SimBackend` interface (`src/unilab/base/backend/base.py`), returning a
`(num_envs,)` scalar magnitude. Each backend implements it according to its
own data layout:

- **MuJoCo** (`src/unilab/base/backend/mujoco/backend.py`): take the norm of
  the 3D force vector returned by `get_sensor_data(name)`.
- **Motrix** (`src/unilab/base/backend/motrix/backend.py`): interpret the
  layout by reduce mode:
  - `reduce="netforce"`: take `[1:4]`, then norm.
  - no reduce, multiple contacts: sum the per-contact forces, then norm.
  - `reduce="maxforce"`: take the strongest contact.

The env's `_read_tactile_force()` then routes contact sensors through
`get_contact_force_magnitude()`, while ordinary scalar sensors keep using
`_extract_sensor_scalar()`.

This keeps the env backend-agnostic and aligns with the high-risk-area
invariant that the env layer only calls methods declared on `SimBackend` —
no feature leakage into the env, and a new backend only has to implement the
interface method. `get_sensor_data` is already a declared `SimBackend`
method, so the proposal extends the same boundary rather than reaching into a
backend subclass.

## Related Files

| File | Role |
| --- | --- |
| `src/unilab/envs/manipulation/sharpa_inhand/base.py` | `_extract_sensor_scalar()`, `_read_tactile_force()` |
| `src/unilab/envs/manipulation/sharpa_inhand/rotation.py` | reward computation, virtual torque |
| `src/unilab/assets/robots/sharpa_wave/right_sharpa_wave.xml` | contact-sensor XML definitions |
| `src/unilab/base/backend/motrix/backend.py` | Motrix `get_sensor_data()` |
| `src/unilab/base/backend/mujoco/backend.py` | MuJoCo `get_sensor_data()` |
| `src/unilab/base/backend/base.py` | `SimBackend` interface |

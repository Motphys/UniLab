# MuJoCo Backend

The MuJoCo backend uses `mujoco-uni` (a fork pinned to `unilabsim/mujoco_uni`).

- Asset format: MJCF (XML).
- Default `dt`: 0.005 s for locomotion, 0.002 s for in-hand manipulation.
- Solver: default Newton with `iterations=30`.

See {py:mod}`unilab.base.backend.mujoco` for the adapter API.

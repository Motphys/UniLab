# Manager Term Reference

Lookup tables for the built-in Manager-Based reward, observation, and
termination terms shipped with UniLab, organized by owner layer:
framework-level terms in `unilab.envs.mdp` and task-owned terms under
`src/unilab/tasks/`. Every name, parameter, and default in these tables is
compiled from the linked source modules; the authoritative signatures are the
API reference.

:::{only} api_ref
```{raw} html
<ul>
  <li>Framework terms: <a href="../../../api_reference/envs/index.html"><code>unilab.envs</code> API reference</a></li>
  <li>Task terms: <a href="../../../api_reference/tasks/locomotion.html"><code>unilab.tasks.locomotion</code> API reference</a>,
      <a href="../../../api_reference/tasks/motion_tracking.html"><code>unilab.tasks.motion_tracking</code> API reference</a></li>
</ul>
```
:::

:::{only} prose_only
The API reference is skipped in prose-only preview builds. Authoritative
signatures live in the source tree:
[unilab/envs/mdp](https://github.com/unilabsim/UniLab/tree/main/src/unilab/envs/mdp),
[unilab/tasks](https://github.com/unilabsim/UniLab/tree/main/src/unilab/tasks).
:::

## How terms are referenced

Terms are declared in the task owner YAML (for example
`src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml`) as `func + params`
entries. `func` is an importable `module.path.name`; all framework-level terms
are re-exported at `unilab.envs.mdp.<name>`, which is the path the owner YAMLs
use. Class terms are instantiated once with `(cfg, env)` at manager
construction and then called per step; plain functions are called per step. In
both cases the manager invokes the term as `func(env, **params)`
(`src/unilab/managers/manager_base.py`, `src/unilab/managers/reward_manager.py`),
so the `params:` mapping must match the term's accepted parameters.

- Reward terms live under the top-level `reward:` mapping with `func`,
  `weight`, and optional `params`:
  ```yaml
  reward:
    action_rate:
      func: unilab.envs.mdp.action_rate_l2
      weight: -0.005
  ```
- Observation terms live under `env.observations.<group>.terms.<name>` with
  `func` and optional `params`:
  ```yaml
  env:
    observations:
      policy:
        terms:
          joint_pos:
            func: unilab.envs.mdp.joint_pos_rel
  ```
- Termination terms live under `env.terminations.<name>` with `func` and
  optional `params`; adding `time_out: true` marks the term as an episodic
  timeout (truncation) instead of a failure (`TerminationTermCfg.time_out` in
  `src/unilab/managers/termination_manager.py`):
  ```yaml
  env:
    terminations:
      time_out:
        func: unilab.envs.mdp.time_out
        time_out: true
  ```

Terms that need a backend capability — named sensors, ray casters, reset
fields — bind that capability at construction and raise when the backend
cannot provide it; they fail closed instead of silently degrading (see the
`bind_sensor_data` binding in `src/unilab/envs/mdp/observations.py` and the
`_ResetRandomizationTerm` contract in `src/unilab/envs/mdp/events.py`).

## Scope

These tables cover reward, observation, and termination terms. Task packages
also own term kinds configured in other YAML sections, which are not listed in
the tables:

- Command and action terms, for example
  `unilab.envs.mdp.commands.UniformVelocityCommandCfg`,
  `unilab.tasks.locomotion.g1.manager_terms.G1VelocityCommandCfg` (G1 velocity
  command with a 0.2 planar-norm dead zone), and
  `unilab.tasks.motion_tracking.common.manager_terms.MotionCommandCfg` /
  `MotionJointPositionActionCfg` (motion reference command and its joint
  position action).
- Curriculum terms, for example
  `unilab.tasks.locomotion.g1.manager_terms.G1PenaltyCurriculum` (declared under
  `curriculum:`; scales negative-weight reward terms between
  `initial_scale`/`min_scale` and `max_scale` as the tracked average episode
  length crosses `level_down_threshold`/`level_up_threshold`, by `degree` per
  update).
- Event (domain-randomization and reset) terms in
  `unilab.envs.mdp.events`, declared under `env.events`.
- Reset owners, for example
  `unilab.tasks.locomotion.g1.manager_terms.G1WalkResetOwner`.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Reward terms
:link: 1-rewards
:link-type: doc
Framework, locomotion, G1, and motion-tracking reward terms with parameters.
:::

:::{grid-item-card} Observation terms
:link: 2-observations
:link-type: doc
State, sensor, command, gait-phase, and ray-scan observation terms.
:::

:::{grid-item-card} Termination terms
:link: 3-terminations
:link-type: doc
Timeout and failure conditions, including fused termination packs.
:::

::::

```{toctree}
:hidden:

1-rewards
2-observations
3-terminations
```

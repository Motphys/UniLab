# Latency and Observation Lag

Sim worlds are instantaneous. Real robots are not. This page covers how
UniLab injects realistic delay during training so policies don't melt down
when they meet hardware.

## The two delays that matter

1. **Action latency** — time between policy emitting `aₜ` and the motor
   responding. Composed of (a) network/transport (5–10 ms), (b) driver
   filter (1–5 ms), and (c) actuator electrical/mechanical response (~τ).
2. **Observation lag** — time between sensor sampling and the value being
   visible to policy input. Composed of (a) sensor processing, (b)
   transport, (c) state estimator dynamics.

Both can be lumped into a *single equivalent delay* for many robots
(typical 15–30 ms total). UniLab supports them separately, because they
are mitigated differently.

## How to inject in sim

### Action latency

Wrap the env (or implement at backend layer) with an N-step action queue:

```python
from unilab.base.np_env import NpEnv

class ActionLatencyWrapper(NpEnv):
    def __init__(self, env, latency_steps_range=(0, 2)):
        ...
        self._queue = deque(maxlen=max(latency_steps_range) + 1)

    def step(self, action):
        self._queue.append(action)
        delayed = self._queue[-self._current_latency - 1]
        return self.env.step(delayed)
```

In UniLab this is typically expressed via DR: sample
`torque_delay_ms ∈ [0, 20]` per episode. See the recipe in
{doc}`domain_randomization_for_real`.

### Observation lag

Buffer observations in a ring and emit `obs[-lag]`:

```python
class ObservationLagWrapper(NpEnv):
    def __init__(self, env, lag_steps_range=(0, 2)):
        ...
        self._buf = deque(maxlen=max(lag_steps_range) + 1)

    def reset(self, ...):
        obs, info = self.env.reset(...)
        self._buf.clear()
        self._buf.append(obs)
        return obs, info

    def step(self, action):
        obs, *rest = self.env.step(action)
        self._buf.append(obs)
        return self._buf[-self._current_lag - 1], *rest
```

::::{admonition} Don't lag everything
:class: tip
**Joystick commands** and **target rotation** are not sensor readings; do
NOT lag them. Lag joint encoders, IMU, and vision-derived signals.
::::

## How to measure on hardware

1. **Action loopback**: command a step `Δ=0.1 rad`. Record the time
   between command timestamp and the moment encoder reading crosses 50% of
   target. That is your effective action latency.
2. **Observation timestamp**: log `(sensor_timestamp, policy_recv_timestamp,
   policy_step_timestamp)` per step. The difference is your observation
   lag (which should be roughly constant; high variance indicates a
   scheduling issue).

## How much DR is enough

DR latency range should **straddle measured hardware latency by at least
50%**. If you measure 18 ms on hardware, train with
`torque_delay_ms ∈ [0, 30]`. Centering the DR on the measurement leaves no
safety margin for drift over a deployment.

## Symptoms of latency mismatch

- **Low-frequency oscillation** of the robot, especially around contact —
  classic under-trained latency.
- **Action saturation** at episode start as the policy "fights" stale
  observations.
- **Drift / overshoot** on velocity tracking — usually obs lag, not action.

## See also

- {doc}`domain_randomization_for_real`
- {doc}`safety_layers`
- {py:mod}`unilab.dr.manager`

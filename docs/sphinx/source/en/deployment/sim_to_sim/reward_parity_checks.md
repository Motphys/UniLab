# Reward Parity Checks Across Backends

Two backends with "the same" reward function rarely produce **numerically
identical** rewards — and that's fine. What you want is **trajectory-level
parity**: the same policy, applied to the same initial state, produces a
similar reward curve.

## The protocol

1. Freeze a **fixed seed**, fixed initial state, fixed action sequence.
   The action sequence can be a sinusoidal sweep over joints, or a replay
   from a real rollout — anything deterministic.
2. Replay it in both backends. Log per-step reward components (not just
   the scalar total).
3. For each reward term, plot `r_backend_A` vs `r_backend_B` over time.
   Compute correlation and mean absolute deviation.

```python
from unilab.training.reward import RewardLogger

rl = RewardLogger(env)
for action in fixed_action_sequence:
    _, _, _, info = env.step(action)
    rl.record(info["reward_components"])
rl.dump("rewards_<backend>.npz")
```

## What good parity looks like

| Term type | Acceptable parity |
|---|---|
| Smooth penalty (`-α‖v − v*‖²`) | Correlation > 0.95, MAD < 5% |
| Contact-conditional bonus (`+β if foot_contact else 0`) | Correlation may be lower; check timing of contact events |
| Termination penalty | Trigger frames should match within ±2 sim steps |

## What's a red flag

- **Diverging reward late in episode.** Usually means the policy explores
  different state distributions in each backend, which usually means
  contact / friction mismatch. See
  {doc}`contact_and_friction_alignment`.
- **One reward term zero in one backend.** Capability gap: the term reads
  a feature one backend doesn't expose. See
  {doc}`known_capability_gaps`.

## Automating it

The repository does not currently include a standalone reward-parity helper in
`scripts/`. When adding parity coverage, keep the test close to the
backend/task-owner boundary: compose both owner YAMLs, reset with a fixed seed,
replay a deterministic action sequence, and assert on the logged reward
components under `tests/`.

## See also

- {doc}`contact_and_friction_alignment`
- {doc}`../../developer_guide/contracts/backend_capability`

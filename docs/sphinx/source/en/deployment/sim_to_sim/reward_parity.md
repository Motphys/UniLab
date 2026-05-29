# Reward Parity Across Backends

Two backends with "the same" reward function rarely produce **numerically
identical** rewards — and that's fine. What you want is **trajectory-level
parity**: the same policy, applied to the same initial state, produces a
similar reward curve.

## The protocol

1. Freeze a **fixed seed**, fixed initial state, fixed action sequence.
   The action sequence can be a sinusoidal sweep over joints, or a replay
   from a real rollout — anything deterministic.
2. Replay it in both backends. Log per-step reward components from
   `info["log"]`; locomotion reward dispatch writes `reward/<term>` entries
   there when reward logging is enabled.
3. For each reward term, compare the two time series and inspect the first
   frame where they diverge.

## What good parity looks like

| Term type | What to inspect |
|---|---|
| Smooth penalties | Same units, frames, and command inputs. |
| Contact-conditional terms | Contact timing and sensor availability in both backends. |
| Termination penalties | The termination mask and final-observation path. |

## What's a red flag

- **Diverging reward late in episode.** Usually means the policy explores
  different state distributions in each backend, which usually means
  contact / friction mismatch. See
  {doc}`contact_and_friction_alignment`.
- **One reward term zero in one backend.** Capability gap: the term reads
  a feature one backend doesn't expose. See
  {doc}`capability_gaps`.

## Automating it

The repository does not currently include a standalone reward-parity helper in
`scripts/`. When adding parity coverage, keep the test close to the
backend/task-owner boundary: compose both owner YAMLs, reset with a fixed seed,
replay a deterministic action sequence, and assert on the logged reward
components under `tests/`.

## See also

- {doc}`contact_and_friction_alignment`
- {doc}`../../developer_guide/contracts/backend_contract`

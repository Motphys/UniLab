# Aligning Contact and Friction Between Backends

Contact handling is a common source of MuJoCo-vs-Motrix drift. This page shows
you how to identify and close the gap.

## Diagnostic: probe contact early

Run the same owner YAML, reset seed, and deterministic action sequence in both
backends. Log contact-dependent reward terms from `info["log"]` and any
backend-owned contact signals the env already uses. If the nominal contact
response differs before policy learning is involved, fix the friction / damping
/ restitution declarations before debugging reward parity.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Friction declared on geom in MuJoCo but on material in Motrix | One backend has μ=0 | Declare on both; verify in scene export |
| Solver iterations too low | MuJoCo sinks into floor | Increase the backend-owned solver setting in the owner YAML |
| Contact-pair specific overrides | Inconsistent slip | Make pair-level overrides explicit in both backend YAMLs |
| Restitution mismatch | Bouncy vs sticky landing | Set explicitly; defaults differ |

## Aligning DR ranges

Once contact is aligned for the **nominal** parameters, audit DR:

- `friction_static_mu` and `friction_dynamic_mu` should be sampled
  identically in both backends.
- If a DR field is a no-op in one backend, log a warning at episode init.
  Silent ignorance leads to silent reward drift.

## See also

- {doc}`4-reward_parity`
- {doc}`../1-sim_to_real/6-domain_randomization`
- {doc}`6-capability_gaps`

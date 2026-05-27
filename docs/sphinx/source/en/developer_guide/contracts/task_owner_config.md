# Task Owner Config Contract

The owner YAML at `conf/<algo>/<task>/<backend>.yaml` is the **identity**
of a (task, backend, algorithm) combination. Rules:

- `training.sim_backend` echoes the filename's backend; it is not an override.
- Cross-backend behaviour differences live in different owner YAMLs, never in Python.
- Owner YAMLs may import shared snippets, but no Python-side conditional dispatch on backend identity.

See ADR-0003 ({doc}`../adr/ADR-0003-task-owner-and-config-compose-contract`).

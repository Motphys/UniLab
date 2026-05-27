# Env Contract

Every UniLab task subclasses {py:class}`unilab.base.np_env.NpEnv` and must obey:

- `obs` is a **dict** of NumPy arrays; never a single tensor.
- `reset(seed=None)` returns `(obs_dict, info_dict)`.
- `step(action)` returns `(obs_dict, reward, terminated, truncated, info)`.
- `obs_groups_spec` declares observation group shapes — wrappers and learners trust this.

See ADR-0005 ({doc}`../adr/ADR-0005-unified-obs-critic-env-and-ipc-contract`) for the IPC implications.

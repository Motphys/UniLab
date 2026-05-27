# Configuration Overrides

UniLab uses Hydra for config composition. The owner YAML for `task=<task>/<backend>` is the source of truth; you override individual fields on the command line.

```bash
uv run train --algo ppo --task go2_joystick_flat --sim motrix \
    training.num_iterations=2000 \
    training.use_amp=true \
    runner.seed=42
```

::::{admonition} What you cannot override
:class: warning

`training.sim_backend` is an **identity field**, not a switch. Use `--sim <backend>` to select the owner YAML — overriding the field directly is rejected by the registry. See {doc}`../../developer_guide/contracts/task_owner_config`.
::::

See the Hydra documentation for the full override syntax.

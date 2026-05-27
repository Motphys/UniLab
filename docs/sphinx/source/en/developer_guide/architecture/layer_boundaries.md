# Layer Boundaries

UniLab enforces strict layering:

1. **Env** — task semantics, reward, observation building.
2. **Backend** — physics simulation; declares capabilities.
3. **Runner** — drives env + algorithm; manages lifecycle.
4. **Algorithm** — PPO/SAC/TD3 + variants.
5. **CLI / Hydra** — composition; never carries business rules.

A function may **only** call into the layer directly below it. Cross-layer calls (e.g. algorithm reaches into backend) are a contract violation. See {doc}`../adr/ADR-0001-runtime-model-and-layer-boundaries`.

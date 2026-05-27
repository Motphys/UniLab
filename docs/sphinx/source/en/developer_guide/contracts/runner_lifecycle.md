# Runner Lifecycle

Every runner — synchronous PPO, async APPO, off-policy SAC/TD3 — follows the same lifecycle:

1. **Init** — registry bootstrap; resolve owner YAML.
2. **Materialize** — build scene + backend (cold path).
3. **Reset** — first env reset; produce initial obs batch.
4. **Train loop** — collector(s) feed learner via IPC.
5. **Checkpoint** — periodic save under `runs/<run>/`.
6. **Shutdown** — drain queues, flush logs, exit cleanly.

Runners may NOT bypass this lifecycle to start ad-hoc collector / learner pairs. See {py:mod}`unilab.training.run`.

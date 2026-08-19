# mjlab Cartpole migration fixture

This test-only fixture is derived from mjlab v1.6.0 commit
`0fb8a681136be94ffc636a3dd423cabb97d91f10`, specifically
`src/mjlab/tasks/cartpole/cartpole_env_cfg.py` and `cartpole.xml`. The source is
Apache-2.0; provenance is retained in the derived files.

| Surface | Status | Migration delta |
| --- | --- | --- |
| Manager dictionaries, term names/order, `func + params` | Compatible | Imports change from `mjlab` to `unilab`; the Balance task keeps all source entries and ordering. |
| Term math and buffers | Adapted | `torch.Tensor` and Torch ops become `np.ndarray` and NumPy ops. |
| Config container | Adapted | The Python config factory becomes one Hydra owner YAML, materialized as a plain `ManagerBasedRlEnvCfg`. |
| Scene/simulation | Adapted | mjlab scene/sim/viewer objects become task-owned MJCF plus `SceneCfg`/`EntityCfg`; the source contact-disable setting is embedded and visual materials are inlined for replicated scenes. |
| Joint effort and reset mutation | Adapted, fixture-local | Shared test-only adapters write through the entity control/reset contracts; they are not public built-ins. |
| mjlab runner, viewer, Torch/Warp runtime | Unsupported | No dependency or fallback is provided. |

This fixture is evidence for #1042, not a production registration or a claim
that arbitrary mjlab tasks run unchanged.

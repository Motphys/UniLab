# Isaac Lab Cartpole migration fixture

This test-only fixture is derived from Isaac Lab commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`, specifically its manager-based
Cartpole config and `joint_pos_target_l2` reward. The source is BSD-3-Clause;
the retained notice is in `task.py`.

The fixture is evidence for #1042, not a production task or a claim that all
Isaac Lab tasks run unchanged.

| Surface | Status | Migration delta |
| --- | --- | --- |
| Manager/term names, `func + params`, dict order | Compatible | Imports change from `isaaclab` to `unilab`; all 12 terms keep their source names and order. |
| Term math and buffers | Adapted | `torch.Tensor` and Torch ops become `np.ndarray` and NumPy ops. |
| Config container | Adapted | Nested `@configclass` objects become one Hydra owner YAML, materialized as a plain `ManagerBasedRlEnvCfg`. |
| Scene | Adapted | USD/`InteractiveSceneCfg` becomes a minimal task-owned MJCF plus `SceneCfg`/`EntityCfg`. |
| Joint effort action | Adapted, fixture-local | A thin action adapter writes the entity control buffer; it is deliberately not exported as a public built-in. |
| Reset mutation | Adapted, fixture-local | The same two reset terms write through the scoped entity reset transaction. |
| PhysX, USD, Isaac renderer | Unsupported | No dependency or fallback is provided. |

The fixture adds no production registry entry, public manager/backend/lifecycle
contract, training-script branch, runner/IPC path, or external runtime dependency.

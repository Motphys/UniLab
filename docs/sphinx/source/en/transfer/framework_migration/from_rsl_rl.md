# Migrating from RSL-RL

You're already using RSL-RL standalone? Good news: UniLab ships RSL-RL PPO
as one of its supported algorithms ({py:mod}`unilab.algos.torch.rsl_rl_ppo`)
and it's nearly drop-in.

## What you gain by moving inside UniLab

1. **Env contract.** RSL-RL leaves env structure to you. UniLab's
   {py:class}`unilab.base.np_env.NpEnv` standardizes obs/info/reset
   signatures, makes parallelism and resets less error-prone.
2. **Task owners.** Hydra-based config compose, plus the registry-driven
   backend / task / algo selection. No more bespoke train scripts per
   robot.
3. **Async runner.** Wrap RSL-RL PPO inside
   {py:mod}`unilab.algos.torch.appo` for higher throughput on machines
   with many CPU cores.
4. **Deployment story.** ONNX export with the right wrapper, safety
   layer documentation, and the
   {doc}`../sim_to_real/overview` pipeline.

## What you don't lose

- Same PPO algorithm and hyperparameters — RSL-RL is wrapped, not
  reimplemented.
- Checkpoint compatibility: `runs/<run>/model_*.pt` files from UniLab can
  be loaded by stock RSL-RL given a matching policy architecture.

## Migration steps

1. Move your env into `unilab.envs.<family>.<task>/`.
2. Convert your training config to a Hydra group under `conf/ppo/<task>/`.
3. Run `uv run train --algo ppo --task <task> --sim <backend>`.
4. Compare reward curves against your standalone RSL-RL baseline.

## When NOT to migrate

If you have an existing pipeline that works, no real-hardware target, and
you mostly need RSL-RL for the algorithm — UniLab is overkill. Use it
when you need any of: multi-backend, async collection, ONNX deployment,
or task registration.

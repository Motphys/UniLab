# W&B and TensorBoard

UniLab logs to both W&B (when `WANDB_API_KEY` is set) and TensorBoard. To toggle:

```bash
uv run train --algo ppo --task <task> --sim <backend> \
    logging.wandb.enabled=false \
    logging.tensorboard.enabled=true
```

See {py:mod}`unilab.logging` for the adapter implementations.

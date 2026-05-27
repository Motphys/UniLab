# Runtime Model

UniLab's runtime model is documented in detail in {doc}`development_standard` and codified by {doc}`../adr/ADR-0001-runtime-model-and-layer-boundaries`. Read those two first.

Quick summary:

```
[ Collector workers (CPU sim) ] ──shared mem──► [ Learner (GPU) ]
        ▲                                              │
        └──────── weight sync ◄────────────────────────┘
```

- Collectors run the env loop + simulator step on CPU; one process per worker.
- The learner consumes batches from shared memory and computes gradients on GPU.
- Weight sync happens asynchronously, bounded by a staleness budget.

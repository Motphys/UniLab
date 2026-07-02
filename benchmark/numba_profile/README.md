# numba_profile — fused `update_state` for `g1_motion_tracking`

Standalone profile answering: **how much does a Numba prange fusion of the
single-threaded "Env overhead" (obs/reward/termination) buy, and can it be done
without wrecking the structured, config-driven reward code?**

Follow-up to issues **#651** (collector active-throughput target), **#663**
(parallelise Env overhead), and **#665** (Numba prange fusion ceiling). This
directory validates the task-specific `g1_motion_tracking` reward + termination
hot slice, implements the structured Numba scheme, and measures **speed** and
**numerical consistency** side by side.

No MuJoCo/Motrix needed: this profile starts after backend state has already been
materialised into typed arrays, then runs the pure array math over robot state
vs. a motion reference. We synthesise those arrays at the real shapes/dtypes.

## Files

| File | Role |
|------|------|
| `spec.py` | Single source of truth — dims, `RewardConfig` scales/stds, thresholds, `TERM_ORDER`. Mirrors `tracking.py` (line refs inline). |
| `state.py` | Deterministic synthetic batch at real shapes (N envs × 14 bodies × 29 DoF), ~2% terminations. |
| `numpy_reference.py` | **Golden oracle** — each `_reward_*` / `_compute_terminations` ported from `tracking.py`, faithful math. |
| `numba_terms.py` | Single-source `@njit(inline="always")` scalar device fns, one per term (`.py_func` debuggable). |
| `numba_fused.py` | `@njit(parallel=True)` prange driver: static superset gated by a scale vector, per-thread log scratch, numpy fallback. |
| `test_parity.py` | Consistency: numba vs numpy on reward, terminations, per-term log; device `.py_func` vs numpy. |
| `bench.py` | Speed: numpy vs numba across `num_envs` and thread counts. |

## Run

```bash
uv run --with numba python -m benchmark.numba_profile.test_parity      # consistency
uv run --with numba python -m benchmark.numba_profile.bench            # speed (8k + 32k)
PROBE_NUM_ENVS=32768 uv run --with numba python -m benchmark.numba_profile.bench
```

`numba` is the only extra dep. Use `uv run --with numba ...` for a one-shot run,
or install it once with `uv pip install numba`.

## Faithfulness — how this maps to the real task

The hot-slice math is copied from `src/unilab/envs/motion_tracking/g1/tracking.py`:

- **11 default reward-scale terms** (`RewardConfig.scales`, tracking.py:44) with
  the exact default `scales` and `std_*`. Default has 8 nonzero-scale terms;
  `motion_ee_body_pos_z` / `motion_joint_pos` / `motion_joint_vel` are `0.0` and
  skipped, exactly as the real loop does. The real `_init_reward_functions`
  registry also contains `undesired_contacts`, but the default config has no
  scale for it, so it is outside this default-scale profile.
- **Reward math** per term ported 1:1 — anchor pos/ori (`_quat_error` via
  `2·acos|dot|`), per-body mean xyz error, joint-limit violation, action-rate L2,
  `exp(-err/std²)`, final `reward *= ctrl_dt` (0.02).
- **Terminations** (`_compute_terminations`, tracking.py:978): anchor-Z,
  anchor-orientation via body-frame gravity-z (tracking.py:276), end-effector-Z.
- **Dims**: `NUM_ACTION=29`, 14 tracked bodies, anchor = `torso_link` (idx 7),
  ee = `{ankles, wrists}` (idx 3,6,10,13).

The one deviation is intentional: we feed **synthetic** robot/motion arrays
instead of stepping physics, because the fusion target is Env overhead, not
physics. Magnitudes are tuned so rewards are non-degenerate and ~2% of envs
terminate, matching the reset/copy pressure used in the #665 probe.

## The structured scheme — speed *and* maintainability

The design keeps every property of the original config-driven reward code while
fusing the machine code:

1. **Single-source terms** (`numba_terms.py`). One scalar `@njit(inline="always")`
   function per term. The kernel inlines them → machine-code fusion (no
   intermediate `(N,·)` arrays), but the source keeps one readable function per
   term, and `fn.py_func` runs/debugs in plain Python. You give up *vectorised
   numpy style* and nothing else.
2. **Static superset + scale vector** (`numba_fused.py`). No codegen, no runtime
   dict (nopython can't). The kernel evaluates the full `TERM_ORDER` superset;
   a dense `scale` vector built on the **cold path** from the config dict gates
   each term (`scale==0` → contributes 0, matching the numpy `continue`).
   **Weights stay in config**, never baked into compiled code.
3. **Registry / single ordering.** `spec.TERM_ORDER` owns name↔index for the
   numpy oracle, the kernel, and the log. `numpy_reference.NUMPY_TERMS` pairs
   each name with its numpy fn; `test_parity` walks them.
4. **Per-thread log scratch.** Per-term reward means accumulate into
   `(nthreads, N_TERMS)` indexed by `get_thread_id()`, summed on the main thread
   after. A shared `log[k] += …` would false-share one cache line and cap
   scaling at ~5x (#665 §2) — the correctness of this is why the parity test
   checks every per-term log value, not just the total.
5. **Consistency as a gate.** `numpy_reference.py` is retained as the oracle;
   parity is `rtol=1e-4` (fastmath/FMA reorders float ops, so bit-exactness is
   not expected). Adding a term = numpy fn + njit fn + one `TERM_ORDER` entry;
   the test enforces they agree.
6. **Fallback.** `update_state(force="auto")` drops to numpy below 256 envs where
   launch/barrier cost dominates.

## Results (Xeon 8568Y+, 160T — standalone profile run)

`update_state` slice only (reward + termination), float32:

| num_envs | numpy | numba 1T (fusion) | numba best | speedup |
|---------:|------:|------------------:|-----------:|--------:|
| 8 192  | 8.63 ms | 2.67 ms (**3.2x**) | 0.33 ms @ 48T | **26.6x** |
| 32 768 | 35.57 ms | 10.65 ms (**3.3x**) | 0.73 ms @ 64T | **49.0x** |

Consistent with #665's thesis: single-core **fusion ≈ 3.2–3.3x** (kills the
per-term intermediate arrays), **parallel ≈ 8–15x** on top in this standalone
profile, and **larger batch scales higher** (barrier/launch cost amortised).
Parity: total reward and every per-term log match to tolerance; terminations
exact.

## Scope / caveats (from #665)

- **This is half the Env overhead.** `reset_done` (backend `set_state`) and the
  4× RNG `standard_normal` for obs noise are separate ceilings Numba prange does
  not address here.
- **Obs assembly** (concat) is left in numpy — it's concat-bound, not the fusion
  win; a real integration would share the same fused kernel boundary.
- Numbers are the `update_state` micro-slice; end-to-end collector throughput
  gains are diluted by physics and reset. #665's motion_tracking overhead −71%,
  ~+72% throughput figure is a projection for that server/profile, not a result
  from an integrated training path.
- **Warmup**: first call JIT-compiles (~0.1 s). `cache=True` persists it to
  `__pycache__/*.nbi` across processes.

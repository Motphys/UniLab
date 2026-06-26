import inspect
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from unilab.base.backend.mujoco import chunk_tuner as ct


def test_make_candidates_never_exceeds_upper():
    # upper = ceil(num_envs/nthread): a chunk beyond it yields fewer chunks than
    # threads -> idle threads -> always slower, so it must never be a candidate.
    for num_envs, nthread in [(4096, 16), (2048, 40), (1024, 40), (4096, 40)]:
        upper = math.ceil(num_envs / nthread)
        cands = ct.make_candidates(num_envs, nthread)
        assert max(cands) == upper, (num_envs, nthread, max(cands), upper)
        assert all(1 <= c <= upper for c in cands)
        assert 1 in cands
        assert cands == sorted(set(cands))


def test_make_candidates_covers_sweet_spot():
    # optimum sits near heur = num_envs/(10*nthread); a candidate must land in the band.
    for num_envs, nthread, lo, hi in [(2048, 40, 3, 9), (1024, 40, 2, 7), (4096, 40, 8, 41)]:
        cands = ct.make_candidates(num_envs, nthread)
        assert any(lo <= c <= hi for c in cands), (num_envs, nthread, cands)


def test_make_candidates_regression_g1walk_optimum_present():
    # for 2048 envs / 40 threads the sweet-spot chunks (~4-6) must be candidates and
    # the always-bad large chunks (> upper) must be excluded.
    cands = ct.make_candidates(2048, 40)
    assert any(4 <= c <= 6 for c in cands), cands
    assert 1024 not in cands and 2048 not in cands


def test_make_candidates_num_envs_one():
    assert ct.make_candidates(num_envs=1, nthread=8) == [1]


def test_make_candidates_fewer_envs_than_threads():
    # nbatch <= nthread -> upper=1 -> only chunk_size=1 is sensible.
    assert ct.make_candidates(num_envs=20, nthread=40) == [1]


def test_make_candidates_rejects_bad_num_envs():
    with pytest.raises(ValueError):
        ct.make_candidates(num_envs=0, nthread=8)


def test_filter_candidates_clamps_and_dedups():
    out = ct.filter_candidates([0, 1, 1, 5, 4096, 9000], num_envs=4096)
    assert out == [1, 5, 4096]


def test_filter_candidates_capping_keeps_low_band_and_top_anchor():
    # capping must NOT drop the low/optimum band (the old log-spacing bug); it trims
    # the coarse middle, keeping the smallest values + the coarsest anchor.
    out = ct.filter_candidates(list(range(1, 60)), num_envs=4096, max_candidates=8)
    assert len(out) <= 8
    assert 1 in out and max(out) == 59
    assert all(c in out for c in (2, 3, 4, 5))


def test_make_and_filter_g1walk_finds_optimum_band():
    cands = ct.filter_candidates(ct.make_candidates(2048, 40), num_envs=2048)
    assert any(4 <= c <= 6 for c in cands)
    assert max(cands) == math.ceil(2048 / 40)
    assert 1024 not in cands and 2048 not in cands


class _FakeModel:
    nq = 30
    nv = 29
    nbody = 16
    njnt = 28
    nu = 12
    ngeom = 40
    nsensordata = 100


def test_device_fingerprint_has_expected_keys():
    fp = ct.device_fingerprint()
    assert set(fp) >= {"system", "machine", "cpu_count"}
    assert isinstance(fp["cpu_count"], int) and fp["cpu_count"] >= 1


def test_model_signature_reads_structural_fields():
    sig = ct.model_signature(_FakeModel(), n_variants=3)
    assert sig == {
        "nq": 30,
        "nv": 29,
        "nbody": 16,
        "njnt": 28,
        "nu": 12,
        "ngeom": 40,
        "nsensordata": 100,
        "n_variants": 3,
    }


def test_make_cache_key_deterministic_and_sensitive():
    common = dict(
        backend_type="mujoco",
        model_sig=ct.model_signature(_FakeModel(), 1),
        device={"system": "Linux", "machine": "x86_64", "cpu_count": 32},
        num_envs=4096,
        nthread=64,
        dtype=np.float32,
        post_step_forward_sensor=False,
        bench_nsteps=4,
    )
    k1 = ct.make_cache_key(**common)
    k2 = ct.make_cache_key(**common)
    assert k1 == k2
    assert ct.make_cache_key(**{**common, "num_envs": 2048}) != k1
    assert ct.make_cache_key(**{**common, "dtype": np.float64}) != k1


def test_cache_path_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(target))
    assert ct.cache_path() == target


def test_cache_path_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("UNILAB_CHUNK_SIZE_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ct.cache_path() == tmp_path / "unilab" / "chunk_size.json"


def test_load_missing_cache_returns_empty(tmp_path):
    assert ct.load_cache(tmp_path / "nope.json") == {}


def test_store_then_load_roundtrip_and_merge(tmp_path):
    path = tmp_path / "c.json"
    ct.store_cache(path, "k1", {"chunk_size": 7})
    ct.store_cache(path, "k2", {"chunk_size": None})
    data = ct.load_cache(path)
    assert data["k1"]["chunk_size"] == 7
    assert data["k2"]["chunk_size"] is None


def test_file_lock_acquires_and_releases(tmp_path):
    lock = tmp_path / "c.lock"
    with ct.file_lock(lock):
        pass
    with ct.file_lock(lock):  # re-acquire after release must not deadlock
        pass


class _FakePool:
    def __init__(self):
        self.seen = []

    def step(
        self,
        state,
        *,
        nstep,
        control=None,
        control_spec=0,
        chunk_size=None,
        return_sensor=False,
        post_step_forward_sensor=False,
    ):
        self.seen.append(chunk_size)
        return state, np.zeros((state.shape[0], 1))


def test_benchmark_visits_all_candidates_plus_baseline():
    pool = _FakePool()
    state = np.zeros((8, 5), dtype=np.float64)
    timings = ct.benchmark_chunk_sizes(
        pool,
        state,
        nstep=2,
        candidates=[1, 2],
        control=None,
        post_step_forward_sensor=False,
        warmup=1,
        reps=2,
    )
    assert set(timings) == {None, 1, 2}
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())
    assert None in pool.seen and 1 in pool.seen and 2 in pool.seen


def test_select_picks_fastest_above_margin():
    timings = {None: 1.0, 128: 0.95, 256: 0.90, 512: 0.80}
    assert ct.select_chunk_size(timings, base=256) == 512


def test_select_keeps_default_when_margin_not_met():
    timings = {None: 1.0, 256: 0.99}
    assert ct.select_chunk_size(timings, base=256) is None


def test_select_tiebreaks_toward_base():
    timings = {None: 1.0, 200: 0.80, 256: 0.805, 4096: 0.81}
    assert ct.select_chunk_size(timings, base=256) == 256


def test_select_adopts_candidate_at_exact_margin():
    # best_t exactly at the boundary (margin=0.5 -> baseline*(1-0.5)=0.5, exact float);
    # spec '>= margin to adopt' means adopt the candidate, not keep the default.
    timings = {None: 1.0, 256: 0.5}
    assert ct.select_chunk_size(timings, base=256, margin=0.5) == 256


def test_emit_falls_back_to_stderr_when_info_disabled(capsys, monkeypatch):
    # Spawn collector subprocesses leave the root logger unconfigured (default
    # level WARNING), which drops INFO. _emit must still surface the line on the
    # terminal via stderr so the candidate benchmark is visible there.
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)
    ct._emit("chunk_size benchmark: default=1.00ms, 4=0.80ms -> chosen=4")
    err = capsys.readouterr().err
    assert "chunk_size benchmark" in err
    assert "chosen=4" in err


def test_emit_uses_logger_when_info_enabled(capsys, caplog):
    # Main process (Hydra-configured, INFO enabled): go through logging, NOT a
    # raw stderr print, so there is no duplicate line.
    caplog.set_level(logging.INFO, logger=ct.logger.name)
    ct._emit("chunk_size: cache hit -> 4")
    assert "chunk_size: cache hit -> 4" not in capsys.readouterr().err
    assert any("cache hit -> 4" in r.message for r in caplog.records)


def test_log_benchmark_table_surfaces_candidates_without_logging(capsys, monkeypatch):
    # End-to-end: with INFO disabled (subprocess), the full candidate table must
    # reach stderr, including every candidate's timing and the chosen value.
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)
    ct._log_benchmark_table({None: 0.006, 1: 0.013, 4: 0.005, 26: 0.007}, chosen=4, default_chunk=2)
    err = capsys.readouterr().err
    assert "default(=2)=6.00ms" in err
    assert "4=5.00ms" in err
    assert "26=7.00ms" in err
    assert "chosen=4" in err


def test_format_candidate_table_sorts_default_first_then_numeric():
    table = ct._format_candidate_table({"None": 9.38, "1": 25.43, "52": 14.59, "4": 9.15}, 5)
    assert table == "default(=5)=9.38ms, 1=25.43ms, 4=9.15ms, 52=14.59ms"


def test_cache_hit_emits_full_candidate_table_without_logging(capsys, monkeypatch, tmp_path):
    # Warm cache + spawn collector (INFO disabled): the cache-hit line must carry the
    # stored per-candidate breakdown, not just the chosen value -- otherwise the
    # candidate info is invisible on every run after the first cold benchmark.
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(
        ct, "benchmark_chunk_sizes", lambda *a, **k: {None: 0.009, 1: 0.025, 4: 0.0091}
    )
    ct.resolve_chunk_size(**_resolve_kwargs())  # miss -> benchmark -> stores per_candidate_ms
    capsys.readouterr()  # discard first-call output

    monkeypatch.setattr(ct.logger, "level", logging.WARNING)  # simulate spawn collector
    ct.resolve_chunk_size(**_resolve_kwargs())  # hit -> must emit the table
    err = capsys.readouterr().err
    assert "cache hit" in err
    assert "default(=1)=9.00ms" in err  # _resolve_kwargs: 8 // (10*4) = 0 -> clamp to 1
    assert "1=25.00ms" in err
    assert "4=9.10ms" in err


def test_cache_hit_without_candidate_table_still_emits_chosen(capsys, monkeypatch):
    # Legacy entries (or any entry missing per_candidate_ms) must not crash and must
    # still report the chosen value.
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)
    ct._emit_cache_hit(4, None, default_chunk=1)
    assert "cache hit -> 4" in capsys.readouterr().err


def test_aggregate_times_returns_min():
    # min rejects background-load spikes (which only inflate, never deflate) and
    # estimates the uncontended per-step cost a dedicated training box actually sees.
    assert ct._aggregate_times([0.003, 0.001, 0.002]) == pytest.approx(0.001)
    assert ct._aggregate_times([0.05, 0.0061, 0.006, 0.0062]) == pytest.approx(0.006)


def test_benchmark_reps_default_is_stable():
    # reps=5 was too few to reject noise on a loaded machine; the default must be
    # large enough that the per-candidate estimate is stable.
    reps = inspect.signature(ct.benchmark_chunk_sizes).parameters["reps"].default
    assert reps >= 15, reps


def test_benchmark_respects_time_budget():
    pool = _FakePool()
    state = np.zeros((8, 5), dtype=np.float64)
    timings = ct.benchmark_chunk_sizes(
        pool,
        state,
        nstep=2,
        candidates=[1, 2, 4, 8],
        control=None,
        post_step_forward_sensor=False,
        warmup=0,
        reps=1,
        time_budget_s=0.0,
    )
    assert set(timings).issubset({None, 1, 2, 4, 8})
    assert len(timings) < 5  # budget fired -> stopped early


def _resolve_kwargs(**over):
    base = dict(
        pool=_FakePool(),
        state=np.zeros((8, 5), dtype=np.float64),
        model=_FakeModel(),
        n_variants=1,
        num_envs=8,
        nthread=4,
        dtype=np.float32,
        post_step_forward_sensor=False,
        bench_nsteps=2,
        manual_chunk_size=None,
        adaptive=True,
    )
    base.update(over)
    return base


def test_resolve_manual_override_wins(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("benchmark must not run for manual override")

    monkeypatch.setattr(ct, "benchmark_chunk_sizes", _boom)
    assert ct.resolve_chunk_size(**_resolve_kwargs(manual_chunk_size=42)) == 42


def test_resolve_off_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("benchmark must not run when adaptive=False")

    monkeypatch.setattr(ct, "benchmark_chunk_sizes", _boom)
    assert ct.resolve_chunk_size(**_resolve_kwargs(adaptive=False)) is None


def test_native_default_chunk_matches_mujoco_formula():
    # BatchEnvPool's default when chunk_size=None: max(1, nbatch // (10 * nthread)).
    assert ct._native_default_chunk(2048, 40) == 5
    assert ct._native_default_chunk(1024, 40) == 2
    assert ct._native_default_chunk(4, 2) == 1  # 4 // 20 = 0 -> clamped to 1
    assert ct._native_default_chunk(100, 1) == 10


def test_benchmark_table_annotates_default_and_chosen_with_native_chunk(capsys, monkeypatch):
    # "default"/"None" must show the actual chunk_size the native default resolves to.
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)
    ct._log_benchmark_table({None: 0.010, 5: 0.009}, chosen=None, default_chunk=5)
    err = capsys.readouterr().err
    assert "default(=5)=10.00ms" in err
    assert "chosen=None(=5)" in err


def test_cache_hit_annotates_default_with_native_chunk(capsys, monkeypatch):
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)
    ct._emit_cache_hit(None, {"None": 10.23, "5": 9.99}, default_chunk=5)
    err = capsys.readouterr().err
    assert "default(=5)=10.23ms" in err
    assert "chosen=None(=5)" in err


def test_resolve_skips_everything_when_nothing_to_tune(monkeypatch, tmp_path, capsys):
    # num_envs <= nthread -> at most one work-chunk -> chunk_size is a no-op. Must
    # return None WITHOUT benchmarking, caching, or emitting a line (this kills the
    # noisy num_envs=1 setup-env benchmark that APPO/off-policy emit).
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(tmp_path / "c.json"))
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return {None: 1.0, 1: 1.0}

    monkeypatch.setattr(ct, "benchmark_chunk_sizes", _spy)
    monkeypatch.setattr(ct.logger, "level", logging.WARNING)  # any _emit would hit stderr

    assert ct.resolve_chunk_size(**_resolve_kwargs(num_envs=1, nthread=1)) is None
    assert ct.resolve_chunk_size(**_resolve_kwargs(num_envs=20, nthread=40)) is None
    assert calls["n"] == 0  # benchmark never attempted
    assert not (tmp_path / "c.json").exists()  # nothing cached
    assert capsys.readouterr().err == ""  # nothing printed


def test_resolve_cache_hit_skips_benchmark(monkeypatch, tmp_path):
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(tmp_path / "c.json"))
    calls = {"n": 0}

    def _bench(*a, **k):
        calls["n"] += 1
        return {None: 1.0, 4: 0.5}

    monkeypatch.setattr(ct, "benchmark_chunk_sizes", _bench)

    first = ct.resolve_chunk_size(**_resolve_kwargs())  # miss -> benchmark
    second = ct.resolve_chunk_size(**_resolve_kwargs())  # hit -> no benchmark
    assert first == second
    assert calls["n"] == 1


def test_resolve_cache_miss_runs_benchmark_and_persists(monkeypatch, tmp_path):
    cache = tmp_path / "c.json"
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(cache))
    monkeypatch.setattr(ct, "benchmark_chunk_sizes", lambda *a, **k: {None: 1.0, 4: 0.5})
    chosen = ct.resolve_chunk_size(**_resolve_kwargs())
    assert chosen == 4
    assert cache.exists()
    assert any(v.get("chunk_size") == 4 for v in ct.load_cache(cache).values())


def test_resolve_cache_hit_none_is_valid(monkeypatch, tmp_path):
    # A prior benchmark concluded native default is best (chunk_size=None).
    # That is a legitimate cached value: a second call must return None and NOT re-benchmark.
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(tmp_path / "c.json"))
    calls = {"n": 0}

    def _bench(*a, **k):
        calls["n"] += 1
        return {None: 0.5, 4: 1.0}  # baseline fastest -> select_chunk_size returns None

    monkeypatch.setattr(ct, "benchmark_chunk_sizes", _bench)

    first = ct.resolve_chunk_size(**_resolve_kwargs())  # miss -> benchmark -> chosen=None
    second = ct.resolve_chunk_size(**_resolve_kwargs())  # hit on key presence -> no benchmark
    assert first is None
    assert second is None
    assert calls["n"] == 1

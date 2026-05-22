from __future__ import annotations

import benchmark.core.mem_profile as mem_profile


def test_build_memory_summary_keeps_env_step_schema() -> None:
    samples = {
        "before_env": {"rss_bytes": 1000, "uss_bytes": 800, "memory_source": "test"},
        "after_benchmark": {
            "rss_bytes": 2600,
            "uss_bytes": 1900,
            "memory_source": "test",
            "rss_source": "test",
            "peak_rss_bytes": 3000,
        },
        "after_close": {"rss_bytes": 2400, "uss_bytes": 1700, "memory_source": "test"},
    }

    summary = mem_profile.build_memory_summary(samples, num_envs=2)

    assert summary["preferred_metric"] == "uss"
    assert summary["total_rss_delta_bytes"] == 1600
    assert summary["total_rss_delta_bytes_per_env"] == 800.0
    assert summary["total_uss_delta_bytes"] == 1100
    assert summary["total_uss_delta_bytes_per_env"] == 550.0
    assert summary["retained_uss_delta_after_close_bytes"] == 900
    assert summary["after_benchmark_peak_rss_bytes"] == 3000
    assert summary["samples"] is samples


def test_format_mb() -> None:
    assert mem_profile.format_mb(None) == "-"
    assert mem_profile.format_mb(1024 * 1024) == "+1.0 MB"
    assert mem_profile.format_mb(1024 * 1024, signed=False) == "1.0 MB"

# Choosing a Backend

| Decision | Pick MuJoCo if… | Pick Motrix if… |
|---|---|---|
| Throughput on many CPU cores | medium | ✅ better |
| Snapshot / playback needed | ❌ | ✅ |
| macOS headless video | needs Xvfb | ✅ native |
| Tightest determinism across versions | ✅ | medium |
| Best asset format compatibility | ✅ | medium |

See also {doc}`../../transfer/sim_to_sim/why_switch`.

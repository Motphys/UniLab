---
orphan: true
---

# ADR-0006 Phase 5 PPO RSS Threshold Amendment

语言: 简体中文

- Status: Accepted
- Date: 2026-07-30
- Owners: RL infrastructure maintainers
- Supersedes: None
- Superseded by: None

## Context

managed MuJoCo/MJWarp rollout 的 Phase 0 threshold manifest 在候选实现前冻结，并由独立 commit、Git blob 和
SHA-256 receipt 保护。Phase 5 PPO benchmark 在两个完整 40-case capture 和后续 b128 focused
capture 中，mjwarp/MuJoCo 的进程树 peak RSS median ratio 稳定落在约 `1.248` 到 `1.2517`。
最新 clean candidate 的 b128 延迟和吞吐均有余量，RSS 只比 `1.25` 上限多约 3.29 MB。

项目 owner 决定接受这部分 allocator/runtime 常驻差异，将 Phase 5 PPO RSS ratio 上限调整为
`1.26`。原 Phase 0 manifest 已被 Phase 4 和其他验收链消费，直接改写它会破坏已有 evidence
的 provenance，也会让修改后的 threshold 与历史结果无法区分。

## Decision

1. Phase 0 threshold manifest 和 freeze receipt 保持逐字节不变。
2. 新增版本化 amendment `g1-phase5-ppo-rss-ratio-v1`，只覆盖 Phase 5 PPO artifact 的进程树
   peak RSS median ratio，将上限从 `1.25` 调整到 `1.26`。
3. amendment 仅适用于 `manager_mjwarp-mjwarp-device-ppo-benchmark-v1` 的
   `device_resident` profile，并覆盖 throughput paired lane 与 behavior lane。所有其他 threshold、
   profile、artifact 和 acceptance phase 继续使用 base manifest。
4. amendment 使用独立 manifest 和 freeze receipt。receipt 绑定 amendment manifest 的
   SHA-256、Git blob 和先于 candidate 的 freeze commit。
5. Phase 5 PPO artifact 必须同时记录 base threshold 和 amendment provenance；candidate commit
   必须是 amendment freeze commit 的严格后代。缺失、scope 不匹配、hash/blob 篡改或祖先关系
   不成立时 fail closed。
6. amendment 不改变 matrix、warmup、进程隔离、worker 顺序、采样、aggregation、retry、GPU
   memory、transfer/sync、训练行为或延迟/吞吐 gate。

## Stable Contracts

- `g1_threshold_manifest.yaml` 的 `host_preferred_metric_ratio_max` 仍为 `1.25`。
- Phase 5 PPO benchmark 只有在 base 与 amendment 两套 receipt 都有效时才能构造执行计划。
- 有效 artifact 的 threshold payload 同时包含 base 和 amendment 的 id/path/hash/freeze commit。
- RSS boundary `<= 1.26` 通过，`> 1.26` 失败；其余 gate 继续读取 immutable base manifest。
- threshold amendment PR 不包含 candidate benchmark 结果或 production runtime 修改。

## Future Amendment Acceptance Criteria

后续 threshold amendment 必须同时满足以下五项；任一项缺失即拒绝，不得仅以当前 candidate
距离阈值很近为理由放宽：

1. 在原 threshold、原 harness 和完整预注册 matrix 下至少完成两次独立 clean capture，并保留原始
   PASS/FAIL artifact；focused run 只能用于定位，不能替代完整复现。
2. child issue 必须给出可审计的根因、对 correctness / transfer / synchronization 等相邻 gate 的影响
   分析，以及新阈值的独立工程预算或统计依据；禁止用“observed maximum 加 epsilon”作为唯一依据。
3. amendment 必须通过独立的 threshold-only PR 在下一次 candidate capture 前冻结；该 PR 不得包含
   production runtime、benchmark harness、matrix、aggregation 规则或 candidate 结果的修改，并继续使用
   SHA-256、Git blob 和 ancestry receipt。
4. amendment 的 metric、artifact、profile、lane 和 phase scope 必须取最小闭包并显式列出；不得按已
   观察到的 batch、seed、worker order 或单次样本做特判。至少两名 maintainer 批准，且其中一名不是
   amendment 作者或 candidate 实现者。
5. freeze 后必须从严格后代 clean commit 重跑完整 matrix 并生成新 artifact；历史失败 artifact 保持
   失败且不得追溯改判。新 artifact 必须同时验证 base 与 amendment provenance，并重新通过所有未改动
   gate。

## Alternatives Considered

- 直接修改 Phase 0 manifest。拒绝原因：会使既有 Phase 4 evidence 失去原始 threshold identity，
  并形成 post-hoc 改写历史。
- 在 benchmark 代码中直接把常量改成 `1.26`。拒绝原因：缺少独立 schema、scope、receipt 和
  ancestry，无法证明结果使用了哪个 threshold 版本。
- 对 b128 单独放宽、其他 batch 保持 `1.25`。拒绝原因：现有 metric contract 是统一的
  end-to-end process RSS ratio；按观测结果特判 batch 会引入 post-hoc selection。

## Consequences

- 后续 Phase 5 candidate 必须基于 amendment freeze commit 之后的 clean integration commit 重采。
- 历史 failed diagnostic artifact 不会自动转为 evidence；必须运行完整 40-case matrix并重新生成
  双 provenance artifact。
- `1.26` 成为 Phase 5 PPO host RSS 的唯一有效上限，进一步修改仍需新的 ADR、child issue、
  threshold-only PR 和先于 candidate 的 freeze receipt。

## Evidence In Repo

- Base manifest: `tests/acceptance/manager_mjwarp/g1_threshold_manifest.yaml`
- Amendment manifest: `tests/acceptance/manager_mjwarp/g1_phase5_ppo_threshold_amendment.yaml`
- Validator: `tooling/acceptance/thresholds.py`
- PPO consumer: `benchmark/rl/benchmark_mjwarp_ppo.py`
- Contract tests: `tests/tools/test_manager_mjwarp_threshold_amendment.py`

## Related Documents

- [managed MuJoCo/MJWarp rollout](https://github.com/unilabsim/UniLab/issues/705)
- [Issue #807](https://github.com/unilabsim/UniLab/issues/807)
- {doc}`ADR Index </adr/README>`
- {doc}`协作流程 </zh_CN/4-developer_guide/5-contributing_workflow>`

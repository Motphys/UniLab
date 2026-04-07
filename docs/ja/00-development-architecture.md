# RL Infrastructure 開発標準

言語: [English](../en/00-development-architecture.md) | [简体中文](../zh_CN/00-development-architecture.md) | 日本語 | [한국어](../ko/00-development-architecture.md)

UniLab は**高性能、モジュール化、contract 駆動**の RL infrastructure リポジトリです。この標準は 1 つの問いだけに答えます: **どのような変更が正しいか**。

エンジニアリング特性: 高性能、構造化、体系性、モジュール性、再利用性、可観測性。

---

## 1. Runtime Model

3 段のゼロコピー pipeline:

```text
CPU Physics Sim ──shm──► Collector / IPC ──shm──► GPU Learner
(MuJoCo/Motrix)          (AsyncRunner)            (torch/mlx)
                                  ▲                   │
                                  └── SharedWeightSync ┘
```

- backend の切り替えは **contract + registry + config** で行い、script 分岐では行わない
- Env 層は numpy / vectorized を維持し、GPU は learner が専有する
- collector と learner は IPC + shared memory で疎結合にし、lifecycle は統一する

---

## 2. Layered Architecture

依存方向は厳密に一方向です。**問題はその責務を持つ層で解決する**必要があります。

| Layer | ディレクトリ | 責務 | 持ってはいけない責務 |
|-------|--------------|------|----------------------|
| L0 Backend | `base/backend/` | 物理 backend 抽象 `SimBackend` | 学習ロジック、reward |
| L1 Env | `envs/`, `base/np_env.py` | MDP 意味論、observation、reward、reset | scheduling、logging policy |
| L2 Config & Registry | `config/`, `base/registry.py`, `conf/` | schema、task / reward 合成、登録 | 散在する業務デフォルト |
| L3 Algo & IPC | `algos/`, `ipc/` | learner、runner、collector、shared-memory 経路 | env / backend の詳細 |
| L4 Scripts | `scripts/` | 組み立てのみ | コア業務ルール |

---

## 3. Design Principles

1. **Contract first**: 局所パッチより先に contract を守る。承重壁は `registry.make`、`NpEnvState.obs: dict`、`reset -> (obs, info)`、`obs_groups_spec`、`SimBackend`、collector / learner の shared-memory protocol。
2. **Own your layer**: scripts は env bug を直さず、env は backend bug を直さない。
3. **Config over branching**: 拡張順序は config schema -> registry -> env / backend adapter 層 -> 最後に script 分岐。
4. **Backend isolation**: MuJoCo / Motrix の差分は backend 実装、env adapter 層、backend-specific profile に閉じ込める。能力差は明示する。
5. **Evidence-graded claims**: `Registered`、`Configured`、`Benchmarked`、`Recommended` を使う。証拠なしに stable support を主張しない。
6. **Validate near risk**: 上位 smoke run は補助であり、近接境界での検証の代替ではない。
7. **Reusable primitives**: 汎用ロジックは `base/` や `utils/` に引き上げ、workflow ごとにコピペしない。

---

## 4. Training Entrypoints

| パス | エントリポイント | 主経路 |
|------|------------------|--------|
| PPO (torch) | `scripts/train_rsl_rl.py` | `registry.make` -> `RslRlVecEnvWrapper` -> `rsl_rl.OnPolicyRunner` |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | `registry.make` -> MLX `RolloutBuffer` -> `PPOTrainer` |
| APPO | `scripts/train_appo.py` | `APPORunner` -> collector -> `SharedOnPolicyStorage` |
| SAC / TD3 | `scripts/train_offpolicy.py` | `OffPolicyRunner` -> collector -> `ReplayBuffer` |

編集前に、自分がどの経路を触っているのかを特定してください。

---

## 5. Configuration

UniLab は dataclass + Hydra を使います。schema は `src/unilab/config/structured_configs.py`、runtime config は `conf/{ppo,appo,offpolicy}/` にあります。

合成順序: `{algo}/config*.yaml` -> `task=...` -> `reward[_{backend}]` -> CLI override -> 必要に応じて `motrix_legacy`。

- reward は明示的に注入する
- backend 選択が task や reward の挙動を変えるなら config で表現する
- 動的 override は CLI を尊重する

---

## 6. Env

拡張フロー:

1. `@registry.envcfg("EnvName")` で config dataclass を登録する
2. `@registry.env("EnvName", sim_backend=...)` で実装クラスを登録する
3. `registry.make(...)` で構築する

Env が**持つ責務**は MDP 意味論、observation 構造、reward、reset、backend データから学習意味論への写像です。Env は学習 orchestration、多プロセス制御、トップレベル logging を持ちません。

---

## 7. Backend

`SimBackend` (`src/unilab/base/backend/base.py`) は base pose / velocity、DOF state、body pose / velocity（world と baselink 座標系）、named sensor を提供しなければなりません。

既知の backend-specific 分岐には `backend_type == "motrix"` による `_process_rigid_body_props` があり、一部の play / debug / video / symmetry 経路はまだ MuJoCo-first です。

---

## 8. Async And Runner

すべての async algorithm は `src/unilab/ipc/async_runner.py` の `AsyncRunner` を共有します。spawn モデル、collector lifecycle、shared-resource cleanup を統一します。

- **APPO**: collector は `SharedOnPolicyStorage` に書き込み、learner は V-trace を使い、actor 重みは `SharedWeightSync` で戻す
- **Off-policy**: collector は `ReplayBuffer` に書き込み、learner がそこから sample し、`SharedWeightSync` で重み同期し、sync / async collection の両方をサポートする

shared runner の外で並列 protocol を複製したり、shared-resource lifecycle を迂回したり、暗黙結合を導入したりしてはいけません。

---

## 9. Validation

| 変更 | 最低限の検証 |
|------|--------------|
| Hydra / task / reward | `make test`（`tests/config/`, `tests/scripts/`） |
| env contract / observation | `make test`（`tests/base/test_np_env.py` など） |
| runner / IPC | `make test`、必要に応じて `make test-slow` |
| 学習メイン経路 | 関連テスト + 1 iteration の smoke run |
| backend 経路 | backend 別 smoke run、必要に応じて slow test |
| docs-only | コマンド、パス、config 名、CI、support claim を手動確認 |

---

## 10. Review Checklist

1. この変更はどの contract に影響するか？
2. もっと下位の層で直すべきではないか？
3. backend / task の挙動は config で表現されているか、それとも script 特判で隠れているか？
4. すべての support claim に registry / config / test / benchmark の証拠があるか？
5. 検証はリスクに最も近い境界で行われたか？

---

## 11. High-Signal Files

- `scripts/train_{rsl_rl,mlx_ppo,appo,offpolicy}.py`
- `src/unilab/base/{registry,np_env}.py`
- `src/unilab/base/backend/base.py`
- `src/unilab/config/structured_configs.py`
- `src/unilab/utils/{reward_utils,obs_utils}.py`
- `src/unilab/ipc/async_runner.py`

---

## Navigation

- Previous: [README](README.md)
- Next: [Getting Started](01-getting-started.md)

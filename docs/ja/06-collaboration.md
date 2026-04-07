# 協業ワークフロー

言語: [English](../en/06-collaboration.md) | [简体中文](../zh_CN/06-collaboration.md) | 日本語 | [한국어](../ko/06-collaboration.md)

リポジトリの docs は安定した標準だけを書きます。実行状況、owner、フェーズ進捗は GitHub の協業オブジェクトに置くべきです。

UniLab をインストールまたは学習したいだけなら、`docs/ja/README.md`、`docs/ja/01-getting-started.md`、`docs/ja/03-training.md` から読み始めてください。

## Work Item Granularity

各 issue は少なくとも次の問いに答えるべきです:

1. 何の問題を解くのか？
2. 期待する deliverable は何か？
3. 完了条件は何か？
4. 実行責任者は誰か？
5. 上流 blocker は何か？

推奨 issue タイプ:

- `bug`
- `work item`: feature / infra / benchmark / test / sim / docs work

## Milestone Structure

各 milestone は次を満たすべきです:

- GitHub 上で milestone object として存在する
- 子 issue を要約する tracking issue を持つ
- 実行詳細は milestone 説明ではなく子 issue に書く
- 「コードが merge された」ではなく、成果物で完了を定義する

典型的な完了成果物:

- green CI
- benchmark 結果または W&B run link
- demo video / ONNX export / checkpoint path
- user-facing 挙動が変わる場合の docs update

## PR Evidence Standard

各 PR は次を満たすべきです:

- driving issue を link する
- user-facing change と training impact を説明する
- 実際に走らせた validation command を列挙する
- `mujoco`、`motrix`、macOS、Linux の間で挙動差があるかを明記する

## Ownership Model

実行 owner には GitHub assignees を使い、review owner には `CODEOWNERS` を使います。安定した GitHub handle がまだない場合は issue を unassigned のままにし、期待する owner を issue body に一時記載してください。

## Navigation

- Previous: [G1 Motion Tracking](05-g1-motion-tracking.md)
- Next: [Contributing](CONTRIBUTING.md)

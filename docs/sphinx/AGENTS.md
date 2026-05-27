# docs/sphinx Agent Guide

这个文件**规范 agent 在 `docs/sphinx/source/` 下撰写和修改文档内容的行为**。

- 目录结构、本地 build、deploy key 等基础设施信息在 [`docs/sphinx/README.md`](README.md)
- 仓库整体编码规范在顶层 `AGENTS.md` / `CLAUDE.md`(本文件只覆盖文档写作)
- 架构契约与术语定义在 `source/zh_CN/developer_guide/development-standard.md` 和 `source/glossary.md`

## 文档采用双语平行结构

```
source/
├── en/                      ← 英文版
│   ├── index.md
│   ├── user_guide/
│   ├── developer_guide/
│   ├── transfer/
│   └── agents/
├── zh_CN/                   ← 中文版(路径与 en/ 1:1 镜像)
│   ├── index.md
│   ├── user_guide/
│   ├── developer_guide/
│   ├── transfer/
│   └── agents/
└── (共享,不写双语)
    ├── adr/                 ← ADR(中英共用,中文为主)
    ├── api_reference/       ← autodoc 拉 src/ docstring(英文)
    ├── glossary.md
    ├── changelog.md
    └── _static/
```

**核心原则**:`en/<path>` 和 `zh_CN/<path>` **路径一一镜像**(`en/user_guide/algorithms/ppo.md` ↔ `zh_CN/user_guide/algorithms/ppo.md`)。改一边时同 PR 检查另一边。

## 核心原则(按优先级)

1. **Evidence only**:只写仓库里能 grep / 读 / 跑出来的事实——registry、conf、test、benchmark、ADR、CI 配置。无法在 `src/` / `conf/` / `tests/` / `scripts/` 中找到的特性,**不要写进文档**。
2. **Ground truth = code**:命名、签名、默认值、CLI 形式以代码为准。代码与文档冲突时,文档错——除非代码的 docstring 错,那就改 docstring,让 autodoc 重新拉。
3. **Verify before claim**:文档里出现的每一个 `src/...` / `conf/...` / `scripts/...` / hydra key,写之前先 `ls` / `grep` 确认存在。`tests/scripts/test_check_docs.py` 兜底,但 agent 不应该依赖它。
4. **Don't invent**:不写"即将推出"、"未来支持"、"计划中"、"roadmap"、"coming soon"。功能要么已实现并可以举证,要么不写。
5. **Concise > comprehensive**:中文用户文档单文件硬上限 **120 行**(`E-reference/01-backend-support-matrix.md` 例外)。超了就拆。
6. **Bilingual parity**:`en/` 和 `zh_CN/` 路径必须一一镜像;改一边先看另一边是否需要同步;**空头(只有标题 + Navigation)比假翻译诚实**。
7. **API 文档 = autodoc**:`source/api_reference/` 下页面只放 `automodule` / `autosummary`。要补 API 说明,改 `src/unilab/**/*.py` 的 docstring,**不要手写**类签名表格。

## 写文档之前(强制步骤)

按顺序做完再写:

1. **选语言归属**——这次要写英文还是中文?决定文件落在 `en/` 还是 `zh_CN/`。如果同时双语,先写主语言版本,另一种语言用占位文件(只有标题 + Navigation)。
2. **定位 section**——user_guide / developer_guide / transfer / agents 哪一档?参考 [`README.md` 的目录结构](README.md#目录结构)。
3. **检查重复**——`find docs/sphinx/source -name "*<topic>*"`,`grep -r <核心术语> docs/sphinx/source/`。已有页就改它,不要另起。
4. **找 ground truth**——
   - 算法 / 任务相关:翻 `conf/<algo>/`、`conf/<algo>/task/<task>/<backend>.yaml`
   - 训练入口:`scripts/train_*.py` + 其 docstring
   - Contract / API:`src/unilab/base/`、`src/unilab/registry.py`、`src/unilab/structured_configs.py`
   - 已有规范:`source/zh_CN/developer_guide/development-standard.md`、各 ADR
5. **找镜像对应**——确认这次要不要同 PR 改另一种语言版本。

## 写文档时(行为规则)

### 引用代码 / 配置 / 脚本

| 场景 | 正确写法 | 反面 |
|------|---------|------|
| 引用源码符号 | `` `NpEnvState` `` 并附路径 `src/unilab/base/np_env.py` | "the env state class"(找不到目标) |
| 引用源文件 | `` `src/unilab/base/np_env.py` `` 反引号包住 | 裸路径或 markdown 链接 |
| 引用脚本 | `uv run scripts/train_rsl_rl.py task=go2_joystick_flat/mujoco` | `python scripts/...` / `uv run python scripts/...`(`check_canonical_commands` 会拒) |
| Hydra override 示例 | 用 `conf/` 下真实任务名 | `task=my_new_task` 这类编造名 |
| 站内 cross-ref(同语言) | MyST `{doc}` 相对路径:`` {doc}`algorithms/ppo` `` | 裸 markdown 链接(部署后路径会变) |
| 站内 cross-ref(跨语言或跨共享区) | 绝对路径 `{doc}` 角色:`` {doc}`/zh_CN/developer_guide/index` `` 或 `` {doc}`/adr/ADR-0003-task-owner-and-config-compose-contract` `` | 相对 `../../../zh_CN/...`(深度不稳定) |
| 链接到 src/ 文件 | GitHub 绝对链接:`https://github.com/unilabsim/UniLab/blob/main/src/unilab/...` | 相对路径(Sphinx build 时解不开) |

### zh_CN 文档 shape(自动检查会卡)

每个 `source/zh_CN/**.md` 都必须满足(`check_zh_cn_doc_shape`):

```markdown
# <标题>

语言: 简体中文            ← 第 3 行,字面量

<正文,≤ 120 行>

## Navigation

- Index: [Documentation](../index.md)
```

- 第 3 行字面量 `语言: 简体中文`
- 必须含 `## Navigation` 段
- 必须有 `- Index: [<text>](<...>/index.md)` 形式的 back-link(指向 `zh_CN/index.md`,深度自适应)
- **例外**:`index.md`(section landing 页)豁免

### ADR(共享,放 `source/adr/`)

新 ADR 用 `cp source/adr/ADR-TEMPLATE.md source/adr/ADR-NNNN-<kebab-slug>.md`。

- `NNNN` 从 0006 起递增(0000 是 index,0001-0005 已占)
- 8 个 governance 字段必填(`check_adr_shape`):
  - `- Status:` `Proposed` / `Accepted` / `Superseded` 之一
  - `- Date:`、`- Owners:`、`- Supersedes:`、`- Superseded by:`
  - `## Alternatives Considered`、`## Evidence In Repo`、`## Related Documents`
- 挂进 `source/zh_CN/developer_guide/index.md` 的 ADR toctree(绝对路径 `/adr/ADR-...`)
- **例外**:`ADR-0000-index.md` 和 `adr/README.md` 是索引页,不受 shape 约束
- ADR 用**中文为主**(治理性内容,避免双语漂移);英文 developer_guide 通过 `{doc}` 角色链过去

### Backend 支持矩阵

`source/zh_CN/user_guide/E-reference/01-backend-support-matrix.md` 由
`scripts/generate_support_matrix.py` 写"生成块"。

- 加了新任务 / 新 backend 支持后,跑:
  ```bash
  uv run scripts/generate_support_matrix.py --write
  ```
- **不要手改生成块**——`check_generated_support_matrix` 会拒
- 生成块以外的散文(背景、读法)可以手写
- 这一页**只有中文版本**;英文 user_guide 通过链接指向

### Toctree 维护

加新页时必须挂进**对应语言的** toctree,否则:
- 用户从导航找不到
- `make strict` 报 `document isn't included in any toctree`

需要更新的位置:
- 语言根 index:`source/en/index.md` 或 `source/zh_CN/index.md` 的 hidden toctree
- Section index:`source/<lang>/<section>/index.md`(如 `en/user_guide/index.md`)
- 删页:`grep -rn "<filename without ext>" docs/sphinx/source/` 清掉所有 `{doc}` 和 toctree 引用,**两种语言都查**

### 跨语言对齐

- 改 `en/<path>/<file>.md` → 检查 `zh_CN/<path>/<file>.md` 是否需要同步
- 加新页 → 在另一语言下建**同名占位文件**(只有标题 + Navigation 也 OK)
- **不要把英文内容粘进 zh_CN 文件当占位**——空头比假翻译诚实
- 翻译时保留原文的代码块、命令、文件路径(命名一致性比翻译流畅性优先)

## 改已有文档时

按场景分:

| 场景 | 操作 |
|------|------|
| API 用法变了 | 改 `src/unilab/**/*.py` 的 docstring;**不需要**碰 `source/api_reference/*`(autodoc 会拉) |
| Contract 变了 | 同步对应 contract 页(双语)+ 起 ADR(或更新现有 ADR 的 `Status:` 为 `Superseded`) |
| Hydra 默认值 / key 变了 | `grep -rn "task=\|algo=\|training\." docs/sphinx/source/` 找受影响处一起改 |
| 删页 | `grep -rn "<filename without ext>" docs/sphinx/source/ tests/`,清完 `{doc}` 和 toctree 引用,**两种语言一起删**,再删文件 |
| 重命名页 | 同上 grep,改链接;Sphinx 没有 alias,旧 URL 会 404,**避免重命名**;双语一起改 |

## 反模式(具体不要写)

- ❌ Marketing 措辞:"blazingly fast"、"next-gen"、"industry-leading"、"state-of-the-art"
- ❌ 编造的 benchmark 数字 / 性能对比表
- ❌ "Coming soon" / "TBD" / 半截内容占位(整页只留标题 + Navigation 比半截内容好)
- ❌ 手写 API 类签名 / 方法表格 → 留给 autodoc
- ❌ 重复 `CLAUDE.md` / `development-standard.md` 里已经说过的规范 → 链过去,不抄
- ❌ Roadmap / 计划页(没 owner / 日期 / 负责模块)
- ❌ 把 README / quickstart 内容复制粘贴到多个文档 → 单一来源,其他地方链过去
- ❌ 用已被移除的 CLI:`unilab train ...`、`unilab eval ...`(`check_canonical_commands` 会拒)
- ❌ `uv run python scripts/...` → 用 `uv run scripts/...`(同上)
- ❌ `training.load_run=...` → 用 `algo.load_run=...`(`check_training_entrypoint_semantics` 会拒)
- ❌ `task=go2_joystick_flat`(没 backend 段)→ 必须 `task=go2_joystick_flat/mujoco` 或 `task=<algo>/<task>/<backend>`
- ❌ 写绝对路径回到旧位置 `docs/users/` / `docs/developers/` / `docs/sphinx/source/user_guide/zh_CN/`——这些目录已不存在
- ❌ **不要把英文页的 toctree 链中文页**(反之亦然)→ 走根 picker 或顶部 banner 跨语言

## PR 之前必跑的 validation

```bash
# 1. Doc 检查测试(快,3 秒)
uv run pytest tests/scripts/test_check_docs.py -q

# 2. Sphinx HTML 构建(本地预览或 toctree 完整性)
cd docs/sphinx
uv pip install -r requirements.txt    # 第一次需要
sphinx-build -b html -n source build/html

# 3. 严格构建(可选,warning → error,用于 release-grade audit)
make -C docs/sphinx strict
```

**通过标准**:

- `test_check_docs.py`:**0 失败**,`collect_doc_errors` 返回 `[]`
- `sphinx-build`:**exit 0**;warning 允许,但每条都要看一眼是否你引入的
- 手验:你 PR 里引用的每个 `src/...` / `conf/...` / `scripts/...` 路径都能 `ls` 到
- 输出确认:`build/html/index.html`(根 picker)、`build/html/en/index.html`、`build/html/zh_CN/index.html` 三个文件都生成

**PR Gate(pure docs 改动)**:`make test-all` **不需要**跑全套,只需上面 1 + 2;但如果 PR 同时碰了 `src/unilab/**/*.py`(包括 docstring),要跑完整 `make test-all`。在 PR body 的 Validation 段写清楚跑了哪些。

## Cookbook(按场景照抄)

### 加一个新算法的文档(双语)

1. 文件(创建两个):
   - 英文 `source/en/user_guide/algorithms/<algo>.md`
   - 中文 `source/zh_CN/user_guide/algorithms/<algo>.md`(或 `zh_CN/user_guide/C-algorithms/0X-<algo>.md` 兼容已有结构)
2. 内容必须含:
   - 算法适用范围(on-policy / off-policy、CPU / GPU 要求)
   - 训练命令——hydra override 用 `conf/<algo>/` 下真实任务名
   - 关键超参的 default 出处:`{file}` 角色指向 `conf/<algo>/<algo>.yaml`
   - 训练入口脚本:`{file}` 指向 `scripts/train_<algo>.py`
3. 挂 toctree:
   - 英文挂 `source/en/user_guide/index.md` 的 Algorithms 块
   - 中文挂 `source/zh_CN/user_guide/index.md` 的对应块
4. 如有英文算法概览表:更新 `source/en/user_guide/algorithms/overview.md`

### 加一个新任务的文档(双语)

1. 文件:
   - 英文 `source/en/user_guide/tasks/<task>.md`
   - 中文 `source/zh_CN/user_guide/D-tasks/0X-<task>.md`(序号顺延)
2. 必须含训练命令、playback 命令、观测 / 动作维度、reward 组成、backend 支持矩阵链接
3. 更新任务索引(`source/zh_CN/user_guide/D-tasks/01-task-index.md`)的"按机器人家族"和"按任务类型"两段(`check_user_doc_architecture` 校验)
4. 跑支持矩阵生成:`uv run scripts/generate_support_matrix.py --write`

### 加一个 ADR(单一份,共享)

1. `cp source/adr/ADR-TEMPLATE.md source/adr/ADR-000N-<slug>.md`
2. 填全 8 个 governance 字段(用中文写正文)
3. `## Evidence In Repo` 段列出至少一条可验证的 commit / file / test
4. 挂进 `source/zh_CN/developer_guide/index.md` 的 ADR toctree(绝对路径 `/adr/ADR-...`)
5. 如果取代了旧 ADR:同时把旧 ADR 的 `Status:` 改成 `Superseded`,`Superseded by:` 指向新 ADR
6. 英文 `source/en/developer_guide/index.md` 已经通过 `{doc}` 链到 `/adr/...`,通常**不用改**

### 改了 `src/` 的 docstring

- 通常**不需要**碰 `source/api_reference/`——autodoc 会拉
- 如果 docstring 里用 `:doc:` / `:func:` 角色 cross-ref,确认目标存在
- 跑 `cd docs/sphinx && sphinx-build -b html source build/html` 看 autodoc 有无 warning

### 加一个新 backend

1. 用户角度:
   - 英文 `source/en/user_guide/backends/<backend>.md`
   - 中文 `source/zh_CN/user_guide/backends/<backend>.md`(或归入 02-simulation-backends.md)
2. 开发角度:如果加了 `SimBackend` 抽象方法,起一个 ADR
3. 跑 `uv run scripts/generate_support_matrix.py --write` 让任务 × backend 矩阵更新
4. 如果有 mock 需求:在 `source/conf.py` 的 `autodoc_mock_imports` 里加该 backend 的 import path

### 翻译现有页面(单向 EN → ZH 或 ZH → EN)

1. `cp source/<src_lang>/<path>/<file>.md source/<dst_lang>/<path>/<file>.md`(路径镜像!)
2. 翻译正文,**保留所有 code block、命令、文件路径、`{doc}` 角色不变**
3. 如果是 `src_lang=zh_CN`,加上中文 shape 头部(`语言: 简体中文`、`## Navigation`)
4. 挂进目标语言对应 section index 的 toctree

## Reference

- **基础设施**(build、deploy key、目录结构): [`docs/sphinx/README.md`](README.md)
- **仓库整体规范**: 顶层 `AGENTS.md` / `CLAUDE.md`
- **架构标准 + 术语**:
  - `source/zh_CN/developer_guide/development-standard.md`
  - `source/glossary.md`
- **协作流程**: `source/zh_CN/developer_guide/collaboration.md`
- **已接受 ADR**: `source/adr/ADR-0000-index.md`
- **doc check 实现**(行为兜底): `tests/scripts/doc_checks.py`、`tests/scripts/test_check_docs.py`

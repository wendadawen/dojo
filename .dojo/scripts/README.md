# 构建脚本

`.dojo/` 下是三类辅助文件：

| 目录 | 放什么 |
|---|---|
| `.dojo/scripts/` | 工具脚本，按任务类别分子目录 |
| `.dojo/templates/` | 页面骨架模板；已暂停类别的在 `archive/` 下 |
| `.dojo/site/` | 首页的 JS 和 CSS，部署时拷成 `_site/assets/` |

某页专属的东西（生成脚本、图片）放进它自己的 `wiki/<name>/`，不放 `.dojo/`。

## 脚本

| 目录 | 服务谁 | CI |
|---|---|---|
| `site/` | 整个站点：页面校验、模板、首页目录、样式，以及词表 `vocab.py`（主题、标签、类型、状态） | 跑 |
| `concept/` | 概念页：从 `research/` 生成页面，检查大纲与正文 | 跑自检 |
| `archive/` | 已暂停的页面类别（目前只有数据流） | 不跑 |

| 脚本 | 作用 |
|---|---|
| `site/vocab.py` | 页面元数据的封闭词表，校验、目录、概念页检查共用 |
| `site/validate.py` | 发布闸门，入口只有一个 |
| `concept/build.py` | 概念页生成器：读 `research/content.md`，产出 `index.html` 和 `overview.html`；`--run` 跑文中代码；生成后自动运行 `check.py` |
| `concept/check.py` | 概念页机械检查：`--outline` 查大纲格式与字数预算；默认查正文的依据、数字、结构、链接 |
| `archive/legacy-validate.py` | 全部 outdated 页面的校验：补回论文、学习记录、数据流的样式表和文案，并关掉对 outdated 页面的跳过 |
| `archive/dataflow-check.py` | 数据流页核查 |
| `archive/dataflow-build.py` | 数据流页生成器 |

### site/validate.py

    # 页面校验（默认；顺带跑内联脚本语法，需要 node，缺失则跳过并提示）
    python3 .dojo/scripts/site/validate.py <页面...>
    python3 .dojo/scripts/site/validate.py --all

    # 专项
    python3 .dojo/scripts/site/validate.py --templates      # 模板
    python3 .dojo/scripts/site/validate.py --copy [--fix]   # 概览文案
    python3 .dojo/scripts/site/validate.py --js <页面...>   # 只查内联脚本
    python3 .dojo/scripts/site/validate.py --css [--confirm] [--fix]   # 死 CSS
    python3 .dojo/scripts/site/validate.py --catalog --output _site/catalog.json

    # 改共享样式前后的渲染比对
    python3 .dojo/scripts/site/validate.py --diff <before-root> <after-root> <页面...> [--detail]

只校验现行页面：标了 `dojo:status=outdated` 的页面整页跳过并提示数量。旧概念页、论文、学习记录、数据流都标了，只有没标 outdated 的页面在校验范围内。`dojo:type` 不是 `concept` 又没标 outdated 的页面报错。`--templates`、`--css`、`--copy` 只管概念页，`--all` 不含各页 `research/` 里的资料快照。

页面校验覆盖：HTML 骨架、模板占位符、重复 id、同页锚点、本地引用、
`research/` 目录、数学字符、结构图、正文居中、元数据词表、模板与共享 CSS
一致性。

改样式时的顺序：先 `--diff` 确认改动只影响预期页面，再 `--css` 找死规则。

### concept/build.py

    # 读 research/content.md，生成 wiki/<name>/index.html 和 overview.html
    python3 .dojo/scripts/concept/build.py wiki/<name>/research/content.md

    # 另外运行文中后面跟着 text 输出块的 Python 代码，逐行比对输出
    python3 .dojo/scripts/concept/build.py wiki/<name>/research/content.md --run

    # 自检（CI 会跑；用虚构数据验证生成器本身可用）
    python3 .dojo/scripts/concept/build.py --selftest

- 文件格式见 `guides/concept/content.md`。格式错误会报出行号：缺「资料」「短页」「长页」之一、长页没有开头或没有章、章的标题带编号、折叠块里写标题、站内链接指向不存在的页面、落单的 `$` 等。
- 每次生成都打印长页折叠块外的字数和短页的字数，超出 `guides/concept.md` 定的上限时写明。长页顶部的阅读时间也按折叠块外的字算。
- 生成完成后自动运行 `check.py`，检查不过时退出码为 1。

### concept/check.py

    python3 .dojo/scripts/concept/check.py wiki/<name>/research/outline.md --outline   # 大纲
    python3 .dojo/scripts/concept/check.py wiki/<name>/research/content.md             # 正文
    python3 .dojo/scripts/concept/check.py --selftest

- 大纲：长页、短页各一节；每个节点五样「标题｜写什么｜类型｜字数｜依据」，「写什么」一句话；类型限正文、公式、例子、折叠块；短页恰好三节；长页最后一个节点叫「适用条件与边界」；字数预算合计不超上限，折叠块不计入。
- 正文：每段文字后面都要有「依据」，指向 `research/sources/` 里存在的文件，位置是标题里的一段文字或行号范围；正文里的数字必须在 sources 里作为完整的数出现（7 不算命中 17），「自行构造」的段落不查；短页恰好三节，不检查标题文字；站内链接只指向现行概念页；字数上限；代码块只能是 python 加紧跟的输出块；「依据：」必须单独成行。
- 事实、范围、重复、文风、读不读得通，脚本不判断，由 `guides/concept/` 里的审查者负责。
- `research/sources/` 不进 git，只提交 `sources/index.md`。

### archive/

已暂停类别的脚本，CI 不跑。用法见各脚本文件头部的说明。

    # 校验所有 outdated 页面
    python3 .dojo/scripts/archive/legacy-validate.py --all-outdated
    python3 .dojo/scripts/archive/legacy-validate.py wiki/<name>/index.html

数据流页的指南已归档（`guides/archive/model-dataflow.md`）。

## 页面从哪来

概念页由生成器读 `research/` 下的 Markdown 填进模板，不手工复制。其余几类模板（论文、学习记录、数据流）只服务已暂停的类别，保留以便已有页面通过校验。

| 模板 | 产出 |
|---|---|
| `.dojo/templates/concept/` | 概念的长页与短页，由 `concept/build.py` 读 `research/content.md` 生成 |
| `.dojo/templates/archive/paper/` | 论文解析页（已暂停） |
| `.dojo/templates/archive/note/` | 学习记录页（已暂停） |
| `.dojo/templates/archive/dataflow/` | 模型前向数据流页（已暂停），`archive/dataflow-build.py` 读 `research/dataflow.md` 生成 |

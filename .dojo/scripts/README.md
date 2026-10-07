# 构建脚本

`.dojo/` 下是三类辅助文件：

| 目录 | 放什么 |
|---|---|
| `.dojo/scripts/` | 工具脚本 |
| `.dojo/templates/concept/` | 概念页的长页和短页模板，由 `concept/build.py` 填入内容 |
| `.dojo/site/` | 首页的 JS 和 CSS，部署时拷成 `_site/assets/` |

## 脚本

| 脚本 | 作用 | CI |
|---|---|---|
| `site/vocab.py` | 主题和标签的词表，校验、目录、概念页检查共用 | |
| `site/validate.py` | 发布闸门，入口只有一个，只做调度 | 跑 |
| `site/page.py` | 页面、模板、内联脚本的校验 | 跑 |
| `site/styles.py` | 死 CSS 检测，以及改共享样式前后的渲染比对 | 跑 `--css` |
| `site/catalog.py` | 扫描 `wiki/*/index.html` 生成首页的 `catalog.json` | 跑 |
| `concept/build.py` | 概念页生成器 | 跑自检 |
| `concept/check.py` | 概念页机械检查 | 跑自检 |

### site/validate.py

    # 页面校验（默认；顺带跑内联脚本语法，需要 node，缺失则跳过并提示）
    python3 .dojo/scripts/site/validate.py <页面...>
    python3 .dojo/scripts/site/validate.py --all

    # 专项
    python3 .dojo/scripts/site/validate.py --templates      # 模板
    python3 .dojo/scripts/site/validate.py --js <页面...>   # 只查内联脚本
    python3 .dojo/scripts/site/validate.py --css [--confirm] [--fix]   # 死 CSS
    python3 .dojo/scripts/site/validate.py --catalog --output _site/catalog.json

    # 改共享样式前后的渲染比对
    python3 .dojo/scripts/site/validate.py --diff <before-root> <after-root> <页面...> [--detail] [--jobs N]

`--all` 不含各页 `research/` 里的资料快照。

页面校验覆盖：HTML 骨架、模板占位符、重复 id、同页锚点、本地引用、
`research/` 目录、数学字符、结构图、正文居中、元数据词表、样式表引用。

改样式时的顺序：先 `--diff` 确认改动只影响预期页面，再 `--css` 找死规则。

`--css --confirm` 在 Chrome 里复核候选规则，`--diff` 用 Chrome 比对渲染，这两项需要装 Chrome。

### concept/build.py

    # 读 research/content.md，生成 wiki/<name>/index.html 和 overview.html
    python3 .dojo/scripts/concept/build.py wiki/<name>/research/content.md

    # 另外运行文中后面跟着 text 输出块的 Python 代码，逐行比对输出
    python3 .dojo/scripts/concept/build.py wiki/<name>/research/content.md --run

    # 自检（用虚构数据验证生成器本身可用）
    python3 .dojo/scripts/concept/build.py --selftest

- 文件格式见 `guides/concept/content.md`。格式错误会报出行号。
- 每次生成都打印长页折叠块外的字数和短页的字数，超出 `guides/concept.md` 定的上限时写明。长页顶部的阅读时间也按折叠块外的字算。
- 生成完成后自动运行 `check.py`，检查不过时退出码为 1。

### concept/check.py

    python3 .dojo/scripts/concept/check.py wiki/<name>/research/outline.md --outline   # 大纲
    python3 .dojo/scripts/concept/check.py wiki/<name>/research/content.md             # 正文
    python3 .dojo/scripts/concept/check.py --selftest

- 大纲：长页、短页各一节；每个节点五样「标题｜写什么｜类型｜字数｜依据」，「写什么」一句话；类型限正文、公式、例子、折叠块；短页恰好三节；长页最后一个节点叫「适用条件与边界」；字数预算合计不超上限，折叠块不计入。
- 正文：每段文字后面都要有「依据」，指向 `research/sources/` 里存在的文件，位置是标题里的一段文字或行号范围；正文里的数字必须在 sources 里作为完整的数出现（7 不算命中 17），「自行构造」的段落不查；短页恰好三节，不检查标题文字；站内链接只指向 `wiki/` 下已有的页面；字数上限；代码块只能是 python 加紧跟的输出块；「依据：」必须单独成行。
- 事实、范围、重复、文风、读不读得通，脚本不判断，由 `guides/concept/` 里的审查者负责。
- `research/sources/` 不进 git，只提交 `sources/index.md`。

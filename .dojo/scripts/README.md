# 构建脚本

## 边界

`.dojo/` 只放**通用设施**，不出现任何模型名或页面名。

| 目录 | 放什么 |
|---|---|
| `.dojo/scripts/` | 通用工具，所有页面复用 |
| `.dojo/templates/` | 页面骨架模板 |
| `.dojo/home/` | 首页资源 |

判断标准：文件里出现具体模型名或页面路径，它就不属于 `.dojo/`，而应放进该页
自己的 `wiki/<name>/` 目录（生成脚本、图片等）。

## 只有四个脚本

分组依据是**在哪跑**，不是技术领域：

- 站点的机械检查、构建，全在 CI 跑，归 `ci-`
- 写某类页面时才用的生成与核查，按页面类型归 `concept-`、`dataflow-`；CI 只跑它们的自检和示范页

| 脚本 | 作用 | 何时跑 |
|---|---|---|
| `ci-01-validate.py` | 发布闸门：页面、模板、内联脚本、样式、首页目录，入口只有一个 | CI + 本地 |
| `concept-01-build.py` | 概念页生成器：读 `research/concept.md`，产出 `index.html` 和 `overview.html`；`--run` 跑文中代码 | 写概念页时；CI 跑 `--selftest` 和示范页 |
| `dataflow-01-check.py` | 数据流页核查：节点名/形状回查源码；形状箭头两边都要有；`--geometry` 查连线；`--shots` 截图（画布、说明、折叠） | 写数据流页时 |
| `dataflow-02-build.py` | 数据流页生成器：读计划文件 `research/dataflow.md`，填进模板产出页面 | 写新数据流页时；CI 跑 `--selftest` 和示范页 |

### ci-01-validate.py

    # 页面校验（默认；顺带跑内联脚本语法，需要 node，缺失则跳过并提示）
    python3 .dojo/scripts/ci-01-validate.py <页面...>
    python3 .dojo/scripts/ci-01-validate.py --all

    # 专项
    python3 .dojo/scripts/ci-01-validate.py --templates      # 模板
    python3 .dojo/scripts/ci-01-validate.py --copy [--fix]   # 概览文案
    python3 .dojo/scripts/ci-01-validate.py --js <页面...>   # 只查内联脚本
    python3 .dojo/scripts/ci-01-validate.py --css [--confirm] [--fix]   # 死 CSS
    python3 .dojo/scripts/ci-01-validate.py --catalog --output _site/catalog.json

    # 改共享样式前后的渲染比对
    python3 .dojo/scripts/ci-01-validate.py --diff <before-root> <after-root> <页面...> [--detail]

页面校验覆盖：HTML 骨架、模板占位符、重复 id、同页锚点、本地引用、
`research/` 目录、数学字符、结构图、正文居中、元数据词表、模板与共享 CSS
一致性。`ALLOWED_TOPICS` / `ALLOWED_TAGS` / `ALLOWED_TYPES` 三张词表也定义在
这个文件里，页面校验与目录构建共用同一份。

改样式时的顺序：先 `--diff` 确认改动只影响预期页面，再 `--css` 找死规则。

### concept-01-build.py

    # 读 research/concept.md，生成 wiki/<name>/index.html 和 overview.html
    python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md

    # 另外运行文中后面跟着 text 输出块的 Python 代码，逐行比对输出
    python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md --run

    # 自检（CI 会跑；用虚构数据验证生成器本身可用）
    python3 .dojo/scripts/concept-01-build.py --selftest

- 文件格式见 `guides/concept/write.md`。只认那里列出的 Markdown 写法，别的写法直接报错。
- 格式错误会报出行号：缺固定小节、核心问题不是 3 到 5 个、章节标题带编号、页面文字里写了源码行号、站内链接指向不存在的页面、落单的 `$` 等。
- 「范围」和「依据」两节只给审查者看，不写进页面。

### dataflow-01-check.py

    # 计划文件：「源码」列回查，标题从 def forward 起，节点指的行不是空行或注释且含节点名
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/research/dataflow.md --source-root <源码目录>

    # 事实：节点名与形状回查源码与权重
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py [--shapes shapes.json] [--names-only]

    # 几何：连线穿框与像素重合
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry

    # 截图：每张图一幅画布、一幅打开说明面板
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --shots <目录>

- 计划文件的回查只看起始行和名字是否出现，不判断画得对不对。
- 其余检查只能读生成器产出的页面（含 `DOJO_FLOW_DATA`）。每次运行都先查视图数据自洽：节点 id 不重复，边、分组、批注引用的节点和下钻目标都存在。
- 名字检查只看节点名里的英文标识符在源码里是否出现，不查归属和顺序。数据类型、`nn`、形状用的字母，以及 `x`、`q`、`k`、`v` 这类单字母在检查里被跳过，不会改写成别的变量名。单字母不在源码里时，这项检查也发现不了。
- 形状检查只比对 `shapes.json` 里有的键，页面里要写出完整键名，层号、专家号写成 `{i}`、`{e}` 时按第 0 个比对；没比对上的键逐个列为提示，一个都没比对上时报错。含大写字母的键（如 `A_log`）匹配不到，要人工核对。
- 带标签的 `--source`（`标签=路径`）只用于标题或标签里含该标签的视图。视图没有对上任何标签时，改用没带标签的源。这次传入的源全都带了标签、又没有对上时，这个视图不检查。
- 几何检查和截图需要 Chrome，没有连线的视图不做几何检查。

### dataflow-02-build.py

     # 读计划文件，生成 wiki/<name>/index.html
     python3 .dojo/scripts/dataflow-02-build.py wiki/<name>/research/dataflow.md

     # 自检（CI 会跑；用虚构模型数据验证生成器本身可用）
     python3 .dojo/scripts/dataflow-02-build.py --selftest

计划文件格式见 `guides/model-dataflow/plan.md`。页面骨架仍是
`.dojo/templates/dataflow/index.html`，这个脚本只负责把数据填进去，不复制骨架。

## 页面从哪来

每类页面都有骨架模板。论文、学习记录页复制 `.dojo/templates/<type>/index.html`，替换
`【】` 占位符，得到 `wiki/<name>/index.html`。概念页和数据流页由生成器读 `research/` 下的
Markdown 填进模板，不手工复制。

| 模板 | 产出 |
|---|---|
| `.dojo/templates/concept/` | 概念页与概览页，由 `concept-01-build.py` 读 `research/concept.md` 生成 |
| `.dojo/templates/paper/` | 论文解析页（含概览页与组件片段） |
| `.dojo/templates/note/` | 学习记录页 |
| `.dojo/templates/dataflow/` | 模型前向数据流页（画布 + 参数面板），由 `dataflow-02-build.py` 读 `research/dataflow.md` 生成 |

流程分别见 `guides/concept.md` 和 `guides/model-dataflow.md`。

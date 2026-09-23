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

## 只有两个脚本

分组依据是**在哪跑**，不是技术领域：

- 站点的机械检查、构建，全在 CI 跑，归 `ci-`
- 数据流页的事实与几何核查需要外部源码和权重，CI 拿不到，只能人工跑，归 `dataflow-`

| 脚本 | 作用 | 何时跑 |
|---|---|---|
| `ci-01-validate.py` | 发布闸门：页面、模板、内联脚本、样式、首页目录，入口只有一个 | CI + 本地 |
| `dataflow-01-check.py` | 数据流页核查：节点名/形状回查源码；`--geometry` 查连线 | 写数据流页时 |

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

### dataflow-01-check.py

    # 事实：节点名与形状回查源码与权重
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py [--shapes shapes.json] [--names-only]

    # 几何：连线穿框与像素重合
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry

流程见 `guides/model-dataflow.md`。

## 页面从哪来

每类页面都有骨架模板，用法一致：复制 `.dojo/templates/<type>/index.html`，替换
`【】` 占位符，得到 `wiki/<name>/index.html`。

| 模板 | 产出 |
|---|---|
| `.dojo/templates/concept/` | 概念页（含概览页与组件片段） |
| `.dojo/templates/paper/` | 论文解析页（含概览页与组件片段） |
| `.dojo/templates/note/` | 学习记录页 |
| `.dojo/templates/dataflow/` | 模型前向数据流页（画布 + 参数面板） |

没有生成脚本：页面本身（HTML + 内嵌 JSON）就是唯一的数据来源。

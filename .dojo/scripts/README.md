# .dojo 目录

## 边界

`.dojo/` 只放**通用设施**，不出现任何模型名或页面名。

| 目录 | 放什么 |
|---|---|
| `.dojo/scripts/` | 通用工具与共享模块，所有页面复用 |
| `.dojo/templates/` | 页面骨架模板 |
| `.dojo/home/` | 首页资源 |

判断标准：文件里出现具体模型名或页面路径，它就不属于 `.dojo/`，而应放进该页
自己的 `wiki/<name>/` 目录（生成脚本、图片等）。

## scripts 分组

文件名前缀表示分组，编号只用于组内排序。

| 前缀 | 职责 | 何时跑 |
|---|---|---|
| `ci-` | 发布闸门：站点级机械校验 | 每次提交，CI 自动跑 |
| `dataflow-` | 数据流页的事实与几何核查（通用，接受页面路径参数） | 写数据流页时 |
| `css-` | 样式清理与渲染比对 | 改共享样式时按需跑 |
| `lib-` | 被其他脚本 import 的模块 | 由调用方执行 |

### ci

| 脚本 | 作用 |
|---|---|
| `ci-01-validate.py` | 页面结构、元数据、本地引用、数学字符、research 目录；`--templates` 校验模板；`--copy` 校验概览文案 |
| `ci-02-check-inline-js.py` | 编译检查页面内联 `<script>`，需要 node |

### dataflow

| 脚本 | 作用 |
|---|---|
| `dataflow-01-verify-facts.py` | 节点名、形状、来源回查源码与权重 |
| `dataflow-02-check-geometry.py` | 无头 Chrome 按像素检查连线穿框与重合 |

两个脚本都接受页面路径，不绑定具体页面。流程见 `guides/model-dataflow.md`。

### css

| 脚本 | 作用 |
|---|---|
| `css-01-unused.py` | 找出没有元素会匹配到的样式规则；`--confirm` 用无头 Chrome 复核后再删 |
| `css-02-render-diff.py` | 逐元素比对两棵站点树的渲染结果 |

改共享样式时先跑 `css-01-unused.py`（静态找候选），加 `--confirm` 在真实浏览器里
确认候选确实匹配不到元素，再删。

### lib

| 脚本 | 作用 |
|---|---|
| `lib-01-catalog-builder.py` | 首页目录构建逻辑与主题/标签/类型词表；被 `ci-01-validate.py` import，也可直接执行生成 `catalog.json` |
| `lib-02-dataflow-page.py` | 数据流页的通用骨架与无脚本回退 |

`lib-` 下的模块名带连字符，不能写 `from ... import ...`，用 `importlib` 加载。

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
各类型的写法见对应的 `guides/*.md`。

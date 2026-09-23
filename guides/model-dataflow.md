# 模型前向数据流

适用于推导模型前向路径并生成交互页面。产出 `wiki/<name>/index.html`，页面由数据驱动，只放 JSON，可缩放、可拖动，节点可下钻。

## 目标

读者沿图从输入走到输出，在任一节点查到模块名、张量形状与依据。

## 材料

需要三类材料，缺哪类就说明缺哪类：

- 官方 `config.json`：层数、hidden、头数、专家数、topk 等结构参数
- 官方源码：执行路径、张量维度、算子顺序
- 权重索引或检查点头：每层实际存在哪些张量、形状如何

三者不一致时以源码为准，并在节点 `detail` 中写明分歧。

## 数据契约

    view = { id, label, title, nodes[], edges[], groups[]?, notes[]? }
    node = { id, name, shape?, detail?, kind, drill? }
    edge = { from, to, label? }

`kind` 四种：`tensor` 张量（白底直角）、`op` 算子（浅蓝圆角）、`cache` 缓存（圆柱）、`port` 跨视图输入输出（虚线框）。

`name` 算子名、`shape` 形状、`detail` 依据，三层均可折行；同一视图内所有节点同宽。`drill` 指向目标视图 id，节点可点击跳转。`groups` 把同一子层的节点圈在一起，只画虚线边界。

## 命名

节点名写源码中实际出现的符号，不用部署侧实现名与自造变量。常见错误：

- `einsum`：源码中不存在
- `VocabParallelEmbedding` / `ParallelLMHead` / `linear_gate`：部署侧命名，对应 `nn.Embedding` / `nn.Linear` / `gate_proj`
- `reduced` / `mlp_out`：自造变量

检查点键名与运行时属性名不一致时在 `detail` 中注明，不把检查点键名当属性名。`name` 用源码符号，`detail` 用中文。

## 粒度

按 `forward` 的规模与控制流决定：

- 几行、无分支：一个模块一格
- 十几行、逻辑线性：在父视图中展开
- 几十行、有循环或条件分支：单独开一个视图，父视图只留一格入口

`torch.where` / `torch.cat` 这类单算子不必逐条罗列；`for` 循环、路由选择这类结构必须展开。

## 视图

- 每个视图只表达一条完整路径，第一个视图为首屏可读的主干。
- 每条边两端都标方向：起点圆点、终点箭头。
- 跨视图的量用 `port` 标出，不假装在视图内部产生。

## 生成

生成器在 `.dojo/scripts/dataflow-02-build.py`，通用，不随模型改。它只做一件事：
把数据填进 `.dojo/templates/dataflow/index.html`，产出 `wiki/<name>/index.html`。

数据是这一页的视图、节点、边、config 分组——读源码和权重整理出来的内容。
准备好后调用生成器：

    import importlib
    build = importlib.import_module("dataflow-02-build")   # 模块名带连字符

    VIEWS = [
        build.view("main", "XxxModel", "主干：XxxModel.forward",
                   nodes=[build.node("x", "input_ids", "int64 [B, T]", "tensor"),
                          ...],
                   edges=[build.edge("x", "emb"), ...],
                   groups=[{"label": "主干", "members": [...], "stroke": "#9db6d8"}]),
    ]

    build.page(
        out=Path("wiki/<name>/index.html"),
        meta={"title": ..., "description": ..., "summary": ...,
              "topics": "模型结构", "tag": "数据流"},
        views=VIEWS,
        config_groups=[("分组名", [("键", "值")])],   # 省略则整块参数面板不生成
        config_note="config.json 原值，共 N 个键",
    )

产出物是自包含的 `index.html`：视图数据内嵌在里面，运行时只外链
`libs/dojo-flow.js` 与 `libs/elk.bundled.js`。页面就是唯一交付物。

`node()` 的 `src` 参数是构建期的可读标注（源码行号或权重键），供人复核，
不写进页面。

生成器有自检：

    python3 .dojo/scripts/dataflow-02-build.py --selftest

页面是自包含的：视图数据以 JSON 形式内嵌在 `index.html` 里（`window.DOJO_FLOW_DATA`），
运行时只外链 `libs/dojo-flow.js` 与 `libs/elk.bundled.js`。

不手写节点坐标与连线折点，不做布局后移动节点的后处理，布局与走线交给 ELK。

## 核查

四项全部通过后交付。

    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py --shapes /tmp/shapes.json
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry
    python3 .dojo/scripts/ci-01-validate.py --all

节点的每个形状、参数、算子都要定位回源码或权重；形状写成纯文本（`bf16 [B, T, 6144]`），不写 LaTeX，页面不加载 KaTeX。`--source` 可带标签，带标签的来源只作用于标题含该标签的视图，例如 MTP 视图的事实来源可以是部署侧的 `mtp.py`。

几何自检在无头 Chrome 中按像素统计每条边的穿框与重合，页面路径是位置参数
（`dataflow-01-check.py wiki/<name>/index.html --geometry`）。切换视图后需等
ELK 异步布局完成再测量。

最后打开页面拖动、切换视图、点击下钻、开合参数面板，确认连线方向可读、无穿框、无重合、文字不溢出、节点尺寸一致。脚本不检查可读性。

## research 目录

需要时建 `wiki/<name>/research/`，只放 `.md`。核对结果写入 `review-1.md`，由未参与写作的核对者逐条回查源码与配置并给出定位。核对者不读取写作记录。

## 发布

未获明确授权，不执行 commit 与 push。

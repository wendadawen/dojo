# 模型前向数据流

适用于推导模型前向路径并生成交互页面的任务。产出 `wiki/<name>/index.html`。

## 目标

读者沿图从输入走到输出，在任一节点查到模块名、张量形状与依据。

## 材料

需要三类材料，缺哪类就说明缺哪类：

- 官方源码：执行路径、张量维度、算子顺序
- 官方 `config.json`：层数、hidden、头数、专家数、topk
- 权重索引或检查点头：每层实际有哪些张量、形状如何

三者不一致时以源码为准，并在节点 `detail` 中写明分歧。

## 数据契约

    view = { id, label, title, nodes[], edges[], groups[]?, notes[]? }
    node = { id, name, shape?, detail?, kind, drill? }
    edge = { from, to, label? }

`kind` 四种：`tensor` 张量（白底直角）、`op` 算子（浅蓝圆角）、`cache` 缓存（圆柱）、`port` 跨视图输入输出（虚线框）。`drill` 指向目标视图 id，节点可点击跳转。

`name` 算子名、`shape` 形状、`detail` 依据，三层均可折行；同一视图内所有节点同宽。

## 命名

节点名写源码中实际出现的符号，不用部署侧实现名与自造变量。常见错误：`einsum`（源码中不存在）、`VocabParallelEmbedding`（部署侧命名，对应 `nn.Embedding`）、`reduced`（自造变量）。

检查点键名与运行时属性名不一致时在 `detail` 中注明，不把检查点键名当属性名。`name` 用源码符号，`detail` 用中文。

## 粒度

- 几行、无分支：一个模块一格
- 十几行、逻辑线性：在父视图中展开
- 几十行、有循环或条件分支：单独开一个视图，父视图只留一格入口

`torch.where` / `torch.cat` 这类单算子不必逐条罗列；`for` 循环、路由选择这类结构必须展开。

## 视图

- 每个视图只表达一条完整路径，第一个视图为首屏可读的主干。
- 每条边两端都标方向：起点圆点、终点箭头。
- 跨视图的量用 `port` 标出。

## 生成

用 `.dojo/scripts/dataflow-02-build.py`，它把数据填进 `.dojo/templates/dataflow/index.html`。生成器与模板都通用，只有数据随模型变。

调用方式见生成器文件头（`node` / `edge` / `view` / `page` 四个接口）。数据、生成、核查是同一次任务里的连续动作：读源码时整理出视图与 config 分组，调用生成器落盘，随即核查。产出物只有 `index.html`，视图数据内嵌在里面，运行时只外链 `libs/dojo-flow.js` 与 `libs/elk.bundled.js`。

不手写节点坐标与连线折点，不做布局后移动节点的后处理，布局与走线交给 ELK。

## 核查

    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py --shapes /tmp/shapes.json
    python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --geometry
    python3 .dojo/scripts/ci-01-validate.py --all

每个形状、参数、算子都要定位回源码或权重；形状写成纯文本（`bf16 [B, T, 6144]`），不写 LaTeX，页面不加载 KaTeX。`--source` 可带标签，带标签的来源只作用于标题含该标签的视图。

几何自检在无头 Chrome 中按像素统计每条边的穿框与重合，页面路径是位置参数。

最后打开页面拖动、切换视图、点击下钻、开合参数面板，确认连线方向可读、无穿框、无重合、文字不溢出、节点尺寸一致。脚本不检查可读性。

## research 目录

需要时建 `wiki/<name>/research/`，只放 `.md`。核对结果写入 `review-1.md`，由未参与写作的核对者逐条回查源码与配置并给出定位。核对者不读取写作记录。

## 发布

页面 `dojo:type` 为 `dataflow`，外链 `../../libs/dojo-flow.css`。元数据包括纯文本 `description`、可含 `$...$` 公式的 `dojo:summary`、类型、主题和标签。

未获明确授权，不执行 commit 与 push。

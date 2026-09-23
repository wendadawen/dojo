# 数据流画图

按规划产物生成页面。落实 outline.md 的视图与节点清单，不重新决定范围或粒度；正确性由后续质检确认。

## 1. 输入与边界

输入为 `research/` 下的 scope.md、evidence.md、outline.md，模板 `.dojo/templates/dataflow/index.html`，生成器 `.dojo/scripts/dataflow-02-build.py`。输入缺失或规划完成条件未满足时，不得开始画图。

不得自行增删视图、改变节点粒度、改变分组归属。画的过程中发现缺口时停止相关部分：缺核心步骤或形状时返回 plan.md 修正；局部措辞和 `detail` 可直接补充。

## 2. 数据契约

    view = { id, label, title, nodes[], edges[], groups[]?, notes[]? }
    node = { id, name, shape?, detail?, kind, drill? }
    edge = { from, to, label? }

`kind` 四种：`tensor` 张量（白底直角）、`op` 算子（浅蓝圆角）、`cache` 缓存（圆柱）、`port` 跨视图输入输出（虚线框）。

一个节点三层：`name` 算子名、`shape` 形状、`detail` 依据。同视图内所有节点同宽。

## 3. 命名

节点名写源码中实际出现的符号，不用部署侧实现名与自造变量。常见错误：

- `einsum`：源码中不存在
- `VocabParallelEmbedding` / `ParallelLMHead`：部署侧命名，对应 `nn.Embedding` / `nn.Linear`
- `reduced` / `mlp_out`：自造变量

检查点键名与运行时属性名不一致时，在 `detail` 中注明，不把检查点键名当属性名。`name` 用源码符号，`detail` 用中文。

`detail` 写可回查的硬信息——权重键、配置开关、分层构成统计，不复述节点名。

## 4. 形状

形状写成纯文本，如 `bf16 [B, T, 6144]`，不写 LaTeX。

写每个维度前确认它的来源：`num_attention_heads` 来自 config，`T` 是输入长度，`hc_mult` 是并行残差流数。混淆维度含义是常见错误。

权重形状写成 `名称 [维度]`，如 `q_proj.weight [2048, 1024]`，键名与 `safetensors` 头一致。

## 5. 连接与布局

- 每条边两端都标方向：起点圆点、终点箭头。
- 跨视图的量用 `port` 标出，不假装在视图内部产生。
- 不手写节点坐标与连线折点，不做布局后移动节点的后处理，布局与走线交给 ELK。

第一条是硬规矩：曾在布局后加了一步「挪节点」的后处理，ELK 算好的折点全部失效，只好退化成自绘走线，结果穿框、重叠、绕大圈全来了。想让图好看，改数据（节点粒度、分组），不改坐标。

## 6. 生成页面

调用生成器，把视图数据与 config 分组写进模板：

    import importlib
    build = importlib.import_module("dataflow-02-build")   # 模块名带连字符

    build.page(out=..., meta={...}, views=VIEWS,
               config_groups=[("分组名", [("键", "值")])])

接口与参数详见生成器文件头。产出物是自包含的 `index.html`：视图数据内嵌在里面，运行时只外链 `libs/dojo-flow.js` 与 `libs/elk.bundled.js`。

## 7. 页面上不做的事

- 不加载 KaTeX：画布与参数面板都不用公式
- 不写行号：`src` 只作构建期标注，不写进页面
- 不写引导语、调试过程、临场评价

## 8. 完成条件

- outline.md 的节点与边全部落实
- 规划中列出的分组与 `drill` 都在页面上
- 页面可生成，无占位符残留

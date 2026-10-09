# 模型前向数据流

把一个模型的前向计算画成交互数据流页：整屏是图，每张图画源码里的一个 forward。图可以缩放，在空白处拖动是平移画面，节点不能拖离布局。点开算子进入下一张图；算子右上角有标记时，点标记打开这个算子的公式和说明。每张图配一个说明面板，第一张先用几句话介绍这个模型。

用户点名要某个模型的数据流时，直接从第 1 步做到交付。

## 流程

1. **收集**：把源码、`config.json`、权重文件头、论文收到 `wiki/<name>/research/sources/`，见[收集](dataflow/collect.md)。
2. **计划**：写 `research/dataflow.md`，定下画哪条路径、分哪几张图、每张图的节点连线公式，见[计划](dataflow/plan.md)。
3. **审计划**：派 5 个独立子代理核（前 4 席对照源码，读者与措辞一席只看计划），见[审计划](dataflow/review-plan.md)。
4. **生成**：运行 `python3 .dojo/scripts/dataflow/build_page.py wiki/<name>/research/dataflow.md`，得到 `wiki/<name>/index.html`。
5. **核查页面**：见[核查](dataflow/check.md)，回查源码位置、节点名与形状、页面本身。
6. **修改**：页面内容改 `dataflow.md` 后重新生成，再按第 5 步核查；改了事实错误就回到第 3 步重派一轮。

每一步完成再进入下一步。`<name>` 用小写英文和连字符。交付要求见下节。

## 交付

- `wiki/<name>/index.html`，head 里有完整的首页元数据。
- `wiki/<name>/research/`：`dataflow.md`、每轮的审查记录、`sources/` 下的 `config.json` 副本；拿得到权重形状时再放 `shapes.json`。
- 回复里写交付说明：生成位置、资料的版本、审查了几轮、已知不足。

## 页面信息

`dataflow.md` 开头的四行页面信息写进页面 head：

- 描述：一句话说清这次前向在算什么，纯文本。
- 摘要：首页卡片上的一两句，可以写 `$...$` 公式。
- 主题：从 `.dojo/scripts/site/vocab.py` 的 `ALLOWED_TOPICS` 里选，逗号分隔。
- 标签：从同一文件的 `ALLOWED_TAGS` 里选，数据流页写「数据流」。

主题和标签都要在词表里，`build_page.py` 会核。

## 示范

`wiki/deepseek-v4-1-dataflow/` 是按这份指南生成的数据流页，`research/dataflow.md` 是它的计划文件。写新页时对照它，格式照它写。

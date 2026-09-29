# 模型前向数据流

适用于把一个模型的前向计算画成交互数据流页面的任务。页面整屏是图，每张图画源码里的一个 forward：第一张是整个模型，从输入到输出；后面每张是模型自己定义的一个类。图可以缩放、拖动，点开节点进到下一张。每张图配一个文字面板，写这张图的关键公式和用到的配置值。

用户点名要某个模型的数据流时，直接从第 1 步做到交付。`wiki/<name>/` 里已有旧页面或旧的 `research/dataflow.md`，都按本指南在这个目录重写，不复述旧稿。不要先删掉旧目录，也不要停下来问是恢复还是重做。

## 步骤

1. 收集资料：源码、`config.json`、权重文件头、论文或技术报告，写进计划文件 `research/dataflow.md` 的「资料」一节。写法见[计划文件](model-dataflow/plan.md)。源码优先用 transformers 里的实现，把所用版本写进资料。transformers 里没有这个模型时，用官方仓库的源码。和其他资料对不上时，以这一步选定的源码为准。
2. 定怎么画，写进 `dataflow.md`：画哪条路径（前提），分哪几张图，每张图有哪些节点和连线，每个节点写什么、对应源码哪一行，面板写哪些公式和配置值。格式见[计划文件](model-dataflow/plan.md)，示范见 `guides/examples/dataflow/`。已有的 `dataflow.md` 不符合写法时，先改到符合再进第 3 步。
3. 审查 `dataflow.md`：先运行 `python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/research/dataflow.md --source-root <源码目录>` 回查源码位置（源码目录如 transformers 的包目录）。脚本退出码为 0 之后再审查。派审查之前先确认三件事：计划里只有一条主路径；没有把 `if`、比较、`for` 画成节点；描述、摘要和前提里没有讲页面怎么组织。这三件没做到就先回到第 2 步改，不要先派审查。每轮按[审查](model-dataflow/review.md)派 20 个独立的 `tclaude -p` 进程。通过条件见同一份文件。
4. 生成页面：这一轮的审查记录已经写好、事实错误都已改完之后，再运行 `python3 .dojo/scripts/dataflow-02-build.py wiki/<name>/research/dataflow.md`，得到 `wiki/<name>/index.html`。
5. 检查页面：`python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --shots <目录>` 逐张图截图，看公式能不能正常显示，面板能不能打开。连线用同一个脚本的 `--geometry` 检查有没有穿过节点或叠在一起。节点名用 `--source <源码文件>` 核对；源码位置在第 3 步用计划文件和 `--source-root` 查。源码拆在多个文件时写成 `--source 标签=路径`。有 `shapes.json` 时再加上 `--shapes wiki/<name>/research/sources/shapes.json`。再运行 `python3 .dojo/scripts/ci-01-validate.py wiki/<name>/index.html`。
6. 修改：页面内容改 `dataflow.md` 后重新生成，不手改生成出的 html。图上的显示问题改完回到第 5 步。局部修改不再派新的一轮 20 人。整条路径重画时回到第 2 步重写计划，再按第 3 步重新派一轮。交付前按 plan.md 的写法把每张图再读一遍。

## 交付

- `wiki/<name>/index.html`，head 里有完整的首页元数据
- `wiki/<name>/research/`：`dataflow.md`、每轮的审查记录、`sources/` 下的 `config.json` 副本；拿得到权重形状时再放 `shapes.json`
- 回复里写交付说明：生成位置、资料的版本、审查了几轮、已知不足

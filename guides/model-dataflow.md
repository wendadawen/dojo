# 模型前向数据流

适用于把一个模型的前向计算画成交互数据流页面的任务。页面整屏是图，每张图画源码里的一个 forward：第一张是整个模型，从输入到输出；后面每张是模型自己定义的一个类。图可以缩放、拖动，点开节点进到下一张。每张图配一个文字面板，写这张图的关键公式和用到的配置值。

用户点名要某个模型的数据流时，直接从第 1 步做到交付。

## 步骤

1. 收集资料：源码、`config.json`、权重文件头、论文或技术报告，写进计划文件 `research/dataflow.md` 的「资料」一节。写法见[计划文件](model-dataflow/plan.md)。源码优先用 transformers 里的实现，把所用版本写进资料。transformers 里没有这个模型时，用官方仓库的源码。和其他资料对不上时，以这一步选定的源码为准。
2. 定怎么画，写进 `dataflow.md`：画哪条路径（前提），分哪几张图，每张图有哪些节点和连线，每个节点写什么、对应源码哪一行，面板写哪些公式和配置值。格式见[计划文件](model-dataflow/plan.md)，示范见 `wiki/deepseek-v4-1-dataflow/`。写完后进入第 3 步。
3. 审查 `dataflow.md`：先运行 `python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/research/dataflow.md --source-root <源码目录>` 回查源码位置（源码目录如 transformers 的包目录）。脚本退出码为 0，并且计划符合[计划文件](model-dataflow/plan.md)的「写法」之后，再按[审查](model-dataflow/review.md)派 20 个独立的子代理。还没符合时回到第 2 步。通过条件写在审查文件里。
4. 生成页面：有一轮审查没有事实错误之后，再运行 `python3 .dojo/scripts/dataflow-02-build.py wiki/<name>/research/dataflow.md`，得到 `wiki/<name>/index.html`。
5. 检查页面：
   - `python3 .dojo/scripts/dataflow-01-check.py wiki/<name>/index.html --shots <目录>` 逐张图截图，看公式能不能正常显示，面板能不能打开。
   - 同一个脚本加 `--geometry`，看连线有没有穿过节点或叠在一起。
   - 节点名用 `--source <源码文件>` 核对。源码拆在多个文件时写成 `--source 标签=路径`。
   - 有 `shapes.json` 时加上 `--shapes wiki/<name>/research/sources/shapes.json`。
   - 再运行 `python3 .dojo/scripts/ci-01-validate.py wiki/<name>/index.html`。
6. 修改：
   - 页面内容改 `dataflow.md` 后重新生成，再按第 5 步检查。
   - 改了事实错误就再派一轮，这一轮自己也要没有事实错误才通过。
   - 整条路径重画时回到第 2 步重写计划，再按第 3 步重新派一轮。
   - 交付前按 plan.md「写法」一节核对每张图的节点、连线和面板。

## 交付

- `wiki/<name>/index.html`，head 里有完整的首页元数据
- `wiki/<name>/research/`：`dataflow.md`、每轮的审查记录、`sources/` 下的 `config.json` 副本；拿得到权重形状时再放 `shapes.json`
- 回复里写交付说明：生成位置、资料的版本、审查了几轮、已知不足

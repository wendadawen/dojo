# UltraEP 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成并满足 plan.md 第 6 节完成条件。论文固定为 arXiv:2606.04101v3，TeX 源码包已下载解压至 /tmp/ultraep/src（v1、v2 已被作者撤回，无版本歧义）。Figure 编号经 arXiv v3 HTML 版核对。

## 大纲落实

- 页面开头：定位摘要（lead 三段）、论文元信息（组件 02，无代码仓库故删除代码行）、核心问题块（4 题各配解答折叠块）、术语速查表（14 项）、贯穿示例（EP=4 最小例子）与到第 1 章的过渡——全部落实。
- 章节：7 个正文章节，标题与职责与 outline.md 一致，h2 编号 1–7 连续，各章 h3 编号 1.1 起连续（第 1 章 5 节、第 2 章 5 节、第 3 章 6 节、第 4 章 9 节、第 5 章 7 节、第 6 章 7 节、第 7 章 3 节）。
- 核心问题覆盖：Q1→第 1 章，Q2→第 2 章，Q3→第 4 章，Q4→第 3 章与第 5 章。
- 前置知识：8 个概念页链接（moe-serving、aux-loss-free-routing、gpu-communication、model-parallelism、gpu-execution-model、chunked-prefill、moonep、deepseek-moe），链接位置在正文首次依赖处；脚本核对全部文件存在。无需递归生成新概念页。
- 贯穿示例推进：页面开头给出负载矩阵与初始不均衡 2.0；第 3 章用于说明槽位布局并做三个代价模型的代入检查；第 4 章走完二分三次探测得 $\tau=6$、不均衡 1.0，并给 round-robin 与本地优先两个对照；第 5 章用 $e_0$ 的 fan-out=2 低于中继阈值 4 呼应 Figure 16。
- 误解与边界：decode 不在范围（第 1.1 节 callout）、1.49$\times$ 与 1.42$\times$/1.56$\times$ 的基线区分（第 6.7 节表格）、94.6% 是均值非下界（第 6.2 节 callout）、不均衡度不可线性映射为吞吐差（第 6.4 节 callout）、系统侧不替代辅助损失（第 1.5 节）——全部落实到 outline 指定位置。
- 评价章节：第 7 章，章首 callout-gray 标明分析性判断，含优点/局限/适用场景与相邻工作三节。
- 过渡：各章末尾均有指向下一章的衔接句（第 2.5 节末指向第 4、5 章并交代先讲第 3 章的理由；第 4 章开头回指第 3 章确定的优化对象；第 5 章开头回指第 3 章的逐层期限）。

## 代码运行

- 第 4 章折叠块（Algorithm 1 复现）：运行命令 `/Users/wendadawen/.workbuddy/binaries/python/versions/3.13.12/bin/python3 /tmp/ultraep/solver.py`，退出码 0。实际输出与页面「预期输出」块逐字符一致，含二分三行（tau=9/7/6 均可行）、最终配额表、重路由后负载 {0:6,1:6,2:6,3:6} 不均衡 1.0、round-robin 对照 {0:6,1:6,2:7,3:5} 不均衡 1.1667、e0 的四行拆分、本地优先 41.7% 对关闭后 62.5%。
- 数字自洽核对脚本：`/tmp/ultraep/verify_numbers.py`，退出码 0，17/17 项通过。由 Figure 11/13/16/17 与 Table 3 的标注值反算正文声称的百分比与倍数，全部吻合（如 EPLB/LPLB/EPLB+/Ours 相对 Megatron-LM 实算 20.1/12.3/28.9/41.9% 对正文 20/12/29/42%；0.33 ms 占比实算 1.8%；求解耗时省 27.45% 对 27.4%；冗余槽少 57.94% 对 57.9%；36/72 MB 实算 35.95/71.9）。该脚本为核对工具，不入页面。
- 两处读数差异已记录在 evidence.md 冲突表：§8.5 正文写 3.1$\times$–5.5$\times$，由 Figure 16 读数反算最低档为 3.0$\times$（0.73/0.24），页面按论文表述引用。

## 原图

11 张全部来自 v3 TeX 源码 `figs/`，`pdftoppm -png -r 190` 渲染后经 sharp 转 webp（quality 86，宽度上限 1500），未裁剪未修改，总计 716 KB。

- assets/img-01.webp ← Figure 1 (intro-overview)，1500×767，浏览器实测加载正常
- assets/img-02.webp ← Figure 2 (bg-rsn)，1220×903，正常
- assets/img-03.webp ← Figure 6 (motive_eplb_imbalance)，1500×648，正常
- assets/img-04.webp ← Figure 7 (tech-expert-layout)，1140×546，正常
- assets/img-05.webp ← Figure 8 (tech-timeline-fwd)，1164×370，正常
- assets/img-06.webp ← Figure 10 (tech-relay)，1262×1188，正常
- assets/img-07.webp ← Figure 15 (eval_solver_perf)，1500×816，正常
- assets/img-08.webp ← Figure 16 (eval_comm_perf)，1500×689，正常
- assets/img-09.webp ← Figure 11 (eval_train_e2e)，1500×606，正常
- assets/img-10.webp ← Figure 13 (eval_latency_breakdown)，1500×740，正常
- assets/img-11.webp ← Figure 17 (eval_prod)，650×253，正常

每张按「引导句 → 图片 → 解释段」组织，解释段开头标注原文 Figure 编号。未入页面的 Figure 3/4/5/9/12/14 的排除理由已写入页面末尾「原图与原文对应」小节。

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/ultraep/index.html` → `validation ok`，退出码 0
- `python3 .dojo/scripts/validate.py wiki/ultraep/overview.html` → `validation ok`，退出码 0
- 占位符扫描：两页均无残留【…】，无 @content / @copy-start / @copy-end / @component 标记
- 结构脚本核对：h2 编号 1–7 连续；各章 h3 编号章内连续无跳号；页面级核心问题 4 题 4 解答；6 个章节问题块题数与解答数相等（3/3/3/4/3/4）；带 id 的 h2/h3 锚点全部可定位
- 内部链接：8 个概念页 + 首页 + 3 个 libs 资源，文件存在性全部通过

## 公式渲染与交互

headless Chrome 实测（`--virtual-time-budget=9000 --window-size=1400,2000`，探针脚本注入副本、结果写入 document.title 后从 dump 的 DOM 读取）：

- `.katex` 节点 391 个
- 正文可见文本中未渲染的 `$...$` 片段：0 处
- `document.images` 12 项，除 lightbox 空容器外 11 张原图 naturalWidth/naturalHeight 均非零
- `<details>` 27 个（20 个问题解答 + 4 个补充折叠块 + 3 个代码/示例块，与页面实际一致）
- 侧边目录条目 66 条，`body > h2` 10 个、`body > h3` 56 个，表格 9 个，代码块 2 个
- 锚点异常列表为空

页面未使用内联 SVG，两处结构图使用组件库的 HTML 结构 A（第 3.4 节前向流程）与结构 B（第 5.1 节通信分层），因此不涉及 `<foreignObject>` 公式渲染与 SVG 标签重叠检查。

## 写作偏差

- 原图 G11（Figure 12，serving RPS–TTFT 曲线）按 outline.md 已记录的理由未入页面，serving 结论用 Figure 12 的标注数值以表格呈现。这是规划阶段已作出的决定，非写作期偏差。
- 第 5.5 节增加了一句 $k=\sqrt{n}$ 使两级宽度相等的推导。属于由论文给出的中继前沿取值直接展开的中间步骤，已在页面末尾「论文事实与分析性判断」小节标注为随文推断。
- 第 1.3 节增加了 EP=8 对 EP=64 的十六分之一算例。论文只给结论与「每 rank 常 2 或 4 个主专家」，该算例为本页构造的说明，已在页面末尾标注。
- 其余无偏差：未增删核心章节、未新增核心问题、未更换贯穿示例、未改变前置知识映射、未把正文必要内容移入折叠块。

## 待办

写作阶段的自查不计入质检轮次。页面已完成写作，待按 `guides/paper/check.md` 由三个独立审查者完成三轮全量审查。

# UltraEP 审查记录（第 2 轮）

- 页面版本：e78602f251ec2d097c27090cd12e85d120f6d2cb（index.html 工作树哈希）
- 论文版本：arXiv:2606.04101v3（TeX 源码 /tmp/ultraep/src/；v1 2026-06-02、v2 2026-06-05 均 withdrawn，v3 2026-06-18 提交，已经 arXiv 页面提交历史核实）
- 审查时间：2026-09-02
- 审查者：独立审查者（本轮未参与写作；未读取 research/ 目录下任何已有文件）
- 已完整阅读章节：index.html 全文（1. 非平稳负载使历史式均衡失准 / 2. 精确负载与 RSN / 3. 内存布局与代价模型 / 4. 配额驱动的实时规划 / 5. tile 流水线与 chunk 中继 / 6. 实验结果 / 7. 方法评价 / 来源与范围说明）、overview.html 全文；论文 main.tex 与 sections/ 全部 7 个源文件、table-setup.tex

## 已执行的核对（供复验）

- 章节映射：论文编译顺序 §1–§10（Introduction / Background / Expert Load Analysis / System Design / Quota-Driven Planning / RSN-Native Communication / Implementation / Evaluation / Related Work / Conclusion），与页面「来源与范围说明」声明一致。
- 数字逐条核对：摘要与 §1 的 94.3%、94.6%/93.9%、1.49×/1.42×/1.56×、不均衡 1.30–4.01→1.01–1.04、3.1–5.5×、92%+、9.6%、2×/11×、0.33 ms/1.8%、+33%/+10%、Table 3 五项及派生百分比（27.4%、57.9%、3.9pp）均与原文一致；+20/+12/+29/+42% 按逐模型比值再平均复算成立。
- 渲染 figs/ 下 PDF 核对读图：Figure 11（20 个 TFLOPS 与不均衡标注值、Ideal 785.4/574.7/637.6）、Figure 13（18 个延迟标注值）、Figure 17（ideal 504、no-balancing 425.0）与页面表格完全一致；Figure 16 五档四方案柱高与页面读数在读图误差内一致，中继启用数 0/0/1/2/3 与横轴标注一致；Figure 15 的 (128,64,1) 组 6.0/8.0 档读数约 1.07/1.09 与 EPLB+ 约 1.40 一致；Figure 1 确有「Exposed overhead (~0.3 ms)」标注；Figure 6 左 Qwen3-235B Layer 68 prefill、右 DeepSeek-V3 Layer 57 training，与页面图注一致。
- 代码块实跑：提取 index.html 第 4 章折叠块 Python 代码运行，输出与页面展示的「预期输出」逐行 diff 完全一致。
- 机械项：8 个概念页相对链接全部存在；唯一 h1.title；无占位符；正文无未走 KaTeX 的 Unicode 数学符号（仅表格分隔符「·」）；KaTeX 与 libs 引用存在；`.dojo/scripts/validate.py wiki/ultraep/index.html` 返回 validation ok；推断与构造内容（贯穿示例、√n 推导、第 7 章评价、三处随文推断）均有显式标注。

## 问题

- [轻微·技术] index.html 第 911、954 行（2.2 节及本章问题第 2 题）：「EP64 意味着一个 EP 组横跨 8 到 16 台服务器」是 64/8 与 64/4 的推算，原文只说 scale-up 域「confined to a single 4/8-GPU node」，未给出台数；该推算透明正确，但未按页面自己的惯例标注为推导。｜引文依据：§2.1「the scale-up domain is confined to a single node with 4 or 8 GPUs」（无 8–16 台表述）｜修复要求：在该句后补「（按 64/8 与 64/4 推算）」或删去台数只保留原文表述｜修复：两处（2.2 节正文、本章问题第 2 题解答）均已在「8 到 16 台服务器」后补「（按 64/8 与 64/4 推算）」｜复验：已重新读取两处段落确认标注存在，validate.py 重过
- [轻微·格式] overview.html 第 44 行：「arXiv 预印本 2026 · 更新于 2026-09-01」中「更新于」主语不明——index.html 第 722 行标注的是论文 v3 提交日 2026-06-18，读者易把 2026-09-01 误读为论文版本日期。｜引文依据：不适用（两页日期口径对照）｜修复要求：改为「页面更新于 2026-09-01 · 论文 v3 提交于 2026-06-18」或等价明确表述｜修复：改为「论文 v3 提交于 2026-06-18 · 本页更新于 2026-09-01」｜复验：已重新读取第 44 行确认
- [轻微·格式] overview.html `<head>` 缺 description、dojo:summary、dojo:type、dojo:topics、dojo:tag 五项元信息；check.md 发布条件要求「页面 `<head>`」包含它们，但未明确是否覆盖 overview.html，validate.py 只校验 index.html（通过），且站内惯例不一（chunked-prefill 等概览页有、moonep 等没有）。｜引文依据：不适用｜修复要求：为 overview.html 补齐五项 meta，或在规范中明确该发布条件仅适用于 index.html｜修复：已为 overview.html `<head>` 补齐 description、dojo:summary（概览口径，不含逐节细节）、dojo:type=paper、dojo:topics=并行与通信,推理系统,模型结构、dojo:tag=MoE 专家并行负载均衡，与 index.html 口径一致｜复验：已重新读取 overview.html `<head>` 确认五项齐全

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（3 条轻微均不影响内容正确性：一条是未标注的透明算术推算，两条是格式规范问题；建议修复后归档）

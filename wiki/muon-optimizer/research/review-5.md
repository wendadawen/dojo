<!-- review-meta
round: 5
page: wiki/muon-optimizer/index.html
reviewed_content_sha256: c701a9c44b5b7d3c
-->
# Muon 优化器审查记录（第 5 轮）

- 页面版本：36a53b92ad95f872be67f2c276fce32b0e069fd4（`git hash-object wiki/muon-optimizer/index.html`）
- 审查时间：2026-09-13 20:20
- 审查者：独立子代理（未参与写作与前序审查；未读取本页 research/）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 动量更新矩阵为什么需要正交化——条件数高与稀有方向被忽略（含本章问题）/ 2. Newton-Schulz 迭代如何近似正交化——只用矩阵乘法把奇异值推向 1（含 2.1 手算例子、三个折叠块、本章问题）/ 3. Muon 的完整更新流程与几何含义——动量、正交化与参数更新（含 3.1 几何区别、3.2 与 Shampoo 的关系、3.3 工程开销与外部证据、本章问题）/ 4. Muon 的适用边界——哪些参数用 Muon，哪些仍用 AdamW（含 4.1 QKV 分开、4.2 RMS 对齐缩放因子、本章问题）/ 来源与范围说明；另读 overview.html。
- 来源核对：主源博客 https://kellerjordan.github.io/posts/muon/（逐句核对）、Moonshot AI《Muon is Scalable for LLM Training》arXiv:2502.16982（Eq.4 与 update RMS 段落）、PyTorch 源码 `torch/optim/_muon.py`（main 与 v2.8.0/v2.9.0 标签）、Large et al. 2024 arXiv:2405.14813。页面可运行代码以 numpy 实际执行，输出与页面"预期输出"逐行一致。

## 问题

- [阻断·技术] 2.1 手算例子结尾正文（index.html 第 246 行）："越接近 1 提升越快（多项式在 1 附近斜率较大）"与同页表格数据相反，亦与主源相反。｜引文依据：同页表格（第 237–242 行）给出小奇异值 $0.1104\to0.2189\to0.4222\to0.7383\to0.9826\to1.0001$，逐步增量 $0.1085/0.2033/0.3161/0.2443/0.0175$，最大增量在中段、末步增量最小（$0.9826\to1.0001$ 仅 $+0.0175$）；$\phi'(x)=2-4.5x^2+2.5x^4$，$\phi'(1)=0$ 而非"斜率较大"。主源博客对应图注原文 "Note the steeper growth around x=0."（增长更陡处接近 $x=0$），与页面陈述方向相反。｜修复要求：删除"越接近 1 提升越快（多项式在 1 附近斜率较大）"，改写为与表格数据一致且与主源一致的表述（增量峰值出现在 $x\approx0.4\!-\!0.7$ 的中段、越接近 1 增量越小；或按主源说明"增长最陡处在 $x=0$ 附近"）。不得保留"多项式在 1 附近斜率较大"。｜修复：｜复验：

- [轻微·技术] 第 2 章谱范数归一化段句末（index.html 第 206 行）：引用标注 `[N4]` 与被引论断不符。该句陈述的是 "正交化满足 $\operatorname{Ortho}(cG)=\operatorname{Ortho}(G)$（对任意非零常数 $c$）"，属 F5；N4 的定义是 $\varepsilon=10^{-7}$。｜引文依据：第 555 行"F5（谱范数归一化与 $\operatorname{Ortho}(cG)=\operatorname{Ortho}(G)$）"；第 558 行"N4（$\varepsilon=10^{-7}$）"；第 335 行本章问题答案同一论断亦标注 `[F5]`。｜修复要求：将第 206 行句末 `[N4]` 改为 `[F5]`（与第 335 行一致）。｜修复：｜复验：

- [轻微·表述] 2.1 手算例子（index.html 第 246 行）句首："注意每步小奇异值的提升："为面向读者的祈使式元话语。｜引文依据：不适用｜修复要求：删除"注意……"祈使式开头，改为陈述句（如"小奇异值逐步提升：$0.11\to\cdots\to1.00$"），不保留提示/祈使语。｜修复：｜复验：

- [轻微·表述] 全文小结段句首（index.html 第 522 行）："最后回顾全文要点，回扣开篇的四个核心问题。"为以文档结构为主语的元话语/自我指代（"全文""开篇"）。｜引文依据：不适用｜修复要求：删除"最后回顾全文要点，回扣开篇的……"这一自我指代框架，直接以四点结论或"四个核心问题的答案如下"起句；保留其后的四段要点内容。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复

补充（已核对通过、不构成问题的项）：主源全部引用编号（C1–C8、F1–F7、N1–N12）均有可定位依据，逐条比对原文一致——含 $35\%$/10-15/24/"12 个记录由 7 位研究者"、1.5B 模型 "10 8xH100-hours vs 13.3 hours"、CIFAR-10 "94% accuracy ... 3.3 to 2.6 A100-seconds"、NanoGPT 目标 "3.28 val loss on FineWeb"、"We find that using momentum is necessary for the best empirical performance."、"a phenomenon ... applied separately to the Q,K,V parameters"（Vlado Boza）、"stably run in bfloat16"/"coupled Newton iteration must be run in at least float32"、FLOP "improved from 6nm² to 4nm² + 2m³" 与上限 "Tm/B"、"instantaneous/accumulation-free Shampoo" 及式 $(G_tG_t^\top)^{-1/4}G_t(G_t^\top G_t)^{-1/4}$、embedding "follows from the modular norm theory (Large et al. 2024)"、output 层 "does not seem to follow from the theory, and is instead driven by empirics."。Moonshot Eq.4 "$0.2\cdot\mathbf O_t\cdot\sqrt{\max(A,B)}$" 与 "AdamW's update RMS is usually around 0.2 to 0.4" 支持 $\gamma=0.2\sqrt{\max(m,n)}$ 及 RMS 对齐叙述。PyTorch `_muon.py` 默认值 lr=1e-3、weight_decay=0.1、momentum=0.95、nesterov=True、ns_coefficients=(3.4445,-4.7750,2.0315)、ns_steps=5、eps=1e-7，`adjust_lr_fn` 未指定即按 "original"（docstring 明示），v2.9.0 存在 `_muon.py` 而 v2.8.0 不存在（"2.9 起内置"成立）。公式可复算：$\|G\|_F=\sqrt{0.82}\approx0.9055$、归一化奇异值 $(0.9939,0.1104)$、基线 5 步 $(1.0001,1)$、调优 5 步 $(0.7529,0.7034)$、$\operatorname{RMS}(UV^\top)=\sqrt{1/\max(m,n)}$、$\gamma=0.2\sqrt2\approx0.283$ 均正确；Shampoo 等价与 $\operatorname{RMS}$ 推导折叠块步骤完整无误。可运行代码以 numpy float64 实跑，输出与页面预期输出完全一致（含"基线 O^T O"接近 I）。页面前置概念链接 `../newton-schulz/index.html`、`../svd/index.html` 及 overview 的 `../per-head-muon/index.html` 均真实存在；页面与 overview 相互链接；无 research/ 文件路径引用、无"（待生成）"占位；`validate.py` 返回 "validation ok"。表述其余部分未发现会话指代（我/我们/你）、调试叙事、临场评价；"本文…"为全站通行用法，未计入问题。
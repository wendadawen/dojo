<!-- review-meta
round: 4
page: wiki/muon-optimizer/index.html
reviewed_content_sha256: 9423546aa63d21a7
-->
# Muon 优化器审查记录（第 4 轮）

- 页面版本：1d0fa0cde10e076b3b2af3eed8cfda9a8f1c77af（index.html 工作树哈希）
- 审查时间：2026-09-13 19:46
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节（按顺序）：引言 / 核心问题 / 最容易误解；1. 动量更新矩阵为什么需要正交化——条件数高与稀有方向被忽略（含本章问题）；2. Newton-Schulz 迭代如何近似正交化——只用矩阵乘法把奇异值推向 1（含 2.1 手算例子「把 diag(0.9,0.1) 拉平」、两个「展开」折叠块、numpy 代码折叠块、本章问题）；3. Muon 的完整更新流程与几何含义——动量、正交化与参数更新（含伪代码折叠块、3.1 与 SGD-momentum/AdamW 的几何区别、3.2 与 Shampoo 的关系、3.3 工程开销与外部证据、本章问题）；4. Muon 的适用边界——哪些参数用 Muon，哪些仍用 AdamW（含表、4.1 QKV 应分开应用、4.2 RMS 对齐缩放因子、PyTorch 默认超参表、本章问题）；来源与范围说明。

## 机械验证（本轮实际执行）

- 代码复现：按页面折叠块「代码：用 numpy 复现 diag(0.9, 0.1) 的 5 步 NS 迭代」原样运行，输出为 `||G||_F = 0.905539`、`谱归一化后奇异值 = [0.9939 0.1104]`、`基线系数 5 步后奇异值 = [1.0001 1.]`、`调优系数 5 步后奇异值 = [0.7529 0.7034]`、`基线 O^T O = [[1. 0.] [0. 1.0003]]`，与页面「预期输出」逐行一致。
- 手算表复算：以全精度初值 0.9/√0.82、0.1/√0.82 逐位施加基线 φ，得 0.2189 → 0.4222 → 0.7383 → 0.9826 → 1.0001，与页面 5 步表一致；调优系数第 1 步 φ(0.1104) = 0.37398 ≈ 0.3740，与页面一致；调优系数大奇异值路径 0.7056 → 1.1083 → 0.7141 → 1.0982 → 0.7034，与「在 [0.7,1.3] 内波动」一致。
- `.dojo/scripts/validate.py wiki/muon-optimizer/index.html` 返回 `validation ok`；来源编号 C1–C8、F1–F7、N1–N12 与正文上标双向一一对应，无悬空、无未引用；页面无指向 `research/` 的路径，无「（待生成）」。

## 问题

- [阻断·技术] 3 章第 2 步（及其伪代码折叠块第 1 步、第 3 章「本章问题」第 1 题解答）：Nesterov 前瞻动量写成使用上一步动量 $\tilde{M}_t = G_t + \beta M_{t-1}$，与页面所引三处来源均不符。来源一致地把「更新后的动量」$\beta M_{t-1}+G_t$ 与当前梯度组合，即 $\tilde{M}_t = G_t + \beta M_t$。｜引文依据：Moonshot《Muon is Scalable for LLM Training》Eq.(1) 正文原文 "In practice, we follow (Jordan et al., 2024) to use a Nesterov-style momentum by putting μ𝐌_t + ∇ℒ_t(𝐖_{t-1}) to the Newton-Schulz iteration instead of 𝐌_t"；PyTorch `torch.optim.Muon` 算法行 "B̃_t ← { g_t + μ B_t, if nesterov=True"；`torch/optim/_muon.py` 第 345 行 `buf.lerp_(grad, 1 - momentum); update = grad.lerp(buf, momentum) if nesterov else buf`（先更新 buf，再用更新后的 buf 组合）。｜修复要求：把前瞻动量改为 $\tilde{M}_t = G_t + \beta M_t$（顺序上先按 $M_t = \beta M_{t-1} + G_t$ 更新动量缓冲，再以 $G_t + \beta M_t$ 作为被正交化对象），并同步修改 3 章第 2 步正文、伪代码折叠块第 1 步与第 3 章「本章问题」第 1 题解答；改后重新核对来源。｜修复：｜复验：
- [重要·技术] 2 章末段与 3.3 节、4.2 节：维度命名 $m/n$ 与「行/列」的对应关系全文不单义。2 章末段写「若 $G$ 的**行数**大于列数（$m > n$）」、伪代码注释写「保证 $m \le n$」，此处以 $m$ 为行数；而 3.3 节写「对 $n \times m$ 矩阵（$m \le n$）」与 4.2 节写「矩阵 $W \in \mathbb{R}^{n \times m}$」「$U \in \mathbb{R}^{n \times r}$，$V \in \mathbb{R}^{m \times r}$」，按标准读法以 $n$ 为行数。同一页面两处对同一对符号给出相反含义。｜引文依据：不适用（页面内部一致性）。｜修复要求：全文统一 $m/n$ 与行/列的对应（例如统一令矩阵为 $n \times m$、行数为 $n$、列数为 $m$），使「行数大于列数」的括号条件、3.3 节的 $n \times m$（$m \le n$）写法与代码注释「保证 $m \le n$」表达同一件事；改后 $r=\min(m,n)$、$\max(m,n)$ 仍成立（该式对 $m,n$ 对称，不需要改公式）。｜修复：｜复验：
- [重要·技术] 4.2 节与 overview.html：「PyTorch 内置版本」两页说法冲突，且 overview 的说法与来源不符。index.html 写「PyTorch `torch.optim.Muon`（2.9 起内置）」，overview.html 第 3 节写「RMS 对齐缩放因子（Moonshot 改进，PyTorch 2.12 采用）」。｜引文依据：PyTorch 2.9 版文档页 `torch.optim.Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None)`（2.9 已内置），与 index.html 正文一致；overview 的 2.12 无来源支持。｜修复要求：把 overview.html 中「PyTorch 2.12 采用」改为「PyTorch 2.9 起内置」，与 index.html 及来源一致。｜修复：｜复验：
- [轻微·技术] 1 章第 3 段与 1 章「本章问题」第 1 题解答：条件数高的经验观察省略了来源限定。来源限定为「transformer 网络的二维参数」且依据是「人工检查」，页面写成无条件的「SGD-momentum 和 Adam 产生的二维参数更新矩阵通常条件数很高」。｜引文依据："based on manual inspection, the updates produced by both SGD-momentum and Adam for the 2D parameters in transformer-based neural networks typically have very high condition number."（Keller Jordan 2024 博客）｜修复要求：补回「transformer 网络」「人工检查」限定，或明确写成「作者对训练得到的更新矩阵人工检查后观察到」；同时在第 1 章「本章问题」解答的经验范围句中补上同一限定。｜修复：｜复验：
- [轻微·技术] 2 章：NS 迭代步数用两个符号表示，未统一。正文 2 章写「应用 $N$ 步 NS 迭代的输出为 $U\,\phi^N(S)\,V^\top$」，而 3.3 节的 $T\cdot m/B$、伪代码「NS 步数 $T=5$」用 $T$ 表示同一量（默认 5）。｜引文依据：不适用（符号单义要求）。｜修复要求：把表示 NS 迭代步数的符号统一（例如统一写作 $T$，或明确说明 F3 中的 $N$ 即步骤数、默认取值 $T=5$）。｜修复：｜复验：
- [轻微·表述] 「来源与范围说明」的「公式与来源（F）」段：出现两处以「本页」为主语的自我指代——「本页折叠块补全过程」「本页折叠块补全 RMS 推导」。｜引文依据：不适用。｜修复要求：改为「推导见上文折叠块补全」「RMS 推导由上文折叠块补全」等不含「本页」主语的写法。｜修复：｜复验：

## 已核对、本轮未发现问题的项（留档）

- NS 单步公式 $G'=(aI+b(GG^\top)+c(GG^\top)^2)G$、SVD 下 $U\phi(S)V^\top$、$N$ 步复合 $U\phi^N(S)V^\top$、正交化目标 $\arg\min_O\|O-G\|_F$、谱范数归一化与 $\operatorname{Ortho}(cG)=\operatorname{Ortho}(G)$、关闭动量的 Shampoo 等价 $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}=UV^\top$，均与主源博客逐字一致；两个折叠块推导可复算。
- RMS 对齐缩放因子 $\gamma=0.2\sqrt{\max(m,n)}$ 与 $\operatorname{RMS}(UV^\top)=\sqrt{1/\max(m,n)}$ 与 Moonshot 原文 Eq.(7) 及三处「AdamW 的 update RMS 通常约 0.2~0.4」一致；$0.2\sqrt2\approx0.283$、$\operatorname{RMS}(I_2)=0.707$ 复算无误。
- FLOP：每步 $2(2nm^2+m^3)$、由 $6nm^2$ 改进为 $4nm^2+2m^3$、开销上界 $Tm/B$、NanoGPT 0.7%（$m=768,B=524288$）、Llama 405B 0.5%（$m=16384,B=16\text{M}$）均与主源博客一致。
- 外部数字：2024-10-15 首次记录、提速 35%、此后 12 个记录由 7 位研究者创造、1.5B transformer 10 vs 13.3 8xH100-hours、CIFAR-10 3.3→2.6 A100-seconds、val loss 3.28 / FineWeb 目标、调优系数与基线系数、$\varepsilon=10^{-7}$、$[0.7,1.3]$、bfloat16 稳定性与 coupled Newton 需 float32，均与来源一致。
- PyTorch 默认超参表（lr=1e-3、weight_decay=0.1、momentum=0.95、nesterov=True、ns_coefficients=(3.4445,-4.775,2.0315)、ns_steps=5、eps=1e-7、adjust_lr_fn 默认 'original'）与官方源码/文档逐项一致。
- 表述维度：正文与折叠块无「我们/你/我」等会话指代，无「值得注意/综上所述/由此可见」类公文连接词，无调试与复现踩坑叙事，无临场评价；「场景」仅作普通用词（「典型 LM 训练场景」）。「本文」为规范允许的自称。

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 3
- 处置：修复（阻断与重要问题修复并复验后再进入下一轮）
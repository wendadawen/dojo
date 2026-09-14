<!-- review-meta
round: 6
page: wiki/muon-optimizer/index.html
reviewed_content_sha256: 8708fe0203bfba6b
-->
# Muon 优化器审查记录（第 6 轮）

- 页面版本：aedbd04c144069547d3f9b685203c05cf79070d5
- 审查时间：2026-09-13 21:16 CST
- 审查者：独立子代理
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 动量更新矩阵为什么需要正交化——条件数高与稀有方向被忽略 / 2. Newton-Schulz 迭代如何近似正交化——只用矩阵乘法把奇异值推向 1（含 2.1、全部折叠块）/ 3. Muon 的完整更新流程与几何含义——动量、正交化与参数更新（含 3.1–3.3）/ 4. Muon 的适用边界——哪些参数用 Muon，哪些仍用 AdamW（含 4.1–4.2）/ 来源与范围说明
- 页面类型：dojo:type=concept（适用 guides/concept/check.md）

## 来源核对（关键数值与原文片段）

- 条件数高｜博客原文：「based on manual inspection, the updates produced by both SGD-momentum and Adam for the 2D parameters in transformer-based neural networks typically have very high condition number. That is, they are almost low-rank matrices, with the updates for all neurons being dominated by just a few directions.」页面 §1 表述一致，且已标注为人工检查得到的经验观察。
- 稀有方向｜原文：「We speculate that orthogonalization effectively increases the scale of other "rare directions"」——页面 142 行如实标注为「作者推测」。
- NS 单步公式与系数｜原文：`G' := aG + b(GG^T)G + c(GG^T)^2G`；基线 `(a, b, c) = (2, -1.5, 0.5)`；调优 `(3.4445, -4.7750, 2.0315)`；`T=5`；`eps = 1e-7`。页面 F1、208 行、PyTorch 表一致。
- 调优原则｜原文：「make a as large as possible, since φ'(0) = a」「ε can be as high as around 0.3 without harming the loss curve」「maximize a subject to lim φ^N(x) ∈ [0.7, 1.3]」。页面 208 行一致。
- FLOP｜原文：「each step of the NS iteration requires 2(2nm^2 + m^3) matmul FLOPs, which is at most 6nm^2」「improved from 6nm^2 to 4nm^2 + 2m^3 FLOPs」（Bernstein, You, Cesista）。页面 430 行一致；复算 5*768/524288=0.73%≈0.7%、5*16384/16000000=0.51%≈0.5%。
- 外部证据｜原文：CIFAR-10「from 3.3 to 2.6 A100-seconds」、1.5B「in 10 8xH100-hours. Using AdamW ... takes 13.3 hours」、「on 10/15/24 ... improved the training speed by 35%」「all 12 ... set by 7 different researchers」。页面 432 行逐项一致。
- Shampoo｜原文：去除 preconditioner accumulation 后 `W_{t+1} = W_t - η (G_tG_t^⊤)^{-1/4}G_t(G_t^⊤G_t)^{-1/4} = W_t - η UV^⊤`，由 Bernstein & Newhouse (2024) 观察到。页面 F6、折叠块一致。
- 参数分组｜原文：「scalar and vector parameters ... as well as the input and output layers, should be optimized by a standard method such as AdamW」「AdamW should be used for the embedding and final classifier head layers」「the modular norm theory (Large et al. 2024)」「A third result is ... applied to their Q, K, V parameters separately」。页面 §4、4.1、折叠块一致。
- bfloat16｜原文：「we find that Newton-Schulz iterations ... can be stably run in bfloat16」；coupled Newton iteration「must be run in at least float32 precision to avoid numerical instability」。页面 214 行一致。
- RMS 对齐｜Moonshot《Muon is Scalable for LLM Training》(arXiv:2502.16982v1, 2025-02-24)：`Lemma 1. For a full-rank matrix parameter of shape [A,B], its theoretical Muon update RMS is √(1/max(A,B))`；`W_t = W_{t-1} - η_t(0.2·O_t·√max(A,B) + λW_{t-1})`（Eq. 4）；「From empirical observations, AdamW's update RMS is usually around 0.2 to 0.4」「Muon can directly reuse the learning rate and weight decay tuned for AdamW」。页面 484–505 行一致；复算 ‖UV^⊤‖_F²=r、RMS=√(r/(nm))=√(1/max(m,n))、0.283×0.7071=0.2 均成立。
- PyTorch 默认｜`torch/optim/_muon.py`：`lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445,-4.7750,2.0315), eps=1e-7, ns_steps=5, adjust_lr_fn=None`（None 按 "original" 处理）；`_adjust_lr` 中 `"match_rms_adamw" → 0.2*sqrt(max(A,B))`。该文件存在于 v2.9.0 / v2.10.0，v2.8.0 无——页面「2.9 起内置」及 N11 表逐项一致。
- 公式复算｜φ_base(0.99388)=1.0000、φ_base(0.11043)=0.2189、φ_tuned(0.11043)=0.3740、φ_tuned(1.0)=0.701；基线 5 步奇异值序列 0.9939/0.1104 → 1.0000/0.2189 → … → 1.0000/1.0001 与页面表格逐位相符。
- 代码执行｜页面 267–302 行的 numpy 代码在 Python3 下实跑，输出逐行等于页面 306–313 行「预期输出」（||G||_F=0.905539、[0.9939 0.1104]、[1.0001 1.]、[0.7529 0.7034]、O^⊤O≈I）。
- 链接与资源｜../newton-schulz/、../svd/、overview.html 与 ../../libs/ 下全部资源均存在；`validate.py wiki/muon-optimizer/index.html` 返回 validation ok；全文无 我/我们/你，无 `$...$` 出现在 alt（唯一 img 为 lightbox，alt==""）。
- 首末一致性｜核心问题 4 题的解答均给出结论并指明所在章节（§1+§2 / §2 / §3 / §4），与 522 行四问收束一致；summary、overview 与正文无数字冲突。

## 问题

- [轻微·表述] §1/§2/§3 章末衔接句（150、320、434 行）：三处过渡为同一句式「本章……。但……——下一章讲……」｜引文依据：不适用｜修复要求：改写其中至少两处，直接陈述前一节结论与下一节问题的依赖关系（例：由「直接做 SVD 太慢」引出「需要用只含矩阵乘法的迭代逼近正交化」），不出现「下一章讲……」式结构预告；style-guide §8 要求不使用固定句式｜修复：｜复验：
- [轻微·技术] §4 参数分组表「卷积核（展平后三维）」行：括注「展平后三维」含混，与同格「原因」列「展平为二维后适用」及来源不符（来源为 4D 卷积参数）｜引文依据：博客「Muon can be used for 4D convolutional parameters by flattening their last three dimensions」｜修复要求：括注改为「卷积核（4D 展平最后三维为 2D）」，与形状列 $[\text{out}, \text{in}\cdot k^2]$ 自洽｜修复：｜复验：
- [轻微·技术] §2 末段（214 行）：「SVD 太慢，Shampoo 使用的 coupled Newton iteration 需要至少 float32 精度才能避免数值不稳定」属来源论断，但未带任何 [Cx]/[Nx] 标注（同段 bfloat16 一句已标 [C8]）｜引文依据：博客「we don't use it because we find that it must be run in at least float32 precision to avoid numerical instability」｜修复要求：为该句补引文编号并在来源章节登记对应条目（可并入 C8 段或新增 C9）｜修复：｜复验：
- [轻微·格式] 来源与范围说明 → 外部数字与实验条件（N）（558 行）：列出 N4（ε=10⁻⁷），但正文对 ε 仅以 [F5] 标注（204、206 行），全文检索无 [N4]，正文与来源双向对应不成立｜引文依据：正文所有上标为 [C1]–[C8]、[F1]–[F7]、[N1,N2,N3,N5]、[N6,N7]、[N8]–[N12]，无 [N4]｜修复要求：删除 N4 条目，或在 204 行 ε 处补标 [N4]；style-guide §6 要求双向对应｜修复：｜复验：
- [轻微·技术] 来源与范围说明 → 公式与来源（F）（555 行）：F7 注「推导参考苏剑林博客（kexue.fm）」未给出具体文章标题或链接，无法定位核对；而该 RMS 推导本身已见于 Moonshot 论文｜引文依据：Moonshot 论文「Lemma 1 ... The proof can be found in the Appendix A」｜修复要求：删除该博客引用，或补全可定位的文章标题与 URL｜修复：｜复验：
- [轻微·格式] §4.2（505 行）与来源说明 N10（558 行）：「对前面的 2×2 例子」用未由 KaTeX 渲染的 ×，N10 的「3.3→2.6」用未渲染的 →；同一页 568 行同类写法为 $\times$（$768 \times 768$），全页写法不一致｜引文依据：不适用｜修复要求：505 行改为 $2\times2$（或「$2 \times 2$ 的例子」），558 行 N10 的数值关系改写为「由 3.3 降至 2.6」或使用 LaTeX；style-guide §11 要求数学运算符统一由 KaTeX 渲染、同页写法一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：修复（无阻断、无重要；上述 6 项轻微均为表述/引文格式层面，逐条修复后即可发布；核心事实论断、公式与代码输出已逐条回源核对并复算通过）

> 本轮所列问题的处理结果见 `minor-fixes.md`。

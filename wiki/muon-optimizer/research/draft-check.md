# Muon 优化器初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均为本轮新生成（覆盖占位），位于 wiki/muon-optimizer/research/
- 模板：.dojo/templates/concept/index.html + overview.html + components.html
- 写作正本：guides/concept/content-examples.md

## 大纲落实

- 页面开头：callout（钩子问题）+ context-box（概念定位）+ learning-goals（4 个学习目标）+ blockquote.meta（主要依据）——已落实。
- S1 动量更新矩阵为什么需要正交化——已落实，含动量更新式、SVD 视角、条件数经验观察、正交化目标 Ortho(G) 与 UVᵀ 等价、贯穿例子 diag(0.9, 0.1) 首次出现。
- S2 Newton-Schulz 迭代如何近似正交化——已落实，含单步公式、SVD 下多项式映射推导（折叠块）、N 步复合、谱范数归一化、基线与调优系数、bfloat16 稳定性、手算前两步、完整 5 步折叠块、调优 vs 基线对照折叠块、可运行代码折叠块。
- S3 Muon 完整更新流程与几何含义——已落实，含五步算法、伪代码折叠块、与 SGD-momentum/AdamW 几何对比表、与 Shampoo 关系及推导折叠块、工程开销与外部证据。
- S4 Muon 适用边界——已落实，含参数分类表、embedding/lm_head 用 AdamW 原因、QKV 分开、RMS 对齐缩放因子及推导折叠块、PyTorch 默认超参表、常见误解回顾。
- 文末来源与教学说明——已落实，含核心论断与来源、核心公式与来源、外部数字与实验条件、教学示例、教学解释与类比边界、教学简化及其限制。
- 每章完成检查——S1/S2/S3/S4 均有完成检查项。
- 章间过渡——S1→S2（正交化目标明确但 SVD 太慢）、S2→S3（组装完整算法并对比）、S3→S4（适用边界）均落实。
- 前置知识引用——Newton-Schulz 正交化标注占位页（待生成），SGD-momentum/AdamW 一句话回顾。
- 贯穿例子——diag(0.9, 0.1) 跨 S1/S2/S3/S4 推进，每次增加新层次。
- 误解和边界——S1 条件数、S2 近似非精确、S4 适用范围与常见误解三处。

## 学习目标闭环

- Q1（Muon 每步做什么、为什么改善更新）：S1 讲清动机（条件数高、少数方向主导、正交化拉平），S3 讲清完整五步流程与动量在正交化前加入。正文完整回答，不依赖折叠块。
- Q2（NS 迭代如何变成矩阵乘法）：S2 讲清单步公式、SVD 下多项式映射、谱范数归一化、系数与步数。正文有公式与手算前两步，完整 5 步与代码在折叠块。正文完整回答。
- Q3（与 SGD-momentum/AdamW 几何区别）：S3 讲清三者更新方向几何结构对比表与文字说明，与 Shampoo 关系。正文完整回答。
- Q4（适用边界）：S4 讲清参数分类表、embedding/lm_head 用 AdamW 原因、QKV 分开、RMS 对齐。正文完整回答。

## 代码运行

- 代码块：wiki/muon-optimizer/index.html 中"可运行代码：用 numpy 复现 diag(0.9, 0.1) 的 5 步 NS 迭代"。
- 运行命令：`python3 /tmp/muon_page_code.py`（代码已提取到临时文件独立运行）。
- 退出码：0。
- 实际输出与页面预期输出一致：
  - `||G||_F = 0.905539` ✓
  - `谱归一化后奇异值 = [0.9939 0.1104]` ✓
  - `基线系数 5 步后奇异值 = [1.0001 1.    ]` ✓
  - `调优系数 5 步后奇异值 = [0.7529 0.7034]` ✓（落在 [0.7, 1.3] 容忍区间）
  - `基线 O^T O ≈ I` ✓（已修正预期输出中 1.0003 与实际一致）
- 手算表格数字（S2 完整 5 步折叠块）与代码输出一致：初始 (0.9939, 0.1104) → 第1步 (1.0001, 0.2189) → 第2步 (1.0000, 0.4222) → 第3步 (1.0000, 0.7383) → 第4步 (1.0000, 0.9826) → 第5步 (1.0000, 1.0001)。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/muon-optimizer/index.html`
- 结果：`validation ok`，退出码 0。
- 命令：`python3 .dojo/scripts/validate.py wiki/muon-optimizer/overview.html`
- 结果：`validation ok`，退出码 0。
- 残留占位符检查：`grep "【"` 无结果；`grep "@copy\|@component"` 无结果；`grep "@content"` 无结果。

## 公式渲染与交互

- KaTeX 公式使用 `$...$`（行内）和 `$$...$$`（展示），与模板 auto-render 配置一致。
- 关键公式：动量更新式 $M_t=\beta M_{t-1}+G_t$、正交化目标 $\operatorname{Ortho}(G)=\arg\min$、NS 单步 $G'=(aI+b(GG^\top)+c(GG^\top)^2)G$、SVD 下 $U\phi(S)V^\top$、Shampoo $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}=UV^\top$、RMS $\sqrt{1/\max(m,n)}$、缩放 $\gamma=0.2\sqrt{\max(m,n)}$。
- 符号全文一致：$M_t$ 动量、$G_t$ 梯度、$O_t$ 正交化更新、$\beta$ 动量系数、$\eta$ 学习率、$U,S,V$ SVD 分量、$\phi$ 多项式、$a,b,c$ 系数、$m,n$ 矩阵形状、$\gamma$ 缩放因子。与 glossary.md 一致。
- 折叠块：6 个 details（NS 推导、完整 5 步手算、调优 vs 基线对照、可运行代码、Shampoo 推导、RMS 推导）+ 1 个伪代码折叠块。全部收起时正文仍回答全部学习目标（已在学习目标闭环核对）。
- 互相链接：index.html 有 overview.html 链接（模板 nav 中）；overview.html 有 index.html 链接。前置概念 newton-schulz 链接为 `../newton-schulz/index.html`（占位页），per-head-muon 链接为 `../per-head-muon/index.html`（已存在）。

## 写作偏差

无。大纲全部章节、学习目标、前置知识、贯穿例子、误解和边界、过渡均已落实，未引入范围外内容。唯一局部修正：代码预期输出中 `O^T O` 对角线第二个元素由 `1.` 修正为 `1.0003`，与实际运行结果一致（不影响机制说明）。

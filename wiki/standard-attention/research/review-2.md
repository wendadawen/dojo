# 标准 Transformer 注意力 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html ac5b744、overview.html ac5b744
- 时间：2026-08-09
- 来源：Vaswani et al. 2017, "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762，通过 ar5iv HTML 版逐条核对 §3.2.1（缩放点积、脚注 4、加性注意力对比）、§3.2.2（多头公式与"不同表示子空间"原文）、§3.2.3（因果遮罩）、§4.1 Table 1（复杂度对比）、Table 3（base 模型超参数）。

## 问题

- [重要·技术] "注意力要解决什么问题" 章复杂度对比表 CNN 行：最大路径长度写"$O(\log_k n)$ 或 $O(n/k)$"，但来源 Vaswani 2017 §4.1 Table 1 中 CNN 行只列 $O(\log_k n)$，无"$O(n/k)$"。原表共四行（Self-Attention、RNN、Convolutional、Self-Attention (restricted)），页面删去了 restricted 行，其路径长度为 $O(n/r)$。页面中的"$O(n/k)$"很可能是把 restricted self-attention 的 $O(n/r)$（邻域 r）误挪到 CNN 行（换字母 r→k），或来自非来源的替代推导；表格下方标注"来源：Vaswani et al. 2017, §4.1 Table 1"让读者以为整行来自原文。修法：CNN 行最大路径长度只保留 $O(\log_k n)$；若要补充"$O(n/k)$ 是标准 CNN（非膨胀）的路径长度"作为教学推导，须另起一行并明确标注"教学补充，非论文 Table 1"；或补回 restricted self-attention 行 $O(n/r)$。 ｜ 修复：采用最小化方案——CNN 行最大路径长度删除"或 $O(n/k)$"，只保留 $O(\log_k n)$，与 Vaswani 2017 §4.1 Table 1 CNN 行一致；overview.html 同步删除"$O(n/k)$ 或"。index.html 来源标注 N2 行本就只写 $O(\log_k n)$，无需改动。validate.py 通过。 ｜ 复验：
- [轻微·盲读] "为什么除以 √d_k" 章"不缩放的后果"段：同一句写"$n$ 个 key 中最大 logit 与典型值差值常达 16+（如 $n=256$ 时 $E[\max]\approx 8\sqrt{2\ln 256}\approx 27$）"。"16+"与同一括号内给出的 27 不一致——按括号内推导，$n=256$ 时最大 logit 期望约 27、典型值约 0、差值约 27 而非 16。小白读者会困惑"到底是 16 还是 27"。修法：将"16+"改为与括号一致的数字（如"差值可达 27 量级"），或说明 16+ 对应更小的 $n$（如 $n=8$）。 ｜ 修复： ｜ 复验：
- [轻微·盲读] "来源与教学说明"章教学简化条目写"不展开加性注意力的完整机制：只在 S3 一句话作对照"。页面章节标题为中文（"注意力要解决什么问题""缩放点积公式""为什么除以 √d_k"等），无 S1/S2/S3 编号标记；小白读者无法定位"S3"指哪一节。修法：将"S3"改为章节标题或锚点链接（如"见<a href="#why-sqrt-dk">为什么除以 √d_k</a>"），或去掉内部编号。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：进入修复（重要问题涉及来源一致性，需修复后复验；轻微不阻断）

## 段 A 盲读小结

扮演完全小白读者按页面顺序阅读。主线理解顺畅：从 RNN"逐步传递"的两个局限（路径 $O(n)$、顺序操作 $O(n)$）引出注意力动机；用数据库类比建立"查询-键-值"直觉并标注失效边界；缩放点积公式拆成四步（$QK^\top$ → $\div\sqrt{d_k}$ → softmax → $\cdot V$）每步给形状与含义；2×2 手算例子把每步数字落到可复算程度；$\sqrt{d_k}$ 的方差推导完整展开（$\mathrm{Var}(q\cdot k)=d_k$ → $\mathrm{std}=\sqrt{d_k}$ → 除后归一）；多头公式、拼接机制、参数量等价（$h\cdot d_k=d_{model}$）清晰；因果遮罩 3×3 例子完整；复杂度瓶颈定位在 $QK^\top$ 产生 $n\times n$ 矩阵；Flash vs Linear 区分清楚（改实现 vs 改公式）。学习目标四条均由正文章节完整回答。卡点为上述两条轻微项，不阻断主线。

## 段 B 对照来源小结

逐条核对 Vaswani 2017 §3.2.1、§3.2.2、§3.2.3、§4.1 Table 1、Table 3：

1. 定义与机制：缩放点积公式 $\text{softmax}(QK^\top/\sqrt{d_k})V$（§3.2.1 Eq.(1)）一致；多头公式 $\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$（§3.2.2 Eq.(2)）一致；因果遮罩"setting to $-\infty$ all values corresponding to illegal connections"（§3.2.3）一致；§3.2.2"不同表示子空间"原文引用逐字一致。
2. 公式与推导：脚注 4 方差推导"$q\cdot k$ has mean 0 and variance $d_k$"一致，页面展开为完整推导（$\mathrm{Var}(q_ik_i)=1$ → 求和 $\mathrm{Var}=d_k$）复算一致；2×2 例子 $QK^\top$、$\div\sqrt{2}$、softmax、$AV$ 逐步复算一致；3×3 遮罩例子 softmax 逐行复算一致（$[1,0,0]$、$[0.401,0.599,0]$、$[0.258,0.316,0.426]$）；softmax 雅可比 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ 复算一致；$\$E[\max]\approx 8\sqrt{2\ln 256}\approx 27$ 复算一致；$e^{16}\approx 8.89\times 10^6$ 复算一致。
3. 可运行代码：页面无可运行代码块（仅有 KaTeX 公式与 ASCII 图示），不适用。
4. 事实与推断：§3.2.1 加性注意力对比"additive attention outperforms dot product attention without scaling for larger $d_k$"一致；base 模型超参 $d_{model}=512$、$h=8$、$d_k=d_v=64$、$W^O\in\mathbb{R}^{512\times 512}$（§3.2.2 与 Table 3）一致；§4.1 Table 1 中 RNN 行（$O(nd^2)$/$O(n)$/$O(n)$）与 Self-Attention 行（$O(n^2d)$/$O(1)$/$O(1)$）一致。**CNN 行发现来源不一致**（见重要问题）：页面添加"$O(n/k)$"且删去 restricted self-attention 行。
5. 前置知识引用：页面标注"概念页 矩阵乘法（待生成）""概念页 向量点积（待生成）""概念页 softmax（待生成）""概念页 方差（待生成）"——均为占位提示，符合规范。
6. 教学简化：2×2 对角 $Q,K$ 简化、3×3 遮罩跳过 $QK^\top$ 计算、方差推导假设独立同分布均值 0 方差 1、softmax 数值稳定实现放折叠块——均标注简化理由与可/不可推出边界，未发现简化导致核心结论失真。$d_k=64$ 饱和数字标注"非论文一手数据，参考 jethroodeyemi.github.io 2026 的实测验证"。
7. 页面功能：KaTeX 公式渲染正常；details 折叠交互正常；侧边目录锚点（why-attention / scaled-dot-product-formula / why-sqrt-dk / multi-head-attention / complexity-and-boundaries / sources-and-teaching-notes）有效。

发现来源不一致 1 项（CNN 路径长度），已列重要问题。

# 低秩分解独立审查（第二次）

- 审查者：独立上下文（AI 模拟）
- 页面版本：eac84c495e17f069f861576250bbf34284ab4630（index.html 工作树哈希）
- 时间：2026-08-09

## 审查范围

- 输入：`wiki/low-rank-projection/index.html`、`wiki/low-rank-projection/overview.html`、`guides/concept/check.md`、`guides/concept/content-examples.md`
- 外部来源：WebSearch "SVD low rank approximation"（Eckart-Young-Mirsky 定理与误差公式）+ "LoRA Hu 2021 arxiv 2106.09685"（LoRA 假设、初始化、r=1 或 2、0.01%）+ "DeepSeek-V2 MLA arxiv 2405.04434"（MLA 公式 Eq.9-11、配置 d_c=512, n_h=128, d_h=128、KV cache 减少 93.3%）
- 未读取 `research/` 目录、未读取仓库中其他概念页（仅验证链接路径存在）

## 段 A 盲读小结

按页面顺序阅读，五个学习目标在正文中均得到完整回答：

1. 大矩阵为何可用两个小矩阵乘积近似、何时误差小 → "矩阵的秩"+"SVD 与最优低秩近似"两章回答
2. SVD 截断如何给出最优低秩近似、误差由什么决定 → "SVD 与最优低秩近似"章回答（Eckart-Young-Mirsky 定理 + 误差公式）
3. LoRA 如何把微调参数减到极小比例、省的是哪些参数 → "LoRA"章回答（省的是 ΔW 的参数，W₀ 冻结）
4. MLA 如何减少 KV cache、缓存的具体是什么 → "MLA"章回答（缓存潜向量 c_t^{KV}，维度 d_c）
5. 适用边界 → "适用边界"章回答（奇异值平坦失效、MLA 有损、K3 full-rank 教训）

术语首现解释充分（秩、奇异值、Frobenius 范数、谱范数、潜向量、下/上投影均有定义）。推导跳步无发现：W=AB⇒rank(W)≤r、误差公式、MLA 低秩瓶颈三处推导均给出了关键变换理由。手算例子（diag(3,1,0.5)）代入过程完整、可复算。折叠块（直觉说明、手算例子）前后均能接回主线，正文不依赖折叠块即可成立。前置链接 ../../wiki/standard-attention/index.html、../../wiki/mla/index.html、../../wiki/kda/index.html 路径均存在。

## 段 B 对照来源小结

1. 定义与机制：[C1] LoRA 低内在秩假设与原文一致；[C2] Eckart-Young-Mirsky 定理对 Frobenius 与谱范数成立、Mirsky 推广到酉不变范数与 Wikipedia 一致；[C3] MLA 低秩 KV 联合压缩与 arXiv:2405.04434v2 §2.1.2 一致；[C4] 失效条件由 Eckart-Young 误差公式直接推出。
2. 公式与推导：[F1] h=W₀x+BAx 与 arXiv:2106.09685v2 §4 Eq.(3) 一致；初始化 A 高斯、B 零使 ΔW=0 与原文一致。[F2] SVD 定义与截断 SVD 形式一致。[F3] 误差公式 ‖W−W_k‖_F=√(Σ_{i>k}σ_i²)、‖W−W_k‖_2=σ_{k+1} 与定理一致。[F4] c_t^{KV}=W^{DKV}h_t（Eq.9）、k_t^C=W^{UK}c_t^{KV}（Eq.10）、v_t^C=W^{UV}c_t^{KV}（Eq.11）及维度标注与原文一致。[F5] 参数计数 mn→r(m+n) 由维度直接推出。[F6] KV cache 计数 MHA=2n_h d_h、MLA=d_c 与原文 Table 1 一致（略去解耦 RoPE 项 d_h^R，已说明）。
3. 可运行代码：页面无可运行代码块，只有 ASCII 围道图与公式，不适用。
4. 事实与推断：[N1] 0.01% 与 10000× 减少对应论文 §1/§2。[N2] r 低至 1 或 2、满秩 d 高达 12288 对应论文 §1 原文。[N3] d_c=512, n_h=128, d_h=128 对应论文 §3.1.2 与 config.json。[N4] 93.3% 对应摘要；Table 7 数字 110.6K→15.6K、860.2K→34.6K 已标注为不同基线整体对比，并诚实澄清与同配置单层 1/64（98.4%）不可直接对照——教学诚实标注到位。
5. 前置知识引用：standard-attention、mla、kda 三个目录均存在，链接层级正确。
6. 教学简化：5 项简化（SVD 算法未展开、α/r 缩放未展开、MLA 略去解耦 RoPE 与矩阵吸收、对角矩阵使 SVD 平凡、未计偏置与优化器状态）均有说明，不影响核心结论。
7. 页面功能：KaTeX 渲染（delimiters 配置 $$...$$ 与 $...$、throwOnError:false）、details 折叠、自动生成目录锚点（h2 均有显式 id）结构正确。

数字复算：d=12288, r=2 → 2dr=49152、d²=150,994,944、比例 0.033% ✓；diag(3,1,0.5) 秩-1 Frobenius 误差 √1.25≈1.118、谱误差 1、秩-2 Frobenius 误差 0.5、谱误差 0.5 ✓；平坦分布 (3,2.9,2.8) 秩-1 误差 √(2.9²+2.8²)=√16.25≈4.03≈4.04 ✓；MLA 512/32768=1/64 ✓；Table 7 残留比例 15.6/110.6≈14.1%、1/64≈1.56% ✓；末尾检查 W=diag(5,2,0.1) 秩-1 Frobenius 误差 √4.01≈2.002、谱误差 2 ✓；d=4096, r=8 比例 2×4096×8/4096²=16/4096≈0.39% ✓。

## 问题

- [轻微·技术] MLA 章节对比表第四行"等价权重秩"：MHA 列写"满秩（$d_h n_h$）"在 DeepSeek-V2 配置下不严谨。MHA 的 $W^K \in \mathbb{R}^{d_h n_h \times d}$，秩 $\le \min(d_h n_h, d)$；DeepSeek-V2 配置 $d_h n_h = 128 \times 128 = 16384 > d = 5120$，实际秩上限为 $d = 5120$，不等于 $d_h n_h = 16384$。不影响核心结论（MLA 秩 $\le d_c = 512 \ll 5120$）。修法：将"满秩（$d_h n_h$）"改为"满秩（$\le \min(d_h n_h, d)$）"或仅写"满秩"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：进入修复

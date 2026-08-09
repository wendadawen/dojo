# LatentMoE 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 来源对照）
- 页面版本：wiki/latent-moe/index.html（1237 行）、overview.html（76 行）
- 时间：2026-08-09
- 来源：[L] arXiv:2601.18089v1（LatentMoE 论文，Elango et al., 2026）；[K3] k3-report.txt §2.3 与 Table 1；[R] Sebastian Raschka "Latent MoE"

## 段 A 盲读

按页面顺序阅读，记录小白读者理解主线上的卡点。

1. 开篇"256 路由专家、每 token 激活 8 个，想扩到 896 个专家、每 token 激活 16 个"——读者需已了解 top-K MoE 基本概念。页面在"开始之前"已声明前置知识要求（moe-serving 第 2、3 章 + DeepSeekMoE），前置链接有效。开篇可继续。

2. S1 引入"all-to-all 通信"未在正文解释，依赖前置页 moe-serving §3。页面已声明此前置，可接受。

3. S2 公式 F2 将 S1 的 $S_k(x)$、$g_i$ 改记为 $T_k(x)$、$p_i$，页面说明"以与 K3 报告 Eq. 11 一致"。改记号有解释，不构成卡点。

4. S3 引入"非线性容量" $U_\text{eff}=K\cdot m$ 时，页面直接给出定义"论文把非线性容量定义为 top-$k$ × 专家中间维度"，随后用于解释设计原则 III 与 V。定义在首次出现处给出，可理解。

5. S3 reinvestment 示例从 S1 的 $k=8$ 切换到 $N=8, k=2$ 作为起点。页面标注"教学示例（贯穿推进）。在 $d=4096, \ell=1024$ 的基础上，取初始 $N=8, k=2$"，明确是新的构造示例，不构成跳步。

6. S4 提及"K2-1T"未说明是什么模型。"万亿参数规模"给出规模上下文，但"K2-1T"作为模型名首现未解释——小白读者不知道 K2 指 Kimi K2。轻微卡点。

7. S5 与 Stable LatentMoE 的关系清晰：主结构一致 + 三件稳定化；K3 Eq. 11 与 F2 的差异（RMSNorm 位置）明确指出。数据流图与对比表帮助理解。

8. 逐题核对学习目标：
   - "为什么路由开销随路由倍数线性增长？LatentMoE 用什么结构变化缓解？" → S1 + S2 完整回答 ✓
   - "共享分支与路由分支各自走什么宽度、经过哪些投影？写出完整层公式。" → S2 公式 F2 + 符号表 + 数据流图 ✓
   - "压缩后节省的开销以什么比例缩小？reinvest 到哪里？" → S3 完整回答 ✓
   - "压缩比能不能无限放大？受什么约束？" → S4 三条约束完整回答 ✓
   - "LatentMoE 与 Stable LatentMoE 是什么关系？" → S5 完整回答 ✓

   全部学习目标由正文章节完整回答。

## 段 B 对照来源

### 1. 定义与机制

- C1（路由分支两端加 $W_\downarrow$/$W_\uparrow$，路由专家在 $\ell$ 维计算，共享专家与 router 仍在 $d$ 维）：[L] Figure 1(b) 与 §2 确认。论文原文"tokens are projected from the model hidden dimension d into a smaller latent dimension ℓ for expert routing and computation"；"all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension d"。一致 ✓
- C2（标准 MoE 全宽路由开销随 top-$k$ 与 $d$ 线性增长）：[L] §1、§2 设计原则 I、II 确认。"Memory bandwidth cost scales with d and m"；"communication cost scales with K and d"。一致 ✓
- C3（路由部分开销按 $d/\ell$ 比例缩小）：[L] Figure 1 caption 原文"reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"。一致 ✓
- C4（Reinvestment 按 $\alpha = d/\ell$ 放大 $N$ 与 $K$）：[L] Figure 1 caption 原文"increase the total number of experts and the top-k active experts per token by the same factor d/ℓ"。一致 ✓
- C5（压缩下限与投影成本）：[L] 设计原则 IV（特征秩下限 $r_\text{eff}$）+ §3 消融确认。一致 ✓
- C6（Stable LatentMoE = LatentMoE + 三件稳定化）：[K3] §2.3 确认。"Stable LatentMoE addresses these two failure modes with three components: an RMSNorm before the up-projection and SiTU-GLU to suppress activation explosion, and Quantile Balancing for load balancing"。一致 ✓
- C7（router 仍在 $d$ 维工作）：[L] §3 确认。"The routing weights p' = Softmax(W'_r · x) are computed from the original token x ∈ ℝ^d"。一致 ✓

### 2. 公式与推导

- F1（标准 MoE $y = \sum g_i E_i$）：引自前置页，与 [L] §1 标准定义一致 ✓
- F2（LatentMoE 层公式）：$u = \sum p_i E_i^\text{routed}(W_\downarrow x)$，$y = \sum E_j^\text{shared}(x) + W_\uparrow u$。与 [K3] Eq. 11 去掉 RMSNorm 后一致。[K3] Eq. 11 原文：$u = \sum_{i \in T_k(x)} p_i E_i^\text{routed}(W_\downarrow x)$，$y = \sum_{j=1}^{N_s} E_j^\text{shared}(x) + W^\uparrow \text{RMSNorm}(u)$。一致 ✓
- F3（路由部分开销缩小比例 $\approx d/\ell$）：[L] Figure 1 caption ✓
- F4（reinvestment：$N \to \alpha N$、$k \to \alpha k$）：[L] Figure 1 caption ✓
- 组合数 $\binom{32}{8} = 10{,}518{,}300$：代码复算一致 ✓
- "约 37 万倍"：$10{,}518{,}300 / 28 = 375{,}653.57$，约 37.6 万倍，页面取"约 37 万倍"为下取整，偏差可接受 ✓

### 3. 可运行代码

代码块（S3 details）已实际执行。输出与页面"预期输出"逐行一致：

```
top-8: 标准 MoE=32768, LatentMoE=8192, 缩小 4.0x
top-16: 标准 MoE=65536, LatentMoE=16384, 缩小 4.0x
C(8,2)=28 → C(32,8)=10518300，组合数增长倍数: 375653.57
投影参数量: 2*4096*1024 = 8388608
```

一致 ✓

### 4. 事实与推断

- N1（K3 配置 $d=7168, \ell=3584, N=896, k=16, N_s=2$，稀疏度 56）：[K3] Table 1 逐项确认。Hidden Dimension=7,168；Latent MoE Dimension=3584 (0.5×)；Routed Experts=896；Experts Active per Token=16；Shared Experts=2。896/16=56 ✓
- N2（Nemotron-3 Super：$4096 \to 1024 \to 4096$，120B/12B）：[R] 确认"Super has 120B total and 12B active parameters""For Super, this was 4096 -> 1024 -> 4096"。但页面来源标注列"[L] §4"——[L] 论文仅提及 Nemotron-3 采用了 LatentMoE 架构（"adopted by the flagship Nemotron-3 Super and Ultra models"），未给出具体配置数字。具体数字来源应为 [R]。
- N3（Nemotron-3 Ultra：$8192 \to 2048 \to 8192$，550B/55B，512 路由专家、top-22、1 共享、中间维度 5120/10240）：[R] 确认 8192→2048→8192、550B/55B。kiadev.net 确认"512 experts, top-22"。"1 共享专家"与"中间维度 5120/10240"未能从搜索到的来源独立验证，页面标注来自 [R]。
- N4（16B 消融：压缩比到 4 质量保持）：[L] §3 确认。"model quality is preserved for compression ratios α ≤ 4. Consequently, we adopt α = 4 for all subsequent experiments" ✓
- N5（投影额外计算约 9%，K2-1T 规模）：[L] 确认。"native Kimi-K2-1T remains close, within up to ~9% of Kimi-K2-1T-LatentMoE, indicating that projection overhead is small" ✓
- N6（iso-accuracy 约 3.5x 加速，350B 参数）：[L] 确认 350B（"≈ 0.35T ≈ 350B parameters"）与减速比范围"1.24×–3.46× slower"。页面取上界 3.46≈3.5，但页面已声明"本页正文未直接引用 N6，仅在此登记""N6 为投影值非实测"。登记可接受 ✓

### 5. 前置知识引用

- ../moe-serving/index.html — 目录存在 ✓
- ../deepseek-moe/index.html — 目录存在 ✓
- ../stable-latent-moe/index.html — 目录存在 ✓
- overview.html 互相链接 — 存在且双向链接 ✓

### 6. 教学简化

- 论文有两个变体 $\ell\text{-MoE}_\text{eff}$（只放大 $N$）与 $\ell\text{-MoE}_\text{acc}$（同时放大 $N$ 与 $K$，推荐）。页面只呈现 acc 变体作为 reinvestment 策略，未提及 eff 变体。教学简化合理（acc 是论文推荐变体），但"教学简化及其限制"小节未登记此简化。
- dispatch 量用"个数"度量、组合数假设无约束选取、投影参数量粗略估计——均已在"教学简化及其限制"中说明 ✓

### 7. 页面功能

- KaTeX 公式渲染：delimiters 配置正确 ✓
- 折叠块 details：S2 MLA 类比、S3 代码、S4 设计原则与 Nemotron 配置——收起后正文主线不依赖其内容 ✓
- 目录锚点：h2/h3 带 id，scroll-margin-top 避开导航 ✓

## 问题

- [轻微·来源] 来源与教学说明 > 外部数字 > N2：Nemotron-3 Super 的具体配置（$4096 \to 1024 \to 4096$、120B/12B）来源标注为"[L] §4；[R]"，但 [L] arXiv:2601.18089 仅在摘要与 §1 提及 Nemotron-3 采用了 LatentMoE 架构，未给出 Super 的具体配置数字。具体数字应只标注 [R]。修法：将 N2 来源改为仅 [R]，或注明"[L] §4 仅确认采用关系，具体配置来自 [R]" ｜ 修复： ｜ 复验：
- [轻微·盲读] S4 约束二："论文在万亿参数规模（K2-1T）的投影分析中估计"——"K2-1T"作为模型名首次出现未解释，小白读者不知道 K2 指 Kimi K2。修法：首次出现时加括注"（Kimi K2，1 万亿参数）" ｜ 修复： ｜ 复验：
- [轻微·技术] S4 约束一："压缩比到 4 时模型质量仍可保持（数字 N4）；再压缩需配合 reinvestment 才不损失"——论文 N4 消融使用的是 $\ell\text{-MoE}_\text{eff}$ 变体（放大 $N$、$K$ 不变），"质量保持"是在有专家数放大的条件下成立的。页面此处未说明该结论已含专家数放大，可能被误读为"纯压缩到 4x 也安全"。虽然 S4 约束三与 S3 已说明 reinvestment 是必要条件，但约束一本身应补"（在专家数同步放大前提下）"。修法：约束一"压缩比到 4 时模型质量仍可保持"后加"（此时专家数已按 $\alpha$ 放大）" ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：进入修复（3 条轻微均为来源标注精确性与首现术语解释，改动量小，不影响核心结论与主线理解）

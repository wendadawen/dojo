# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-6.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**3** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**21** 条
- 有理由不改：**0** 条

## 修复与判定说明

Cleared the accumulated 轻微 backlog on wiki/kda. Reviewed all six review-*.md records; found 26 raw minor findings, 24 distinct after de-duplication. Only 3 were still live: (1) R6 — [C1] and [C10] had no body <sup> reference, added [C1] at the KDA definition (index.html L108) and [C10] at the §2 transpose sentence (L250); (2) R6 — orchestration meta lead-ins in §2, deleted the standalone "先说清每个符号，再分三步读。" paragraph (L241) and rewrote "公式分三步读，顺序由矩阵乘法的结合律决定：" to a direct "Eq.1 里三个作用的先后顺序，由矩阵乘法的结合律决定：" while preserving the associativity claim; (3) R2/R3 FP32 inference — already gone from index.html but the identical twice-reported claim survived in the sibling overview.html (L50 "若用 FP32 训练，negative-softplus 也能用但成本更高"), removed it to keep index/overview consistent. The other 21 distinct minor issues were verified already absent from the current page (fixed in prior rounds) and were not re-edited: S5/§5.1 example cross-reference, ShortConv attribution, A_h early reference, "第 2-4 行" wording, W_α^{↑↓} low-rank note, V[t] shape d_k→d_v, e^{160}≈3×10^69, α₁=(1,1,1,1) boundary note, h3 "（N）" naming, §3 meta sentence, §3.4 "权衡值得", description "源码核对", [C9] g_min notation, §2 $V'$ mis-sourced to delta-rule, six chapter-end transitions, §3.4 "最直接/很清楚", 本页/本文 self-reference, "前身" wording, "快 C 倍", [C2] <code> Unicode math, and "构造示例。" trailing spaces. No item was left unfixed; no surviving finding conflicted with the guides.

# -*- coding: utf-8 -*-
"""第 2 轮审查问题的修复脚本（causal-mask/index.html）。

对每个 old 串断言命中 1 次，再做精确替换。
"""
import sys

PAGE = "/Users/wendadawen/code/dojo/wiki/causal-mask/index.html"

with open(PAGE, encoding="utf-8") as f:
    html = f.read()

edits = []

# ── 问题 1（重要）：C6 机制引用改到 NoPE 正文 §2/§8 与附录 C.1，摘要句仅支持"NoPE 有效" ──
edits.append((
    '<p><sup>[C6]</sup> 因果掩码打破排列对称性、是 NoPE 结构前提：§3.2.3 因果掩码可见集合性质（C1）的推论；NoPE 论文（Kazemnejad et al., NeurIPS 2023, arXiv:2305.19466）摘要——"explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences"。NoPE 完整论证见 <a href="../../wiki/nope/index.html">NoPE</a>。</p>',
    '<p><sup>[C6]</sup> 因果掩码打破排列对称性、是 NoPE 结构前提：§3.2.3 因果掩码可见集合性质（C1）的推论。"NoPE 无需显式位置编码也能有效并泛化到更长序列"由 NoPE 论文（Kazemnejad et al., NeurIPS 2023, arXiv:2305.19466）摘要末句支持——"explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences"。而"decoder-only Transformer 因因果掩码而非排列不变、因而无需显式位置信息"这一机制陈述见该论文 §2 Background——"decoder-only Transformers with causal attention mask are not permutation invariant and can model sequences even without explicit position information (Tsai et al., 2019)"；§8 Related Work——"Decoder-only Transformers, due to their causal attention mask, are not order-agnostic and can operate without explicit positional information"；附录 C.1——"relies on the causal attention mask in the decoder-only Transformer and the softmax function to recover absolute positions"。NoPE 完整论证见 <a href="../../wiki/nope/index.html">NoPE</a>。</p>',
))

# ── 问题 2（轻微·可读性）：第 1 章解释 §3.1 的 offset 半句，把"可见 ≤ i"与"依赖 < i"对齐 ──
edits.append((
    '论文 §3.1 把这一约束表述为"the predictions for position $i$ can depend only on the known outputs at positions less than $i$"。<sup>[C1, C4]</sup></p>',
    '论文 §3.1 把这一约束表述为"the predictions for position $i$ can depend only on the known outputs at positions less than $i$"。<sup>[C1, C4]</sup></p>\n\n<p>引文里还有半句需要解释——"the output embeddings are offset by one position"。decoder 的输入不是原目标序列，而是把它右移一位（开头补一个起始符号）得到的序列：输入位置 $i$ 承载的是目标序列第 $i-1$ 个 token。因此"输入位置 $i$ 可 attend 到 $1..i$"与"预测位置 $i$ 只依赖小于 $i$ 的已知输出"是同一件事——可见的输入位置 $1..i$ 承载目标序列第 $0..i-1$ 个 token（其中第 $0$ 个是起始符号），真正的已生成输出是第 $1..i-1$ 个，正好是"小于 $i$"的那些。第 4 章"一次喂入整条目标序列"喂入的正是这条右移一位的输入序列，掩码保证每个输入位置只看到自己的前缀。</p>',
))

# ── 问题 3（轻微·技术）：第 2 章末段标注为工程惯例示例并补成立条件 ──
edits.append((
    '<p>实现上，精确 $-\\infty$ 是数学极限。工程中可用足够大的负数（如 $-10^5$）或框架提供的 <code>-inf</code> 达到同样效果——在浮点精度下被掩位置的 $e^{s_j}$ 也恰好下溢为 $0$，权重为 $0$。本文手算用 $-\\infty$ 表达数学含义。这一节是辅助说明，不影响机制结论。</p>',
    '<p>实现上，精确 $-\\infty$ 是数学极限。以下是<strong>工程惯例示例</strong>（非论文原文）：工程中可用足够大的负数（如 $-10^5$）或框架提供的 <code>-inf</code> 达到同样效果——在浮点精度下被掩位置的 $e^{s_j}$ 恰好下溢为 $0$，权重为 $0$。成立条件是未掩分数在 $\\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值；否则把 $-10^5$ 加到一个同量级的未掩分数上并不构成足够大的负数，掩码会失效。本文手算用 $-\\infty$ 表达数学含义。这一节是辅助说明，不影响机制结论。</p>',
))

# ── 问题 3 配套：本章问题答案同步补成立条件 ──
edits.append((
    '<p>精确 $-\\infty$ 是数学极限，工程中可用足够大的负数（如 $-10^5$）或框架提供的 <code>-inf</code>：浮点精度下被掩位置的 $e^{s_j}$ 下溢为 $0$、权重为 $0$，效果与 $-\\infty$ 相同。</p>',
    '<p>精确 $-\\infty$ 是数学极限，工程中可用足够大的负数（如 $-10^5$）或框架提供的 <code>-inf</code>：浮点精度下被掩位置的 $e^{s_j}$ 下溢为 $0$、权重为 $0$，效果与 $-\\infty$ 相同。这是工程惯例示例，成立条件是未掩分数在 $\\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值。</p>',
))

# ── 问题 3 配套：把该工程惯例示例收入"辅助解释与类比边界" ──
edits.append((
    '不展开群论含义。</p>',
    '不展开群论含义。第 2 章"工程中可用足够大的负数（如 $-10^5$）或 <code>-inf</code> 代替 $-\\infty$"是工程惯例示例（非论文原文），成立条件是未掩分数在 $\\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值。</p>',
))

for i, (old, new) in enumerate(edits, 1):
    n = html.count(old)
    if n != 1:
        print(f"ASSERT FAIL edit#{i}: 命中 {n} 次，期望 1 次")
        print("  old 前 80 字：", old[:80])
        sys.exit(1)
    html = html.replace(old, new)
    print(f"edit#{i} ok")

with open(PAGE, "w", encoding="utf-8") as f:
    f.write(html)
print("written", PAGE)

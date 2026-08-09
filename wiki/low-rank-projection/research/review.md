# 低秩分解 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型 GLM-5.2）
- 页面版本：index.html `8e8a5b1e6238715374fbff6799f4038240cb7a09` / overview.html `6597dd7400d3949c7d75fc159eb373d553a95ea6`
- 时间：2026-08-09

## 来源对照范围

授权对照来源为 WebSearch "SVD low rank approximation" 与 LoRA 论文 arXiv:2106.09685v2。
- LoRA 全部引用 [C1][F1][N1][N2] 已对照 arXiv:2106.09685v2 原文逐条核对，全部一致。
- SVD/Eckart-Young 引用 [C2][F2][F3] 已对照 Wikipedia "Low-rank approximation" / "Singular value decomposition" 及多个二手来源（CMU 讲义、inferensys、grokipedia、mlpod），定理陈述与误差公式一致。
- MLA/DeepSeek-V2 引用 [C3][F4][F6][N3][N4] 来源 arXiv:2405.04434 不在本次授权对照范围内，仅做内部一致性核查；其事实正确性未由本次独立审查背书，相关盲读问题仍可独立成立。

## 问题

- [重要·盲读] index.html §5 MLA（line 881-885）：先算"单层简化"比例 $d_c/(2 n_h d_h) = 512/32768 = 1/64$（即 98.4% 减少），紧接着引用论文 Table 7 数据"110.6K→15.6K（85.9%）/ 860.2K→34.6K（96.0%）/ 整体 93.3%"，但 1/64（≈1.56% 残留）与 15.6/110.6（≈14.1% 残留）相差近 9 倍，与 34.6/860.2（≈4.0% 残留）也相差 2.5 倍。页面只在 [F6] 备注"略去较小的解耦 RoPE 项 $d_h^R$"，而 $d_h^R$ 远不足以填补此差距。实际原因是 Table 7 是 DeepSeek-V2 MoE（MLA，$n_h{=}128,d_h{=}128,d_c{=}512$）与不同 MHA 基线（DeepSeek 67B 等，配置不同）的跨架构对比，并非同架构单层切换 MHA→MLA 的比例。页面把两个层级不同的比例并列呈现且未说明关系，盲读时会得出"理论 1/64、实测只有 1/7"的困惑。修法：在引用 Table 7 数值前加一句明确说明——"上面的 1/64 是 DeepSeek-V2 同配置下单层、忽略 RoPE 项的理论比例；下面 Table 7 的 110.6K→15.6K 等是 DeepSeek-V2 MoE 与不同 MHA 基线（含 DeepSeek 67B）的整体对比，$n_h, d_h, d_c$、层数与模型规模均不同，因此减少比例不能直接与 1/64 对照"。 ｜ 修复：已在 1/64 理论比例与 Table 7 实测数据之间加层级说明，明确 1/64 为同配置单层忽略 RoPE 项的理论比例，Table 7 数据为跨架构（不同 MHA 基线、不同规模）整体对比，并点出残留比例（1.56% vs 14.1%）不可直接对照 ｜ 复验：

- [轻微·盲读] index.html §3 SVD（line 750）："U 是 $m \times m$ 正交矩阵、V 是 $n \times n$ 正交矩阵"中"正交矩阵"首次出现未解释。小白读者不知道正交矩阵是什么、为何 SVD 要 U、V 正交。该性质在 Eckart-Young 证明（"正交矩阵不改变 Frobenius 范数"）和误差公式推导中被隐含使用。修法：首次出现处加一句括注"正交矩阵指列向量两两垂直且长度都为 1 的方阵，等价于 $U^\top U = I$"，或链接到线性代数前置概念页。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §2 矩阵的秩（line 704、706）：秩的定义"列（或行）张成空间的维度"以及"每一列都是 $A$ 的列向量的线性组合"中，"张成空间"和"线性组合"两个线性代数基础术语首次出现未单独解释。页面紧接给出"独立方向的个数"作为直觉，能缓解但未消除术语卡点。修法：在"张成空间"首次出现处加括注"（即这些列向量经线性组合能得到的全部向量的集合）"或在前置知识位置链接到线性代数概念页。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §3 vs §4（line 760、802）：§3 用 $W = AB$（A 为 $m \times r$ 高矩阵、B 为 $r \times n$ 宽矩阵），§4 LoRA 用 $\Delta W = BA$（B 为 $d \times r$ 高矩阵、A 为 $r \times d$ 宽矩阵）。两节中"高矩阵"的命名相反（§3 叫 A、§4 叫 B），对照阅读时容易混。§4 命名遵循 LoRA 原论文（$\Delta W = BA$），无错误，但页面未提示命名约定的切换。修法：在 §4 首次定义 $\Delta W = BA$ 处加一句"注意这里 B、A 的顺序遵循 LoRA 原论文，与 §3 中 $W=AB$ 的 A、B 顺序相反；两节都是 $m \times r$ 与 $r \times n$ 矩阵的乘积，仅命名不同"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §4 LoRA（line 822-826）：r=2 例子计算 $49152/150994944 \approx 0.033\%$，分母是单个权重矩阵 $d^2$；紧接着引用论文"可训练参数可低至原模型的 $0.01\%$"，分母是整个 GPT-3 175B 模型（约 1750 亿参数）且仅适配 $W_q, W_v$ 部分矩阵。两个比例分母不同，盲读时会困惑"自己算的 0.033% 为何大于论文报告的 0.01%"。修法：在引用 0.01% 前加一句"注意 0.01% 的分母是整个 GPT-3 175B 的参数总量（约 1750 亿），且论文实际只适配 $W_q, W_v$ 而非全部权重；上面 0.033% 是单个权重矩阵内的比例，分母不同所以数值更大"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §6 适用边界（line 928）："一个来自 K3 的真实教训"——K3（Kimi K3）首次出现未交代是什么模型、与 Kimi Linear 的关系，仅靠链接到 KDA 概念页。盲读时若不点开链接，"K3 为什么要把 low-rank 改成 full-rank"的语境缺失。修法：在"K3"首次出现处加括注"（Kimi K3，月之暗面 Kimi 系列模型之一）"或一句话背景。 ｜ 修复： ｜ 复验：

## 来源对照逐项核查（段 B 记录）

### LoRA 论文 arXiv:2106.09685v2

- [C1] §1 原文："the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'"。页面 line 946 引用一致 ✓
- [F1] §4.1 Eq.(3) 原文："h = W₀x + ΔWx = W₀x + BAx"。页面 line 806、952 一致 ✓
- [F1] §4.1 原文："We use a random Gaussian initialization for A and zero for B, so ΔW = BA is zero at the beginning of training. We then scale ΔWx by α/r"。页面 line 828 引用一致；页面 line 974 已说明 α/r 缩放属工程简化，未失真 ✓
- [F1] Figure 1 caption 原文："Our reparametrization. We only train A and B."。页面 line 952 一致 ✓
- [N1] Abstract 原文："reduce the number of trainable parameters by 10,000 times"；§2 原文："the number of trainable parameters |Θ| can be as small as 0.01% of |Φ₀|"。页面 line 826、960 引用一致 ✓
- [N2] §1 原文："a very low rank (i.e., r in Figure 1 can be one or two) suffices even when the full rank (i.e., d) is as high as 12,288"。页面 line 727、822、961 引用一致 ✓
- 论文 §4.2 与 §5.1 说明实际适配 $W_q, W_v$（r=4 时 10000× 减少），页面未在 LoRA 章节正文展开此矩阵选择细节，属合理简化（与学习目标"省的是哪些参数"无冲突，因为页面明确"只训练 A、B"）。

### SVD / Eckart-Young（Wikipedia + 多来源）

- [F2] SVD $W = U\Sigma V^\top$ 与截断 SVD $W_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$：Wikipedia "Singular value decomposition" 一致 ✓
- [F3] $\|W - W_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots}$、$\|W - W_k\|_2 = \sigma_{k+1}$：Wikipedia "Low-rank approximation"（Frobenius 与 spectral 两证明小节）、CMU 讲义 Theorem 3.1、inferensys、grokipedia、mlpod 多源一致 ✓
- [C2] Eckart-Young-Mirsky 定理：Wikipedia "Singular value decomposition" 用"Eckart–Young theorem"（两姓氏），"Low-rank approximation" 用"Eckart–Young–Mirsky theorem"（三姓氏，含 Mirsky 1960 推广到酉不变范数）。页面 line 762 用三姓氏形式，line 947 同时引用两个 Wikipedia 页面，归属正确 ✓
- 手算例子 $W = \mathrm{diag}(3,1,0.5)$：σ₁=3, σ₂=1, σ₃=0.5；秩-1 Frobenius 误差 $\sqrt{1+0.25}=\sqrt{1.25}\approx 1.118$、谱误差 1；秩-2 Frobenius 误差 0.5、谱误差 0.5——复算一致 ✓
- 检查问题 $W=\mathrm{diag}(5,2,0.1)$：秩-1 Frobenius 误差 $\sqrt{4+0.01}\approx 2.002$、谱误差 2——页面 line 967 复算一致 ✓
- 检查问题 $d=4096, r=8$：$2 \times 4096 \times 8 / 4096^2 = 65536/16777216 \approx 0.39\%$——页面 line 967 复算一致 ✓
- §3 line 784 平坦奇异值例子 $\sqrt{2.9^2+2.8^2} = \sqrt{16.25} \approx 4.031$，页面写"$\approx 4.04$"，末位四舍五入略偏（更准确为 4.03），属轻微数值误差，不影响结论。

### MLA/DeepSeek-V2（arXiv:2405.04434，本次不在授权对照范围）

- [C3][F4][F6][N3][N4] 无法对照授权来源独立验证；页面引用的公式形式、维度符号、配置数字、Table 7 数值内部一致，但事实正确性未经本次审查背书。
- 盲读层面问题（1/64 vs Table 7 比例不匹配）独立于来源对照成立。

## 学习目标闭环核查

- 目标 1（大矩阵为什么可以用两小矩阵近似、误差何时小）：§2 秩 + §3 SVD 误差公式完整回答 ✓
- 目标 2（SVD 截断为何最优、误差由什么决定）：§3 Eckart-Young 定理 + 误差公式完整回答 ✓
- 目标 3（LoRA 如何减参、省的是哪些参数）：§4 完整回答（冻结 $W_0$、只训练 $A, B$，参数 $d^2 \to 2dr$）✓
- 目标 4（MLA 如何减 KV cache、缓存的具体是什么）：§5 完整回答（缓存潜向量 $c_t^{KV}$ 而非 K/V，维度 $d_c$）✓
- 目标 5（低秩近似的失效条件）：§6 完整回答（奇异值平坦时失效 + K3 实例 + MLA 有损）✓

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：进入修复（无阻断；1 项重要为 MLA 比例层级未说明，加一句话即可修复，不需改变研究范围或教学大纲；5 项轻微均为首次出现术语或命名约定说明，加括注即可）。MLA 章节的事实核查因 DeepSeek-V2 论文不在授权来源范围而保留，修复方在处理 MLA 重要项时可一并重新对照 arXiv:2405.04434 原文确认 Table 7 与 93.3% 的语境。

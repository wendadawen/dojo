# Delta 规则与 DeltaNet 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html git ac5b744
- 时间：2026-08-09
- 说明：本轮审查者未参与初稿生成与第一轮审查。段 A 盲读按页面顺序进行；段 B 对照来源为 Schlag 2021 ICML（arXiv:2102.11174）、Yang 2024 NeurIPS（arXiv:2406.06484）、Yang/Kautz/Hatamizadeh ICLR 2025（arXiv:2412.06464）。

## 段 A 盲读

按页面顺序阅读，记录主线卡点。

- 顶部 callout 与 context-box 给出本文定位、前置概念（线性注意力，占位）、现实应用（Kimi K3 KDA），小白能定位。学习目标 5 条具体可检验。
- §1 retrieval error 推导：由 $S_t = \sum_i v_i k_i^\top$ 展开到 $S k_j = v_j(k_j^\top k_j) + \sum_{i\neq j} v_i(k_i^\top k_j)$，步骤完整。$d$ 维最多容纳 $d$ 个正交向量、$L>d$ 必然碰撞的论证清楚。手算例子 $d=2, L=3$ 逐步算到 $S k_2 = (2,3)^\top \neq v_2$，可跟随。
- §2 紧凑公式：每个符号（$S, k, v, \beta, \sigma, W_*$）首次出现均有定义；形状检查逐项验证维度自洽；$\beta=0/\beta=1$ 边界先做快速检查再展开。命名溯源把 Widrow-Hoff 与 DeltaNet 的"作用对象差异"讲清。手算一步 $\beta=1$ 完整。
- §3 等价改写：定义 $v^{\text{old}}, v^{\text{new}}$，代入推导放在折叠块，主线用结论。几何直觉 $I - kk^\top$ 的幂等性、对称性各验证一次，2 维示意图直观。
- §4 边界：$\beta=0/\beta=1/\beta\in(0,1)$ 三段 + 误解排查表，把"投影需 $\beta=1$ 且 $\|k\|=1$"的前提列清。$\beta=0.5$ 手算例子返回 $v^{\text{new}}=(0.5,0.5)^\top$，可验证。
- §5 对比：4 种更新规则对比表 + 三组对比 + Gated DeltaNet 退化 + MAD 实验表 + 1.3B PPL + 工程实现。退化分析（$\alpha\to1$ / $\beta\to0$ / $\alpha\to0$）完整。
- 折叠块（推导细节、代码）均为补充，收起后主线可独立成立。

学习目标核对：
1. key 碰撞原因 + delta 规则如何解决 —— §1 + §2 回答 ✓
2. 手算一步 + 符号含义 + $\beta$ 作用 —— §2 回答 ✓
3. 等价形式 + 为什么等价 —— §3 回答（含折叠块推导）✓
4. $\beta\to1/\beta\to0$ 退化 + 前提 —— §4 回答 ✓
5. DeltaNet vs 线性注意力/Mamba2/Gated DeltaNet 差异 + 为什么加 $\alpha$ —— §5 回答 ✓

## 段 B 对照来源

### 1. 定义与机制
- C1 线性注意力加性递归 $S_t = S_{t-1} + v_t k_t^\top$：Yang 2024 NeurIPS §2.1 原文一致 ✓
- C2 $L>d$ key 碰撞：Yang 2024 §2.1 原文 "a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key 'collisions' when L>d, as pointed out by Schlag et al." —— 逐字核对一致 ✓
- C3 Delta 规则紧凑公式：Yang 2024 §2.2 与 Table 4 DeltaNet 行 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ —— 一致 ✓
- C4 等价形式 $v^{\text{old}} = S_{t-1}k_t$、"先擦后写"：Yang 2024 §2.2 推导 $S_t = S_{t-1} - v_t^{\text{old}}k_t^\top + v_t^{\text{new}}k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ —— 一致 ✓
- C5 与 Widrow-Hoff 关系：Schlag 2021 摘要称 "delta rule-like programming instruction"，页面引用 §4.2 "akin to the famous error-correcting delta-rule" 与作用对象差异说明，与来源一致 ✓
- C6 generalized Householder / 投影性质：Yang 2024 §2.2/§3.1 "generalized Householder transformation" 一致 ✓；L2 归一化 ✓（来源 §3.3 明确 $k_t = \text{SiLU}(W_K x_t)/\|\cdot\|_2$）
- C7 Gated DeltaNet 公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$：与 arXiv:2412.06464 方法对比表 Gated DeltaNet 行逐字一致 ✓；Mamba2 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ 一致 ✓
- Gated DeltaNet 设计动机引述 "gating enables rapid memory erasure while the delta rule facilitates targeted updates"：与 arXiv:2412.06464 摘要逐字一致 ✓
- Gated DeltaNet 退化：$\alpha\to1$ 退化为 DeltaNet ✓（来源 Table 2 明确 GDN $\alpha_t=1$ 特例为 DeltaNet）；$\beta\to0$ 退化为 $\alpha_t S_{t-1}$（Mamba2 在 $v_t=0$ 时的特殊情形）—— 页面表述比来源更精确（指出 $v_t=0$），正确 ✓

### 2. 公式与推导
- F1 retrieval 展开 $S k_j = v_j + \sum_{i\neq j}(k_i^\top k_j) v_i$：代数恒等式，手算复核正确 ✓
- §3 折叠块等价推导（合并 $v^{\text{old}}k^\top$ 系数 $-1+(1-\beta)=-\beta$、分配律）：逐步复核正确 ✓
- §3 投影幂等性 $P^2 = I - 2kk^\top + k(k^\top k)k^\top = I - kk^\top$（用 $\|k\|=1$）：正确 ✓
- F2/F3 边界与退化：直接代入，正确 ✓
- 符号首次出现处均有定义，全文含义一致 ✓

### 3. 可运行代码
- 折叠块代码（numpy 实现 C3 紧凑公式 + C4 等价形式 + 线性注意力对比）提取执行，实际输出：
  - Linear S_2 = [[1,0],[1,0]]，S_2@k_2 = [1,1] ✓
  - Delta S_1 = [[1,0],[0,0]]，Delta S_2 = [[0,0],[1,0]]，S_2@k_2 = [0,1] = v_2 ✓
  - Equivalent form S_2 = [[0,0],[1,0]]，Matches compact form: True ✓
  - 与页面"预期输出"逐行一致 ✓

### 4. 事实与推断
- C8/MAD benchmark（N2）：DeltaNet 100/35.7/100/100/52.8/42.2/71.8、Mamba 90.4/6.7/90.1/86.3/89.5/52.7/69.3 等 —— 与 Yang 2024 §4.1 Table 2（经作者博客交叉核对）逐数一致 ✓；页面 Average 列说明"含未展示 Compress 取均值"且复核 $(100+35.7+100+100+52.8+42.2)/6=71.8$、$(90.4+6.7+90.1+86.3+89.5+52.7)/6=69.3$ 正确 ✓
- N1 1.3B/100B PPL（DeltaNet 16.87/12.21、Mamba 17.06/13.89、GLA 17.25/14.92、Transformer++ 16.85/13.44）：NeurIPS PDF Table 3 经 WebFetch 无法解析（PDF 二进制流），未能从网页独立复核具体数值；但页面定性结论"DeltaNet 略优于 Mamba 和 GLA，与 Transformer++ 接近"与论文 §4.2 原文 "DeltaNet outperforms the strong Mamba/GLA baselines in terms of both perplexity" 一致 ✓。数值未发现矛盾，仅独立复核受限。
- 教学示例均标注"教学示例"，未冒充来源数据 ✓

### 5. 前置知识引用
- 线性注意力链接 `../../wiki/linear-attention/index.html` 标注"占位：该页面尚未生成"，符合占位提示要求 ✓
- overview.html 与 index.html 互相链接 ✓

### 6. 教学简化
- key 未归一化（§1 例子）已说明"为便于手算"，并指出工程实现 L2 归一化 ✓
- $d=2$ 极小维度、未展开 WY 表示/多头/Triton kernel 均在"教学简化及其限制"说明 ✓
- 类比边界（"先擦后写"、"橡皮擦 vs 晒褪色"）均列出失效边界 ✓

### 7. 页面功能
- KaTeX 公式渲染、折叠交互、目录锚点结构正常（机械项以 validate.py 为准，本轮未运行 validate.py）。

## 问题

- [轻微·技术] §3 几何直觉段 + 来源 C6：页面将 Yang 2024 §3.3 表述以引号直接引用为 "only erases information in one subspace while keeping the other d-1 subspace intact"，来源原文为 "erasing information in one subspace while preserving the other d−1 subspaces"（when $\beta_t=1$, $I-k_tk_t^\top$ becomes a projection matrix）。差异：页面加 "only"、"keeping...intact" 替换 "preserving"、"subspace"（单）替换 "subspaces"（复）。含义忠实，但引号内非逐字。修法：改为逐字引用 "erasing information in one subspace while preserving the other d−1 subspaces"，或去掉引号改为意译。 ｜ 修复： ｜ 复验：
- [轻微·盲读] §1 手算例子：先称"理想输出是 $v_2 = (0,1)^\top$"，后又称"理想项是 $2v_2 = (0,2)^\top$（外积定义下的自然量级）"。"理想输出"与"理想项"措辞相近，小白可能短暂困惑到底目标是 $v_2$ 还是 $2v_2$（后文已解释因 $k_2$ 未归一化导致系数为 2）。修法：把首次"理想输出是 $v_2$"改为"希望检索到的值是 $v_2$"，与后文"理想项"在用词上区分开。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（进入修复环节处理 2 条轻微问题后发布；轻微问题不阻断发布门控，但建议修复以提升引用准确度与盲读顺畅度）
- 复核说明：N1 的 1.3B PPL 具体数值因 NeurIPS PDF Table 3 无法经网页解析而未能独立逐数复核，定性结论已与论文原文核对一致，未发现矛盾；其余核心论断、公式、代码、MAD 数字均逐条核对通过。

# GLU（门控线性单元）独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html git ac5b744
- 时间：2026-08-09
- 说明：本轮审查者未参与初稿生成与第一轮审查。段 A 盲读按页面顺序进行；段 B 对照来源为 Dauphin et al. 2017 ICML（arXiv:1612.08083）、Shazeer 2020（arXiv:2002.05202）。

## 段 A 盲读

按页面顺序阅读，记录主线卡点。

- 顶部 callout 用"每个维度自己决定放多少"切入门控动机。context-box 给出是什么/解决什么/提出场景/家族角色。学习目标 5 条具体可检验。
- §1 为什么需要门：常规"线性+固定激活"缺两样——按维度调节能力、不杀梯度的非线性；GTU 用 sigmoid/tanh 当门会杀梯度。两分支结构图直观。sigmoid $\sigma(z)=1/(1+e^{-z})$、$\otimes$ 逐元素乘作为基础记号内联定义。
- §2 公式与手算：GLU 定义 $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，每个符号逐项解释；$*$ 为卷积（前馈语境退化为矩阵乘，已说明为教学简化）。三个边界检查（门全 1/全 0/中间值）。手算例子 $x=[1.0,0.5]$、$W=I$、$V=\mathrm{diag}(1,-1)$：$\sigma(1.0)\approx0.7311$、$\sigma(-0.5)\approx0.3775$，输出 $\approx[0.7311,0.1888]$，并翻译为"第一维放行 73%、第二维放行 38%"。极端对照（$V=\mathrm{diag}(10,-10)$）在折叠块验证门→1/门→0 退化。
- §3 梯度通路：$\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$，明确声明 $\nabla\mathbf{X}$ 为上游梯度。第一项缩放因子是门值 $\sigma(\mathbf{X})$（非导数），第二项含 $\sigma'$。GTU 对照（第一项带 $\tanh'$）。对比表清晰。链式法则推导在折叠块。边界（门饱和到 0 时通路也断）有说明。
- §4 家族：Bilinear 去 $\sigma$；ReGLU/GEGLU/SwiGLU 换激活（Shazeer 记法，激活在 $W$ 分支）。记号差异表（Dauphin $\sigma$ 在 $V$ 分支 vs Shazeer $\sigma$ 在 $W$ 分支，$W\leftrightarrow V$ 等价）。FFN 三矩阵 + $2/3$ 缩放推导（$3\cdot d\cdot d_{ff}'=2\cdot d\cdot d_{ff}\Rightarrow d_{ff}'=\tfrac23 d_{ff}$），参数量验算在折叠块。
- §5 经验结论与边界：Shazeer Table 1 八行数字。GEGLU/SwiGLU 最优。"divine benevolence"原文引述。GLU 不解决什么（不保证普适更好/不替代归一化/不做跨位置聚合/不保证全局最优/门全关退化为零）+ 常见误解四条。
- 折叠块（极端对照、梯度推导、参数量验算）均为补充，收起后主线可独立成立。
- 本页无可运行代码块（仅图示、公式、表格），无代码需执行。

学习目标核对：
1. 一句话说清 GLU 在做什么 + 解决什么问题 —— §1+§2 回答 ✓
2. 公式解释非线性 + 梯度线性通路（与 GTU 对比）—— §3 回答 ✓
3. 手算 GLU 输出 + 翻译"哪个维度被放行/压低" —— §2 回答 ✓
4. 家族派生 + Dauphin/Shazeer 记号差异 —— §4 回答 ✓
5. FFN 缩 $2/3$ 维度原因 + GLU 不保证什么 —— §4+§5 回答 ✓

## 段 B 对照来源

### 1. 定义与机制
- C1 GLU 定义 $h(\mathbf{X})=(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$，$\sigma$ 在 $V$ 分支：Dauphin §2 原文 "a linear projection $\mathbf{X}*\mathbf{W}+\mathbf{b}$ modulated by the gates $\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})" —— 逐字一致 ✓
- C2/C3 GLU 梯度通路与"乘性跳连"：Dauphin §3 Eq.(3) $\nabla[\mathbf{X}\otimes\sigma(\mathbf{X})]=\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\mathbf{X}\otimes\sigma'(\mathbf{X})\nabla\mathbf{X}$，原文 "This can be thought of as a multiplicative skip connection" —— 逐字一致 ✓
- C4 GTU 梯度含 $\tanh'$ 与 $\sigma'$：Dauphin §3 Eq.(2) $\nabla[\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})]=\tanh'(\mathbf{X})\nabla\mathbf{X}\otimes\sigma(\mathbf{X})+\sigma'(\mathbf{X})\nabla\mathbf{X}\otimes\tanh(\mathbf{X})$ —— 一致 ✓
- "for the activated gating units" 限定语：Dauphin §3 原文 "without downscaling for the activated gating units in $\sigma(\mathbf{X})$" —— 一致 ✓
- C5 Bilinear 去 $\sigma$、归因 Mnih & Hinton 2007：Dauphin §5.3 ✓（Shazeer 也确认 "attribute to [4]"）
- C6 ReGLU/GEGLU/SwiGLU 换激活变体：Shazeer §2 Eq.(5) $\mathrm{ReGLU}=\max(0,xW+b)\otimes(xV+c)$、$\mathrm{GEGLU}=\mathrm{GELU}(xW+b)\otimes(xV+c)$、$\mathrm{SwiGLU}=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)$，激活在 $W$ 分支 —— 一致 ✓
- C7 FFN_GLU 三矩阵 $\mathrm{FFN}_{\mathrm{GLU}}=(\sigma(xW)\otimes xV)W_2$：Shazeer §2 Eq.(6) —— 一致 ✓；$2/3$ 缩放：Shazeer §2 "reduce the number of hidden units $d_{ff}$ by a factor of $2/3$" ✓；§3.1 实例 $d_{ff}:3072\to2048$ ✓
- C8 所有变体优于基线、GEGLU/SwiGLU 最优、无理论解释：Shazeer Table 1 + §4 "divine benevolence" ✓
- C9 Dauphin $\sigma$ 在 $V$ 分支 vs Shazeer $\sigma$ 在 $W$ 分支：Dauphin Eq.(1) $(\mathbf{X}*\mathbf{W}+\mathbf{b})\otimes\sigma(\mathbf{X}*\mathbf{V}+\mathbf{c})$ vs Shazeer Eq.(4) $\sigma(xW+b)\otimes(xV+c)$ —— 一致，$\otimes$ 可交换故 $W\leftrightarrow V$ 等价 ✓

### 2. 公式与推导
- F1 GLU 定义：Dauphin §2 Eq.(1) ✓
- F4 GLU 梯度：Dauphin §3 Eq.(3) ✓；折叠块链式法则推导（$Y_i=A_iB_i$，$\partial Y_i/\partial X_i=\sigma(X_i)+X_i\sigma'(X_i)$）手算复核正确 ✓
- F3 GTU 梯度：Dauphin §3 Eq.(2) ✓
- 手算例子：$\sigma(1.0)=1/(1+e^{-1})\approx0.7311$、$\sigma(-0.5)\approx0.3775$、输出 $[1.0,0.5]\otimes[0.7311,0.3775]\approx[0.7311,0.1888]$ —— 复核正确 ✓
- 极端对照 $V=\mathrm{diag}(10,-10)$：$\sigma(10)\approx0.99995$、$\sigma(-5)\approx0.00669$，输出 $\approx[1.0,0]$ —— 复核正确 ✓
- 参数量等式：双矩阵 $2\cdot d\cdot d_{ff}$、三矩阵 $3\cdot d\cdot d_{ff}'$，令等得 $d_{ff}'=\tfrac23 d_{ff}$；$d=768,d_{ff}=3072$ 验算 $2\times768\times3072=4{,}718{,}592=3\times768\times2048$ —— 复核正确 ✓

### 3. 可运行代码
- 本页无可运行代码块（仅图示、公式、表格），无代码需执行。不适用。

### 4. 事实与推断
- N1 Shazeer Table 1（524,288 步 heldout log-perplexity）：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633 —— 与原文 Table 1 逐数一致 ✓；GEGLU(1.633) 与 SwiGLU(1.636) 为最优两个 ✓
- "divine benevolence" 原文结语：Shazeer §4 "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." —— 逐字一致 ✓
- 实验条件（T5 base、$d_{model}=768$、12 头、基线 $d_{ff}=3072$、GLU 变体 $d_{ff}=2048$、segment-filling）：与 Shazeer §3.1 一致 ✓
- 教学示例（手算 $x=[1.0,0.5]$、极端对照、参数量验算）均标注"教学构造，不代表真实模型参数"✓

### 5. 前置知识引用
- Swish/GELU/ReLU 作为激活名出现而不展开，明确标注"属各自概念页"✓
- sigmoid、$\otimes$、$xW+b$、链式法则作为基础记号内联定义，不建独立概念页（教学简化说明已论述理由）✓
- overview.html 与 index.html 互相链接 ✓

### 6. 教学简化
- $*$（卷积）退化为矩阵乘、梯度分析用简化形式 $Y=\mathbf{X}\otimes\sigma(\mathbf{X})$、Swish/GELU 不展开 均在"教学简化及其限制"说明，并区分"可推出/不可推出"✓
- "门=逐维度阀门""乘性跳连""门控决定 vs 输入幅值副作用"均在"教学解释与类比边界"列出失效边界（门非物理阀门、门饱和到 0 通路也断、不等同 ResNet 加性跳连）✓

### 7. 页面功能
- KaTeX 公式渲染、折叠交互、目录锚点结构正常（机械项以 validate.py 为准，本轮未运行 validate.py）。

## 问题

（无）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布
- 复核说明：核心论断（GLU 定义、梯度通路、GTU 对比、家族派生、记号差异、FFN 缩放、经验结论）、全部公式推导、外部数字（Table 1 八行、"divine benevolence"引述、$d_{ff}:3072\to2048$）、手算例子与参数量验算均逐条核对通过，与两篇来源完全一致；本页无可运行代码。未发现需修复的问题。

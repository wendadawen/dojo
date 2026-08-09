# RoPE 旋转位置编码 · 核心论断与证据

编号约定：C 论断 / F 公式 / N 数字。仅覆盖核心内容。

## C 论断（事实性陈述）

### C1：RoPE 只作用于 Q 和 K，不作用于 V
- **论断内容**：RoPE 在每个注意力层内部、softmax 之前对 Q 和 K 施加位置相关旋转；V 不被旋转。
- **来源定位**：Su et al. 2021, arXiv:2104.09864v5, §3.2, Eq.(22)–(25)：`q̃_m = R_m^d W_q x_m`、`k̃_n = R_n^d W_k x_n`、`v_n = W_v x_n`（V 公式无旋转矩阵）。
- **适用条件**：标准 softmax 注意力。
- **置信状态**：已确认。

### C2：RoPE 用绝对位置构造旋转，但内积只依赖相对位置
- **论断内容**：旋转矩阵 `R_m` 由绝对位置 `m` 决定，但 `(R_m q)^T (R_n k) = q^T R_{n−m} k`，结果只依赖相对位置 `m−n`。
- **来源定位**：Su et al. 2021, §3.1, Eq.(1)–(7)；摘要："encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency"。
- **适用条件**：`R_m` 满足 `R_m^T R_n = R_{n−m}`，即旋转矩阵群的性质。
- **置信状态**：已确认。

### C3：RoPE 没有可学习参数
- **论断内容**：`θ_i = base^{-2i/d}`、`base = 10000` 均为固定常数；旋转矩阵预算即用，不参与训练。
- **来源定位**：Su et al. 2021, §3.2, Eq.(12)；与 Vaswani et al. 2017 的正弦位置编码频率公式同源。
- **适用条件**：原始 RoPE 定义；扩展方法（YaRN 等）可能引入可学习缩放，不在此论断范围。
- **置信状态**：已确认。

### C4：θ_i 形成几何级数，i 小则 θ 大（旋转快），i 大则 θ 小（旋转慢）
- **论断内容**：`θ_i = 10000^{-2i/d}`，i 从 0 到 d/2−1；θ₀ = 1（最快，每位置旋转约 1 弧度），θ_{d/2−1} ≈ 0.0001（最慢，需数千位置才完成一周）。
- **来源定位**：Su et al. 2021, §3.2, Eq.(12)；多源综述印证（agentica.wiki、unseel.com/cs/rotary-position-embedding）。
- **适用条件**：base=10000 是论文默认；不同模型可能调整 base（如 LLaMA 系列保持 10000，部分扩展工作用更大值）。
- **置信状态**：已确认。

### C5：远程衰减性质——期望内积随 |m−n| 增大趋于 0
- **论断内容**：在 q、k 分量独立同分布且均值为 0、方差为 1 假设下，`E[q^T R_{n−m}^d k] = Σ_{i=0}^{d/2−1} cos((n−m)θ_i)`，由 Riemann-Lebesgue 引理当 |m−n| → ∞ 时趋于 0。
- **来源定位**：Su et al. 2021, §3.3（"Long-term decay"）, Eq.(15)–(21)；论文图 2 给出 d=128、不同 |m−n| 的衰减曲线。
- **适用条件**：q、k 分量 i.i.d. 假设；真实模型中分量分布非 i.i.d.，衰减为期望性质而非逐点保证。
- **置信状态**：已确认。

### C6：RoPE 主流开源 LLM 普遍采用
- **论断内容**：LLaMA（所有版本）、Mistral、Falcon、Qwen、PaLM、GPT-NeoX、Gemma 等开源 LLM 使用 RoPE 作为默认位置编码。
- **来源定位**：多源综述印证（agentica.wiki, unseel.com, spawn08.github.io）。具体每个模型的引用需查对应论文 / 配置文件，本页正文只陈述"主流采用"事实，不逐一展开。
- **适用条件**：截至 2024 年发布的开源模型；闭源模型（GPT/Claude）位置编码方案未公开。
- **置信状态**：已确认（"主流采用"层面）；逐模型引用为辅助级。

### C7：超出训练长度简单外推性能下降，需 YaRN/PI 等扩展
- **论断内容**：训练长度外，`m θ_i` 进入未训练的角度区间，简单外推性能下降；Position Interpolation（Chen et al. 2023）缩放 m、YaRN（Peng et al. 2023）按频率分组缩放，是常见扩展方法。
- **来源定位**：Chen et al. 2023, arXiv:2306.13595（PI）；Peng et al. 2023, arXiv:2309.00071（YaRN）；Su et al. 2021 原论文摘要提到"flexibility of sequence length"但未声称无限外推。
- **适用条件**：原始 RoPE 不做缩放时；扩展方法的具体机制不在本页展开。
- **置信状态**：已确认（存在性）；扩展方法细节属相邻概念。

### C8：K3 的 MLA 层不施加旋转（NoPE），RoPE 是理解此设计选择的前置
- **论断内容**：Kimi K3 的 MLA 层在结构上保留 RoPE 接口（拆出 rot 分量），但 `config.json` 中 `mla_use_nope=true`，所有 MLA 层不施加位置编码；这是因为 RoPE 与矩阵吸收冲突（DeepSeek-V2 用解耦 RoPE 化解，K3 直接用 NoPE 简化）。
- **来源定位**：Kimi K3 技术报告 §2.1.2（Gated MLA）、§3.4（Long-Context Extension）；config.json `mla_use_nope: true`；已有页面 [`MLA`](../mla/index.html) S3、[`NoPE`](../nope/index.html)、[`Kimi K3`](../kimi-k3/index.html) §2.1.2 已对此论证。
- **适用条件**：K3 的 MLA 层；K3 的 KDA 层与 MLA 角色不同。
- **置信状态**：已确认（与已有页面一致）。

## F 公式

### F1：2 维旋转矩阵
$$R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$
- **来源**：Su et al. 2021, §3.1, Eq.(5)。
- **适用**：位置 m 的 (q₀, q₁) 对；θ 是该对的频率。

### F2：旋转后内积（2 维）
$$\langle R_m q, R_n k \rangle = (q_0 k_0 + q_1 k_1)\cos((n-m)\theta) + (q_0 k_1 - q_1 k_0)\sin((n-m)\theta)$$
- **来源**：Su et al. 2021, §3.1, Eq.(4)。
- **适用**：单 2 维对。

### F3：相对位置保持性质
$$R_m^T R_n = R_{n-m}$$
- **来源**：Su et al. 2021, §3.1, Eq.(7)。
- **适用**：旋转矩阵群性质，由三角恒等式直接推出。

### F4：d 维分块对角旋转矩阵
$$R_m^d = \mathrm{diag}(R_m^{(0)}, R_m^{(1)}, \ldots, R_m^{(d/2-1)}), \quad R_m^{(i)} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$$
- **来源**：Su et al. 2021, §3.2, Eq.(9)。
- **适用**：d 为偶数。

### F5：d 维内积的相对位置保持
$$\langle R_m^d q, R_n^d k \rangle = q^T R_{n-m}^d k$$
- **来源**：Su et al. 2021, §3.2, Eq.(11)。
- **适用**：由 F3 在每个 2×2 块上分别成立。

### F6：频率公式
$$\theta_i = \mathrm{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1, \quad \mathrm{base} = 10000$$
- **来源**：Su et al. 2021, §3.2, Eq.(12)。
- **适用**：原始 RoPE；变体可能调整 base 或 i 的范围。

### F7：远程衰减期望
$$\mathbb{E}[\langle R_m^d q, R_n^d k \rangle] = \sum_{i=0}^{d/2-1} \cos((n-m)\theta_i)$$
- **来源**：Su et al. 2021, §3.3, Eq.(16)。
- **适用**：q、k 分量 i.i.d.、均值 0、方差 1。

### F8：远程衰减极限
$$\frac{1}{d/2} \sum_{i=0}^{d/2-1} \cos((n-m) \cdot 10000^{-2i/d}) \to 0 \quad \text{当 } |n-m| \to \infty$$
- **来源**：Su et al. 2021, §3.3, Eq.(21)；由 Riemann-Lebesgue 引理。
- **适用**：d 较大时的积分近似。

## N 数字

### N1：base 默认值 10000
- **数值**：`base = 10000`。
- **来源**：Su et al. 2021, §3.2, Eq.(12)；与 Vaswani et al. 2017 的正弦位置编码同源。
- **条件**：原始 RoPE 默认；扩展方法可能调整。

### N2：d=128 时最快与最慢频率
- **数值**：θ₀ = 10000^0 = 1（每位置旋转约 1 弧度，约 57.3°）；θ₆₃ = 10000^{-126/128} ≈ 0.0001（约每 62832 位置完成一周，即 2π/0.0001）。
- **来源**：F6 代入 d=128 计算；多源综述印证（unseel.com 提到 54000）。
- **条件**：d=128（LLaMA 系列头维度典型值）。
- **置信状态**：已确认（直接代入公式）；54000 与 62832 的差异源于近似精度，本页用 62832。

### N3：手算例子数值（教学示例）
- **数值**：取 q=(1,0)、k=(1,0)、θ=π/4、m=1、n=2；旋转后 R_m q = (cos(π/4), sin(π/4)) = (√2/2, √2/2)、R_n k = (cos(π/2), sin(π/2)) = (0, 1)；内积 = √2/2 · 0 + √2/2 · 1 = √2/2 ≈ 0.7071；按 F2 验证：(1·1+0·0)cos((2−1)π/4) + (1·0−0·1)sin((2−1)π/4) = cos(π/4) = √2/2 ≈ 0.7071，一致。
- **来源**：教学构造，便于手算。
- **条件**：θ=π/4 是教学数字，非真实模型频率。
- **置信状态**：教学示例。

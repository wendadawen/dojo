# 位置编码基础 · 核心论断与证据

核心论断编号（C 论断 / F 公式 / N 数字）只覆盖核心内容。每个论断记录内容、来源定位、适用条件与置信状态。

## C 论断

### C1：自注意力对输入序列行重排不变（排列等变），不含位置信息

- **论断内容**：在没有任何位置信息时，把输入序列的行（token 顺序）任意重排，注意力的输出也对应重排，每个位置上的输出值不变；模型无法区分 token 的先后顺序。
- **来源定位**：Vaswani et al. 2017, "Attention Is All You Need", arXiv:1706.03762, §3.5 第 1 段（"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence."）。排列等变性的形式化论证见已有页面 [`标准 Transformer 注意力`](../standard-attention/index.html) §复杂度与边界（permutation-invariant 性质）。
- **适用条件**：双向自注意力且不施加任何位置信息时成立；decoder-only 加因果掩码时位置隐式存在（见 NoPE）。
- **置信状态**：已确认。

### C2：绝对正弦位置编码公式（Vaswani 2017 §3.5）

- **论断内容**：位置 pos 的位置编码向量由 sin/cos 给出，偶数维 sin、奇数维 cos，频率随维度索引 i 几何递减；该向量加到 token 嵌入上注入位置信息。
- **来源定位**：Vaswani et al. 2017, §3.5, Eq.(3) 与 Eq.(4)。原文："PE_{(pos,2i)} = sin(pos/10000^{2i/d_model})", "PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_model})"。
- **适用条件**：d_model 为偶数；pos 为非负整数位置索引。
- **置信状态**：已确认。

### C3：多频率选择让不同维度编码不同尺度的位置信息

- **论断内容**：i 小的维度频率大（波长短，变化快，编码局部/精细位置）；i 大的维度频率小（波长长，变化慢，编码全局/粗略位置）；波长从 2π（i=0）到 20000π（i=d_model/2−1）跨越多个数量级。
- **来源定位**：Vaswani et al. 2017, §3.5 第 3 段（"The wavelengths ... range from 2π to 10000·2π"）；多源综述印证（learnixo.io、alessioborgi.github.io、iclr-blogposts 2025）。
- **适用条件**：base=10000，d_model 为偶数。
- **置信状态**：已确认。

### C4：正弦 PE 的线性性质——PE(pos+k) 是 PE(pos) 的线性函数

- **论断内容**：对任意固定偏移 k，PE(pos+k) 可表示为 PE(pos) 的线性变换（旋转）；这是模型能从绝对编码中读出相对位置的来源。
- **来源定位**：Vaswani et al. 2017, §3.5 第 4 段（"We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_{pos}."）。
- **适用条件**：同一频率对（同一 i）内的两个维度；k 为固定整数偏移。
- **置信状态**：已确认。

### C5：Vaswani 2017 base 与 big 模型都用正弦位置编码；可学习是实验项，两者结果相近

- **论断内容**：原论文的 base 和 big 模型都使用正弦位置编码；可学习位置编码只作为 Table 3 row (e) 的对比实验，报告两者"nearly identical results"，论文最终选正弦（因可能允许外推）。
- **来源定位**：Vaswani et al. 2017, §3.5 第 5 段（"We also experimented with using learned positional embeddings [9] and found that the two versions produced nearly identical results (see Table 3 row (e))."）；Table 3 row (e)：learned EN-DE BLEU 26.74(base)/27.72(big)，sinusoidal 27.3(base)/28.4(big)。引用 [9] = Gehring et al. 2017, ConvS2S, arXiv:1705.03122。经 The Annotated Transformer (nlp.seas.harvard.edu) 核实。
- **适用条件**：Vaswani 2017 论文范围内的实验设置。
- **置信状态**：已确认（纠正"big 模型用可学习"的误传）。

### C6：可学习绝对位置编码的形式与外推限制

- **论断内容**：可学习方案把位置向量换成一个 L×d_model 的可学习参数表（L 为最大序列长度），每个位置一个可学习向量；超出 L 即无对应参数、外推失败；BERT、GPT-2 采用此方案。
- **来源定位**：Vaswani et al. 2017, §3.5 引用 [9] Gehring et al. 2017；iclr-blogposts 2025 综述（"learned positional embeddings ... applied in BERT (2018) and GPT (2019) ... the upper bound L limited the method's ability to extrapolate"）；learnixo.io 对比表。
- **适用条件**：序列长度 ≤ L。
- **置信状态**：已确认。

### C7：T5 相对位置 bias 的机制——加在注意力分数上、分桶、每头独立各层共享

- **论断内容**：T5 不在输入嵌入上加位置向量，而是在注意力分数 softmax 之前加一个与 query-key 相对距离 i−j 绑定的标量偏置；相对距离分桶（bucketing），每桶一个可学习标量；每个注意力头有独立的 bias，所有层共享同一套 bias；对超出训练最大距离的相对距离 clamp 到最大桶。
- **来源定位**：Raffel et al. 2019, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR Vol 21, §3.2（"We use a simplified form of relative positional bias ... added to the attention logits"）；T5 源码 `t5/relative_position_bias.py` 的 `_relative_position_bucket`；多源综述印证（research.mental-momentum.ai、blog.csdn.net/qq_42675275）。
- **适用条件**：T5 encoder-decoder 架构；最大距离约 128（分桶后约 32 桶）。
- **置信状态**：已确认。

### C8：绝对方案编码"第几位"，相对方案编码"相差几位"

- **论断内容**：绝对位置编码的向量与位置 pos 绑定（第 pos 位的向量只依赖 pos），作用在输入嵌入；相对位置编码的偏置与两个位置的差 i−j 绑定，作用在注意力分数。两者作用点不同、编码对象不同。
- **来源定位**：Vaswani et al. 2017 §3.5（绝对方案作用点）；Raffel et al. 2019 §3.2（相对方案作用点）；iclr-blogposts 2025 综述明确区分两类。
- **适用条件**：一般性区分。
- **置信状态**：已确认。

### C9：RoPE 是"绝对构造、相对效果"，NoPE 依赖因果掩码隐式提供位置

- **论断内容**：RoPE 用绝对位置 m、n 构造旋转，但内积只依赖 m−n（详见 wiki/rope）；NoPE 不施加任何显式位置编码，decoder-only 的因果掩码隐式提供位置信息（详见 wiki/nope）。
- **来源定位**：Su et al. 2021, arXiv:2104.09864 §3.1（RoPE）；已有页面 [`NoPE`](../nope/index.html)、[`RoPE`](../rope/index.html) 已论证。
- **适用条件**：RoPE 需 softmax 注意力与偶数维度；NoPE 仅适用 decoder-only。
- **置信状态**：已确认（引用已有页面）。

### C10：K3 的 MLA 层选 NoPE，因 RoPE 与矩阵吸收冲突

- **论断内容**：Kimi K3 的 Gated MLA 层在结构上保留 RoPE 接口（rot 分量），但 config 中 mla_use_nope=true，所有 MLA 层不施加位置编码；原因是 RoPE 施加在压缩潜变量上会破坏 MLA 的矩阵吸收前提；K3 的位置信息由 KDA 隐式提供，MLA 直接外推到 1M token 无需修改位置编码参数。
- **来源定位**：已有页面 [`Kimi K3`](../kimi-k3/index.html) §2.1.2、§3.4；[`MLA`](../mla/index.html) §3（矩阵吸收与 RoPE 解耦）；[`NoPE`](../nope/index.html)。
- **适用条件**：K3 架构范围内。
- **置信状态**：已确认（引用已有页面）。

## F 公式

### F1：绝对正弦位置编码公式

$$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}}),\quad PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}})$$

来源：Vaswani et al. 2017, §3.5, Eq.(3) 与 Eq.(4)。

### F2：位置编码注入方式

$$x'_m = x_m + PE_m$$

其中 $x_m$ 是位置 $m$ 的 token 嵌入，$PE_m$ 是位置 $m$ 的正弦/可学习位置向量。来源：Vaswani et al. 2017 §3.5（"we add the positional encodings to the input embeddings"）。

### F3：正弦 PE 的线性性质（相对位置可读出）

对同一频率对（维度 $2i$、$2i+1$），记 $\omega_i=1/10000^{2i/d_{model}}$，则

$$PE_{(pos+k,2i)}=PE_{(pos,2i)}\cos(k\omega_i)+PE_{(pos,2i+1)}\sin(k\omega_i)$$
$$PE_{(pos+k,2i+1)}=PE_{(pos,2i+1)}\cos(k\omega_i)-PE_{(pos,2i)}\sin(k\omega_i)$$

来源：Vaswani et al. 2017 §3.5 第 4 段（陈述形式）；由 sin/cos 和角公式直接推出。

### F4：T5 相对 bias 加在注意力分数上

$$\text{score}(i,j)=q_i\cdot k_j + b_{\text{bucket}(i-j)}$$

其中 $b_{\text{bucket}(i-j)}$ 是与相对距离分桶绑定的可学习标量。来源：Raffel et al. 2019 §3.2。

### F5：可学习绝对位置编码

$$PE \in \mathbb{R}^{L\times d_{model}},\quad x'_m = x_m + PE_m$$

其中 $PE$ 是可学习参数表，$L$ 是最大序列长度。来源：Vaswani 2017 §3.5 引用 [9] Gehring et al. 2017。

## N 数字

### N1：Vaswani 2017 Table 3 row (e) learned vs sinusoidal 对比

- learned（base）：EN-DE BLEU 26.74
- sinusoidal（base）：EN-DE BLEU 27.3
- learned（big）：EN-DE BLEU 27.72
- sinusoidal（big）：EN-DE BLEU 28.4
- 来源：Vaswani et al. 2017 Table 3 row (e)；经 The Annotated Transformer 核实。
- 结论：两者相近，sinusoidal 略优。

### N2：波长范围

- i=0：波长 $2\pi$（约 6.28 位置转一圈）
- i=d_model/2−1（d_model=512 时 i=255）：波长 $10000\cdot 2\pi$（约 62832 位置转一圈）
- 来源：Vaswani et al. 2017 §3.5 第 3 段。

### N3：d_model 默认 512

- 来源：Vaswani et al. 2017 §3.2.2 / Table 3（base 模型 $d_{model}=512$）。

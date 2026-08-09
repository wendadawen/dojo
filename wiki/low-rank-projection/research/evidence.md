# 核心论断与证据：低秩分解

编号约定：C 论断 / F 公式 / N 数字。仅覆盖核心内容。

## C 论断

### C1
- 论断：过参数化模型实际上处在低内在维度上；权重更新在适应过程中也具有低"内在秩"，因此可以用低秩矩阵近似权重更新。
- 来源定位：LoRA 论文（Hu et al., 2021, arXiv:2106.09685v2）§1 Introduction："the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank', leading to our proposed Low-Rank Adaptation."
- 适用条件：预训练模型的微调场景，权重更新量相对于原权重是低秩的。
- 置信状态：已确认。

### C2
- 论断：截断 SVD 给出的 $W_k$ 是所有秩不超过 $k$ 的矩阵中对 $W$ 的最优近似（Frobenius 范数与谱范数意义下）。
- 来源定位：Eckart-Young-Mirsky 定理。Wikipedia "Low-rank approximation"：定理最初在向量空间中提出，后被 Eckart 和 Young 于 1936 年重新发现，Mirsky 推广到任意酉不变范数。Wikipedia "Singular value decomposition"："$\tilde{M}$ is the best approximation of $M$ by any matrix of rank less than or equal to $t$, under the Frobenius norm ... This is known as the Eckart–Young theorem, as it was proved by those two authors in 1936."
- 适用条件：范数为 Frobenius 范数或谱范数（更一般为酉不变范数）；矩阵为实或复矩阵。
- 置信状态：已确认。

### C3
- 论断：MLA 的核心是低秩 K-V 联合压缩，用下投影把隐藏状态压成低维潜向量，只缓存潜向量，K/V 由上投影重建。
- 来源定位：DeepSeek-V2 论文（arXiv:2405.04434v2）§2.1.2 "Low-Rank Key-Value Joint Compression"："The core of MLA is the low-rank joint compression for keys and values to reduce KV cache"。
- 适用条件：MLA 注意力架构；重建的 K/V 与 MHA 直接投影的不等价（有损）。
- 置信状态：已确认。

### C4
- 论断：低秩近似有效的前提是矩阵奇异值快速衰减；若矩阵接近满秩，截断会丢掉大量信息。K3 把 Kimi Linear 的 low-rank gate 改为 full-rank gate，正是因为低秩门控表达力不足。
- 来源定位：Eckart-Young 误差公式（C2 的推论）+ KDA 概念页已核对的 config.json `use_full_rank_gate = true`（Kimi K3 技术报告）。
- 适用条件：一般低秩近似；KDA 的 full-rank 转变是 LLM 架构中的具体实例。
- 置信状态：已确认。

## F 公式

### F1
- 公式：$h = W_0 x + \Delta W x = W_0 x + BAx$，其中 $\Delta W = BA$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。
- 来源定位：LoRA 论文 §4 Eq.(3)："For $h = W_0 x$, our modified forward pass yields: $h = W_0 x + \Delta W x = W_0 x + BAx$"。Figure 1 标注 "We only train $A$ and $B$"。
- 适用条件：$W_0$ 冻结不训练，$A, B$ 可训练；初始化 $A$ 为随机高斯、$B$ 为零使 $\Delta W = 0$。
- 置信状态：已确认。

### F2
- 公式：SVD $W = U\Sigma V^\top$，$\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_r)$，$\sigma_1 \ge \cdots \ge \sigma_r > 0$；截断 SVD $W_k = \sum_{i=1}^{k}\sigma_i u_i v_i^\top$。
- 来源定位：Wikipedia "Singular value decomposition"（定义）+ "Low-rank approximation"（$A_k := \sum_{i=1}^{k}\sigma_i u_i v_i^\top$）。
- 适用条件：任意实/复矩阵；$u_i, v_i$ 为 $U, V$ 的第 $i$ 列。
- 置信状态：已确认。

### F3
- 公式：Frobenius 范数误差 $\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^{\min(m,n)}\sigma_i^2}$；谱范数误差 $\|W - W_k\|_2 = \sigma_{k+1}$。
- 来源定位：Eckart-Young-Mirsky 定理。Wikipedia "Low-rank approximation" §"Proof of Eckart–Young–Mirsky theorem (for spectral norm)" 与 §"(for Frobenius norm)"：$W_k$ 是最优解，误差由被截掉的奇异值决定。
- 适用条件：$W_k$ 为截断 SVD；范数为 Frobenius 或谱范数。
- 置信状态：已确认。

### F4
- 公式：$c_t^{KV} = W^{DKV} h_t$（Eq.9），$k_t^C = W^{UK} c_t^{KV}$（Eq.10），$v_t^C = W^{UV} c_t^{KV}$（Eq.11）。$W^{DKV} \in \mathbb{R}^{d_c \times d}$，$W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$。
- 来源定位：DeepSeek-V2 论文 §2.1.2 Eq.(9)(10)(11)。
- 适用条件：MLA 架构；$d_c \ll d_h n_h$ 时为低秩压缩。
- 置信状态：已确认。

### F5
- 公式：低秩分解参数计数。原矩阵 $W \in \mathbb{R}^{m \times n}$：$mn$ 个参数。分解 $W \approx AB$，$A \in \mathbb{R}^{m \times r}$、$B \in \mathbb{R}^{r \times n}$：$mr + rn = r(m+n)$ 个参数。当 $r \ll \min(m,n)$ 时 $r(m+n) \ll mn$。
- 来源定位：由矩阵乘法维度直接推出；LoRA 论文 §4 隐含使用（$d^2 \to 2dr$）。
- 适用条件：$r \ll \min(m,n)$。
- 置信状态：已确认（基本算术）。

### F6
- 公式：KV cache 参数计数。MHA 每 token 每层：$2 n_h d_h$（K 和 V 各 $n_h d_h$）。MLA 每 token 每层：$d_c$（只缓存潜向量 $c_t^{KV}$，加上较小的解耦 RoPE 项）。
- 来源定位：DeepSeek-V2 论文 §2.1.4 Table 1：MHA = $2 n_h d_h l$，MLA = $(d_c + d_h^R) l$。
- 适用条件：每层每 token；$l$ 为层数，本页取 $l=1$ 比较单层。
- 置信状态：已确认。

## N 数字

### N1
- 数字：LoRA 在 GPT-3 175B 上，可训练参数可低至原模型的 $0.01\%$；论文报告"reduce the number of trainable parameters by 10,000 times"。
- 来源定位：LoRA 论文 §2："the number of trainable parameters $|\Theta|$ can be as small as $0.01\%$ of $|\Phi_0|$"；§1："reduce the number of trainable parameters by 10,000 times"。
- 实验条件：GPT-3 175B，特定 rank 配置。
- 置信状态：已确认。

### N2
- 数字：LoRA 论文指出 rank $r$ 可以是一或二就足够，即使满秩 $d$ 高达 12288。
- 来源定位：LoRA 论文 §1："a very low rank (i.e., $r$ in Figure 1 can be one or two) suffices even when the full rank (i.e., $d$) is as high as 12,288"。
- 实验条件：GPT-3 175B 的 $d = 12288$。
- 置信状态：已确认。

### N3
- 数字：DeepSeek-V2 的 MLA 配置 $n_h = 128$，$d_h = 128$，$d_c = 512$；MHA 每 token 每层缓存 $2 n_h d_h = 32768$，MLA 的潜向量维度 $d_c = 512$。
- 来源定位：DeepSeek-V2 论文 §3.1.2："we set the number of attention heads $n_h$ to 128 and the per-head dimension $d_h$ to 128. The KV compression dimension $d_c$ is set to 512"。
- 实验条件：DeepSeek-V2 模型配置。
- 置信状态：已确认。

### N4
- 数字：DeepSeek-V2 报告 MLA 相比 MHA 减少了 93.3% 的 KV cache；Table 7 显示 Small MoE 从 110.6K 降到 15.6K（每 token），Large MoE 从 860.2K 降到 34.6K。
- 来源定位：DeepSeek-V2 论文摘要："reduces the KV cache by 93.3%"；§2.1.4 Table 7。
- 实验条件：DeepSeek-V2 整体模型（含解耦 RoPE 项与多层）；93.3% 为整体降幅。
- 置信状态：已确认。

### N5（教学示例，非外部数字）
- 数字：教学示例矩阵 $W = \mathrm{diag}(3, 1, 0.5)$，奇异值 $\sigma_1=3, \sigma_2=1, \sigma_3=0.5$。秩-1 近似误差 $\|W-W_1\|_F = \sqrt{1^2+0.5^2} = \sqrt{1.25} \approx 1.118$；秩-2 近似误差 $\|W-W_2\|_F = 0.5$。
- 来源：人为构造，目的是让 SVD 平凡（$U=V=I$）以便手算误差公式。
- 置信状态：教学示例。

# Gated DeltaNet · 术语表

登记全文所有首次出现的术语、缩写和符号：名称、首次出现位置、定义或含义。

## 术语

| 名称 | 首次出现 | 定义或含义 |
|---|---|---|
| Gated DeltaNet | 页面标题 | 本页主概念；把 DeltaNet 的 delta rule 与 Mamba2 的标量衰减门 α_t 统一的线性注意力递归模型 |
| Gated Delta Networks | S1 或来源说明 | 论文原始标题（Yang 2025 ICLR arXiv:2412.06464），与 Gated DeltaNet 指同一模型 |
| GDN | S5 或实验小节 | Gated DeltaNet 的缩写，论文 §3.3、§4 混合架构用此缩写 |
| delta rule / Delta 规则 | S1 | DeltaNet 的递归更新规则 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，详见 `wiki/delta-rule/` |
| DeltaNet | S1 | 把线性注意力的加性累加替换为 delta rule 的模型（Yang 2024 NeurIPS arXiv:2406.06484） |
| Mamba2 | S1 | 状态空间模型，递归 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$，标量 α_t 全局衰减 |
| 线性注意力 | S1（间接） | 用核函数把注意力从 O(N²) 降到 O(N) 的注意力族，详见 `wiki/linear-attention/` |
| 状态空间模型 / SSM | S1 | Mamba2 所属家族，本页只引用 Mamba2 公式，不展开 SSM 理论 |
| KDA / Kimi Delta Attention | S5 | Kimi K3 的线性注意力变体，基于 Gated DeltaNet 但 α_t 改 channel-wise + lower-bounded decay，详见 `wiki/kda/` |
| chunkwise 并行算法 | S4 | 把序列切成 chunk，chunk 内并行计算、chunk 间递归传状态的训练算法 |
| WY 表示 | S4 | Bischof & Van Loan 1985 提出的 Householder 矩阵乘积紧凑形式，Gated DeltaNet 并行算法的基础；本页不展开 |
| Tensor Core | S4 | GPU 上专门做半精度矩阵乘法的硬件单元，chunk size 需为 16 的倍数才能用 |
| 滑动窗口注意力 / SWA | S5 | 混合架构 H1/H2 中与 Gated DeltaNet 组合的局部注意力，本页一句话提及 |
| 混合架构 H1/H2 | S5 | Gated DeltaNet + SWA（H1）或 Gated DeltaNet + Mamba2 + SWA（H2），本页一句话提及 |
| S-NIAH | S5 | Single-Needle in a Haystack，单针检索基准，Yang 2025 ICLR Table 3 |
| data-dependent / 数据相关 | S2 | α_t、β_t 由当前输入 x_t 经 sigmoid 计算得到，而非固定常数 |
| data-independent decay | S2（对比） | 早期线性注意力（如 RetNet）用固定常数 γ 衰减，非数据相关 |

## 符号

| 符号 | 首次出现 | 定义或含义 |
|---|---|---|
| $S_t$ | S1 | 第 t 步的记忆状态矩阵 $\in \mathbb{R}^{d_v \times d_k}$ |
| $S_{t-1}$ | S1 | 上一步的状态矩阵 |
| $d_v$ | S2 | value 维度 |
| $d_k$ | S2 | key 维度；本页默认 $d_v = d_k = d$ |
| $I$ | S2 | 单位矩阵，与 $k_t$ 同维 $\in \mathbb{R}^{d_k \times d_k}$ |
| $k_t$ | S2 | 第 t 步的 key 向量 $\in \mathbb{R}^{d_k}$，由 $x_t$ 经 $W_K$ 投影 |
| $v_t$ | S2 | 第 t 步的 value 向量 $\in \mathbb{R}^{d_v}$，由 $x_t$ 经 $W_V$ 投影 |
| $q_t$ | S2 | 第 t 步的 query 向量 $\in \mathbb{R}^{d_k}$，由 $x_t$ 经 $W_Q$ 投影 |
| $o_t$ | S2 | 第 t 步的输出 $= S_t q_t$ |
| $\alpha_t$ | S2 | Gated DeltaNet 的数据相关标量衰减门 $\in (0,1)$，由 sigmoid 输出，控全局衰减 |
| $\beta_t$ | S2 | 数据相关标量写入强度 $\in (0,1)$，由 sigmoid 输出，控单点覆写 |
| $\sigma(z)$ | S2 | sigmoid 函数 $= 1/(1+e^{-z})$ |
| $W_Q, W_K, W_V, W_\alpha, W_\beta$ | S2 | 可学习参数矩阵 |
| $x_t$ | S2 | 第 t 步的输入 |
| $\gamma_{[t]}^j$ | S4（折叠） | chunkwise 形式中的局部累积衰减乘积 $\prod \alpha_i$，本页只提及不展开 |
| $\Gamma_{[t]}$ | S4（折叠） | chunkwise 形式中的衰减矩阵 $(\Gamma_{[t]})_{ij} = \gamma_i \gamma_j$，本页只提及不展开 |

## 符号一致性约束

- $\alpha_t$ 在本页始终指**标量**（Gated DeltaNet 形式）；KDA 的 channel-wise $\alpha_t$ 在 S5 明确标注为"KDA 的 α_t"以区分。
- $\beta_t$ 在本页始终指**标量**写入强度，与 `wiki/delta-rule/` 一致。
- $S_t$ 形状 $\mathbb{R}^{d_v \times d_k}$，与 `wiki/delta-rule/` 一致（行数 $d_v$、列数 $d_k$）。
- 输出 $o_t = S_t q_t$，与 `wiki/delta-rule/` 一致。

# MLA 术语表

全文所有首次出现的术语、缩写和符号。保证全文含义一致。

## 缩写与机制名

| 名称 | 首次出现 | 定义/含义 |
|---|---|---|
| MLA | 页面开头 | Multi-head Latent Attention，多头潜注意力。DeepSeek-V2 §2.1 提出的注意力机制，把每 token 的 K/V 压成低维潜向量缓存。 |
| MHA | S1 | Multi-Head Attention，标准多头注意力。每个头独立产生一份 K、V。 |
| GQA | S4 | Grouped-Query Attention。n_g 组 query 头共享一组 K/V。 |
| MQA | S4 | Multi-Query Attention。GQA 的 n_g=1 特例，所有头共享一组 K/V。 |
| KDA | S5 | Kimi K3 中的 Kernelized Delta Attention，线性注意力族的一种。本文只作为 K3 混合架构对照提及。 |
| NoPE | S5 | No Position Encoding，不施加位置编码。K3 §2.1.2 第二段说明 K3 的 MLA 层用 NoPE。 |
| RoPE | S3 | Rotary Position Embedding，旋转位置编码。把位置 t 编码为旋转矩阵 R_t 作用在 q 和 k 上。详见 rope 概念页（占位）。 |
| KV cache | S1 | 自回归推理时为前序 token 保留的 K 和 V（或 MLA 下的 c_t^{KV} 与 k_t^R），用于下一步注意力计算。 |

## 符号（按首次出现顺序）

| 符号 | 首次出现 | 含义 | 形状（DeepSeek-V2 / 贯穿例子） |
|---|---|---|---|
| $\mathbf{h}_t$ | S1 | 第 t 个 token 在注意力层的输入向量 | $\mathbb{R}^d$（d=5120 / 4） |
| $d$ | S1 | 嵌入维度（hidden size） | 5120 / 4 |
| $n_h$ | S1 | 注意力头数 | 128 / 2 |
| $d_h$ | S1 | 每个注意力头的维度 | 128 / 4 |
| $l$ | S1 | Transformer 层数（cache 计算用） | 60 / 1 |
| $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ | S1 | MHA 的 query/key/value（所有头拼接） | $\mathbb{R}^{d_h n_h}$ |
| $W^Q, W^K, W^V$ | S1 | MHA 的 q/k/v 投影矩阵 | $\mathbb{R}^{d_h n_h \times d}$ |
| $W^O$ | S1 | 输出投影矩阵 | $\mathbb{R}^{d \times d_h n_h}$ |
| $\mathbf{c}_t^{KV}$ | S1 | KV 联合压缩的潜向量 | $\mathbb{R}^{d_c}$（512 / 3） |
| $d_c$ | S1 | KV 压缩维度，$d_c \ll d_h n_h$ | 512 / 3 |
| $W^{DKV}$ | S1 | KV 下投影矩阵 | $\mathbb{R}^{d_c \times d}$ |
| $\mathbf{k}_t^C, \mathbf{v}_t^C$ | S1 | content 部分的 key/value（由 c_t^{KV} 重建） | $\mathbb{R}^{d_h n_h}$ |
| $W^{UK}, W^{UV}$ | S1 | K/V 上投影矩阵 | $\mathbb{R}^{d_h n_h \times d_c}$ |
| $\mathbf{c}_t^Q$ | S1 | Query 压缩的潜向量 | $\mathbb{R}^{d_c'}$（1536 / 教学中略） |
| $d_c'$ | S1 | Query 压缩维度 | 1536 |
| $W^{DQ}, W^{UQ}$ | S1 | Query 下/上投影矩阵 | $\mathbb{R}^{d_c' \times d}$、$\mathbb{R}^{d_h n_h \times d_c'}$ |
| $\mathbf{q}_{t,i}, \mathbf{k}_{t,i}, \mathbf{v}_{t,i}$ | S1 | 第 i 个头的 q/k/v | $\mathbb{R}^{d_h}$ |
| $\mathbf{o}_{t,i}$ | S1 | 第 i 个头的注意力输出 | $\mathbb{R}^{d_h}$ |
| $\mathbf{u}_t$ | S1 | 注意力层最终输出（W^O 之前拼接形式） | $\mathbb{R}^d$ |
| $\mathbf{q}'$ | S2 | 矩阵吸收后的等价 query，$\mathbf{q}' = W^{UK T} \mathbf{q}$ | $\mathbb{R}^{d_c}$ |
| $R_t$ | S3 | 位置 t 对应的 RoPE 旋转矩阵（不同 t 不同） | $\mathbb{R}^{d_h^R \times d_h^R}$ |
| $\mathbf{q}_t^R$ | S3 | 承载 RoPE 的多头 query（每头一份） | $\mathbb{R}^{d_h^R n_h}$ |
| $\mathbf{k}_t^R$ | S3 | 承载 RoPE 的共享 key（所有头共享同一份） | $\mathbb{R}^{d_h^R}$ |
| $d_h^R$ | S3 | 解耦 query/key 的每头维度 | 64 / 2 |
| $W^{QR}$ | S3 | 解耦 query 投影矩阵 | $\mathbb{R}^{d_h^R n_h \times d_c'}$ |
| $W^{KR}$ | S3 | 解耦 key 投影矩阵 | $\mathbb{R}^{d_h^R \times d}$ |
| $[\cdot; \cdot]$ | S3 | 向量拼接（concatenation） | — |
| $\tilde{\mathbf{o}}_t$ | S5 | K3 中未 gate 的 MLA 输出（即 MHA 公式的 $\mathbf{u}_t$） | $\mathbb{R}^d$ |
| $W^g$ | S5 | K3 output gate 的投影矩阵，满秩 | $\mathbb{R}^{d \times d}$ |
| $y_t$ | S5 | K3 MLA 层最终输出 | $\mathbb{R}^d$ |

## 易混淆记号说明

- $\mathbf{c}_t^{KV}$ 与 $\mathbf{c}_t^Q$ 是两个不同潜向量：前者缓存（cache 来源），后者不缓存（每次前向重算，只为 S2 矩阵吸收服务）。
- $\mathbf{k}_t^R$ 无下标 i——所有头共享同一份；$\mathbf{q}_t^R$ 有下标 i——每头一份。这是 DeepSeek-V2 §2.1.3 的设计，正文 S3 须明示。
- $\tilde{\mathbf{o}}_t$（K3 ungated output）等于 MHA 公式的 $\mathbf{u}_t$；K3 加 gate 后才记为 $y_t$。S5 引入。
- $d_c$ 与 $d_c'$：$d_c$ 是 KV 压缩维度（决定 cache 大小），$d_c'$ 是 Query 压缩维度（不进 cache，只服务吸收）。两者独立。

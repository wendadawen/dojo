# DSA 术语表

全文术语、缩写与符号以本表为准，含义不得漂移。

## 缩写

| 缩写 | 全称 | 首次出现位置 | 含义 |
|---|---|---|---|
| DSA | DeepSeek Sparse Attention | 页面开头 | 本页概念本体；一种可训练的细粒度稀疏注意力机制 |
| MLA | Multi-head Latent Attention | 第 4 章 | DeepSeek 的注意力变体，把 KV 压缩成低秩 latent；DSA 在其上实例化。引用概念页 |
| MQA | Multi-Query Attention | 第 4 章 | 全部 query 头共享同一份 key/value；此处指 MLA 的 MQA 模式。引用概念页 |
| MHA | Multi-Head Attention | 第 4 章 | 标准多头注意力；此处也指 MLA 的 MHA 模式与短序列的 masked MHA 路径 |
| KV cache | key-value cache | 第 1 章 | 推理时缓存的历史 key/value 条目 |
| FP8 | 8 位浮点 | 第 3 章 | 低精度数值格式，indexer 在此精度下计算。引用概念页 |
| KL | Kullback-Leibler divergence | 第 5 章 | 衡量两个概率分布差异的量；此处作为 indexer 的训练损失 |
| RoPE | Rotary Position Embedding | 第 6 章 | 旋转位置编码；indexer 使用独立的 RoPE 通道。引用概念页 |
| NSA | Native Sparse Attention | 第 8 章 | 另一篇论文提出的可训练稀疏注意力；DSA 引用其 kernel 层约束 |
| DCP | decode context parallel | 第 6 章 | vLLM 中把 KV cache 按 rank 切分的并行方式 |
| PD | prefill-decode | 不出现 | 本页不使用该缩写 |

## 符号

| 符号 | 首次出现位置 | 含义 |
|---|---|---|
| $L$ | 第 1 章 | 序列长度 |
| $t$ | 第 3 章 | 当前 query token 的位置下标 |
| $s$ | 第 3 章 | 被打分的历史位置下标，取遍 $t$ 之前（含 $t$ 自身） |
| $\mathbf{h}_t$ | 第 3 章 | 位置 $t$ 的 hidden state，$\mathbf{h}_t \in \mathbb{R}^d$ |
| $d$ | 第 3 章 | hidden state 维度 |
| $I_{t,s}$ | 第 3 章 | index score，lightning indexer 给出的位置 $s$ 对查询 $t$ 的相关性分数 |
| $H^I$ | 第 3 章 | indexer 头数。V3.2 中为 64 |
| $j$ | 第 3 章 | indexer 头的索引，$j = 1, \ldots, H^I$ |
| $d^I$ | 第 3 章 | indexer 每头的维度。V3.2 中为 128 |
| $\mathbf{q}^I_{t,j}$ | 第 3 章 | indexer 第 $j$ 头在位置 $t$ 的 query 向量，$\in \mathbb{R}^{d^I}$，由 $\mathbf{h}_t$ 导出 |
| $\mathbf{k}^I_s$ | 第 3 章 | indexer 在位置 $s$ 的 key 向量，$\in \mathbb{R}^{d^I}$，由 $\mathbf{h}_s$ 导出。注意 key 不分头，全部 indexer 头共用同一个 $\mathbf{k}^I_s$ |
| $w^I_{t,j}$ | 第 3 章 | 第 $j$ 头的标量权重，$\in \mathbb{R}$，由 $\mathbf{h}_t$ 导出 |
| $k$ | 第 4 章 | 每个 query token 选取的 KV token 数。V3.2 中为 2048。与 key 向量的字母 k 区分：本页凡表示选取数量时写作斜体 $k$，表示 key 时一律带上标与下标写作 $\mathbf{k}^I_s$ |
| $\mathbf{c}_s$ | 第 4 章 | 位置 $s$ 的 key-value 条目；在 V3.2 中即 MLA 的 latent 向量 |
| $\mathbf{u}_t$ | 第 4 章 | 位置 $t$ 的注意力输出 |
| $\mathrm{Top\text{-}k}(I_{t,:})$ | 第 4 章 | 对 $I_{t,:}$ 取最大的 $k$ 个分数构成的集合 |
| $\mathcal{S}_t$ | 第 5 章 | 位置 $t$ 选中的位置集合，$\mathcal{S}_t = \{s \mid I_{t,s} \in \mathrm{Top\text{-}k}(I_{t,:})\}$ |
| $p_{t,:}$ | 第 5 章 | warm-up 的目标分布，由主注意力分数跨头求和后沿序列维 L1 归一化得到，$\in \mathbb{R}^t$ |
| $\mathcal{L}^I$ | 第 5 章 | indexer 的训练损失（KL 散度） |
| $D_{\mathrm{KL}}(\cdot \| \cdot)$ | 第 5 章 | KL 散度，第一个参数为目标分布 |
| $O(\cdot)$ | 第 1 章 | 复杂度记号 |

## 需要保持一致的表述

| 对象 | 统一说法 | 禁止的说法 |
|---|---|---|
| lightning indexer | 统一写作"lightning indexer"或"indexer"，中文语境可写"闪电索引器"但需在首次出现时给出英文原名 | 不写"索引头""打分头""侦察兵" |
| fine-grained token selection | 统一写作"细粒度 token 选择"或"top-k 选择" | 不写"token 筛选器""稀疏门控" |
| index score | 统一写作"index score"或"相关性分数" | 不写"注意力分数"（会与主注意力分数混淆） |
| 主注意力 | 指 MLA 那一路完整注意力，统一写作"主注意力" | 不写"真注意力""正式注意力" |
| 稀疏训练阶段 | 统一写作"sparse training 阶段" | 不与"续训"混用指代同一阶段（续训指两阶段整体） |
| 续训 | 指 dense warm-up 与 sparse training 两阶段整体 | 不用它单指其中一个阶段 |
| DSA 的复杂度 | 主注意力 $O(Lk)$，indexer $O(L^2)$，须同时出现 | 不单说"DSA 是线性复杂度" |
| KV cache 与 DSA 的关系 | 统一表述为"DSA 不减少 KV cache" | 不写"DSA 降低显存占用" |

## 版本口径

| 名称 | 含义 |
|---|---|
| DeepSeek-V3.1-Terminus | DSA 续训的起点模型，V3.1 的最后一个版本 |
| DeepSeek-V3.2-Exp | 首次公开 DSA 的实验版本（2025 年 9 月） |
| DeepSeek-V3.2 | 正式版；架构与 V3.2-Exp 完全相同 |

本页讲机制时不区分 V3.2-Exp 与 V3.2；涉及具体评测时间或实现仓库时按 evidence.md 的来源标注。

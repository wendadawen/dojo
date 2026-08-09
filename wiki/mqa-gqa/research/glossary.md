# MQA 与 GQA 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致。

## 术语与缩写

| 术语 / 缩写 | 首次出现 | 定义或含义 |
|---|---|---|
| MHA（Multi-Head Attention，多头注意力） | S1 | Vaswani 2017 的标准注意力，h 个 query 头各有独立 K/V 头；本页基线，前置页 standard-attention 已讲清 |
| MQA（Multi-Query Attention，多查询注意力） | 页面开头 | Shazeer 2019 提出，所有 query 头共享同一组 K/V；"multi" 修饰 query 头数量保持多头，不是"增加 query 头" |
| GQA（Grouped-Query Attention，分组查询注意力） | 页面开头 | Ainslie 2023 提出，query 头分 G 组、每组共享一组 K/V；MHA（G=h）与 MQA（G=1）的插值 |
| GQA-G | S3 | 表示有 G 个 KV 组的 GQA；GQA-1=MQA、GQA-h=MHA |
| KV cache | S1 | 自回归推理时缓存的前序 token 的 K 和 V 张量，供后续 token 的注意力查询使用；大小 $2 h d_k n l$（MHA） |
| uptraining | S3 | Ainslie 2023 提出的从已有 MHA 检查点转 GQA/MQA 的配方：均值池化 K/V 投影 + 5% 继续预训练 |
| mean pooling（均值池化） | S3 | uptraining 第一步：把 MHA 检查点里 h 个 K/V 投影矩阵按组取均值得到 G 个；优于选第一个或随机初始化 |
| α（alpha） | S3 | uptraining 继续预训练的计算量比例，论文取 α=0.05（5%） |
| 算术强度（arithmetic intensity） | S1 | FLOP/byte 比值，衡量每字节数据对应的计算量；低于 GPU 脊点时受带宽限制 |
| HBM（High Bandwidth Memory） | S1 | GPU 的高带宽显存，KV cache 存于此、每步解码加载 |
| MLA（Multi-head Latent Attention） | S5 | DeepSeek-V2 提出的另一种 KV 压缩注意力，走低秩压缩而非共享头；本页是其前置 |
| 增量解码 / 自回归推理（incremental decoding） | S1 | 逐 token 生成、每步只能看到已生成部分的推理方式；与训练时的全序列并行相对 |
| roofline 模型 | S1（折叠） | 描述算术强度与峰值算力关系的性能模型，算术强度低于脊点时受带宽限制 |

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $h$ | S1 | query 头数（MHA 与 MQA 的 query 头都为 h） |
| $G$ | S3 | GQA 的 KV 组数；$1\le G\le h$，$G=1$ 为 MQA、$G=h$ 为 MHA |
| $d$ | S1（折叠） | 模型隐藏维度 $d_{model}$ |
| $d_k$ | S1 | 每个 head 的 key 维度（通常也等于 value 维度 $d_v$） |
| $n$ | S1 | 当前序列长度（已生成的 token 数） |
| $l$ | S1 | Transformer 层数 |
| $b$ | S1（折叠） | batch size |
| $P^Q,P^K,P^V,P^O$ | S2 | MHA 的 query/key/value/output 投影张量；Shazeer 2019 §2.2 的 einsum 记法 |
| $X$ | S2（引用前置） | 输入序列，$\mathbb{R}^{n\times d}$ |
| $d_c$ | S5 | MLA 的压缩潜向量维度（对照用，不展开机制） |
| $d_h^R$ | S5 | MLA 的解耦 RoPE 维度（对照用，不展开机制） |
| $\Theta(\cdot)$ | S1 | 渐近上界记法，描述随参数增长的方式 |
| $2 h d_k$ | S1 | MHA 每 token 每 layer 的 KV cache 元素数（h 个 K + h 个 V，各 $d_k$ 维） |
| $2 d_k$ | S2 | MQA 每 token 每 layer 的 KV cache 元素数（1 个 K + 1 个 V） |
| $2 G d_k$ | S3 | GQA 每 token 每 layer 的 KV cache 元素数（G 个 K + G 个 V） |
| $(d_c + d_h^R)$ | S5 | MLA 每 token 每 layer 的 KV cache 元素数（潜向量 + 解耦 RoPE key） |

## 来源简称

| 简称 | 完整引用 |
|---|---|
| Shazeer 2019 | Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150, 2019. |
| Ainslie 2023 | Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023, arXiv:2305.13245. |
| Vaswani 2017 | Vaswani et al. "Attention Is All You Need." NeurIPS 2017, arXiv:1706.03762.（前置页 standard-attention 的主要依据） |
| DeepSeek-V2 | DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." 2024. §2.1（MLA 提出）。 |

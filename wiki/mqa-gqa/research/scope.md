# MQA 与 GQA 内容范围

## 1. 概念歧义处理

- 名称：Multi-Query Attention（MQA）与 Grouped-Query Attention（GQA）。
- 同名问题：MQA 在某些语境下可指 Multi-Question Answering（多问题问答任务），但在 Transformer 注意力机制语境下无歧义，指 Shazeer 2019 提出的共享 K/V 头注意力变体。GQA 在注意力机制语境下无歧义，指 Ainslie 2023 提出的分组查询注意力。
- 缩写混用：MQA 中的 "Multi-Query" 容易被读成"多个查询"，实际含义是"多个 query 头共享同一组（单一的）key/value"——"multi" 修饰的是 query 头数量，"query" 指查询头侧保持多头。GQA 中的 "Grouped" 指 query 头被分组、每组共享一组 K/V。
- 裁定：本文采用 Shazeer 2019 arXiv:1911.02150 §3 的 MQA 定义，与 Ainslie 2023 arXiv:2305.13245 §2 的 GQA 定义。两者同一谱系（K/V 头数从 h 到 1 的连续插值），合并在一页讲。

## 2.1 概念含义

- 概念名称：Multi-Query Attention（MQA，多查询注意力）与 Grouped-Query Attention（GQA，分组查询注意力）。
- 一句话定义：MQA 让所有 query 头共享同一组 key/value 头，GQA 把 query 头分成 G 组、每组共享一组 key/value 头，两者都以减少自回归推理时需要加载的 KV cache 大小为目标，GQA 是 MHA 与 MQA 之间的插值。
- 正式定义：
  - MQA：Shazeer 2019 §3——"Multi-query attention is identical except that the different heads share a single set of keys and values." 即 $P^Q\in\mathbb{R}^{h\times d\times k}$、$P^O\in\mathbb{R}^{h\times d\times v}$ 保持多头，但 $P^K\in\mathbb{R}^{d\times k}$、$P^V\in\mathbb{R}^{d\times v}$ 去掉头维度，全模型只有一组 K 和一组 V。
  - GQA：Ainslie 2023 §2——"Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head." $P^K,P^V\in\mathbb{R}^{G\times d\times k/v}$，G 个组各一组 K/V；GQA-1 等价于 MQA，GQA-h（$G=h$）等价于 MHA。
- 本文语境：以 Shazeer 2019 的内存带宽分析为动机起点，讲清 MQA 机制与代价；再用 Ainslie 2023 的 GQA 讲插值与 uptraining 配方；最后用 MLA 的前置视角点出"共享头"与"低秩压缩"是两条不同的 KV 压缩路线。本文是 MLA 概念页的前置。

### 包括什么

- 自回归推理时 MHA 的内存带宽瓶颈：KV cache 随头数 h 与序列长度 n 线性增长，每步解码都要把整个 cache 从 HBM 加载一遍，算术强度低（Shazeer 2019 §2.4.1 性能分析）。
- MQA 的机制：共享一组 K/V，把每 token 的 KV cache 从 $2 h d_k$ 降到 $2 d_k$（Shazeer 2019 §3）。
- MQA 的代价：质量轻微下降、大模型上训练不稳定（Ainslie 2023 §1、Appendix A）。
- GQA 的机制：query 头分 G 组、每组共享一组 K/V，KV cache 为 $2 G d_k$（Ainslie 2023 §2）。
- GQA 的 uptraining 配方：从已有 MHA 检查点出发，把 h 个 K/V 投影矩阵均值池化为 G 个，再用原预训练计算量的 α=5% 继续预训练（Ainslie 2023 §2.1、§3.1）。
- GQA 的实验结论：GQA-8 在 T5-XXL 上质量接近 MHA、速度接近 MQA（Ainslie 2023 Table 1、Figure 6）。
- 手算对比：4 头 MHA / 2 组 GQA / 1 组 MQA 的每 token KV cache 元素数，展示 MHA→GQA→MQA 是 K/V 头数从 h 到 1 的连续谱。
- MHA→GQA→MQA→MLA 的关系定位：前三者在"共享 K/V 头"这一条路上，MLA 走"低秩压缩"另一条路（DeepSeek-V2 §2.1.4 Table 1 的对照）。

### 不包括什么

- MLA 的内部机制（压缩公式、矩阵吸收、解耦 RoPE）：属 MLA 概念页，本文只在最后一节点出"共享头 vs 低秩压缩"的区别，不展开 MLA 机制。
- Flash Attention / Linear Attention 的机制：与 MQA/GQA 正交（Flash 改 IO 实现不改公式，Linear 改公式降阶），本文只在边界处一句话对照。
- 训练 kernel 实现（KV 广播、PagedAttention、FlashAttention 对 MQA 的适配）：属 GPU 工程页。
- 具体模型采用情况表（PaLM/Falcon/Llama 用了哪种）：只在文末一句话提及 GQA-8 已成主流，不展开各模型配置。

### 相邻概念

- Multi-head Attention（MHA）：MQA/GQA 的基线，前置概念 standard-attention 已讲清。本文复用其 $Q=XW^Q,K=XW^K,V=XW^V$ 与多头公式。
- Multi-head Latent Attention（MLA）：同为 KV cache 压缩目标，但走低秩压缩而非共享头。后续概念，本文最后一点做前置衔接，不展开。
- Flash Attention：改 IO 实现不改公式，与 MQA/GQA 正交，可叠加。不纳入。
- Linear Attention：改 softmax 为核分解降阶，与共享头无关。不纳入。

## 2.2 学习目标

### Q1：为什么标准 MHA 在自回归推理时受内存带宽限制——KV cache 如何随头数与序列长度增长？

- 完成答案：读者应能说明——训练时整个序列并行计算、KV 可一次性算出；自回归推理时每生成一个 token 都要把之前所有 token 的 K/V 从 HBM 加载一遍，这部分数据量为 $2 h d_k n l$（h 头、$d_k$ 每头维度、n 序列长、l 层），随 n 与 h 线性增长；Shazeer 2019 §2.4.1 的性能分析给出内存访问与算术的比值为 $\Theta(n/d + 1/b)$，当 n 接近 d 或 batch b 小时比值接近 1，GPU 在等内存而非算；MQA/GQA 减的是这个比值里的 n/d 项（通过减小 h 的系数）。
- 为什么是核心目标：不理解瓶颈在哪就无法理解 MQA/GQA 在减什么；不理解"带宽受限而非算力受限"就会误以为减少计算量才能加速。
- 依赖内容：MHA 的多头结构与 KV 来源（前置页 standard-attention）、自回归推理的逐 token 生成流程、Shazeer 2019 的性能分析。

### Q2：MQA 如何通过共享一组 K/V 减少 KV cache，代价是什么？

- 完成答案：读者应能写出 MQA 的张量形状——$P^Q\in\mathbb{R}^{h\times d\times d_k}$、$P^O\in\mathbb{R}^{h\times d\times d_v}$ 保持多头，但 $P^K\in\mathbb{R}^{d\times d_k}$、$P^V\in\mathbb{R}^{d\times d_v}$ 去掉头维度；解释"所有 query 头读同一组 K 和 V"如何把每 token 每层 cache 从 $2 h d_k$ 降到 $2 d_k$（减少 h 倍）；并指出 Shazeer 2019 报告的代价——质量轻微下降，Ainslie 2023 Appendix A 进一步指出大模型上 MQA 训练不稳定。
- 为什么是核心目标：MQA 是 GQA 的极端（G=1），不理解 MQA 就无法理解 GQA 在插值什么。
- 依赖内容：MHA 的多头投影公式、KV cache 的来源（Q1）、Shazeer 2019 §3 的定义。

### Q3：GQA 如何在 MHA 与 MQA 之间插值，uptraining 如何从已有 MHA 检查点得到 GQA？

- 完成答案：读者应能说明——GQA 把 h 个 query 头分成 G 组，每组共享一组 K/V，KV cache 为 $2 G d_k$；$G=h$ 时等价于 MHA，$G=1$ 时等价于 MQA，$1<G<h$ 是中间插值。uptraining 两步：（a）把 MHA 检查点里 h 个 K/V 投影矩阵按组均值池化成 G 个（Ainslie 2023 §2.1 指出均值池化优于选第一个或随机初始化）；（b）用原预训练计算量的 α=5% 继续预训练（Ainslie 2023 §3.1）。Ainslie 2023 在 T5-XXL 上实验：GQA-8 质量 47.1 接近 MHA 47.2、速度 0.28s 接近 MQA 0.24s（Table 1），G=8 是选定的折中点（Figure 6）。
- 为什么是核心目标：GQA 是当前主流 LLM 的实际选择，不理解 uptraining 就无法理解"为什么不用 MQA 直接训"和"已有 MHA 模型如何升级"。
- 依赖内容：MHA 与 MQA 的机制（Q1、Q2）、uptraining 的两步流程、Ainslie 2023 的实验数据。

### Q4：手算 4 头 MHA / 2 组 GQA / 1 组 MQA 的每 token KV cache，并说明三者的连续谱关系？

- 完成答案：读者应能用固定 $h=4$、$d_k=64$、$l=1$ 层的教学数字算出：MHA 每 token cache = $2\times 4\times 64 = 512$ 元素；GQA-2 每 token cache = $2\times 2\times 64 = 256$ 元素；MQA 每 token cache = $2\times 1\times 64 = 128$ 元素；10 个 token 时 MHA=5120、GQA-2=2560、MQA=1280。并说明三者是 K/V 头数从 h 到 1 的连续谱，GQA 的 G 是谱上的旋钮，每减一个 K/V 头 cache 减 $2 d_k$。
- 为什么是核心目标：把"减少 h 倍"落到可手算的数字，是判断 MQA/GQA 是否值得复杂度的依据。
- 依赖内容：KV cache 的元素构成（每头一份 K 与一份 V）、三种机制的 K/V 头数。

### Q5：MQA/GQA 与 MLA 的根本区别是什么，为什么说 MQA/GQA 是理解 MLA 的前置？

- 完成答案：读者应能指出——MQA/GQA 通过"共享 K/V 头"减 cache，被缓存的每个 K/V 仍是完整的 $d_k$ 维 head 向量，只是头数从 h 降到 G 或 1；MLA 不共享头，而是把所有头的 K/V 联合压成一个 $d_c$ 维潜向量 $c_t^{KV}$，推理时再用学习到的上投影重建各头的 K/V（DeepSeek-V2 §2.1.2）。机制完全不同：共享头是"少存几份完整 head"，低秩压缩是"把所有 head 信息压到一个低维潜向量"。MQA/GQA 是前置，因为理解了"KV cache 来自每头一份 K 和 V"才能理解 MLA 在压缩什么；也理解了"共享头有质量代价"才能理解 MLA 为什么另起一条路。
- 为什么是核心目标：本页的最终定位是 MLA 前置，不点出区别读者无法把本页与 MLA 页衔接。
- 依赖内容：MQA/GQA 的机制（Q2、Q3）、MLA 的对照定位（DeepSeek-V2 §2.1.4 Table 1，只引对照不展开机制）。

## 2.3 内容分级

### 核心内容（缺一不可，对应学习目标）

- 自回归推理时 MHA 的 KV cache 来源与内存带宽瓶颈——Q1 直接依赖；结论：每 token 每层 cache $2 h d_k$，每步解码全量加载，瓶颈在带宽不在算力。
- Shazeer 2019 §2.4.1 的性能分析 $\Theta(n/d + 1/b)$——Q1 直接依赖；结论：n/d 项是 h 的系数，MQA 减的就是它。
- MQA 的张量形状定义与共享机制——Q2 直接依赖；结论：K/V 去掉头维度，全模型一组。
- MQA 的 KV cache 公式 $2 d_k l$（每 token）——Q2、Q4 直接依赖。
- MQA 的代价（质量下降、训练不稳定）——Q2 直接依赖。
- GQA 的分组机制与 KV cache 公式 $2 G d_k l$——Q3、Q4 直接依赖。
- GQA-1=MQA、GQA-h=MHA 的插值关系——Q3、Q4 直接依赖。
- uptraining 两步配方（均值池化 + 5% 继续预训练）——Q3 直接依赖。
- Ainslie 2023 Table 1 的 T5-XXL 实验数字（MHA/MQA/GQA-8 的速度与质量）——Q3 直接依赖。
- G=8 折中点的选择依据（Figure 6）——Q3 直接依赖。
- 4 头 MHA / 2 组 GQA / 1 组 MQA 的手算对比——Q4 直接依赖。
- MHA→GQA→MQA 的连续谱关系——Q4 直接依赖。
- MQA/GQA（共享头）与 MLA（低秩压缩）的机制区别——Q5 直接依赖。
- DeepSeek-V2 §2.1.4 Table 1 四种机制 cache 公式对照——Q5 直接依赖。

### 辅助内容（消除关键理解障碍）

- 训练并行 vs 推理串行的对比——Q1 的辅助可视化，让读者理解为什么训练快推理慢。
- 内存带宽 vs 算力的增长剪刀差（GPU FLOPS 增长快于带宽）——Q1 的背景，说明瓶颈为什么越来越严重。
- "广播 K/V"的工程实现问题——Q2 的辅助，说明 MQA 在 kernel 层需要专门适配，但属工程细节不展开。
- 均值池化 vs 选第一个 vs 随机初始化的消融（Ainslie 2023 Figure 4）——Q3 的辅助，说明 uptraining 配方的选择依据。
- MQA 训练不稳定性（Ainslie 2023 Appendix A）——Q2、Q3 的辅助，说明 GQA 的一部分动机。

### 扩展内容

- 纳入：GQA-8 已成主流 LLM 标配（Llama 2 70B、Llama 3、Mistral 等采用）——一句话背景，不展开各模型配置。
- 排除：MLA 的压缩公式、矩阵吸收、解耦 RoPE——属 MLA 概念页。
- 排除：Flash Attention 的分块 IO 机制——属 GPU 工程页。
- 排除：Linear Attention 的核分解推导——属另一概念页。
- 排除：各模型的具体 num_heads / num_kv_heads 配置表——属模型架构页。

## 2.4 前置知识映射

| 前置概念 | 被哪些目标依赖 | 概念页状态 | 递归层级 |
|---|---|---|---|
| standard-attention（标准 MHA） | Q1（KV cache 来源、多头投影）、Q2（MQA 改了什么）、Q4（cache 元素构成）、Q5（与 MLA 对照的基线） | 已生成 wiki/standard-attention/index.html | 第 0 层（已有） |
| mla（Multi-head Latent Attention） | Q5（对照区别） | 已生成 wiki/mla/index.html，本页作为其前置 | 第 0 层（已有，仅引用对照） |

注：本页是 MLA 概念页的前置。MLA 页的 scope.md §2.4 已将 mqa-gqa 登记为"未生成，占位提示"——本页生成后该占位可由 MLA 页的编排者后续替换为正式链接。本文不展开 MLA 机制，只在 Q5 与最后一节点出区别并给出 MLA 页链接。

## 2.5 明确不展开的内容

- MLA 的低秩压缩公式与矩阵吸收：与"共享头"是另一条路，属 MLA 概念页；本文只在 Q5 与最后对照表给出 cache 公式层面的区别。
- Flash Attention 的分块 IO：与 MQA/GQA 正交（一个改实现、一个改张量形状），可叠加；属 GPU 工程页。
- Linear Attention 的核分解：与 KV cache 压缩无关，属另一概念页。
- 训练 kernel 适配（K/V 广播、PagedAttention 对 MQA 的支持）：属 GPU 工程页。
- 各 LLM 的具体配置（Llama 2 70B 的 num_kv_heads=8 等）：属模型架构页，本文只在文末一句话提及 GQA-8 成为主流。
- Shazeer 2019 的具体 BLEU 与速度倍数：PDF 一手数据未取到精确数值，本文只引抽象结论（"much faster...minor quality degradation"）与分析结果（$\Theta(n/d+1/b)$ 减少 h 倍），不引用未核实的二手数字。

## 2.6 常见误解和适用边界

### 误解 M1

- 错误理解：MQA 是"多个 query 头"的注意力，比 MHA 头更多。
- 正确结论：MQA 的 "multi-query" 指 query 头保持多头（与 MHA 一样多），但 K/V 头缩减为 1 组被所有 query 头共享。"multi" 修饰 query 头数量不变，不是"增加 query 头"。MHA 与 MQA 的 query 头数都是 h，区别只在 K/V 头数（h vs 1）。
- 形成原因：名称字面意思容易被读成"多查询"。
- 影响目标：Q2。

### 误解 M2

- 错误理解：MQA/GQA 减少的是计算量（FLOPS）。
- 正确结论：MQA/GQA 主要减的是自回归推理时每步从 HBM 加载 KV cache 的数据量与显存占用，即内存带宽与容量。计算上每个 query 头仍要做一次 $q\cdot k$ 与 $\sum v$，只是 K/V 被多个 query 头复用——算术强度（FLOP/byte）反而提升，从带宽受限转为更接近算力受限。训练时 MQA 的总 FLOPS 与 MHA 接近（K/V 投影变小但占比小）。
- 形成原因：把"推理变快"直接归因于"算得少了"。
- 影响目标：Q1、Q2。

### 误解 M3

- 错误理解：GQA 就是 MQA 的改进版，两者并列。
- 正确结论：GQA 是 MHA 与 MQA 的插值，$G=h$ 时等价于 MHA、$G=1$ 时等价于 MQA。MQA 是 GQA 的一个端点（$G=1$），不是并列关系。MHA→GQA→MQA 是 K/V 头数从 h 到 1 的连续谱，G 是谱上的旋钮。
- 形成原因：论文标题并列，容易忽略 GQA 包含 MQA 为特例。
- 影响目标：Q3、Q4。

### 误解 M4

- 错误理解：MLA 就是 GQA 的极端（头数减到很低）。
- 正确结论：MLA 不共享头，而是把所有头的 K/V 联合压成一个 $d_c$ 维潜向量 $c_t^{KV}$，再用上投影为每个头重建 K/V。共享头（MQA/GQA）缓存的仍是完整的 $d_k$ 维 head 向量，只是份数减少；低秩压缩（MLA）缓存的是所有 head 信息的低维潜表示。机制完全不同。DeepSeek-V2 §2.1.4 Table 1 给出对照：MQA cache = $2 d_k l$、MLA cache = $(d_c + d_h^R) l$，量级不同（DeepSeek-V2 配置下 MLA 比 MQA 还小），但更重要的是 MLA 质量优于 MHA 而 MQA 质量略低于 MHA。
- 形成原因：两者目标相同（减 KV cache），容易被归为一类。
- 影响目标：Q5。

### 适用边界

- MQA/GQA 解决：自回归推理时 KV cache 随头数 h 与序列长度 n 线性增长带来的内存带宽与显存压力；保留全局 softmax 注意力结构。
- MQA/GQA 不解决：注意力本身的 $O(n^2)$ 算力——每个 query 头仍对全部前序 token 做 softmax 加权和，只是 K/V 复用。也不解决训练时的激活内存。
- 成立条件：头数 h 是 KV cache 的主要系数时才有意义（即大模型、长上下文、多头设置下）；小模型、短序列、单头时收益可忽略。
- 不满足时：若 h=1（本来就是单头），MQA 与 MHA 完全相同，无收益；若 batch 极大使得 $1/b$ 项主导 $\Theta(n/d+1/b)$，减 n/d 项的收益被稀释（这时应增大 batch 而非改注意力）。
- GQA 的 uptraining 条件：需要已有 MHA 检查点，且 α=5% 的继续预训练计算量对大模型仍非平凡（T5-XXL 约 600 TPUv3 chip-days，Ainslie 2023 §3.1）。

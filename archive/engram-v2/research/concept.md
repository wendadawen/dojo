# Engram

- 描述：Engram 是在 Transformer 层间加入的条件记忆模块，用最近几个 token 的哈希值查一张大嵌入表，再由当前隐藏状态门控后加回主干。
- 摘要：把「最近 $n$ 个 token 的哈希 → 嵌入表一行」做成与 MoE 并列的稀疏轴：查表只取表中极少数行，开销为常数，取出的向量再经上下文门控 $\alpha_t$ 写回隐藏状态。
- 主题：模型结构，内存与缓存
- 标签：模型架构

## 资料

- Xin Cheng 等，Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models，arXiv:2601.07372v2，2026，[https://arxiv.org/abs/2601.07372](https://arxiv.org/abs/2601.07372)
  - 摘录（Abstract）：While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic 𝑁 -gram embedding for O (1) lookup.
  - 摘录（Abstract）：Mechanistic analyses reveal that Engram relieves the backbone’s early layers from static reconstruction, effectively deepening the network for complex reasoning.
  - 摘录（Abstract）：Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead.
  - 摘录（1. Introduction）：Specifically, language modeling entails two qualitatively different sub-tasks: compositional reasoning and knowledge retrieval. While the former demands deep, dynamic computation, a substantial portion of text—such as named entities and formulaic patterns—is local, static, and highly stereotyped
  - 摘录（1. Introduction）：Since standard Transformers (Vaswani et al., 2017) lack a native knowledge lookup primitive, current LLMs are forced to simulate retrieval through computation. For instance, resolving a common multi-token entity requires consuming multiple early layers of attention and feed-forward networks
  - 摘录（1. Introduction）：This process essentially amounts to an expensive runtime reconstruction of a static lookup table, wasting valuable sequential depth on trivial operations that could otherwise be allocated to higher-level reasoning.
  - 摘录（1. Introduction）：Whereas conditional computation sparsely activates parameters to process dynamic logic (Bengio et al., 2013; Shazeer et al., 2017), conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge.
  - 摘录（1. Introduction）：local context serves as a key to index a massive embedding table via constant-time O (1) lookups
  - 摘录（1. Introduction）：Unlike MoE’s dynamic routing, Engram employs deterministic IDs to enable runtime prefetching, overlapping communication with computation.
  - 摘录（2.1 Overview）：the module processes each position 𝑡 in two functional phases: retrieval and fusion. First, as detailed in Section 2.2, we extract and compress suffix 𝑁 -grams to deterministically retrieve static embedding vectors via hashing. Subsequently, in Section 2.3, these retrieved embeddings are dynamically modulated by the current hidden state and refined via a lightweight convolution.
  - 摘录（2.1 Overview）：given an input sequence 𝑋 = ( 𝑥1 , . . . , 𝑥𝑇 ) and hidden states H ( ℓ ) ∈ R𝑇 × 𝑑 at layer ℓ,
  - 摘录（2.2 Tokenizer Compression）：standard subword tokenizers prioritize lossless reconstruction, often assigning disjoint IDs to semantically equivalent terms (e.g., Apple vs. ␣apple)
  - 摘录（2.2 Tokenizer Compression）：we pre-compute a surjective function P : 𝑉 → 𝑉 ′ that collapses raw token IDs into canonical identifiers based on normalized textual equivalence (using NFKC (Whistler, 2025), lowercasing, etc.).
  - 摘录（2.2 Tokenizer Compression）：Formally, for a token at position 𝑡 , we map its raw ID 𝑥𝑡 to a canonical ID 𝑥𝑡′ = P ( 𝑥𝑡 ) to form the suffix 𝑁 -gram 𝑔𝑡,𝑛 = ( 𝑥𝑡′− 𝑛+1 , . . . , 𝑥𝑡′ ).
  - 摘录（2.2 Multi-Head Hashing）：Directly parameterizing the combinatorial space of all possible 𝑁 -grams is intractable.
  - 摘录（2.2 Multi-Head Hashing）：To mitigate collisions, we employ 𝐾 distinct hash heads for each 𝑁 -gram order 𝑛. Each head 𝑘 maps the compressed context to an index within an embedding table E𝑛,𝑘 (of prime size 𝑀𝑛,𝑘 ) via a deterministic function 𝜑𝑛,𝑘 :
  - 摘录（2.2 式 1）：𝑧𝑡,𝑛,𝑘 ≜ 𝜑𝑛,𝑘 ( 𝑔𝑡,𝑛 ), e𝑡,𝑛,𝑘 = E𝑛,𝑘 [ 𝑧𝑡,𝑛,𝑘 ].
  - 摘录（2.2 Multi-Head Hashing）：In practice, 𝜑𝑛,𝑘 is implemented as a lightweight multiplicative-XOR hash. We construct the final memory vector e𝑡 ∈ R𝑑mem by concatenating all retrieved embeddings:
  - 摘录（2.2 式 2）：e𝑡 ≜ ∥ᴺₙ₌₂ ∥ᴷₖ₌₁ e𝑡,𝑛,𝑘 .
  - 摘录（2.3 Context-aware Gating）：The retrieved embeddings e𝑡 serve as context-independent priors. Being static, however, they inherently lack contextual adaptability and may suffer from noise due to hash collisions or polysemy
  - 摘录（2.3 Context-aware Gating）：we utilize the current hidden state h𝑡 —which has aggregated global context via preceding attention layers—as a dynamic Query, while the retrieved memory e𝑡 serves as the source for both Key and Value projections:
  - 摘录（2.3 式 3）：k𝑡 = W 𝐾 e𝑡 , v𝑡 = W𝑉 e𝑡
  - 摘录（2.3 Context-aware Gating）：where W 𝐾 , W𝑉 are learnable projection matrices. To ensure gradient stability (Dehghani et al., 2023), we apply RMSNorm (Zhang and Sennrich, 2019) to the Query and Key before computing the scalar gate 𝛼𝑡 ∈ (0, 1):
  - 摘录（2.3 式 4）：𝛼𝑡 = 𝜎 ( RMSNorm(h𝑡 ) ⊤ RMSNorm(k𝑡 ) / √𝑑 ).
  - 摘录（2.3 Context-aware Gating）：The gated output is defined as ṽ𝑡 = 𝛼𝑡 · v𝑡 . This design enforces semantic alignment: if the retrieved memory e𝑡 contradicts the current context h𝑡 , the gate 𝛼𝑡 tends toward zero, effectively suppressing the noise.
  - 摘录（2.3 Context-aware Gating）：Finally, to expand the receptive field and enhance the model’s non-linearity, we introduce a short, depthwise causal convolution (Gu et al., 2022; Peng et al., 2023).
  - 摘录（2.3 式 5）：Y = SiLU Conv1D(RMSNorm( Ṽ)) + Ṽ,
  - 摘录（2.3 式 5 前）：Using a kernel size 𝑤 (set to 4), dilation 𝛿 (set to the max 𝑁 -gram order) and SiLU activation (Elfwing et al., 2018), the final output Y is computed as:
  - 摘录（2.3 Context-aware Gating）：The Engram module is integrated into the backbone via a residual connection: H ( ℓ ) ← H ( ℓ ) + Y, followed by the standard Attention and MoE. Crucially, Engram is not applied to every layer
  - 摘录（2.4 Integration with Multi-branch Architecture）：we implement a parameter-sharing strategy: a single sparse embedding table and a Value projection matrix W𝑉 are shared across all 𝑀 branches, whereas 𝑀 distinct Key projection matrices {W 𝐾( 𝑚 ) } 𝑚 𝑀 =1 are employed to enable branch-specific gating behaviors.
  - 摘录（2.5 System Efficiency）：unlike MoE which relies on runtime hidden states for dynamic routing, Engram employs a deterministic retrieval mechanism based solely on input token IDs, naturally decoupling parameter storage from computation.
  - 摘录（2.5 System Efficiency）：During training, to accommodate large-scale embedding tables, we employ standard model parallelism by sharding the tables across available GPUs. An All-to-All communication primitive is used to gather active rows in the forward pass and dispatch gradients in the backward pass, enabling the total memory capacity to scale linearly with the number of accelerators.
  - 摘录（2.5 System Efficiency）：During inference, this deterministic nature enables a prefetch-and-overlap strategy. Since memory indices are known prior to the forward pass, the system can asynchronously retrieve embeddings from abundant host memory via PCIe. To effectively mask communication latency, the Engram module is placed at specific layers within the backbone, leveraging the computation of preceding layers as a buffer to prevent GPU stalls.
  - 摘录（2.5 System Efficiency）：while placing Engram deeper extends the compute window available for hiding latency, our ablation in Section 6.2 shows that modeling performance favors early intervention to offload local pattern reconstruction. Therefore, the optimal placement must simultaneously satisfy both modeling and system latency constraints.
  - 摘录（2.5 System Efficiency）：natural language 𝑁 -grams inherently follow a Zipfian distribution (Chao and Zipf, 1950; Piantadosi, 2014), where a small fraction of patterns accounts for the vast majority of memory accesses. This statistical property motivates a Multi-Level Cache Hierarchy: frequently accessed embeddings can be cached in faster storage tiers (e.g., GPU HBM or Host DRAM), while the long tail of rare patterns resides in slower, high-capacity media (e.g., NVMe SSD).
  - 摘录（3.1 Compute-matched formulation）：𝑃tot : total trainable parameters, excluding vocabulary embedding and LM head. 𝑃act : activated parameters per token. This quantity determines the training cost (FLOPs). 𝑃sparse ≜ 𝑃tot − 𝑃act : the inactive parameters, which represents the “free” parameter budget available for scaling model size without incurring computational cost (e.g., unselected experts or unretrieved embeddings).
  - 摘录（3.1 Compute-matched formulation）：For Engram, only a constant number of slots are retrieved per token, so scaling the number of embedding slots increases 𝑃tot without increasing per-token FLOPs.
  - 摘录（3.1 Allocation ratio）：We define the allocation ratio 𝜌 ∈ [0, 1] as the fraction of the inactive-parameter budget assigned to MoE expert capacity: 𝑃MoE(sparse) = 𝜌 𝑃sparse , 𝑃Engram = (1 − 𝜌) 𝑃sparse . (7)
  - 摘录（3.1 Allocation ratio）：𝜌 = 1 corresponds to a pure MoE model (all inactive parameters are routed experts). 𝜌 < 1 reduces the number of routed experts and reallocates the freed parameters to Engram embedding slots.
  - 摘录（3.1 Results and Analysis）：Figure 3 (left) reveals a consistent U-shaped relationship between validation loss and the allocation ratio 𝜌. Remarkably, the Engram model achieves comparable performance to the pure MoE baseline ( 𝜌 = 100%) even when the MoE allocation is reduced to just 𝜌 ≈ 40%
  - 摘录（3.1 Results and Analysis）：Furthermore, the pure MoE baseline proves suboptimal: reallocating roughly 20%–25% of the sparse parameter budget to Engram yields the best performance.
  - 摘录（3.1 Results and Analysis）：Crucially, the location of this optimum is stable across regimes ( 𝜌 ≈ 75%–80%), suggesting a robust allocation preference across the examined scales (under fixed sparsity).
  - 摘录（3.1 Results and Analysis）：Quantitatively, in the 10B regime (𝐶 = 6 × 1020 ), validation loss improves from 1.7248 (at 𝜌 = 100%) to 1.7109 near the optimum of 𝜌 ≈ 80% (Δ = 0.0139).
  - 摘录（3.1 Experimental protocol）：We evaluate this trade-off at two compute regimes and maintain a constant sparsity ratio 𝑃tot / 𝑃act ≈ 10 across both settings: 𝐶 = 2 × 1020 FLOPs: 𝑃tot ≈ 5.7B and 𝑃act = 568M. 𝐶 = 6 × 1020 FLOPs: 𝑃tot ≈ 9.9B and 𝑃act = 993M.
  - 摘录（3.1 Results and Analysis）：MoE-dominated ( 𝜌 → 100%): The model lacks dedicated memory for static patterns, forcing it to inefficiently reconstruct them through depth and computation. Engram-dominated ( 𝜌 → 0%): The model loses conditional computation capacity, hurting tasks that require dynamic, context-dependent reasoning; memory cannot replace computation in this regime.
  - 摘录（3.2 Results）：Across the explored range, the curve follows a strict power law (linear in log-space), indicating that Engram provides a predictable scaling knob: larger memory continues to pay off without requiring additional computation.
  - 摘录（3.2 Experimental protocol）：we attach an Engram table and sweep the number of slots 𝑀 from 2.58 × 105 to 1.0 × 107 (adding up to ≈ 13 billion parameters).
  - 摘录（4.2 Experimental Results）：Interestingly, these gains are not limited to knowledge-intensive tasks (e.g., MMLU: +3.0, MMLU-Pro: +1.8, CMMLU: +4.0), where memory capacity is intuitively beneficial. We observe even more significant improvements in general-reasoning domains (e.g., BBH: +5.0, ARC-Challenge: +3.7, DROP: +3.3), as well as code and mathematical reasoning
  - 摘录（5. Long Context Training）：By offloading local dependency modeling to static lookups, the Engram architecture preserves valuable attention capacity for managing global context.
  - 摘录（6.1.2 Representational Alignment and Effective Depth）：The consistent off-diagonal shift, which aligns with the LogitLens results (Section 6.1.1), confirms that Engram achieves deeper representations at earlier layers. This validates our central hypothesis: by bypassing early-stage feature composition via explicit lookups, Engram is functionally equivalent to increasing the model’s effective depth.
  - 摘录（6.2 Where should memory be injected?）：Injecting Engram early allows it to offload local pattern reconstruction before the backbone expends computational depth, aligning with the backbone’s natural hierarchical processing (Ghandeharioun et al., 2024; Jin et al., 2025; Li and Subramani, 2025; Tenney et al., 2019). However, this incurs a cost in gating precision: early hidden states have not yet aggregated sufficient global context via attention, and the parallel branches lack the representational divergence required for fine-grained modulation (Xie et al., 2025; Zhu et al., 2025).
  - 摘录（6.2 Where should memory be injected?）：The sweep shows that Layer 2 achieves the best single-layer performance (Val Loss = 1.770), outperforming Layer 1 and degrading as the insertion point moves deeper.
  - 摘录（6.2 Where should memory be injected?）：While Layer 2 is optimal under a single injection constraint, we find that dividing the same 1.6B memory into two smaller modules (achieved by reducing the embedding dimension 𝑑mem ) and placing them at Layers 2 and 6 performs even better (Val Loss = 1.768).
  - 摘录（6.2 Where should memory be injected?）：This layered design reconciles the trade-off by combining early intervention with rich, late-stage contextual gating.
  - 摘录（6.3 Sensitivity Analysis）：we evaluate the model by completely suppressing the sparse embedding output during inference while keeping the backbone unchanged.
  - 摘录（6.2 Which components matter?）：We find that three components yield the most significant gains: (i) branch-specific fusion within the multi-branch backbone, (ii) context-aware gating, and (iii) tokenizer compression. Removing any of these causes the largest regressions in validation loss.
  - 摘录（6.2 Which components matter?）：Other changes have smaller effects: removing the lightweight depthwise convolution only marginally degrades performance. Allocating capacity to 4-grams is slightly suboptimal under a fixed 1.6B budget—likely because it dilutes capacity from the more frequent 2/3-gram patterns—though we do not rule out that higher-order 𝑁 -grams become beneficial at larger memory scales.
  - 摘录（6.3 Sensitivity Analysis）：Crucially, this post-hoc ablation induces a training–inference inconsistency, potentially introducing noise in complex, mixed-capability tasks.
  - 摘录（6.3 Sensitivity Analysis）：Factual knowledge benchmarks suffer a catastrophic collapse, retaining only 29–44% of the original performance (e.g., TriviaQA at 29%), confirming that the Engram module acts as the primary repository for parametric knowledge. Conversely, reading comprehension tasks are remarkably resilient, retaining 81–93% (e.g., C3 at 93%), suggesting that context-grounded tasks rely primarily on the backbone’s attention mechanism rather than Engram.
  - 摘录（6.4 System Efficiency）：This yields a strictly deterministic memory access pattern: indices for the next Engram lookup are fixed once the token sequence is known and can be computed before the corresponding layer executes.
  - 摘录（6.4 Results）：offloading a 100B-parameter embedding table incurs a negligible throughput penalty, peaking at only 2.8% on the 8B backbone. This confirms that the compute intensity of early dense blocks provides a sufficient temporal window to mask the retrieval latency. Crucially, the effective communication volume per step scales with the number of activated slots rather than the total embedding table size.
  - 摘录（6.5 Case Study）：The gating mechanism consistently activates (shown in red) upon completing local, static patterns. In English, we observe strong activations on multi-token named entities (e.g., “Alexander the Great”, “the Milky Way”) and formulaic phrases (e.g., “By the way”, “Princess of Wales”).
  - 摘录（Figure 7）：Because Engram operates on suffix 𝑁 -grams (here 𝑁 = 3), a high activation on a specific token 𝑥𝑡 implies that the preceding tokens culminating in that token (e.g., the phrase ending at 𝑡 ) are recognized as a static pattern effectively retrieved from memory.
  - 摘录（7. Related Work）：Originating from Shannon’s framework (Shannon, 1948), 𝑁 -gram models rely on local history to predict tokens, traditionally employing smoothing techniques (Katz, 1987; Kneser and Ney, 1995) to mitigate data sparsity.
  - 摘录（4.1 Experimental Setup）：For optimization, the embedding parameters are updated using Adam (Kingma, 2014) with a learning rate scaled by 5× and no weight decay
  - 摘录（7. Related Work）：OverEncoding (Huang et al., 2025a) fails to yield meaningful improvements on sparse MoE backbones even under a non-isoparametric setting.

## 短页

### 是什么

Engram 是插在 Transformer 某些层之间的条件记忆模块。它取以当前位置结尾的最近几个 token 组成的片段（n-gram），用哈希函数把片段算成一个整数行号，到一张很大的嵌入表（每行是一个可学习向量）里取出对应的行，再由当前位置的隐藏状态（该层为每个位置算出的向量）决定取出的向量写回多少。

整张表里只有被取到的少数行参与一次计算，每个 token 的查表开销是常数。「条件」指取哪一行由输入 token 决定。这与 MoE（混合专家：把前馈层换成多个并行的子网络，即专家，每个 token 只激活其中少数几个）的条件计算相对：MoE 按输入选择激活哪些专家去计算，Engram 按输入选择读取哪些存储的向量。两者都是稀疏的：总参数很多，每个 token 只用到其中一小部分。

### 为什么需要

语言建模里有两类工作：需要深层动态计算的组合推理，和对固定内容的检索。命名实体、固定搭配这类文本是局部的、静态的、高度刻板的。标准 Transformer 没有直接查表的原语，只能靠前面若干层注意力和前馈网络把这类内容逐层拼出来，等于在运行时重建一张静态查找表，占用了本可用于推理的层数。

MoE 只扩展条件计算这一条稀疏轴。给静态内容单独配一块按需读取的存储，既把这类重建从主干里拿走，也让参数规模可以脱离计算量增长：每个 token 只读取表中常数个行，表再大，每个 token 的计算量也不变。

### 大概怎么工作

对每个位置，先把等价写法的 token（如大小写不同）合并成同一编号，取以它结尾的最近几个 token 的片段；每个片段经多个哈希头各算出一个行号（每个头有自己的哈希函数和自己的表，用多个头是为了减轻不同片段撞到同一行的碰撞），取出的向量拼成记忆向量。然后用当前隐藏状态与记忆向量变换后的相似度，经函数 $\sigma$ 压成一个 0 到 1 的标量门，门值小说明记忆与上下文不符，取出的内容被压低。门控后的结果再经一个小卷积，以残差的方式加回隐藏状态。

行号只取决于 token，不取决于隐藏状态，所以可以提前算出并预取，让取行与前面各层的计算重叠。

## 长页

Engram 是一个加在 Transformer 主干上的记忆模块，对应的概念是条件记忆：按输入 token 稀疏地读取一张静态嵌入表，读到的向量交给主干使用。它由两个阶段组成，检索负责从 token 序列得到表中的向量，融合负责决定这些向量写入隐藏状态的多少。

它要解决的问题是：主干的前几层被迫把多 token 实体、固定搭配这类静态内容当作计算问题来重建，浪费了深度。Engram 把这部分改成查表，查表的代价与表的大小无关，于是存储容量成为一条独立于计算量的扩展轴，与 MoE 的专家数并列。局部依赖由查表接手后，注意力的容量可以留给全局上下文。MoE 是把前馈层换成多个并行的专家子网络、每个 token 只激活其中几个的结构，见 [DeepSeekMoE](wiki/deepseek-moe)。

### 检索：从 token 序列到嵌入表的一行

#### 压缩 token，取后缀 n-gram

分词器按能无损还原文本的原则编号，语义相同的词可能有不同编号，例如「Apple」与带前导空格的「 apple」。Engram 先预计算一个满射函数 $\mathcal{P}:V\to V'$（$V$ 是原词表，$V'$ 是压缩后的词表），按 Unicode 规范化（NFKC，把兼容写法合并为同一形式）、转小写等规则把原始 token 编号 $x_t$ 映射成规范编号 $x'_t=\mathcal{P}(x_t)$，使等价写法落到同一编号。

位置 $t$ 处的 $n$ 元后缀片段取以它结尾的最近 $n$ 个规范编号：$g_{t,n}=(x'_{t-n+1},\dots,x'_t)$。这里 $n$ 取 2 到 $N$ 的各个阶数，每一阶各有自己的表。n-gram 本身的概念见 [n-gram](wiki/ngram)。

#### 多头哈希，得到行号和向量

所有可能的 n-gram 组合数太大，无法给每个片段一行。Engram 用哈希函数把片段映射到有限大小的表里。为减轻不同片段落到同一行（碰撞）造成的混淆，每一阶有 $K$ 个哈希头，第 $k$ 个头有自己的表 $E_{n,k}$（行数为 $M_{n,k}$）和确定性哈希函数 $\varphi_{n,k}$：

$$z_{t,n,k}=\varphi_{n,k}(g_{t,n}),\qquad \mathbf{e}_{t,n,k}=E_{n,k}[z_{t,n,k}]$$

$z_{t,n,k}$ 是行号，$\mathbf{e}_{t,n,k}$ 是取出的行向量。所有阶、所有头取出的向量拼接成记忆向量 $\mathbf{e}_t$。表是可学习的参数：训练时前向取出被用到的行，反向把梯度送回这些行。

::: 拼接公式
设最高阶为 $N$，每阶 $K$ 个头，记忆向量为全部取出向量的拼接：

$$\mathbf{e}_t=\mathop{\Vert}_{n=2}^{N}\mathop{\Vert}_{k=1}^{K}\mathbf{e}_{t,n,k}\in\mathbb{R}^{d_{\mathrm{mem}}}$$

其中 $\Vert$ 表示向量拼接，$d_{\mathrm{mem}}$ 是拼接后的总维数。
:::

### 融合：上下文门控

取出的 $\mathbf{e}_t$ 只由 token 决定，与上下文无关，所以它只是一个先验，可能带有哈希碰撞或一词多义造成的噪声。Engram 用当前隐藏状态 $\mathbf{h}_t$ 做查询（它已经通过前面的注意力层汇集了全局上下文），用 $\mathbf{e}_t$ 生成键和值：

$$\mathbf{k}_t=W_K\mathbf{e}_t,\qquad \mathbf{v}_t=W_V\mathbf{e}_t$$

$W_K,W_V$ 是可学习的投影矩阵。门是一个标量：

$$\alpha_t=\sigma\!\left(\frac{\mathrm{RMSNorm}(\mathbf{h}_t)^{\top}\mathrm{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right)$$

$\sigma$ 是把值压进 $(0,1)$ 的函数，$d$ 是隐藏状态的维数（隐藏状态矩阵的列数），$\alpha_t\in(0,1)$。键 $\mathbf{k}_t$ 用来算门，值 $\mathbf{v}_t$ 是被门控后写回的内容。RMSNorm 的做法见 [RMSNorm](wiki/rmsnorm)，这里用它稳定梯度。门控后的输出是 $\tilde{\mathbf{v}}_t=\alpha_t\mathbf{v}_t$。检索到的记忆与当前上下文矛盾时，$\alpha_t$ 趋近 0，噪声被压住。

门控输出序列再经一个短的深度因果卷积（见 [深度卷积](wiki/depthwise-conv)，用来扩大感受野），结果 $Y$ 以残差方式加回主干：$H^{(\ell)}\leftarrow H^{(\ell)}+Y$。其中 $H^{(\ell)}$ 是第 $\ell$ 层各位置隐藏状态按行堆成的矩阵，残差连接见 [残差连接](wiki/residual-connection)。之后照常进入该层的注意力和 MoE。Engram 不是每层都放。

::: 卷积式
卷积用于扩大感受野并增加非线性。设门控值序列为 $\tilde V$：

$$Y=\mathrm{SiLU}\big(\mathrm{Conv1D}(\mathrm{RMSNorm}(\tilde V))\big)+\tilde V$$
:::

### 与 MoE 之间的参数分配

要比较两条稀疏轴，先区分总参数 $P_{\mathrm{tot}}$、每个 token 实际用到的激活参数 $P_{\mathrm{act}}$，以及差值 $P_{\mathrm{sparse}}=P_{\mathrm{tot}}-P_{\mathrm{act}}$。MoE 中未被选中的专家、Engram 中未被取到的行，都属于 $P_{\mathrm{sparse}}$，不增加每个 token 的计算量。

在总参数和计算量都固定时，把 $P_{\mathrm{sparse}}$ 中比例 $\rho$ 分给 MoE 专家，其余分给 Engram 表。$\rho=1$ 是纯 MoE，$\rho<1$ 是减少专家并把腾出的参数改成表的行数。

验证损失随 $\rho$ 呈 U 形：两端都不如中间。纯 MoE 缺少存放静态模式的专用存储，只能用深度和计算去重建它们；$\rho$ 趋近 0 则失去条件计算的容量，需要动态上下文推理的任务受损，记忆无法取代计算。最优点附近，约有 20% 到 25% 的稀疏参数预算分给 Engram，在固定稀疏度、所考察的两个计算量规模下位置稳定。

另一种设定是固定主干、只增加表的行数：在所扫描的行数范围内，验证损失随行数呈幂律下降（对数坐标下线性），更大的表持续带来收益且不增加计算。

::: 分配实验的条件与数字
实验在两个计算量下进行，保持总参数与激活参数之比约为 10。较大一组（约 6e20 FLOPs，总参数约 9.9B）中，验证损失从 $\rho=100\%$ 时的 1.7248 降到 $\rho\approx80\%$ 附近的 1.7109。行数扫描中，行数从 2.58e5 增至 1.0e7。
:::

### 放在哪一层，以及系统开销

#### 放置位置的权衡

放得早，Engram 能在主干耗费深度之前接手局部模式的重建；但此时隐藏状态还没有经过足够的注意力，作查询的上下文信息不足，门的精度下降。放得晚，门的查询更好，但接手重建的机会变少。只放一层时，放得较早最优，更深则变差；把同样的预算拆成两个较小的模块、分放在早层和较晚的层，结果更好，论文认为这样兼顾了早期接手和后期的上下文门控。

这种接手使主干前几层不必再从零组合特征，论文的机制分析认为其效果相当于有效深度增加。

#### 确定性寻址带来的系统收益

MoE 的路由取决于运行时的隐藏状态，Engram 的行号只取决于输入 token 编号，序列一确定，各层要查的行就能在该层执行之前算出。

训练时，大表分片放在多块 GPU 上，前向用 All-to-All（各设备互相交换数据的集合通信）取出被用到的行，反向送回梯度，表容量随设备数线性增长。推理时，可以提前异步地经 PCIe 从容量大的主机内存取行，与前面各层的计算重叠。模块放在较深的层，可供掩盖延迟的计算窗口更长，这与建模上偏好早放相互制约。每步的通信量只随被取到的行数增长，与整张表的大小无关。

n-gram 的出现频率服从 Zipf 分布，少数模式占了绝大多数访问。据此可以做多级缓存：高频行放在更快的存储（如 GPU 显存或主机内存），低频的长尾放在大容量慢速介质（如固态硬盘）。

### 一个手算例子

::: 手算过程与代码
设每个（阶，头）各有一张 7 行的表，三个 token 依次为「Alexander」「The」「Great」，规范编号映射为 Alexander→4，The→2，the→2，Great→3，即「The」与「the」被压缩成同一编号。在位置 $t=3$（从 1 数；代码里从 0 数，写作 `t = 2`；「Great」处），2 元片段是 $(2,3)$，3 元片段是 $(4,2,3)$。

玩具哈希：第 $k$ 个头从 $h=k+1$ 出发，对片段中的每个编号 $x$ 依次令 $h\leftarrow(3h+x)\bmod 7$，表有 7 行，最后的 $h$ 就是行号。

2 元片段 $(2,3)$：头 0 从 1 起，$3\cdot1+2=5$，再 $3\cdot5+3=18\bmod7=4$，行号 4；头 1 从 2 起，$3\cdot2+2=8\bmod7=1$，再 $3\cdot1+3=6$，行号 6。

3 元片段 $(4,2,3)$：头 0 得行号 2，头 1 得行号 1。四个行向量拼接成 $\mathbf{e}_t$。例子在行号处停止，不展开 $W_K\mathbf{e}_t$ 的计算，下面直接假设 $\mathbf{k}$。若原文写成「alexander the great」，压缩后的编号序列相同，行号也相同。

门：设隐藏状态 $\mathbf{h}=(2,2)$，门值只算到 $\alpha$ 为止。若 $\mathbf{k}=(3,3)$，两者归一化后都是 $(1,1)$，点积为 2，除以 $\sqrt2$ 得 1.414，$\alpha=\sigma(1.414)\approx0.80$。若 $\mathbf{k}=(-3,-3)$，点积为 $-2$，$\alpha=\sigma(-1.414)\approx0.20$，点积的正负决定 $\alpha$ 偏大还是偏小，$\alpha$ 小时该记忆被压低。这里省略了 RMSNorm 中防止除零的小常数。

```python
import math

P = {"Alexander": 4, "the": 2, "The": 2, "Great": 3}
M = 7

def phi(gram, k):
    h = k + 1
    for x in gram:
        h = (h * 3 + x) % M
    return h

def suffix(tokens, t, n):
    ids = [P[w] for w in tokens]
    return tuple(ids[t - n + 1:t + 1])

tokens = ["Alexander", "The", "Great"]
t = 2
for n in (2, 3):
    g = suffix(tokens, t, n)
    print(n, g, [phi(g, k) for k in (0, 1)])
print(suffix(["Alexander", "the", "Great"], t, 3) == suffix(tokens, t, 3))

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def rms_norm(v):
    r = math.sqrt(sum(x * x for x in v) / len(v))
    return [x / r for x in v]

def gate(h, k):
    a, b = rms_norm(h), rms_norm(k)
    return sigmoid(sum(x * y for x, y in zip(a, b)) / math.sqrt(len(h)))

h = [2.0, 2.0]
print(round(gate(h, [3.0, 3.0]), 2), round(gate(h, [-3.0, -3.0]), 2))
```

```text
2 (2, 3) [4, 6]
3 (4, 2, 3) [2, 1]
True
0.8 0.2
```
:::

### 适用条件与边界

#### 适用条件

实验观察到，门在多 token 命名实体和固定短语结束的位置明显激活。

#### 代价与限制

- 推理时把 Engram 输出整个压掉后，事实问答类任务只保留原性能的 29% 到 44%，阅读理解类任务保留 81% 到 93%。论文据此认为事实性知识主要存在表里，并推测依赖上下文的任务主要靠主干的注意力。这项测试是推理时的事后压制，与训练不一致，复杂的混合能力任务上结果可能带噪声。
- 哈希让不同片段可能共用同一行，取出的向量会带噪声；多个哈希头和门控只能缓解，不能消除。
- 行号只由最近几个 token 决定，片段之外的上下文进不了寻址，上下文适配只能靠门控事后完成；预取和缓存的收益也以这种只看 token 的寻址为前提。
- 阶数不是越高越好：在固定的表预算下，加入 4 元片段略差，原因可能是分走了更常见的 2 元、3 元模式的容量；更大的表规模下更高阶是否有益尚未排除。

#### 常见误解

- 它不是统计意义上的 n-gram 语言模型：表中是用优化器更新的嵌入向量，用哈希而不是穷举片段来寻址，碰撞由多头和门控缓解。
- 「记忆」的收益不限于知识类任务：论文的实验中推理类和代码数学类任务也有提升。

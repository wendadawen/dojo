# DSA 核心论断与证据

## 来源清单

| 代号 | 来源 | 定位方式 |
|---|---|---|
| S1 | arXiv:2512.02556《DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models》，LaTeX 源码（`sections/01-method.tex`，2025-12-02 提交版） | 章节名 + 源码行号 |
| S2 | DeepSeek-V3.2-Exp 官方开源推理实现，HuggingFace `deepseek-ai/DeepSeek-V3.2-Exp`，commit `194c67e12b1b0d6df0ef373ddcf215bc84027409`（2025-11-18） | 文件 + 行号 |
| S3 | 同上仓库 `config.json`（同一 commit） | 字段名 |
| S4 | vLLM 仓库 `vllm-project/vllm`，commit `5ac2684976ee22c04fe0d2f968c6cf6096b383f2` | 文件 + 行号 |

S1 为机制与训练的正本；S2、S3 为实现细节正本；S4 为工程落地的第二实现，用于交叉验证与补充服务侧细节。

优先级说明：机制、公式、训练超参一律以 S1 为准；S1 未写明的实现细节（配置数值、kernel 计算链、cache 布局）以 S2/S3 为准；S4 仅用于说明"另一套生产实现如何落地"，不用于覆盖 S1/S2 的结论。

## 公式

### F1：index score（lightning indexer 打分）

$$I_{t, s} = \sum_{j=1}^{H^I} w_{t, j}^I \cdot \mathrm{ReLU}\left(\mathbf{q}^{I}_{t, j} \cdot \mathbf{k}^{I}_{s}\right)$$

- 来源定位：S1 `sections/01-method.tex` L13–15（Equation 1）。符号说明见同文件 L16–17：$H^I$ 为 indexer 头数；$\mathbf{q}^I_{t,j} \in \mathbb{R}^{d^I}$ 与 $w^I_{t,j} \in \mathbb{R}$ 由查询 token $\mathbf{h}_t$ 导出；$\mathbf{k}^I_s \in \mathbb{R}^{d^I}$ 由在前的 token $\mathbf{h}_s$ 导出。
- 适用条件：因果设定，$s$ 取遍 $t$ 之前（含自身）的位置。
- 置信状态：已确认（LaTeX 源码原文）。
- 实现对应：S2 `inference/model.py` L478–480 —— `weights = self.weights_proj(x.float()) * self.n_heads ** -0.5`，随后 `fp8_index(q_fp8, weights, k_cache, k_scale_cache)`。注意实现中 $w^I_{t,j}$ 额外乘了 $1/\sqrt{H^I}$ 与 softmax scale，这属于实现层的缩放约定，论文公式未写出。

### F2：稀疏注意力输出

$$\mathbf{u}_t = \mathrm{Attn}\left(\mathbf{h}_t, \left\{ \mathbf{c}_s \mid I_{t, s} \in \mathrm{Top\text{-}k}\left(I_{t, :}\right) \right\}\right)$$

- 来源定位：S1 `sections/01-method.tex` L23–25（Equation 2）。
- 适用条件：$\mathbf{c}_s$ 指 key-value 条目；在 V3.2 中即 MLA 的 latent 向量（S1 L40）。
- 置信状态：已确认。
- 实现对应：S2 `inference/model.py` L584–586（prefill）与 L601–602（decode）—— 用 `scatter_` 构造 index mask，未被选中的位置置 $-\infty$ 后加到 scores 上，再做 softmax。即实现上通过掩码等价实现"只对 top-k 做注意力"。

### F3：dense warm-up 阶段的 indexer 损失

$$\mathcal{L}^{I} = \sum_t D_{\mathrm{KL}}\left(p_{t,:} \,\|\, \mathrm{Softmax}\left(I_{t,:}\right)\right)$$

- 来源定位：S1 `sections/01-method.tex` L58–60（Equation 3）。目标分布 $p_{t,:} \in \mathbb{R}^t$ 的构造见 L55–56：对第 $t$ 个 query token，先把主注意力分数跨全部注意力头求和，再沿序列维度做 L1 归一化。
- 适用条件：warm-up 阶段保持稠密注意力，冻结除 lightning indexer 外的全部模型参数（S1 L53–54）。
- 置信状态：已确认。

### F4：sparse training 阶段的 indexer 损失

$$\mathcal{L}^{I} = \sum_t D_{\mathrm{KL}}\left(p_{t,\mathcal{S}_t} \,\|\, \mathrm{Softmax}\left(I_{t,\mathcal{S}_t}\right)\right), \quad \mathcal{S}_t = \left\{ s \mid I_{t,s} \in \mathrm{Top\text{-}k}\left(I_{t,:}\right) \right\}$$

- 来源定位：S1 `sections/01-method.tex` L66–69（Equation 4）。
- 适用条件：该阶段引入 top-k 选择并优化全部参数；KL 只在被选中集合 $\mathcal{S}_t$ 上计算。
- 置信状态：已确认。

## 数字

| 编号 | 数字 | 内容 | 来源定位 | 置信状态 |
|---|---|---|---|---|
| N1 | $H^I = 64$ | lightning indexer 头数 | S3 `index_n_heads: 64`；S2 `model.py` L88 默认值 64 | 已确认 |
| N2 | $d^I = 128$ | indexer 每头维度 | S3 `index_head_dim: 128`；S2 `model.py` L89 | 已确认 |
| N3 | $k = 2048$ | 每个 query token 选取的 KV token 数 | S1 L72（"select 2048 key-value tokens for each query token"）；S3 `index_topk: 2048`；S2 `model.py` L90 | 已确认（论文与实现一致） |
| N4 | 学习率 $10^{-3}$ | warm-up 阶段 indexer 学习率 | S1 L61 | 已确认 |
| N5 | 1000 步 / 每步 16 条 128K 序列 / 合计 2.1B token | warm-up 规模 | S1 L62 | 已确认 |
| N6 | 学习率 $7.3\times10^{-6}$ | sparse training 阶段学习率 | S1 L72 | 已确认 |
| N7 | 15000 步 / 每步 480 条 128K 序列 / 合计 943.7B token | sparse training 规模 | S1 L73 | 已确认 |
| N8 | 128K | 续训起点 checkpoint 的上下文长度 | S1 L46 | 已确认 |
| N9 | 61 层 / 128 个注意力头 / hidden 7168 | V3.2-Exp 主模型规模（用于说明 indexer 相对主注意力的量级） | S3 `num_hidden_layers: 61`、`num_attention_heads: 128`、`hidden_size: 7168` | 已确认 |
| N10 | kv_lora_rank 512、qk_rope_head_dim 64 | MLA latent 与 rope 维度（用于对比 indexer 单价） | S3 | 已确认 |
| N11 | 2 USD / GPU 小时，H800 集群 | 推理成本图的估算口径 | S1 L91 | 已确认 |

## 论断

### C1：DSA 是 V3.2 相对 V3.1-Terminus 的唯一架构改动

- 论断内容：与 DeepSeek-V3.1 的最后一个版本 V3.1-Terminus 相比，DeepSeek-V3.2 的唯一架构改动是通过续训引入 DeepSeek Sparse Attention。
- 来源定位：S1 `sections/01-method.tex` L7。
- 适用条件：指架构层面；训练流程与后训练另有变化。
- 置信状态：已确认。

### C2：V3.2 与 V3.2-Exp 架构完全相同

- 论断内容：DeepSeek-V3.2 使用与 DeepSeek-V3.2-Exp 完全相同的架构。
- 来源定位：S1 `sections/01-method.tex` L6（"uses exactly the same architecture as DeepSeek-V3.2-Exp"）；宏定义见 `main.tex` L76（`\newmodel` = DeepSeek-V3.2）。
- 适用条件：架构层面。
- 置信状态：已确认。这条支撑本页不区分两个版本讲机制。

### C3：DSA 原型由 lightning indexer 与细粒度 token 选择两部分组成

- 论断内容：DSA 的原型主要由一个 lightning indexer 和一个 fine-grained token selection mechanism 构成。
- 来源定位：S1 L10。
- 置信状态：已确认。

### C4：候选集跨 query 头共享的原因是 kernel 效率

- 论断内容：在 kernel 层面，每个 key-value 条目必须被多个 query 共享才能保证计算效率；因此 DSA 实现在 MLA 的 MQA 模式上，每个 latent 向量（MLA 的 key-value 条目）被该 query token 的全部 query 头共享。
- 来源定位：S1 L39–40，其中 kernel 层约束引用 `yuan-etal-2025-native`（即 NSA）。
- 适用条件：这是 V3.2 的实例化选择，论文表述为"for the consideration of continued training from V3.1-Terminus"（L38）。
- 置信状态：已确认。
- 实现对应：S2 `model.py` L590–606（MQA decode 路径），index mask 形状为 `(bsz, 1, end_pos)`，在头维度上 unsqueeze 后广播到所有头（L602）。

### C5：选 ReLU 作为激活函数是吞吐考虑

- 论断内容：indexer 中选择 ReLU 作为激活函数出于吞吐（throughput）考虑。
- 来源定位：S1 L18。
- 置信状态：已确认。注意论文只说"throughput consideration"，未展开机制解释；页面上任何进一步解释（如"避免跨序列归一化"）须标记为教学解释而非来源结论。

### C6：indexer 头数少且可用 FP8 实现，因此计算效率突出

- 论断内容：由于 lightning indexer 头数少且可以在 FP8 下实现，其计算效率非常突出。
- 来源定位：S1 L19。
- 置信状态：已确认。
- 实现对应：S2 `inference/kernel.py` L254–274 `fp8_index` 的 docstring 给出完整计算链：`fp8 q @ fp8 k -> fp32 logits`；`relu(fp32 logits) * q_s (weights) -> fp32 logits`；`fp32 logits -> fp32 logits_sum`；`fp32 logits_sum * k_s (e8m0) -> fp32 index_score`。kernel 主体见 L199–251。

### C7：主注意力复杂度从 $O(L^2)$ 降到 $O(Lk)$，但 indexer 仍是 $O(L^2)$

- 论断内容：DSA 把主模型的核心注意力复杂度从 $O(L^2)$ 降到 $O(Lk)$（$k \ll L$）；尽管 lightning indexer 的复杂度仍为 $O(L^2)$，但它相比 V3.1-Terminus 中的 MLA 需要的计算量少得多。
- 来源定位：S1 L87–88。
- 适用条件：$k \ll L$。
- 置信状态：已确认。这条同时支撑"省了什么"和"没省成线性"两个结论。

### C8：短序列 prefill 用 masked MHA 模式模拟 DSA

- 论断内容：对于短序列 prefill，官方专门实现了一个 masked MHA 模式来模拟 DSA，它在短上下文条件下能达到更高效率。
- 来源定位：S1 L92。
- 置信状态：已确认。
- 实现对应（第二实现，S4）：`vllm/model_executor/layers/attention/sparse_mla_attention.py` L279–282 —— `use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not ...sparse_mla_force_mqa)`；消费方见 `vllm/model_executor/layers/attention/mla_attention.py` L789、L804。S4 的阈值判据（序列长度不超过 topk）与 S1 的定性表述一致，但具体阈值来自 S4 实现，不得写成 S1 的结论。

### C9：DSA 不减少 KV cache，indexer 还额外需要自己的 key cache

- 论断内容：DSA 只改变每个 query 读取哪些 KV 条目，全部历史位置的 KV 条目仍须保留；此外 lightning indexer 需要维护一份自己的 key cache。
- 来源定位：S1 未直接给出这条否定性结论，依据实现证据成立——S2 `inference/model.py` L453–454 注册 indexer 专有的 `k_cache`（FP8，形状 `(max_batch_size, max_seq_len, head_dim)`）与 `k_scale_cache`；同文件 L541–542 MLA 自身的 `kv_cache`/`pe_cache` 覆盖全部 `max_seq_len` 位置，未随 top-k 缩减。S4 中对应独立的 `DeepseekV32IndexerCache`（`vllm/models/deepseek_v32/attention.py` L83–88）。
- 适用条件：结论为"DSA 本身不减少 KV cache"，与 MLA 的低秩压缩带来的 KV 节省是两件事，不可混淆。
- 置信状态：已确认（基于两套实现的一致证据）。页面须标注这条是从实现推出的结论，而非论文原文声称。

### C10：indexer 与主注意力的优化信号相互隔离

- 论断内容：sparse training 阶段把 indexer 的输入从计算图上 detach 以做独立优化；indexer 的训练信号只来自 $\mathcal{L}^I$，主模型的优化只依据语言建模损失。
- 来源定位：S1 L70–71。
- 置信状态：已确认。

### C11：引入 DSA 后未观察到明显性能回退

- 论断内容：V3.2-Exp 在显著提升长序列计算效率的同时，与 V3.1-Terminus 相比在短上下文和长上下文任务上均未观察到明显的性能下降。
- 来源定位：S1 L78–79（Parity Evaluation）。
- 适用条件：评测时间为 2025 年 9 月，对比对象为 V3.1-Terminus，且两者训练设置严格对齐。
- 置信状态：已确认。页面只取这条定性结论，不复述具体分数。

### C12：indexer 使用独立的 RoPE 通道，且 rope 应用方式与主注意力不同

- 论断内容：lightning indexer 的 query/key 由独立投影得到（`wq_b`、`wk`），有自己的 LayerNorm，并对前 `qk_rope_head_dim` 维施加 RoPE；indexer 中的 rope 不是 interleaved 的，与主注意力的应用方式不同。
- 来源定位：S2 `inference/model.py` L445–447（投影与 norm）、L462–471（rope 拆分与应用），其中 L463、L469 两处注释明确 "rope in indexer is not interleaved"，调用 `apply_rotary_emb(..., False)`；对比主注意力 L564、L568 使用默认 interleaved。
- 适用条件：实现层事实，S1 未提及。
- 置信状态：已确认。
- 第二实现对应：S4 `vllm/models/deepseek_v32/attention.py` L314–319 为 indexer 单独构造 `indexer_rope_emb`，并注释 "Lightning indexer uses its own RoPE; interleave maps to non-NeoX"。

### C13：index score 计算前对 q、k 施加 Hadamard 变换

- 论断内容：官方实现在量化到 FP8 之前对 indexer 的 q 与 k 施加了一次 Hadamard 变换（`rotate_activation`）。
- 来源定位：S2 `inference/model.py` L472–473 调用 `rotate_activation`，其定义见 L428–432（`fast_hadamard_transform.hadamard_transform`，scale 为 `hidden_size ** -0.5`）。
- 适用条件：实现层事实，S1 未提及；属于低精度量化的常见预处理（把异常值能量摊开），页面若给出这一动机须标记为教学解释。
- 置信状态：已确认（代码事实）；动机解释为推断，需标注。

### C14：DCP 切分下用局部 top-k 合并出全局 top-k 是精确的

- 论断内容：在 decode context parallel（KV cache 按 rank 切分）下，只交换每个 rank 的局部 top-k 候选即可精确得到全局 top-k：若某 token 属于全局 top-k，则全局至多有 $k-1$ 个 token 排在它之前，其所属 rank 上至多也有这么多个排在它之前，故它必然也在本 rank 的局部 top-k 内。
- 来源定位：S4 `vllm/model_executor/layers/sparse_attn_indexer.py` L74–91（`_merge_dcp_topk_global` 的 docstring 给出该论证）。
- 适用条件：属于 vLLM 实现层的工程结论，非 S1 内容；且该合并路径在 S4 中要求 CuteDSL 且 `index_topk ∈ {512, 1024, 2048}`（L67–72）。
- 置信状态：已确认（代码与注释）。页面引用时须标注来源为 vLLM 实现，不得写成论文结论。

## 存在冲突或证据不足的项

无。核心论断均处于"已确认"状态。

两处需要在页面上明确标注性质的内容：

1. C9（DSA 不省 KV cache）是基于两套实现的推出结论，论文未以否定句形式声称。
2. C5 的机制展开、C13 的动机属于教学解释，须按写作规范标记，不得写成来源结论。

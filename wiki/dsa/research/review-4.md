<!-- review-meta
round: 4
page: wiki/dsa/index.html
reviewed_content_sha256: e81808414712d3eb
-->
# DeepSeek Sparse Attention（DSA）审查记录（第 4 轮）

- 页面版本：442ac9af286aa91dac528f93197169bde5f380dd
- 审查时间：2026-09-13 19:35
- 审查者：独立子代理（未参与写作；未读取 research/ 下的规划、修复记录与前序审查结果）
- 已完整阅读章节：题头与摘要、「核心问题」、「常见误解」、「进入本页前需要的基础」、1. 长上下文里，注意力的代价长在哪里 / 2. 两条减负路线，以及 DSA 选的那条 / 3. lightning indexer 如何用极低成本给每个位置打分（含 3.1 手算、3.2 五个来源）/ 4. top-k 选择，以及候选集为什么必须跨头共享（含 4.1）/ 5. indexer 的打分能力是训出来的（含 5.1、5.2）/ 6. 工程落地（含 6.1 前向链路、6.2 FP8 index kernel、6.3 DCP top-k 合并）/ 7. 省了什么、没省什么、什么时候不该用（含 7.1–7.5）/ 8. 与相邻方法的定位 / 来源与范围说明；含全部 details 折叠块、代码块、表格与图注

## 一、来源核对记录（逐条引文依据）

### S1 = arXiv:2512.02556（arXiv HTML 正文）

| 页面论断 | 核对到的原文片段 |
|---|---|
| C1/C2 唯一架构改动、V3.2 与 V3.2-Exp 架构相同 | "DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training." |
| C3 由 lightning indexer 与细粒度 token 选择组成 | "…comprises two main components: a lightning indexer and a fine-grained token selection mechanism." |
| C4 kernel 层共享约束（引 NSA）→ 选 MLA 的 MQA 模式 | "At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency ([Yuan et al., 2025])." / "For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA…" / "…we implement DSA based on the MQA mode of MLA, where each latent vector…will be shared across all query heads of the query token." |
| C5 ReLU 出于吞吐考虑 | "we choose ReLU as the activation function for throughput consideration" |
| C6 头数少 + 可 FP8 → 效率突出 | "Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable." |
| C7 主注意力 O(L²)→O(Lk)、indexer 仍 O(L²) | "DSA reduces the core attention complexity of the main model from O(L²) to O(Lk), where k (≪L) is the number of selected tokens." / "Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus." |
| C8 短序列 prefill 的 masked MHA | "Note that for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions." |
| C10 indexer 输入 detach | "detach the indexer input from the computational graph for separate optimization" |
| C11 三组 parity 证据 | 标准 benchmark："In September 2025, we evaluate DeepSeek-V3.2-Exp on a suite of benchmarks… we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks."；人类偏好："Both DeepSeek-V3.1-Terminus and DeepSeek-V3.2-Exp share an identical post-training strategy" + "Elo scores, obtained from evaluations conducted on 10 November 2025, are closely matched."；长上下文："…AA-LCR, in which DeepSeek-V3.2-Exp scores four points higher than DeepSeek-V3.1-Terminus in reasoning mode." + Fiction.liveBench "consistently outperforms DeepSeek-V3.1-Terminus across multiple metrics" |
| C15 续训数据分布对齐 | 起点 "starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K"；数据 "the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus" |
| F1（Eq.1）index score | I_{t,s} = Σ_{j=1}^{H^I} wⁱ_{t,j}·ReLU(qⁱ_{t,j}·kⁱ_s)：与页面公式逐项一致 |
| F2（Eq.2）稀疏输出 | u_t = Attn(h_t, {c_s \| I_{t,s} ∈ Top-k(I_{t,:})})：一致；k=2048 见 "select 2048 key-value tokens for each query token" |
| F3（Eq.3）dense warm-up 损失 | ℒᴵ = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:}))；p_{t,:} 由主注意力分数 "summing across all attention heads" 后 "L1-normalized along the sequence dimension" 得到，p_{t,:}∈ℝ^t：一致 |
| F4（Eq.4）sparse training 损失 | ℒᴵ = Σ_t D_KL(p_{t,𝒮_t} ‖ Softmax(I_{t,𝒮_t}))，𝒮_t={s \| I_{t,s} ∈ Top-k(I_{t,:})}：一致 |
| N 训练超参 | warm-up：lr 10⁻³、1000 步、每步 16 条 128K 序列、合计 2.1B token；sparse：lr 7.3×10⁻⁶、15000 步、每步 480 条 128K 序列、合计 943.7B token —— 与页面表格逐格一致（页面 943.7B/2.1B ≈ 450 倍的换算正确：449.4） |
| N 成本图口径 | "These costs are estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour."；定性结论 "DSA achieves a significant end-to-end speedup in long-context scenarios" |
| 页面 51 行「DSA 论文引用 NSA」 | 该段确实带 (Yuan et al., 2025) 引用；正文其余引用为 DeepSeek-AI (2024a) MLA、Shazeer (2019) MQA |

### S2/S3 = 官方开源推理实现（HF deepseek-ai/DeepSeek-V3.2-Exp，commit 194c67e）

| 页面论断 | 核对到的源码 |
|---|---|
| 因果掩码 triu_(1)、对角线不屏蔽，故 s 含 t 自身 | `inference/model.py:903`：`mask = torch.full((seqlen, seqlen), float("-inf"), device=...).triu_(1) if seqlen > 1 else None` |
| k_cache（FP8）与 k_scale_cache 独立注册 | `model.py:453-454`：`register_buffer("k_cache", … float8_e4m3fn)`、`register_buffer("k_scale_cache", … float32)` |
| 主 KV cache 为 kv_cache/pe_cache、按最大序列长度 | `model.py:541-542`；`mla.py` 侧对应 `kv_cache`/`pe_cache` 分配 |
| C12 query 复用主注意力低秩 latent、key 独立投影 | `Indexer.forward`：`q = self.wq_b(qr)`、`k = self.k_norm(self.wk(x))`；`MLA.forward:560`：`qr = self.q_norm(self.wq_a(x))`，第 583 行把 `qr` 传给 indexer |
| 两处 "rope in indexer is not interleaved" 注释 | `model.py:463` 与 `:469` 原文即 "rope in indexer is not interleaved"；主注意力在同一文件 `:564/:568` 用默认 `interleaved=True`，确有差别 |
| RoPE 只作用于前 64 维 | `rope_head_dim = args.qk_rope_head_dim`（config=64），`q_pe, q_nope = torch.split(q, [self.rope_head_dim, head_dim - rope_head_dim], -1)` |
| C13 rotate_activation = 带 1/√d 缩放的 Hadamard | `rotate_activation`：`hadamard_transform(x, scale=hidden_size ** -0.5)`，量化前分别作用于 q 与 k |
| wⁱ 额外乘 1/√Hᴵ 与 softmax 缩放后合入 q_s | `weights = self.weights_proj(x.float()) * self.n_heads ** -0.5`；`weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale`（实现中还折入了 query 侧量化 scale，页面称"合进 q_s"属实） |
| 四步计算链 docstring | `inference/kernel.py:fp8_index` docstring 恰为四行：`fp8 q @ fp8 k -> fp32 logits` / `relu(fp32 logits) * q_s (weights) -> fp32 logits` / `fp32 logits -> fp32 logits_sum` / `fp32 logits_sum * k_s (e8m0) -> fp32 index_score`；kernel 签名 q[b,m,h,d]、q_s[b,m,h]、k[b,n,d]、k_s[b,n] → o[b,m,n]，与页面伪代码的形状标注一致 |
| 掩码实现（未选中置 −∞） | `model.py:584/601`：`index_mask = torch.full(..., float("-inf")).scatter_(-1, topk_indices, 0)`，与页面描述一致 |

### S4 = vLLM（commit 5ac2684）

| 页面论断 | 核对到的源码 |
|---|---|
| use_dense_mha 判据 | `sparse_mla_attention.py:279-282`：`use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not self.vllm_config.attention_config.sparse_mla_force_mqa)`；消费侧 `mla_attention.py:789/804` 另有 `use_dense_mha or use_masked_mha` 等判断，"消费侧还有进一步的条件判断"属实 |
| DCP 局部 top-k 合并的精确性论证 | `sparse_attn_indexer.py:_merge_dcp_topk_global` docstring："A token in the global top-K must also be in its owning rank's local top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at most that many on its own rank), so exchanging only the per-rank local candidates is exact -- equivalent to all-gathering the full logit matrix, but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole score row." —— 页面的"卡数 × k 个候选"与"精确等价"均由该句直接支持 |
| 实现约束 CuteDSL / index_topk∈(512,1024,2048) | 同文件："DCP sparse-indexer merge requires CuteDSL; install it or disable DCP." / "DCP sparse-indexer merge requires index_topk in (512, 1024, 2048)" |
| 独立的 DeepseekV32IndexerCache | `vllm/model_executor/models/deepseek_v2.py:616`：`class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase)` |
| indexer 接收主注意力低秩 latent q_c 的调用位置 | `mla.py:206`：`self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`；`mla.py:175`：`q_c = self.q_a_layernorm(q_c)` —— 页面标注的行号准确 |

### 页面内可运行代码

按页面原样执行（Python 3 + NumPy），输出与「预期输出」逐字一致：`I_8,1=2.00 / I_8,2=2.50 / I_8,3=0.25 / I_8,4=0.50 / I_8,5=0.00 / I_8,6=0.75 / I_8,7=4.00 / I_8,8=3.00`；`k=3 时选中的位置: [2, 7, 8]`；`k=8 时选中的位置: [1, 2, 3, 4, 5, 6, 7, 8]（序列长 8，未裁剪）`。手算表格与 3.1/4 章正文的分数字符串、top-3 集合均与运行结果一致。

### 页内可复算项（构造示例）

- 8 个 index score 复算无误，且两两不等（top-k 无并列）；ReLU 截断出现在 s=3、s=4，与正文说明一致。
- warm-up 目标分布示例：4 个头 × 8 位置，每头分数各自求和为 1.00；跨头求和得 (0.60, 0.80, 0.15, 0.12, 0.18, 0.30, 1.05, 0.80)，总和 4.00；除以 4.00 得 (0.1500, 0.2000, 0.0375, 0.0300, 0.0450, 0.0750, 0.2625, 0.2000)，求和 1.00。逐项复算与页面一致；"位置 7 最高、位置 4 最低"成立。
- 128K = 131072 位置、61 层 / 128 头 / hidden 7168 / kv_lora_rank 512 / qk_rope_head_dim 64 与 config.json 一致（index_n_heads=64、index_head_dim=128、index_topk=2048 亦一致）。
- `.dojo/scripts/validate.py wiki/dsa/index.html` 返回 `validation ok`；8 个前置概念链接（standard-attention、mla、mqa-gqa、rope、quantization-basics、linear-attention、kda、hisparse）全部存在，页面无 research/ 路径引用，无「（待生成）」占位；overview.html 与 index.html 互链；7.2 引用的 HiSparse 页面确有"主机内存 / 每请求每层 / LRU"的相应内容。

## 二、问题

- [轻微·表述] 正文 7 处以「本页」为主语的行文自指（第 161、195、252、353、803、863、865 行）。｜引文依据：不适用｜修复要求：改为中性主语或直接陈述——第 161 行「本页讨论的是另一件事」→「这里讨论的是另一件事」；第 195 行「这一比较是本页据两者机制推出的」→「这一比较是依据两者机制推出的」；第 252 行「本页按实现口径处理」→「以下按实现口径处理」；第 353 行「本页在提到它时也会用"top-k 选择"…」→「下文提到它时也会用…」；第 803 行「本页只取其定性结论」→「此处只取其定性结论」；第 863 行「本页只用到 NSA 支撑的那一条约束」→「这里只用到 NSA 支撑的那一条约束」；第 865 行「本页的 DSA 特指…」→「这里的 DSA 特指…」。第 146 行的标题「进入本页前需要的基础」与「来源与范围说明」第 917 行属导航/范围声明，可保留。｜修复：｜复验：
- [轻微·表述] 全文 6 处元话语与"提醒读者"式插入语：第 150 行「下面先把代价的结构说清楚——只有知道钱花在哪里，才能判断该砍哪一刀」、第 366 行「但要注意这只是参考实现的表达方式」、第 598 行「要注意的是 indexer 一路最终只输出一件东西」、第 600 行「链路中有三处值得单独说明」、第 610 行「本节后续正文直接以这四步的顺序指代」、第 732 行「需要注意的是，vLLM 中这条合并路径有实现约束」。｜引文依据：不适用｜修复要求：删除或改写为直接陈述，例如第 598 行改为「indexer 一路最终只输出一件东西——选中的位置；它不参与输出的加权计算」、第 732 行改为「vLLM 中这条合并路径有实现约束：…」，第 600 行改为「链路中有三处需要单独说明的部件」。｜修复：｜复验：
- [轻微·技术] 4.1 末尾 callout（第 410 行）「"每头独立选择"在表达能力上不会更差，但它在 kernel 上跑不快，因此没有被采用」中，"表达能力上不会更差"是无来源支持的推断，论文该处只给出 kernel 层共享约束这一理由，而页面在别处（第 195、604 行）对同类推断均明确标注为推断，此处未标注。｜引文依据：论文原文仅 "At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency ([Yuan et al., 2025])"，未涉及表达能力比较。｜修复要求：删去该分句，或改为与第 195、604 行一致的显式推断标注（如「这一判断依据两者机制推出，论文只给出 kernel 层理由」）。｜修复：｜复验：
- [轻微·技术] C12 的引用位置不精确：页面写「S4 vllm/model_executor/layers/mla.py 中 indexer_rope_emb 的构造与注释」，但 mla.py 中只有字段赋值（`:92` `self.indexer_rope_emb = mla_modules.indexer_rotary_emb`）与调用（`:206`），既无 rope 构造也无相关注释，全文件不含 "interleav" 字样。｜引文依据：vLLM 中 indexer 的 rope 实际构造在 `vllm/model_executor/models/deepseek_v2.py:1122-1127`：`self.indexer_rope_emb = get_rope(qk_rope_head_dim, …, is_neox_style=not getattr(config, "indexer_rope_interleave", False))`（默认 neox 即非 interleaved），随后经 `:1158 indexer_rotary_emb=self.indexer_rope_emb` 传入 `MLAModules`。｜修复要求：把该处引用改为 `vllm/model_executor/models/deepseek_v2.py` 中 `indexer_rope_emb = get_rope(..., is_neox_style=not config.indexer_rope_interleave)` 的构造（mla.py 仅为字段转接）。论断本身（vLLM 为 indexer 单独构造 rope 模块）经核对成立，无需改动正文。｜修复：｜复验：
- [轻微·可读性] 7.4 小标题「效果上的代价」与本节内容不符：该节通篇报告的是论文未观察到明显性能回退（"we do not observe substantial performance degradation…"）及其成立条件，正文并未给出任何效果上的代价。｜引文依据：论文原文 "While DeepSeek-V3.2-Exp significantly improves computational efficiency on long sequences, we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks."｜修复要求：把小标题改为与内容相符的表述（如「7.4 效果上是否回退，以及结论的成立条件」），或在本节内补出论文报告的实际效果损失（若有）。｜修复：｜复验：

## 三、结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布。核心论断（唯一架构改动、indexer 公式与四步计算链、top-k 选择与跨头共享、两阶段续训的超参与损失、O(L²)→O(Lk) 与 indexer 仍 O(L²)、KV cache 不减少、短序列回退、parity 条件）全部有可定位的引文依据；页面内可运行代码实际执行输出与「预期输出」逐字一致；公式与构造示例全部可复算；未发现同一页内互相矛盾或数字与来源不符之处。5 项轻微问题为表述与引用精度问题，不阻断发布（其中第 3、4 项建议在本轮一并修复）。

> 本轮所列问题的处理结果见 `minor-fixes.md`。

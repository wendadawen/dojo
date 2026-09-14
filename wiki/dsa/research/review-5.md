<!-- review-meta
round: 5
page: wiki/dsa/index.html
reviewed_content_sha256: 08c4f92ecfc3bb6b
-->
# DeepSeek Sparse Attention（DSA）审查记录（第 5 轮）

- 页面版本：index.html `477c674f3f2434b6da620d80294ff0f8668317bd`（overview.html `b041495d282ef9ce1b0e9fa46f52c8ffd9190473`）
- 审查时间：2026-09-14 16:54
- 审查者：独立子代理（未参与写作与历次修复，本轮仅使用 index.html、overview.html、页面引用的外部来源与 guides/concept/check.md、guides/concept/style-guide.md；未读取 research/ 下任何文件）
- 已完整阅读章节（按顺序，含全部折叠块与图注）：head（description / dojo:summary / dojo:type / 主要依据）→ 引言 → 核心问题（页面级 5 条）→ 常见误解 → 进入本页前需要的基础 → 1. 长上下文里，注意力的代价长在哪里（+本章问题）→ 2. 两条减负路线，以及 DSA 选的那条（+本章问题）→ 3. lightning indexer 如何用极低成本给每个位置打分（3.1 在贯穿例子上手算、3.2 便宜来自五处、+本章问题）→ 4. top-k 选择，以及候选集为什么必须跨头共享（4.1、+本章问题）→ 5. indexer 的打分能力是训出来的（5.1、5.2、+本章问题）→ 6. 工程落地：从 hidden state 到稀疏注意力的完整链路（6.1、6.2、6.3、+本章问题）→ 7. 省了什么、没省什么、什么时候不该用（7.1–7.5、+本章问题）→ 8. 与相邻方法的定位（+本章问题）→ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 来源核对依据（逐条回源，本轮实核）

S1 = arXiv:2512.02556（2025-12-02 提交版），正文取自 https://ar5iv.labs.arxiv.org/html/2512.02556 ：

- C1/C2：「the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training」「DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp」。
- C3：「The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism」；组件名确为 fine-grained token selection mechanism。
- C4：「For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA」「we implement DSA based on the MQA mode of MLA」；kernel 约束确引 NSA（Yuan et al., 2025，文献 [30]）：「At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency」。与页面「两层理由」的拆分一致。
- C5：「We choose ReLU as the activation function for throughput consideration.」
- C6：「the lightning indexer has a small number of heads and can be implemented in FP8 … its computational efficiency is remarkable」；全文 FP8 仅此一处，确未对主注意力精度作任何声称，页面 3.2 表格第三列「未给出主注意力的精度对照」成立。
- C7：「DSA reduces the core attention complexity of the main model from 𝒪(L²) to 𝒪(Lk)」「Although the lightning indexer still has a complexity of 𝒪(L²)」「it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus」。
- C8：「for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA」，其作用为「can achieve higher efficiency under short-context conditions」。
- C10：「we detach the indexer input from the computational graph for separate optimization」，indexer 只由 ℒ^I 训练、主模型只由语言建模损失优化。
- C11：Parity Evaluation 三段实核——Standard Benchmark（「In September 2025 … compare it with DeepSeek-V3.1-Terminus showing similar performance」／「we do not observe substantial performance degradation」）；Human Preference（「ChatbotArena as an indirect evaluation framework」「Both models share one post-training strategy」「Elo scores, obtained from evaluations conducted on 10 November 2025, are closely matched」）；Long Context Eval（「DeepSeek-V3.2-Exp scores four points higher than DeepSeek-V3.1-Terminus in reasoning mode」／Fiction.liveBench「consistently outperforms … across multiple metrics」）。页面 7.4 三组证据、时间点与两条对照条件的归属均无误。
- C15：「the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus」；续训起点「Starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K」。
- F1–F4 编号实核：Eq.1 `I_{t,s}=Σ_j w^I_{t,j}·ReLU(q^I_{t,j}·k^I_s)`、Eq.2 `u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})`、Eq.3 dense warm-up KL、Eq.4 sparse training KL（含 `S_t = {s | I_{t,s} ∈ Top-k(I_{t,:})}`）。四条公式页面均为原样照录、无代数变形。
- N：dense warm-up「learning rate 10⁻³」「train the indexer for only 1000 steps」「16 sequences of 128K tokens」「a total of 2.1B tokens」；sparse training「learning rate of 7.3×10⁻⁶」「15000 steps」「480 sequences of 128K tokens」「943.7B tokens」「we select 2048 key-value tokens for each query token」；Figure 3 口径「H800 clusters」「a rental price of 2 USD per GPU hour」「(a) Prefilling / (b) Decoding」，定性结论取「DSA achieves a significant end-to-end speedup in long-context scenarios」——页面 7.4 末段「只取定性结论（长上下文下端到端显著加速）、不复述曲线读数」与原文相符。

S2/S3 = huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp，commit `194c67e12b1b0d6df0ef373ddcf215bc84027409`（经 HF API 核对：main 分支 targetCommit 与该 sha 完全一致，commit 存在）：

- S3 config.json 实核：index_n_heads 64、index_head_dim 128、index_topk 2048、num_hidden_layers 61、num_attention_heads 128、hidden_size 7168、kv_lora_rank 512、qk_rope_head_dim 64——与页面 N 小节及正文（$H^I=64$、$d^I=128$、$k=2048$、128 头、61 层、hidden 7168、KV latent 512、每头 key 128+64）逐个吻合。
- S2 inference/model.py：`Indexer` 注册 `k_cache`（`torch.float8_e4m3fn`，形状 [max_batch, max_seq_len, head_dim]）与 `k_scale_cache`（float32）→ C9；`q = self.wq_b(qr)`（:460）、`k = self.k_norm(self.wk(x))`（:466-467）→ C12；两处注释 `# rope in indexer is not interleaved`（:463、:469）→ C12「两处注释」为实数；`rotate_activation` 定义 `hadamard_transform(x, scale=hidden_size ** -0.5)`（:428-432）且调用在 `act_quant` 之前（:472-475）→ C13「带 1/√d 缩放的 Hadamard」成立；`weights = self.weights_proj(x.float()) * self.n_heads ** -0.5`、再 `* q_scale * self.softmax_scale`（:478-479）→ 正文「$w^I_{t,j}$ 额外乘 $1/\sqrt{H^I}$ 与 softmax 缩放并合进 q_s」成立；`MLA` 注册 `kv_cache`(kv_lora_rank)/`pe_cache`(qk_rope_head_dim)（:551-552）、`qr = self.q_norm(self.wq_a(x))`（:560）→ C9/C12；`mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)`（:903）→ 页面「triu_(1) 从对角线上方开始屏蔽、对角线不屏蔽，s 取遍 t 之前的位置以及 t 自身」成立。
- S2 inference/kernel.py：`fp8_index` docstring 四步原文为「fp8 q @ fp8 k -> fp32 logits / relu(fp32 logits) * q_s (weights) -> fp32 logits / fp32 logits -> fp32 logits_sum / fp32 logits_sum * k_s (e8m0) -> fp32 index_score」→ 6.2 四步与伪代码的 q_s[batch,m,h]、k[batch,n,d]、k_s[batch,n]、输出[batch,m,n] 全部对得上。

S4 = vllm-project/vllm，commit `5ac2684976ee22c04fe0d2f968c6cf6096b383f2`：

- `sparse_mla_attention.py`：`use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not self.vllm_config.attention_config.sparse_mla_force_mqa)`（:279-282），`self.topk_tokens = vllm_config.model_config.hf_config.index_topk`（:117）→ 页面 7.3 判据成立（「不超过 index_topk」即 hf_config.index_topk）。
- `mla_attention.py`：消费侧 `use_mha = (use_dense_mha or use_masked_mha) and not sparse_mla_force_mqa`（:804-806）→ 页面「消费侧还有进一步的条件判断」成立。
- `sparse_attn_indexer.py`：`_merge_dcp_topk_global` docstring「A token in the global top-K must also be in its owning rank's local top-K (at most topk_tokens - 1 tokens rank globally above it, hence at most that many on its own rank), so exchanging only the per-rank local candidates is exact … but it ships dcp_world_size * topk_tokens candidates instead of the whole score row」→ C14 与 6.3 正文逐句对应；`_assert_cutedsl_dcp_merge_supported` 实核「requires CuteDSL」与「index_topk in (512, 1024, 2048)」。
- `deepseek_v2.py`：`self.indexer_rope_emb = get_rope(... is_neox_style=not getattr(config, "indexer_rope_interleave", False))` 起于 :1122（页面标注的行号正确）；`DeepseekV32IndexerCache` 定义于 :616。
- `layers/attention/mla.py`：`q_c = self.q_a_layernorm(q_c)`（:175）、`self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`（:206）→ C12 行号标注正确。

机械核对与复算：

- 可运行代码：从页面抽取 `<pre><code class="language-python">` 段并实际执行，8 行逐位置输出与页面「预期输出」逐字符一致，末两行 `k=3 时选中的位置: [2, 7, 8]`、`k=8 时选中的位置: [1, 2, 3, 4, 5, 6, 7, 8]（序列长 8，未裁剪）` 与页面一致。
- 贯穿例子复算：8 个 index score（2.00/2.50/0.25/0.50/0.00/0.75/4.00/3.00）与降序序列、$\{2,7,8\}$ 选中集合一致；位置 3、4 的 ReLU 截断示例一致。
- warm-up 目标分布复算：跨头和 (0.60, 0.80, 0.15, 0.12, 0.18, 0.30, 1.05, 0.80)，总和 4.00，逐项除得 $p_{8,:}=(0.1500,0.2000,0.0375,0.0300,0.0450,0.0750,0.2625,0.2000)$，各项和为 1.0——与页面完全一致。
- 规模比复算：943.7 / 2.1 ≈ 449.4，「约 450 倍」成立。
- `.dojo/scripts/validate.py wiki/dsa/index.html` → `validation ok`（exit 0）。
- 链接与前置概念：全部本地 href/src 解析成功；standard-attention、mla、mqa-gqa、rope、quantization-basics、linear-attention、kda、hisparse 八个被引概念页 index.html 均存在；全页无「（待生成）」，无 `alt` 属性含 `$...$`（唯一 `<img>` 为 lightbox 占位，alt 为空）。
- 双向引用：正文 `<sup>[Cx]</sup>` 实际使用 C1–C15，来源章 C 节定义 C1–C15，一一对应无缺无余；`<sup>[Fx]</sup>` 使用 F1–F4，F 节定义 F1–F4。
- 章节交叉引用：正文所引「1./2./3./3.1/4./5./7./7.1/7.2/8.」各标题与目标 h2/h3 标题逐字一致。
- 符号单义：$k$（选取数量）与 $\mathbf{k}^I_s$（key 向量）已在 4 章显式区分；$d^I$、$H^I$、$w^I$ 全页写法一致；`dojo:summary` 内的 `$O(L^2)$`、`$O(Lk)$`、`$L\le k$` 均可用 KaTeX 渲染。公式可复算、符号全文单义。
- 简化条件：贯穿例子（8 位置 / $k{=}3$ / $H^I{=}2$ / $d^I{=}2$）与 warm-up 目标分布的数字均在首次出现处标注「人为构造」，并列入「构造示例」小节；掩码等价性、FP64 代码、工程链路图的省略项都在「简化条件及其限制」中写明可支持/不可推出的结论。

## 问题

- [轻微·表述] 「7. 省了什么、没省什么、什么时候不该用」章首（index.html:760）：独立成段的一句话「把账算清楚。」属口语化短句，与全页其余章节开篇的陈述性写法不一致，未承担信息量。｜引文依据：不适用｜修复要求：删除该段，或改写为不含口语色彩的陈述性引导句（如说明本章按「计算 / 显存 / 边界」三项分别结账）。｜修复：｜复验：
- [轻微·表述] 「2. 两条减负路线，以及 DSA 选的那条」第 3 段（index.html:195，同一句亦见于 overview.html:41）：括号插入语「（这一点可从机制直接核对）」面向读者说明如何验证，属元话语；该句所在段已在正文给出被选中位置照旧走完整 softmax 的机制说明，插入语无额外信息，且在 overview 中原样重复。｜引文依据：不适用｜修复要求：删除该括号插入语（两处同删）；若需保留「该结论不依赖论文原文」这层信息，改用来源标注方式（如 `<sup>[F2]</sup>`）。｜修复：｜复验：
- [轻微·表述] 「5. indexer 的打分能力是训出来的」本章问题第 3 题 summary（index.html:529）与「7.5 适用边界小结」（index.html:809）：两处用口语化的「掉点」（「效果显著掉点」「会显著掉点」），而同一题正文（index.html:530）与 7.4 用的是「性能下降」，同一概念全页写法不一。｜引文依据：不适用｜修复要求：两处「掉点」统一改为「性能下降」或「效果下降」，与 530 行、797 行用词一致。｜修复：｜复验：
- [轻微·格式] 「来源与范围说明」下「外部数字与实验条件（N）」小节（index.html:899-900）：该小结目无 N1/N2… 编号，正文（如 index.html:263 的 $k=2048$、$H^I=64$、$d^I=128$，index.html:500-503 的两阶段超参与 token 量）也没有任何 `<sup>[Nx]</sup>` 上标，与 style-guide 第 6 节「正文使用 `<sup>[Cx]</sup>` 上标引用（C=论断、F=公式、N=数字），与来源章节双向对应」不符；本页 C 节与 F 节均已做到双向对应，仅 N 节缺失。｜引文依据：不适用｜修复要求：按 C/F 的做法给 N 条目编号（N1…Nn）并在正文对应数字处补 `<sup>[Nx]</sup>`；若确认 N 小节有意不编号，则需在本节内写明该做法及理由。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（仅 4 项轻微表述与格式项；阻断与重要均为 0，本页核心结论、公式、数字、引文编号、跨页一致性、代码输出均经回源核对无误，不阻塞发布）
- 来源核对摘要：S1 的 C1–C8、C10、C11、C15 与 F1–F4、N 全部找到原文片段；S2/S3 的 commit 与 config.json、model.py、kernel.py 关键行逐条对上；S4 的四个文件与页面标注的行号（deepseek_v2.py:1122、mla.py:175/206）逐条对上；唯一旧编号存疑项（vLLM `is_neox_style` 行号）本轮实测亦准确。图内无数值、无 SVG，无「图注读数与刻度不符」类问题（本页结构图为 HTML 文字块，非坐标图）。

统计：阻断 0 / 重要 0 / 轻微 4
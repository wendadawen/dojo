<!-- review-meta
round: 3
page: wiki/dsa/index.html
reviewed_content_sha256: 2e0cc4d08acba829
-->
# DSA 审查记录（第 3 轮）

- 页面版本：`git hash-object wiki/dsa/index.html` = `405c9fd66d1bf07af759642cc47bd28e1e3a22b0`（sha256 前 16 位 `9105a70b40f419eb`）；`overview.html` 一并审查
- 审查时间：2026-09-10 21:39
- 审查者：独立审查者（未参与写作，未读取 `wiki/dsa/research/` 下任何文件）
- 已完整阅读章节：核心问题、常见误解、1 长上下文里，注意力的代价长在哪里、2 两条减负路线，以及 DSA 选的那条、3 lightning indexer 如何用极低成本给每个位置打分（3.1、3.2）、4 top-k 选择，以及候选集为什么必须跨头共享（4.1）、5 indexer 的打分能力是训出来的（5.1、5.2）、6 工程落地：从 hidden state 到稀疏注意力的完整链路（6.1、6.2、6.3）、7 省了什么、没省什么、什么时候不该用（7.1–7.5）、8 与相邻方法的定位、来源与范围说明；`overview.html` 全文。全文 35 个 `<details>` 折叠块逐块展开阅读。

## 机械验证结果

命令：`/usr/bin/python3 .dojo/scripts/validate.py wiki/dsa/index.html`

```text
validation ok: /Users/wendadawen/code/dojo/wiki/dsa/index.html
EXIT=0
```

另外核对的项：

| 项 | 结果 | 依据 |
|---|---|---|
| `validate.py`（index） | 通过 | 见上，exit 0 |
| `validate.py`（overview） | 通过 | `validation ok: .../overview.html` |
| 引用双向闭合 | 通过 | 正文 28 处 `<sup>[Cx]</sup>`；出现编号 C1–C15、F1–F4 共 19 个，与「来源与范围说明」中的定义集完全相等（有引用无定义 0，有定义无引用 0） |
| 相邻双上标 | 通过 | `</sup>\s*<sup>` 匹配 0 处 |
| Unicode 数学字符 | 通过 | 非代码区 U+2212、U+00D7 均 0 次；仅出现 U+2192（5 处，流程图「→」箭头标签）与 U+00B7（1 处，阅读时间的「·」分隔符），二者非数学符号。`validate.py` 亦未报错 |
| TAB | 通过 | `<body>` 内 0 个制表符 |
| 占位符 | 通过 | 「待生成 / TODO / TBD / 占位 / FIXME / XXX」均 0 次 |
| `<head>` 五项元数据 | 通过 | `description`（纯文本）、`dojo:summary`（含 `$O(L^2)$`、`$L\le k$`，LaTeX 书写）、`dojo:type=concept`、`dojo:topics=注意力机制`（属 AGENTS.md:29 八类词表内）、`dojo:tag=注意力机制` |
| overview ↔ index 互链 | 通过 | `overview.html:38` → `index.html`；`index.html:654` → `overview.html` |
| 前置概念链接有效 | 通过 | `../standard-attention/`、`../mla/`、`../mqa-gqa/`、`../rope/`、`../quantization-basics/`、`../linear-attention/`、`../kda/` 七个 `index.html` 均存在 |
| 可运行代码实跑 | 通过 | 抽取 `python` 代码块用 `/usr/bin/python3` 实跑（exit 0），stdout 与页面「预期输出」逐字符一致（去尾换行后 `==` 判定 True），含 `k=3 时选中的位置: [2, 7, 8]` 与 `k=8 时选中的位置: [1, 2, 3, 4, 5, 6, 7, 8]（序列长 8，未裁剪）` |
| 本地资源 | 通过 | `katex.min.css/js`、`auto-render.min.js`、`prism-primer-light/dark.css`、`prism.min.js`、`prism-python.min.js` 全部存在 |
| 折叠块配对与命名 | 通过 | `<details>` 35 / `</details>` 35 / `<summary>` 35；前缀分布 解答 30、补充 2、代码 2、展开 1，无一例外 |
| 问题块完整性 | 通过 | `核心问题` 1 组（5 题）+ 8 个章节 `本章问题`（3/3/3/3/3/3/4/3 题）；9 组 `ol.chapter-questions` 共 35 个 `<li>`，每个 `<li>` 均配 1 个 `<details>`，无只列问题未作答 |
| 章节编号连续 | 通过 | h2 为 1–8 连续无跳号；h3 在所属章内从 1 连续递增；`来源与范围说明` 的 6 个 h3 使用规范固定命名 |
| 代码块数学符号豁免 | 通过 | `<pre>`/`<code>` 内的 `q^I_{t,j}` 等为代码原文，不受 style-guide §11 约束 |

补充说明（不构成本轮问题）：`overview.html` 未带 `description` / `dojo:*` 元数据。抽样 `mla`、`mqa-gqa`、`latent-moe`、`pcp-dcp` 的 `overview.html` 同样未带（`rope` 仅有 `description`），仓库内 81 个 `overview.html` 一致，且 `validate.py` 对 `overview.html` 返回通过，按现有约定不计为问题。

## 问题

- [轻微·技术] 来源与范围说明 C12 条：vLLM 侧出处「`vllm/model_executor/layers/mla.py` 中 `indexer_rope_emb` 的构造与注释」定位不准确｜引文依据：`mla.py:92` 实际内容只有赋值 `self.indexer_rope_emb = mla_modules.indexer_rotary_emb`，该文件无与之相关的注释；真正的构造在另一文件 `vllm/model_executor/models/deepseek_v2.py:1122`：`self.indexer_rope_emb = get_rope(qk_rope_head_dim, max_position=max_position_embeddings, rope_parameters=config.rope_parameters, is_neox_style=not getattr(config, "indexer_rope_interleave", False))`。同条其余内容均核对通过：`mla.py:206` 确为 `self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`，`:175` 确为 `q_c = self.q_a_layernorm(q_c)`；正文「vLLM 同样为 indexer 单独构造了一份 rope 模块」这一实质结论由 `deepseek_v2.py:1122` 完整支持，只是 C12 标注的文件与「注释」名实不符｜修复要求：把 C12 的 vLLM 出处由 `layers/mla.py` 改为 `executor/models/deepseek_v2.py:1122`，并将「的构造与注释」改述为 `is_neox_style=not indexer_rope_interleave`（或删去「与注释」三字）；正文 6.1 节无需改动｜修复：｜复验：

本轮未发现阻断或重要问题。以下为已核对但判定成立、不列入问题的高风险项，附核对片段备查：

- C1/C2（`arxiv.org/html/2512.02556v1` § 2.1 首段）：「DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, the last version of DeepSeek-V3.1, the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training.」——同时支持 C1 与 C2，页面对「唯一架构改动」未扩大范围。
- C3/F1/F2（§ Prototype of DSA）：「a lightning indexer and a fine-grained token selection mechanism」；Eq. 1 `I_{t,s}=\sum_{j=1}^{H^I} w_{t,j}^{I}\cdot \text{ReLU}(\mathbf{q}^{I}_{t,j}\cdot \mathbf{k}^{I}_{s})`；Eq. 2 `\mathbf{u}_t=\text{Attn}(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t,s}\in\text{Top-k}(I_{t,:})\})`。页面公式与符号清单与原文式一致，未做代数变形。同段「we choose ReLU as the activation function for throughput consideration」「has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable」支持 C5、C6。
- C4（§ Instantiate DSA Under MLA）：「For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA. At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA mode of MLA」——页面把「续训」与「kernel 共享约束」拆成两层归因，与原文一致。
- C7/C8（§ Inference Costs）：「DSA reduces the core attention complexity of the main model from O(L^2) to O(Lk)… Although the lightning indexer still has a complexity of O(L^2), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus.」「for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions.」；成本口径「benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour」与页面 7.4 一致。
- C8 的 vLLM 判据：`attention/sparse_mla_attention.py` 中 `use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not self.vllm_config.attention_config.sparse_mla_force_mqa)`，`prefill_max_seq_len` 由 `seq_lens_cpu[num_decodes : num_decodes + num_prefills].max()` 得到；消费方 `forward_mha` 另有 `force_dense` / `force_masked` 覆盖分支。页面 7.3、本章问题 3 的表述（最大序列长度不超过 `index_topk`、未强制走 MQA、消费侧还有进一步条件判断）逐项吻合。
- C10/F3/F4 与 N 组超参：Eq. 3、Eq. 4 原文与页面一致；「For warm-up, we use a learning rate of 10^{-3}. We train the indexer for only 1000 steps, with each step consisting of 16 sequences of 128K tokens, resulting in a total of 2.1B tokens.」「we use a learning rate of 7.3×10^{-6}, and select 2048 key-value tokens for each query token… 15000 steps… 480 sequences of 128K tokens… 943.7B tokens」「we detach the indexer input from the computational graph for separate optimization」全部匹配。复算：16×1000×131072≈2.1B、480×15000×131072≈943.7B、943.7/2.1≈450 倍，与页面 5.2 一致。
- C11（§ Parity Evaluation 三段）：「In September 2025… do not observe substantial performance degradation… on both short- and long-context tasks」「their Elo scores, obtained from evaluations conducted on 10 November 2025, are closely matched… share an identical post-training strategy」「AA-LCR, in which DeepSeek-V3.2-Exp scores four points higher than DeepSeek-V3.1-Terminus in reasoning mode… Fiction.liveBench… consistently outperforms」——页面 7.4 的三个时间点、两个对照条件与三组证据逐一对应。
- C15（§ 2.1.1 Continued Pre-Training 首段）：「Starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K… For both stages, the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus.」——逐词对应页面 C15 引文与 N 组「续训起点 128K」。
- C9/C12/C13 与 F 条实现差异（S2 `inference/model.py` commit `194c67e`）：`Indexer.__init__` 注册 `k_cache`（`float8_e4m3fn`）与 `k_scale_cache`（`float32`），`MLA.__init__` 注册 `kv_cache` / `pe_cache`，支持 C9；`Indexer.forward` 中 `q = self.wq_b(qr)`、`k = self.wk(x); k = self.k_norm(k)`（页面写为复合式 `self.k_norm(self.wk(x))`，语义等价）、`qr = self.q_norm(self.wq_a(x))`（`MLA.forward:560`）、两处 `# rope in indexer is not interleaved` 注释（`:463`、`:469`）、`rotate_activation` 定义为 `hadamard_transform(x, scale=hidden_size ** -0.5)`（即 1/√d）、`weights = self.weights_proj(x.float()) * self.n_heads ** -0.5` 再 `* q_scale * self.softmax_scale`，分别支持 C12、C13 与 F 条中「w^I 额外乘 1/√(H^I) 与 softmax 缩放」的实现差异说明；`apply_rotary_emb(..., interleaved: bool = True)` 默认值与 indexer 显式传 `False` 支持「与主注意力应用方式不同」。
- C6 的四步计算链（S2 `inference/kernel.py`）：`fp8_index` docstring 原文四行为「fp8 q @ fp8 k -> fp32 logits / relu(fp32 logits) * q_s (weights) -> fp32 logits / fp32 logits -> fp32 logits_sum / fp32 logits_sum * k_s (e8m0) -> fp32 index_score」，`fp8_index_kernel` 内 `T.gemm(...)`、`T.max(logits,0)*q_s_frag`、`T.reduce_sum(logits, logits_sum, dim=1)`、`logits_sum *= k_s_frag` 逐步对应。页面 6.2「步骤 4 不对应公式任何一项」成立。
- C14（S4 `layers/sparse_attn_indexer.py`）：`_merge_dcp_topk_global` docstring 原文「A token in the global top-K must also be in its owning rank's local top-K (at most `topk_tokens - 1` tokens rank globally above it, hence at most that many on its own rank), so exchanging only the per-rank local candidates is exact… but it ships `dcp_world_size * topk_tokens` candidates instead of the whole score row.」；`_assert_cutedsl_dcp_merge_supported` 中 `if k not in (512, 1024, 2048): raise RuntimeError(...)` 与「requires CuteDSL」支持页面 6.3 与折叠块中的实现约束说明。
- 掩码与公式的关系（S2 `model.py:583-588`、`:600-605`）：`index_mask = torch.full((bsz, seqlen, seqlen), float("-inf")).scatter_(-1, topk_indices, 0)`；`scores += index_mask…`；`scores = scores.softmax(dim=-1)`。页面 4 节「置 −∞、加回分数、再 softmax」与「掩码版本仍把全部位置算了一遍」均成立。同文件 `:903` `mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)`，对角线不屏蔽，支持页面 3 节「$s$ 取遍 $t$ 之前的位置以及 $t$ 自身、贯穿例子中位置 8 也参与打分」这一与论文口径不同的实现口径说明（页面已明确标注为「按实现口径」）。
- S3（同 commit 仓库根 `config.json`）：`num_hidden_layers=61`、`hidden_size=7168`、`num_attention_heads=128`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`index_n_heads=64`、`index_head_dim=128`、`index_topk=2048`，与页面 N 组及 3.2 节对照表逐项一致（`inference/model.py` 的 `ModelArgs` 默认值仅为小模型示例值，故 128 头/61 层/7168 必须由 `config.json` 支撑，已核对）。
- 构造示例复算：3.1 节 8 个位置的手算分数、4 节降序与 top-k 选中集 {2, 7, 8}、5.1 节 4 头求和向量 (0.60, 0.80, 0.15, 0.12, 0.18, 0.30, 1.05, 0.80) 与 L1 归一化结果 (0.1500, 0.2000, 0.0375, 0.0300, 0.0450, 0.0750, 0.2625, 0.2000)（和恰为 1）全部复算正确；8 个分数两两不等，无并列歧义，与「构造示例」小节声明的三个构造条件一致。
- 简化与推断标注：ReLU/softmax 规约差异、Hadamard 动机、排序容忍度三项均在「辅助解释与类比边界」中标注为教学解释或推断，并写明不可推出的结论；C9、C14、C8 的 vLLM 判据均标明「从实现推出」「非论文内容」。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（附条件，见下）

### check.md §5 发布条件逐条判定

| 条件 | 判定 | 依据 |
|---|---|---|
| 三轮审查均已完成且每轮由独立审查者执行 | 本轮为第 3 轮、由独立审查者执行；前两轮是否独立无法自证 | 本报告独立完成，未读 `research/` |
| 每条来源论断都有引文依据记录；无法核对者已删除或降级 | 满足 | 19 个编号（C1–C15、F1–F4）逐条定位并记录原文片段或关键数值；推断类陈述已显式标注为教学解释/推断 |
| 所有阻断和重要问题均已关闭 | 满足 | 本轮 0 阻断 0 重要 |
| 遗留轻微问题具有明确的接受理由 | 满足（1 条，理由见下） | 见「轻微问题接受理由」 |
| 全部学习目标由正文章节完整回答 | 满足 | 核心问题 5 题分别指向第 1–2、3、4、5、7 章，各章正文覆盖 |
| 两级问题块均有解答折叠块 | 满足 | 1 组核心问题 + 8 组本章问题，共 35 题 35 个解答，无只列不作答 |
| 数学符号全部 LaTeX、结构图为 HTML 或内联 SVG | 满足 | `validate.py` 通过；非代码区无 U+2212/U+00D7 等数学字符；4 处流程图为 HTML `fd-*` 结构，公式由 KaTeX 渲染于 HTML 文本节点 |
| `validate.py` 返回成功 | 满足 | exit 0 |
| 可运行代码结果与页面描述一致 | 满足 | 实跑 stdout 与「预期输出」逐字符一致 |
| 关键论断与数字已重新核对来源 | 满足 | 见上「机械验证结果」后的逐条片段 |
| `<head>` 五项元数据有效且 `dojo:topics` 在词表内 | 满足（index.html） | 五项齐备，`注意力机制` 属 AGENTS.md:29 八类词表 |
| `overview.html` 与 `index.html` 相互链接 | 满足 | 双向链接均在 |
| 引用概念链接有效或有明确占位 | 满足 | 7 个前置概念页均存在，无「（待生成）」占位 |
| 递归生成的前置概念页已完成各自质检 | **无法确认** | 见下 |

**轻微问题接受理由**：C12 条对 vLLM 出处的文件定位偏差（`layers/mla.py` 应为 `models/deepseek_v2.py:1122`，且该处无注释）不改变任何结论——正文 6.1 节「vLLM 为 indexer 单独构造了一份 rope 模块」由 `deepseek_v2.py:1122` 的 `get_rope(..., is_neox_style=not getattr(config, "indexer_rope_interleave", False))` 完整支持，且索引器 RoPE 非 interleaved 这一核心论断另有 S2 两处代码注释与实现口径双重支撑。该缺陷只影响来源溯源时的定位效率，不影响读者理解、不构成误导，故接受为遗留轻微问题；建议在下一次对该页做任何修改时顺手改正。

**关于「递归生成的前置概念页已完成各自质检」**：无法确认。可核对到的事实是——本页正文引用的 7 个前置概念页（`standard-attention`、`linear-attention`、`kda`、`mla`、`mqa-gqa`、`rope`、`quantization-basics`）均已存在且链接有效，其 `research/` 目录各有 `review.md` 与 `review-2.md`（`rope` 另有 `review-1.md`、`review-3.md`）。按 `review-{轮次}.md` 的命名约定读，其中 6 页只留下两轮记录，是否三轮齐备无法从文件名确认；本轮不读其他页面的 `research/` 内容（与本页独立性同源），也不读 `wiki/dsa/research/`。另外，按本页「来源与范围说明」与页面互链看，这 7 页均在本页写作前已存在，不属于本页递归生成的产物，该条是否适用于本页存在判读分歧。此条不涉及本页内容正确性，建议由编排者核对前置页质检记录后确认。

**发布建议**：除上述一条无法确认的元条件外，`check.md` §5 其余全部条件均满足；本页可作为发布候选项，等待编排者确认前置概念页质检状态后放行。

<!-- review-meta
round: 1
page: wiki/dsa/index.html
reviewed_content_sha256: 2e0cc4d08acba829
-->
# DeepSeek Sparse Attention（DSA）审查记录（第 1 轮）

- 页面版本：`index.html` 工作树哈希 `c1cc1b5c08ffad0ced514ecb780615dc592f9312`；`overview.html` 工作树哈希 `5d299b2f2859bdd240d3c097ecc8ccf4ff552c71`
- 审查时间：2026-09-10 19:09
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查；未读取 `wiki/dsa/research/` 下任何文件）
- 已完整阅读章节（按顺序）：h1 与 blockquote.meta → 核心问题（5 条，含解答）→ 1 长上下文里，注意力的代价长在哪里 → 2 两条减负路线，以及 DSA 选的那条 → 3 lightning indexer 如何用极低成本给每个位置打分（3.1、3.2）→ 4 top-k 选择，以及候选集为什么必须跨头共享（4.1）→ 5 indexer 的打分能力是训出来的（5.1、5.2）→ 6 工程落地：从 hidden state 到稀疏注意力的完整链路（6.1、6.2、6.3）→ 7 省了什么、没省什么、什么时候不该用（7.1–7.5）→ 8 与相邻方法的定位 → 来源与范围说明（六个 h3）。全部 35 个 `<details>` 折叠块均已展开逐句阅读。`overview.html` 全文读完。
- 外部来源：arXiv:2512.02556v1 HTML 全文（`https://arxiv.org/html/2512.02556v1`，已下载为纯文本逐段核对）；`deepseek-ai/DeepSeek-V3.2-Exp` commit `194c67e12b1b0d6df0ef373ddcf215bc84027409` 的 `inference/model.py`、`inference/kernel.py`、根目录 `config.json`；vLLM commit `5ac2684976ee22c04fe0d2f968c6cf6096b383f2` 的 `vllm/model_executor/layers/sparse_attn_indexer.py`、`vllm/model_executor/layers/attention/sparse_mla_attention.py`、`vllm/model_executor/layers/attention/mla_attention.py`、`vllm/models/deepseek_v32/attention.py`、`vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py`。

## 机械验证结果

**1. `validate.py`（实际执行）**

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/dsa/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/dsa/index.html
EXIT=0
```

**2. 引用编号双向闭合（脚本核对）**

```
used in body: []
defined in source sec: ['C1'…'C14','F1'…'F4']（18 条）
used-not-defined: []
defined-not-used: ['C1'…'C14','F1'…'F4']（全部 18 条）
```

全页 `<sup>` 出现 **0 次**。正文中实际存在的引用标注为 8 处中文括号：`（C6）`（3.2）、`（C5）`（3.2）、`（C4）`（4.1）、`（C10）`（5.2）、`（C12）`（6.1）、`（C13）`（6.1）、`（C6）`（6.2）、`（C14，来源为 vLLM 实现，非论文内容）`（6.3）。即：正文与来源章节之间没有上标形式的双向对应，且 C1、C2、C3、C7、C8、C9、C11（含全部核心论断所在句）在正文中没有任何引用标注。

**3. 相邻双上标 `<sup>[X][Y]</sup>`**

`<sup>` 总数为 0，不存在相邻双上标，也不存在合并写法；该项因第 2 条不适用。

**4. 来源章节 h3 固定命名（脚本核对）**

实际为：`论断与来源（C）`、`公式与来源（F）`、`外部数字与实验条件`、`构造示例`、`辅助解释与类比边界`、`简化条件及其限制`（6 个，顺序与规范一致）。其中第 3 个缺固定后缀 `（N）`（规范固定命名为 `外部数字与实验条件（N）`）。

**5. 残留占位符**

对 `待生成`、`待补`、`TODO`、`FIXME`、`TBD`、`XXX`、`占位` 全页检索：均 0 次。`（待生成）` 式占位不存在。

**6. 可运行代码（实际执行）**

页内 `language-python` 代码块 1 个，提取后实跑：

```
$ /usr/bin/python3 /tmp/dsa_code.py
s=1  head1: dot=+2.00 relu=2.00  head2: dot=+0.00 relu=0.00  I_8,1=2.00
…
s=8  head1: dot=+1.00 relu=1.00  head2: dot=+4.00 relu=4.00  I_8,8=3.00

k=3 时选中的位置: [2, 7, 8]
k=8 时选中的位置: [1, 2, 3, 4, 5, 6, 7, 8]（序列长 8，未裁剪）
```

与页面「预期输出」块逐字符一致（含 `+2.00`/`-1.00` 符号位与两行 top-k 结果）。代码仅依赖 numpy，含全部输入定义。

**7. 链接与页面互链**

`index.html` 内 12 条本地链接、`overview.html` 内 7 条本地链接全部指向存在的文件（`../standard-attention/`、`../mla/`、`../mqa-gqa/`、`../rope/`、`../quantization-basics/`、`../linear-attention/`、`../kda/` 均有 `index.html`）；`../../libs/` 下 KaTeX/Prism 六个资源文件存在。`index.html → overview.html` 与 `overview.html → index.html` 双向链接均存在。

**8. head 元信息**

`description`（纯文本）、`dojo:summary`（含 `$O(L^2)$`/`$O(Lk)$` 公式）、`dojo:type=concept`、`dojo:topics=注意力机制`、`dojo:tag=注意力机制` 齐备；`注意力机制` 在 `AGENTS.md` 固定大类词表内。

**9. 公式与结构（其他脚本核对）**

公式定界符之外的 Unicode 数学字符：`validate.py` 通过，未发现（标题、summary、正文、列表、表格均用 `$...$`）。结构图 3 处全部为 HTML 结构（`div.flow-diagram/.fd-row/.fd-col/.fd-box/.fd-arrow`，CSS 已在页内定义），无等宽字符框线图。正文用到的 42 个 class 中仅 `language-python`、`language-text`（Prism 保留类）与 `overview-link`（无样式，无副作用）未在 CSS 中定义。用词检查：无「我们」「你们」「读者」「显然」「容易得到」，自称统一为「本页」（7 次）。

### 来源论断核对明细（check.md §2.2 四步核对）

| 编号 | 页面位置与表述 | 来源定位 | 实际原文片段 / 关键数值 | 判定 |
|---|---|---|---|---|
| C1 | 引言/概述：DSA 是 V3.2 相对 V3.1-Terminus 的唯一架构改动 | S1 §2.1 首段 | "Compared with DeepSeek-V3.1-Terminus, the last version of DeepSeek-V3.1, the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training." | 一致 |
| C2 | 来源章节与 overview：V3.2 与 V3.2-Exp 架构完全相同 | S1 §2.1 首句 | "DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp." | 一致 |
| C3 | §4 首段：DSA 由 lightning indexer 与 fine-grained token selection 两部分组成 | S1 §2.1 Prototype of DSA | "The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism." | 一致 |
| C4 | §4.1：续训是实例化在 MLA 之上的理由；kernel 层每个 KV 条目须被多个 query 共享是选用 MQA 的理由，该约束引用 NSA | S1 §2.1 Instantiate DSA Under MLA | "For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA … At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA mode of MLA …, where each latent vector will be shared across all query heads of the query token." | 一致（`Yuan et al., 2025` 即 NSA） |
| C5 | §3.2：ReLU 出于吞吐考虑，论文未展开 | S1 §2.1 | "We choose ReLU as the activation function for throughput consideration." | 一致 |
| C6 | §3.2 与 §6.2：indexer 头数少且可 FP8 实现故效率突出；四步计算链 | S1 §2.1；S2 `inference/kernel.py` `fp8_index` docstring（L260–273） | 论文："Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable."；docstring："fp8 q @ fp8 k -> fp32 logits / relu(fp32 logits) * q_s (weights) -> fp32 logits / fp32 logits -> fp32 logits_sum / fp32 logits_sum * k_s (e8m0) -> fp32 index_score" | 一致（四步与伪代码逐条对应） |
| C7 | §7 表与 §7.1：主注意力 $O(L^2)\to O(Lk)$，indexer 仍 $O(L^2)$ | S1 §2.3 Inference Costs | "DSA reduces the core attention complexity of the main model from 𝒪(L²) to 𝒪(Lk), where k (≪ L) is the number of selected tokens. Although the lightning indexer still has a complexity of 𝒪(L²), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus." | 一致 |
| C8 | §7.3：短序列 prefill 用 masked MHA 模式模拟 DSA | S1 §2.3 末段；S4 `sparse_mla_attention.py` L279–282；消费方 `mla_attention.py` L789 | "Note that for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions."；vLLM：`use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not self.vllm_config.attention_config.sparse_mla_force_mqa)`；消费方 `use_dense_mha = getattr(prefill, "use_dense_mha", False)` | 一致 |
| C9 | §6.1、§7.2：DSA 不减少 KV cache，indexer 另有独立 key cache（从实现推出） | S2 `inference/model.py` L453–454、L541–542；S4 `DeepseekV32IndexerCache` | `self.register_buffer("k_cache", torch.zeros(..., dtype=torch.float8_e4m3fn))`、`"k_scale_cache"`；MLA：`register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank))`、`"pe_cache"`；vLLM `vllm/model_executor/models/deepseek_v2.py: class DeepseekV32IndexerCache(...)` | 一致（页面已标为「从实现推出」，未写成论文结论） |
| C10 | §5.2：indexer 输入 detach，两条优化信号隔离 | S1 §2.1.1 Sparse Training Stage | "It is worth noting that we detach the indexer input from the computational graph for separate optimization. The training signal of the indexer is from only ℒ^I, while the optimization of the main model is according to only the language modeling loss." | 一致 |
| C11 | §7.4 与核心问题答案：三组证据（标准 benchmark / ChatbotArena / 独立长上下文评测） | S1 §2.2 三段 | Standard Benchmark："In September 2025, we evaluate DeepSeek-V3.2-Exp on a suite of benchmarks … we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks."；Human Preference："Both DeepSeek-V3.1-Terminus and DeepSeek-V3.2-Exp share an identical post-training strategy, and their Elo scores, obtained from evaluations conducted on 10 November 2025, are closely matched."；Long Context Eval："AA-LCR, in which DeepSeek-V3.2-Exp scores four points higher than DeepSeek-V3.1-Terminus in reasoning mode. In the Fiction.liveBench evaluation … consistently outperforms DeepSeek-V3.1-Terminus across multiple metrics." | 一致（见问题 2：成立条件的措辞需收紧） |
| C12 | §6.1：indexer key 由 hidden state 独立投影、query 复用主注意力低秩 latent、RoPE 通道独立且非 interleaved | S2 `inference/model.py` L460、L466–467、L463/L469、MLA L560；S4 `models/deepseek_v32/attention.py` L313–318、L380、L434 | `q = self.wq_b(qr)`；`k = self.wk(x)` / `k = self.k_norm(k)`；两处注释 `# rope in indexer is not interleaved`；MLA：`qr = self.q_norm(self.wq_a(x))`；vLLM：`# Lightning indexer uses its own RoPE; interleave maps to non-NeoX.` + `self.indexer_rope_emb = get_rope(..., is_neox_style=not getattr(config, "indexer_rope_interleave", False))`（默认 True ⇒ 非 interleaved，主注意力同文件 `is_neox_style=False`）、`indexer_k_rope_cos_sin_cache = self.indexer_rope_emb.cos_sin_cache`、`index_q = self.indexer.wq_b(q_c)[0]` | 一致（注释恰为 2 处） |
| C13 | §6.1：量化前调用 `rotate_activation`，即带 $1/\sqrt{d}$ 缩放的 Hadamard 变换 | S2 `inference/model.py` L428–432 | `def rotate_activation(x): … from fast_hadamard_transform import hadamard_transform; hidden_size = x.size(-1); return hadamard_transform(x, scale=hidden_size ** -0.5)` | 一致（动机已标为教学推断） |
| C14 | §6.3：DCP 下只交换局部 top-k 候选即得精确全局 top-k（来源为 vLLM 实现） | S4 `sparse_attn_indexer.py` L83–94 docstring | "A token in the global top-K must also be in its owning rank's local top-K (at most `topk_tokens - 1` tokens rank globally above it, hence at most that many on its own rank), so exchanging only the per-rank local candidates is exact -- equivalent to all-gathering the full logit matrix, but it ships `dcp_world_size * topk_tokens` candidates instead of the whole score row." | 一致（含「卡数 × k」这一传输量的原文依据） |
| F1 | §3 index score 公式 | S1 §2.1 Eq. 1 | "I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\text{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right)"；符号说明同段：$H^I$ 头数；$\mathbf{q}^I_{t,j}\in\mathbb{R}^{d^I}$ 与 $w^I_{t,j}\in\mathbb{R}$ 由 $\mathbf{h}_t$ 导出；$\mathbf{k}^I_s\in\mathbb{R}^{d^I}$ 由 $\mathbf{h}_s$ 导出 | 一致（逐符号与页面 `<ul>` 对应，未做变形） |
| F2 | §4 稀疏注意力输出公式 | S1 §2.1 Eq. 2 | "𝐮_t = Attn(𝐡_t, { 𝐜_s \| I_{t,s} ∈ Top-k(I_{t,:}) })" | 一致 |
| F3 | §5.1 dense warm-up 损失 | S1 §2.1.1 Eq. 3 | "ℒ^I=\sum_{t}\mathbb{D}_{KL}(p_{t,:} ‖ Softmax(I_{t,:}))"；构造："we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution p_{t,:}∈ℝ^t." | 一致 |
| F4 | §5.2 sparse training 损失 | S1 §2.1.1 Eq. 4 | "ℒ^I=\sum_{t}\mathbb{D}_{KL}(p_{t,𝒮_t} ‖ Softmax(I_{t,𝒮_t}))，𝒮_t={s \| I_{t,s} ∈ Top-k(I_{t,:})}" | 一致 |
| N-1 | $H^I=64$、$d^I=128$、$k=2048$ | S3 `config.json`（同 commit 根目录） | `index_n_heads = 64`、`index_head_dim = 128`、`index_topk = 2048`；S2 `ModelArgs.index_topk: int = 2048` | 一致 |
| N-2 | 主注意力 128 头 / 61 层 / hidden 7168 / kv_lora_rank 512 / qk_rope_head_dim 64 | S3 `config.json` | `num_attention_heads=128`、`num_hidden_layers=61`、`hidden_size=7168`、`kv_lora_rank=512`、`qk_rope_head_dim=64` | 一致 |
| N-3 | warm-up lr $10^{-3}$、1000 步、每步 16 条 128K 序列、合计 2.1B token | S1 §2.1.1 Dense Warm-up Stage | "we use a learning rate of 10^{-3}. We train the indexer for only 1000 steps, with each step consisting of 16 sequences of 128K tokens, resulting in a total of 2.1B tokens." | 一致 |
| N-4 | sparse lr $7.3\times10^{-6}$、15000 步、每步 480 条 128K 序列、合计 943.7B token、$k=2048$ | S1 §2.1.1 Sparse Training Stage | "we use a learning rate of 7.3×10^{-6}, and select 2048 key-value tokens for each query token. We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens." | 一致（943.7/2.1≈449 ⇒ 页面「约 450 倍」正确） |
| N-5 | 续训起点为已扩到 128K 的 base checkpoint | S1 §2.1.1 首段 | "Starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K, we perform continued pre-training followed by post-training to create DeepSeek-V3.2." | 一致 |
| N-6 | 成本曲线口径：H800 集群、2 USD/GPU 小时 | S1 §2.3 | "These costs are estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour." | 一致（页面只取定性结论，未复述读数） |
| N-7 | vLLM DCP 合并的实现约束：CuteDSL + `index_topk ∈ {512,1024,2048}` | S4 `sparse_attn_indexer.py` L53–71 | "# The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL stable-topk selector); there is no PyTorch fallback. The first cut targets Blackwell/Hopper with index_topk in (512, 1024, 2048) …"；`if k not in (512, 1024, 2048): raise RuntimeError(...)` | 一致 |
| 计算示例 A | §1 配对次数：$1+\cdots+8=36$、$1+\cdots+16=136$、$L(L+1)/2$ | 构造数据 | 复算 36 / 136 ✓；且与实现口径（含 $t$ 自身）自洽 | 复算一致 |
| 计算示例 B | §3.1 八个 index score 与 top-3 | 构造数据 | 逐项复算：2.00 / 2.50 / 0.25 / 0.50 / 0.00 / 0.75 / 4.00 / 3.00；top-3={2,7,8} ✓；实跑代码输出逐字符一致 | 复算一致 |
| 计算示例 C | §5.1 目标分布（4 头 × 8 位置 → 跨头求和 → L1 归一化） | 构造数据 | 逐列求和 (0.60, 0.80, 0.15, 0.12, 0.18, 0.30, 1.05, 0.80)，总和 4.00；归一化 (0.1500, 0.2000, 0.0375, 0.0300, 0.0450, 0.0750, 0.2625, 0.2000)，和 = 1.0 ✓；每行原始分数各自和为 1.0 ✓ | 复算一致 |

**机制描述的口径核对（§2.2「机制描述也是来源论断」）**：§3 关于「$s$ 取遍 $t$ 之前的位置以及 $t$ 自身」的加注，依据 S2 的 `Transformer` 因果掩码 `torch.full((seqlen, seqlen), float("-inf")).triu_(1)`（对角线不屏蔽）+ `Indexer.forward` 中 `index_score += mask`、`topk(min(self.index_topk, end_pos))`，成立；§4 关于「掩码版本数学等价但拿不到性能收益」的表述与 S2 `index_mask = torch.full(...).scatter_(-1, topk_indices, 0)` + `scores.softmax(-1)` 的实现一致，且页面已明确标出这是参考实现的表达方式。

## 问题

- [重要·格式] 全页（`<head>` 之后的正文与来源章节）引用机制不符合固定写法：全页 `<sup>` 出现 0 次，正文 8 处引用写作中文括号 `（C6）`（3.2）、`（C5）`（3.2）、`（C4）`（4.1）、`（C10）`（5.2）、`（C12）`（6.1）、`（C13）`（6.1）、`（C6）`（6.2）、`（C14，来源为 vLLM 实现，非论文内容）`（6.3），来源章节的 C1–C14、F1–F4 共 18 条编号无任何一条与正文形成上标对应；同时 C1、C2、C3、C7、C8、C9、C11 所承载的论断（唯一架构改动、与 V3.2-Exp 架构相同、DSA 两个组成部分、$O(L^2)\to O(Lk)$、短序列 masked MHA、两份 cache、三组 parity 证据）在正文中连括号形式的标注也没有。｜引文依据：不适用（格式项，依据 `guides/concept/style-guide.md` §6「正文使用 `<sup>[Cx]</sup>` 上标引用（C=论断、F=公式、N=数字），与来源章节双向对应」；`validate.py` 不检查该项，故 `validation ok` 不能作为通过依据）｜修复要求：(1) 把上述 8 处中文括号引用改写为 `<sup>[C4]</sup>` 形式；(2) 为 C1/C2/C3/C7/C8/C9/C11 所指句子补上对应上标；(3) 使 18 条 C/F 编号在正文各至少出现一次上标，无正文对应的编号从来源章节删除。｜修复：正文 8 处中文括号引用（3.2 的「（C6）」「（C5）」，4.1 的「（C4）」，5.2 的「（C10）」，6.1 的「（C12）」「（C13）」，6.2 的「（C6）」，6.3 的「（C14，来源为 vLLM 实现，非论文内容）」）全部改写为「<sup>[Cx]</sup>」；补 C1、C2 于引言段（「DSA 是 DeepSeek-V3.2 相对 DeepSeek-V3.1-Terminus 的唯一架构改动<sup>[C1]</sup>，正式版 V3.2 与先行发布的 V3.2-Exp 架构完全相同<sup>[C2]</sup>」，已与 S1 § DeepSeek Sparse Attention 首段原文逐句核对），C3 于 §4 首段，C7 于 §7 表「主注意力计算复杂度」行的依据列与 §7.1 末句，C8 于 §7.3，C9 于 §6.1 与 §7.2 各一处，C11 于 §7.4 两段与 §7.4 本章问题答案；F1 于 §3 index score 句、F2 于 §4 输出公式句、F3/F4 于 §5.1/§5.2 损失句。C11 所需的第二条对照条件在论文中不属 Parity Evaluation 段，故在来源章节新增编号 C15（C 段末尾），并逐字引用 S1 § Continued Pre-Training 首段原文；来源章节内残留的「（C2）」（简化条件及其限制）亦统一为「<sup>[C2]</sup>」。全页 <sup> 由 0 增至 28。｜复验：`/usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/dsa/index.html` → `validation ok`，`EXIT=0`；闭环脚本（正文 = 「<h2 id="sources-and-teaching-notes">」之前，来源 = 之后）输出 `used in body (19): C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 F1 F2 F3 F4`、`defined in src (19): C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 F1 F2 F3 F4`、`used-not-defined: []`、`defined-not-used: []`（两个差集均为空）；`相邻双上标 </sup><sup>: []`；`grep -c '（C[0-9]' index.html` → `0`。
- [轻微·技术] §7.4 第 1 段与来源章节「外部数字与实验条件」把「两者的训练设置被严格对齐——正是这种对齐才让『性能差异来自稀疏注意力』这一归因成立」写成「未观察到明显回退」这一结论的成立条件；S1 §2.2 的 Standard Benchmark 段只写了评测时间与对比对象，未提训练设置对齐，「严格对齐」是超出来源的表述。｜引文依据：S1 §2.2 Standard Benchmark：「In September 2025, we evaluate DeepSeek-V3.2-Exp on a suite of benchmarks … and compare it with DeepSeek-V3.1-Terminus showing similar performance.」；Human Preference：「Both DeepSeek-V3.1-Terminus and DeepSeek-V3.2-Exp share an identical post-training strategy」；§2.1.1：「For both stages, the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus.」｜修复要求：把该成立条件改写为来源可直接支持的两条并各自标注出处——(a) 两者后训练策略相同（Human Preference 段）；(b) 两阶段续训的数据分布与 V3.1-Terminus 的 128K 长上下文扩展数据完全对齐（Continued Pre-Training 首段）。保留「归因成立依赖对照条件」的论述，但不得继续使用来源没有的「训练设置严格对齐」。｜修复：§7.4 第 1 段删除来源没有的「两者的训练设置被严格对齐」，改为两条来源可直接支持的对照条件：「两者的后训练策略相同<sup>[C11]</sup>」出自 S1 §2.2 Human Preference 段（“share an identical post-training strategy”），「两阶段续训的训练数据分布与 V3.1-Terminus 的 128K 长上下文扩展数据完全对齐<sup>[C15]</sup>」出自 S1 §2.1.1 Continued Pre-Training 首段（“the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus”），两句均经 WebFetch `https://arxiv.org/html/2512.02556v1` 逐字复核；「归因成立依赖对照条件」的论述保留。同句改写同步应用于 §7.4 本章问题答案、「外部数字与实验条件（N）」节末句（改为「见 S1 § Parity Evaluation；两项对照条件见 C11 与 C15」）与 overview.html「关键结论与边界」第 4 条。｜复验：`grep -c '严格对齐' index.html` → `0`，`grep -c '严格对齐' overview.html` → `0`（修复前分别为 3 与 1）；`validate.py` 对 index.html、overview.html 均输出 `validation ok`，`EXIT=0`；改写后的两条条件与论文原文逐句对应。
- [轻微·技术] §3.2「便宜来自五处」表第 2 行标签为「每头维度」，但主注意力列填的是「latent 512 维 + rope 64 维」：其中 512 维（`kv_lora_rank`）是跨该 token 全部 query 头共享的 KV latent，不是任何单个头的维度；主注意力每头 key 维度是 `qk_nope_head_dim`(128) + `qk_rope_head_dim`(64)。同一行 indexer 列的 $d^I=128$ 是每头 query 维度，两列量纲不一致，会让人误判 MLA 的 KV 结构与对比口径。｜引文依据：S3 `config.json`：`kv_lora_rank = 512`、`qk_nope_head_dim = 128`、`qk_rope_head_dim = 64`；S1 §2.1 Instantiate DSA Under MLA：「each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token」｜修复要求：把行标签改为同时适用于两列的口径（如「每个位置的 key/条目维度」），indexer 列保持 $d^I=128$、主注意力列写「512（共享 latent）+ 64（rope）」，或改为「每头 key 维度：$128+64$」并另注 512 维 latent 为跨头共享。二者择一，不得保留「每头维度」配 512 维 latent 的组合。｜修复：§3.2 表第 2 行标签由「每头维度」改为「key 维度」，indexer 列写为「$d^I = 128$（每头），全部头共用同一 key」，主注意力列由「latent 512 维 + rope 64 维」改为「每头 key $128 + 64$（nope + rope）；另有跨头共享的 KV latent 512 维」；§3.2 本章问题答案同口径的「每头维度低（128 对 latent 512 + rope 64）」改为「key 维度低（indexer 每头 128 维；主注意力每头 key $128 + 64$，另有跨头共享的 512 维 KV latent）」，核心问题答案中的「每头维度低」一并改为「key 维度低」。数值口径依据 S3 `config.json` 的 `qk_nope_head_dim = 128`、`qk_rope_head_dim = 64`、`kv_lora_rank = 512`，以及 S1 § Instantiate DSA Under MLA“each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token”。｜复验：`grep -c '每头维度' index.html` → `0`（修复前 3 处）；`grep -n 'key 维度' index.html` 命中 §3 核心问题答案、§3.2 表与 §3.2 答案三处，两列量纲一致；`validate.py` → `validation ok`，`EXIT=0`。
- [轻微·技术] 开篇把 128K 当作 128000：正文「已经读进 128K token 的上下文，现在要生成第 128001 个 token……前面 128000 个位置逐一比对」。128K 上下文惯例为 131072 个位置，与紧随其后的序号 128001 无法同时成立。｜引文依据：S1 §2.1.1「each step consisting of 16 sequences of 128K tokens」（128K 为论文中的序列长度单位，非 128000）｜修复要求：把该段改为不与 128K 严格换算冲突的表述（如「约 12.8 万个位置」「已经读进十几万 token」），或统一改用 131072 并相应调整 token 序号；§1 中「128K 上下文下」一句随之统一。｜修复：开篇整段改写为「一个模型已经读进 128K（$131072$ 个位置）的上下文，现在要生成下一个 token。这一步里，注意力要把新 token 的查询与前面全部位置逐一比对，算出同等数量的分数，再按分数加权求和。下一个 token 到来时，这件事从头再做一遍，而且位置数还要再多一个。」——不再出现 128000 与 128001 这组与 128K 互相矛盾的换算；§1 及其本章问题中的「128K 上下文下」保留为论文使用的序列长度单位（S1 §2.1.1“16/480 sequences of 128K tokens”），不再与纯对序号并存。｜复验：`grep -c '128000\|128001' index.html` → `0`；`grep -o '\$131072\$' index.html` → `$131072$`（1 处，包在 `$...$` 内由 KaTeX 渲染）；`validate.py` 的 unrendered-math 检查零命中，`validation ok`，`EXIT=0`。
- [轻微·格式] 来源章节第 3 个 h3 写作 `外部数字与实验条件`，缺固定后缀 `（N）`。｜引文依据：不适用（`guides/concept/style-guide.md` §1：来源章节下 h3 使用固定命名，`外部数字与实验条件（N）`）｜修复要求：标题改为 `外部数字与实验条件（N）`（该节已含全部外部数字，无需增删内容）。｜修复：`<h3>外部数字与实验条件</h3>` 改为 `<h3>外部数字与实验条件（N）</h3>`，节内文字与数字未增删。｜复验：六个固定命名逐一计数的脚本输出 `{'论断与来源（C）': 1, '公式与来源（F）': 1, '外部数字与实验条件（N）': 1, '构造示例': 1, '辅助解释与类比边界': 1, '简化条件及其限制': 1}`；`validate.py` → `validation ok`，`EXIT=0`。
- [轻微·格式] callout 颜色语义不符：正文使用 `.callout-gray` 2 处（「进入本页前需要的基础」、§8「名称歧义」）与 `.callout-red` 1 处（§3.2「便宜不等于复杂度降低」），而 `style-guide.md` §3 规定「red/green/gray 不作为 callout 颜色」（red 留给 misconceptions、green 留给 learning-goals）；`.callout-blue` 1 处（§4.1「这是一处算法设计被硬件约束反向决定的例子」）内容属深度分析，按 §3 应为 purple（blue 仅用于开篇引入/范围说明且每篇最多 1 个）。另外页面存在明确的常见误解（「DSA 把注意力降到线性」「DSA 省 KV cache」「DSA 与 FlashAttention 混同」）却未使用规范指定的 `section.misconceptions` 组件（§2 列出该组件，两处误解现由 red/yellow callout 承载）。｜引文依据：不适用（`style-guide.md` §2、§3）｜修复要求：gray 两处与 red 一处改为 yellow（边界条件/易误解澄清）或 purple（源码验证/深度分析）；blue 一处改 purple；若将这些误解收拢为 `section.misconceptions` 区块，则 §7.2、§8 的对应 callout 移除，正文结论句保留。｜修复：`.callout-gray` 2 处（开篇「进入本页前需要的基础」、§8「名称歧义」）与 `.callout-red` 1 处（§3.2「便宜不等于复杂度降低」）按语义改为 `.callout-yellow`（边界条件/易误解澄清），`.callout-blue` 1 处（§4.1「算法设计被硬件约束反向决定」）改为 `.callout-purple`（深度分析）。同时按规范补上此前缺失的 `<section class="misconceptions">`（h2「常见误解」，置于 learning-goals 之后），收拢三处常见误解——「DSA 把注意力复杂度降到线性」「DSA 省 KV cache」「DSA 与 FlashAttention 是同类方法」——并各自指向 7.1、7.2、8 章；按该路径要求，§7.2 的 callout 与 §8「名称歧义」callout 移除（§7.2 结论句保留，名称歧义内容改写为 §8 正文段落「<p>名称上需要区分：…</p>」，信息未丢失）。｜复验：`grep -o '<div class="callout callout-[a-z]*">' index.html` 输出 `yellow, yellow, purple, yellow, purple`（`grep -c 'callout callout-gray\|callout callout-red\|callout callout-blue' index.html` → `0`）；`grep -c '<section class="misconceptions">' index.html` → `1`；`validate.py` → `validation ok`，`EXIT=0`。
- [轻微·格式] 开篇 `div.context-box`「全文贯穿例子」逐项预告手算参数（序列长度 8、$k=3$、indexer 2 头 2 维、数字性质），而 `style-guide.md` §4 明确「不设置专门 context-box 预告手算参数」。｜引文依据：不适用（`style-guide.md` §4 第 3 段：「不设置专门 context-box 预告手算参数。简化条件说明中不单独列举所有手算参数。」）｜修复要求：删除该 context-box，把「贯穿例子」参数放到 §3.1「在贯穿例子上手算」的构造示例段首次出现处（该处已写 $H^I=2$、$d^I=2$、位置 8，需补 $k=3$、序列长 8 与「人为构造、不代表真实权重」的说明）；全页对贯穿例子的首次引用相应改为该节。｜修复：删除开篇 `<div class="context-box">`（「全文贯穿例子」四项参数预告），把贯穿例子的参数定义移到 §3.1 首次出现处，改写为「构造示例。取序列长度 8、$k = 3$、$H^I = 2$、$d^I = 2$，当前查询在位置 8（真实模型中 $k = 2048$、$H^I = 64$、$d^I = 128$，序列长 128K）。以下数字全部人为构造，不代表真实权重或推荐值。」；§2 对贯穿例子的首次引用改为「在贯穿例子上看这个差别（该例子的序列长度、$k$ 与 indexer 规模在「3.1 在贯穿例子上手算」一节定义）」。｜复验：`grep -c 'class="context-box"' index.html` → `0`；`grep -n '构造示例。取序列长度 8' index.html` 命中 §3.1 一处且含 $k = 3$ 与「不代表真实权重」说明；`validate.py` → `validation ok`，`EXIT=0`。
- [轻微·格式] §1–§8 共 8 个 `<h3>本章问题</h3>` 均无 `id`，页面脚本按标题文本生成 id（`h.textContent.trim().replace(...)`），8 个 h3 得到同一 id「本章问题」：自动目录会生成 8 条同名条目，点击任一条都因 `getElementById` 取首个匹配而跳到第 1 章的「本章问题」，滚动高亮也会同时点亮这 8 条。该问题影响「目录锚点正常」（check.md §2.2.8）。｜引文依据：不适用（页面脚本第 1539–1565 行与 1567–1579 行逻辑；1.754/2.808/3.916/4.1003/5.1085/6.1308/7.1391/8.1446 行处 `<h3>本章问题</h3>` 无 id）｜修复要求：为 8 个「本章问题」h3 补唯一 id（如 `ch1-questions`…`ch8-questions`）并让目录链接指向唯一锚点，或在目录生成时排除本章问题 h3（保留 h2 与编号 h3）。｜修复：8 个 `<h3>本章问题</h3>` 依次补唯一 id `ch1-questions`…`ch8-questions`；同时修订页面脚本，使自动目录生成（`headings.forEach` 内提前 `return`）与滚动高亮（`sections` 加 `.filter`）都排除「本章问题」h3，目录不再出现 8 条同名条目，h2 与编号 h3 保留。｜复验：脚本输出 `本章问题 h3 id: ['1','2','3','4','5','6','7','8']`，`grep -c '<h3>本章问题</h3>' index.html` → `0`；`validate.py` 的 duplicate-id 与「anchor points to missing id」检查均零命中，`validation ok`，`EXIT=0`。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 7
- 处置：修复
- 依据：核心结论（DSA 的两组成部分、index score 定义、$O(L^2)\to O(Lk)$ 与 indexer 仍 $O(L^2)$、两阶段续训及其全部超参与 token 量、短序列 masked MHA、DSA 不省 KV cache、DCP 局部 top-k 合并精确性）逐条定位到 S1/S2/S3/S4 的原文片段或代码行并写下原文，未发现与来源冲突或扩大适用范围的论断；两处纯推断（不省显存、Hadamard 动机）已按规范标注为推断或标注来源，构造示例与教学解释均已与实际来源结论分离。因此无阻断问题，主要结论可建立在正文之上。唯一重要问题为全页引用格式（`<sup>[Cx]</sup>` 缺失、8 条 C 编号无正文对应），属可机械修复项，修复后需复验「正文出现的编号集合 == 来源章节定义的编号集合」。轻微问题中「表格量纲标签」「128K=128000」「训练设置严格对齐」三项涉及数值或来源措辞，修复后需按上表重新核对来源；「callout 颜色」「context-box 预告参数」「来源 h3 命名」「本章问题锚点」四项为格式项，修复后重新运行 `validate.py` 即可。
- 未核对项：无。全部 C/F/N 论断与 3 个构造示例均给出了原文片段、代码行或复算过程。


## 修复后复验汇总（第 1 轮修复）

**1. `validate.py`（实际执行）**

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/dsa/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/dsa/index.html
EXIT=0
```

`overview.html` 同样 `validation ok`，`EXIT=0`。

**2. 引用编号双向闭合与其他机械项（脚本核对）**

```
used in body  (19): C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 F1 F2 F3 F4
defined in src (19): C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 F1 F2 F3 F4
used-not-defined: []
defined-not-used: []
相邻双上标 </sup><sup>: []
公式定界符外 Unicode 数学字符: []
TAB 字符数: 0
占位符（待生成/待补/TODO/FIXME/TBD/XXX/占位）: 0
来源章节 h3 固定命名: 六项各 1 处，含「外部数字与实验条件（N）」
callout 实际用量: yellow ×3、purple ×2，无 gray/red/blue callout
miscellaneous: `.context-box` 0 处、`严格对齐` 0 处（index 与 overview）、`（C\d+）` 式引用 0 处
```

**3. 页面可运行代码（实际执行）**

从 `index.html` 提取唯一 `language-python` 代码块写入 `/tmp/dsa_code.py`，执行 `/usr/bin/python3 /tmp/dsa_code.py`：退出码 0，输出与页面「预期输出」块逐字符一致（比对脚本输出 `逐字符一致: True`），末两行为 `k=3 时选中的位置: [2, 7, 8]` 与 `k=8 时选中的位置: [1, 2, 3, 4, 5, 6, 7, 8]（序列长 8，未裁剪）`。

**4. 处置说明**

无无法修复的条目，也没有因来源不支持而删除的页面内容；新增的编号 C15 仅为已在正文使用的对照条件补定义，不涉及论断删改。

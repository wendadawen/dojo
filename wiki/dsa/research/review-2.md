<!-- review-meta
round: 2
page: wiki/dsa/index.html
reviewed_content_sha256: 2e0cc4d08acba829
-->
# DSA 审查记录（第 2 轮）

- 页面版本：`git hash-object wiki/dsa/index.html` = `fe5cbf22f11f94610fd59d026c2915dd4200e8fa`；`wiki/dsa/overview.html` = `3bfc3ebc6674797ed2b8ecac670f5b771c1b4503`（两文件均为工作树修改未提交，仓库 HEAD `0f4a841`）
- 审查时间：2026-09-10 20:30
- 审查者：独立子代理（未参与写作与前序轮次审查；未读取 `wiki/dsa/research/` 下任何文件）
- 已完整阅读章节：引言 → 核心问题 → 常见误解 → 前置基础 callout → 1. 长上下文里，注意力的代价长在哪里 → 2. 两条减负路线，以及 DSA 选的那条 → 3. lightning indexer 如何用极低成本给每个位置打分（3.1 在贯穿例子上手算、3.2 便宜来自五处）→ 4. top-k 选择，以及候选集为什么必须跨头共享（4.1）→ 5. indexer 的打分能力是训出来的（5.1、5.2）→ 6. 工程落地：从 hidden state 到稀疏注意力的完整链路（6.1、6.2、6.3）→ 7. 省了什么、没省什么、什么时候不该用（7.1–7.5）→ 8. 与相邻方法的定位 → 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；`overview.html` 全文。含全部 35 个 `<details>` 折叠块（已逐个展开阅读）。

## 机械验证结果

| 项目 | 结果 |
|---|---|
| `.dojo/scripts/validate.py wiki/dsa/index.html` | `validation ok`，退出码 0 |
| `.dojo/scripts/validate.py wiki/dsa/overview.html` | `validation ok`，退出码 0 |
| 引用双向闭合 | 通过。正文上标 27 处，解析出 C1–C15、F1–F4 共 19 个 id；来源章节定义同样为 19 个。正文用而未定义 0，定义而未用 0 |
| 相邻双上标 | 0 处 |
| Unicode 数学字符（含 U+2212、U+00D7） | 0 处。逐字符扫描公式定界符与 `<pre>`/`<code>` 之外的全页文本，未发现 U+2212、U+00D7、U+2211、U+2202、U+221A、U+2248、U+2208、U+2299、U+03B1–U+03C3 等。剩余 5 个 U+2192（→）出现在 6.1 流程图的节点标签内（如 `<code>wq_b(qr)</code> → $\mathbf{q}^I$`），属流程图连线文字而非数学运算符，与本规范 A5 正例中 `&#8594;` 的用法一致，不计为违规 |
| TAB | 0 个 |
| 占位符 | 0 处（检索「待生成/TODO/TBD/占位/XXX/待补」均 0 命中） |
| 可运行代码实跑 | 通过。抽出 6.2 节 `python` 代码块实跑退出码 0，输出与页面「预期输出」文本块**按行逐字符一致**（唯一差异为进程 stdout 末尾多一个换行，HTML 预期块本身不含行尾换行）；`k=3` 选中 `[2, 7, 8]`、`k=8` 全选，8 行 index score 数值与 3.1 表格、4 章选中集合完全对应 |
| 结构图形式 | 纯 HTML `.flow-diagram` 结构（3 个），无 SVG、无 `<foreignObject>`、无框线字符（U+2500–U+257F 命中 0）；图内公式写在节点标签的 `$...$` 内 |
| head 元信息 | `description`（纯文本）、`dojo:summary`（含 KaTeX 公式）、`dojo:type=concept`、`dojo:topics=注意力机制`、`dojo:tag` 均存在；topics 通过 validate.py 词表校验 |
| 本地资源 | `libs/` 下 katex.min.css、katex.min.js、auto-render.min.js、prism.min.js、prism-python.min.js、prism-primer-light.css、prism-primer-dark.css 均存在 |
| 前置概念链接 | `standard-attention`、`mla`、`mqa-gqa`、`rope`、`quantization-basics`、`linear-attention`、`kda` 七个 `index.html` 均存在，无「（待生成）」占位 |
| index ↔ overview 互链 | index.html 导航含 `overview.html`（快速阅读）；overview.html 含 `index.html`（深度教学），双向成立 |
| 问题块 | 页面级「核心问题」5 条 + 8 个章节「本章问题」（3/3/3/3/3/3/4/3 条），共 30 个问题全部有 `解答：` 折叠块；5 个核心问题答案均指明完整论证所在章节 |
| 公式/示例复算 | 见下「来源核对」与「数值复算」 |

### 来源核对（逐条，check.md §2.2 四步）

来源 S1 = arXiv:2512.02556（HTML 版 `arxiv.org/html/2512.02556v1`，2025-12-02 提交）；S2 = `huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp` @ `194c67e12b1b0d6df0ef373ddcf215bc84027409`（`inference/model.py`、`inference/kernel.py`、根目录 `config.json`）；S3 = 同 commit `config.json`；S4 = `github.com/vllm-project/vllm` @ `5ac2684976ee22c04fe0d2f968c6cf6096b383f2`。所有条目均已在标注位置取得原文片段：

- **C1**「DSA 是 V3.2 相对 V3.1-Terminus 的唯一架构改动」→ S1 §2.1 首句：*"the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training."* ✓
- **C2**「V3.2 与 V3.2-Exp 架构完全相同」→ S1：*"DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp."* ✓
- **C3**「DSA 由 lightning indexer 与细粒度 token 选择两部分组成」→ S1：*"The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism."* ✓
- **C4**「续训→实例化在 MLA 之上；kernel 层约束→选用 MQA 模式，引用 NSA」→ S1 §"Instantiate DSA Under MLA"：*"For the consideration of continued training from DeepSeek-V3.1-Terminus, we instantiate DSA based on MLA … At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency ([Yuan et al., 2025]). Therefore, we implement DSA based on the MQA ([Shazeer, 2019]) mode of MLA…"*；参考文献 `bib.bib1` = (Yuan et al., 2025) *"Native sparse attention: hardware-aligned and natively trainable sparse attention"*, ACL 2025 —— 即 NSA，页码标注的 NSA 归因成立 ✓
- **C5**「选 ReLU 出于吞吐考虑」→ S1：*"We choose ReLU as the activation function for throughput consideration."* ✓
- **C6**「头数少且可 FP8 故效率突出」→ S1 紧接上句：*"Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable."*；四步计算链 → S2 `kernel.py` `fp8_index` docstring：`fp8 q @ fp8 k -> fp32 logits` / `relu(fp32 logits) * q_s (weights) -> fp32 logits` / `fp32 logits -> fp32 logits_sum` / `fp32 logits_sum * k_s (e8m0) -> fp32 index_score` ✓（与页面伪代码四步逐一对应，张量形状 `q (b,m,h,d)`、`q_s (b,m,h)`、`k (b,n,d)`、`k_s (b,n)`、输出 `(b,m,n)` 亦一致）
- **C7**「主注意力 $O(L^2)\to O(Lk)$、indexer 仍 $O(L^2)$」→ S1 §"Inference Costs"：*"DSA reduces the core attention complexity of the main model from O(L²) to O(Lk), where k (≪L) is the number of selected tokens."*；*"Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus."* ✓
- **C8**「短序列 prefill 用 masked MHA 模式模拟 DSA」→ S1：*"for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions."*；vLLM 判据 → S4 `vllm/model_executor/layers/attention/sparse_mla_attention.py:279`：`use_dense_mha=(prefill_max_seq_len <= self.topk_tokens and not self.vllm_config.attention_config.sparse_mla_force_mqa)`，消费方 `mla_attention.py:789` `use_dense_mha = getattr(prefill, "use_dense_mha", False)` ✓（页面「消费侧还有进一步的条件判断」成立）
- **C9**「不减少 KV cache、indexer 需额外 key cache」→ S2 `inference/model.py:453-454` `register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn))`、`k_scale_cache`；`:541-542` MLA 注册 `kv_cache`（`kv_lora_rank`）/`pe_cache`（`qk_rope_head_dim`），均按 `max_seq_len` 分配；S4 `deepseek_v2.py:616` `class DeepseekV32IndexerCache`，`:696` `self.k_cache = DeepseekV32IndexerCache(...)` ✓
- **C10**「indexer 输入 detach、两条优化信号隔离」→ S1：*"we detach the indexer input from the computational graph for separate optimization."*；*"The training signal of the indexer is from only ℒ^I, while the optimization of the main model is according to only the language modeling loss."* ✓
- **C11**「未观察到明显性能回退」→ S1 §"Parity Evaluation"：标准 benchmark *"evaluated in September 2025 … comparable performance"*、*"we observe no substantial performance degradation in both short- and long-context tasks"*；人类偏好 *"Elo scores taken on November 10, 2025"* 且 *"same post-training strategy"*；长上下文 *"4 points higher than DeepSeek-V3.1-Terminus in reasoning mode"*（AA-LCR）、Fiction.liveBench *"consistently better on multiple metrics"* ✓ 四组数字/日期全部命中
- **C12**「key 由 hidden state 独立投影、query 复用主注意力低秩 latent、RoPE 独立且非 interleaved」→ S2 `inference/model.py:460` `q = self.wq_b(qr)`、`:466-467` `k = self.wk(x); k = self.k_norm(k)`、`:560` `qr = self.q_norm(self.wq_a(x))`、`:463`/`:469` 两处注释 `# rope in indexer is not interleaved`；S4 `deepseek_v2.py:1122-1127` `self.indexer_rope_emb = get_rope(..., is_neox_style=not getattr(config, "indexer_rope_interleave", False))`、`:1158` `indexer_rotary_emb=self.indexer_rope_emb`；索引器接收主注意力 query latent 的调用 → S4 `vllm/model_executor/layers/mla.py:206` `self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`（`q_c` 即 `:175 q_c = self.q_a_layernorm(q_c)` 的低秩 latent）✓ 实质成立；路径未在来源注中给出，见问题 5
- **C13**「量化前施加 Hadamard 变换」→ S2 `inference/model.py:428-432`：`def rotate_activation(x): … return hadamard_transform(x, scale=hidden_size ** -0.5)`，`:472-473` 在 `act_quant` 前对 q、k 各调用一次 ✓（页面「带 $1/\sqrt{d}$ 缩放」与 `scale=hidden_size**-0.5` 一致）
- **C14**「DCP 下局部 top-k 合并的精确性」→ S4 `vllm/model_executor/layers/sparse_attn_indexer.py:83-93` docstring：*"A token in the global top-K must also be in its owning rank's local top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at most that many on its own rank), so exchanging only the per-rank local candidates is exact -- equivalent to all-gathering the full logit matrix, but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole score row."*；实现约束 → `:53-69` CuteDSL-only 且 `index_topk in (512, 1024, 2048)` ✓
- **C15**「续训数据分布与 V3.1-Terminus 的 128K 扩展数据完全对齐」→ S1 §"Continued Pre-Training"：*"For both stages, the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus."* ✓ 引文逐字一致
- **F1** 式(1) $I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\cdot\mathrm{ReLU}(\mathbf{q}^I_{t,j}\cdot\mathbf{k}^I_s)$ → S1 §2.1 Equation 1 逐字符一致；符号说明（$H^I$、$\mathbf{q}^I_{t,j}\in\mathbb{R}^{d^I}$、$w^I_{t,j}\in\mathbb{R}$ 由 $\mathbf{h}_t$ 导出、$\mathbf{k}^I_s\in\mathbb{R}^{d^I}$ 由 $\mathbf{h}_s$ 导出）同段 ✓
- **F2** 式(2) $\mathbf{u}_t=\mathrm{Attn}(\mathbf{h}_t,\{\mathbf{c}_s\mid I_{t,s}\in\mathrm{Top\text{-}k}(I_{t,:})\})$ → S1 Equation 2 一致 ✓
- **F3** 式(3) $\mathcal{L}^I=\sum_t D_{\mathrm{KL}}(p_{t,:}\,\|\,\mathrm{Softmax}(I_{t,:}))$ → S1 Eq. 3 一致；`$p_{t,:}\in\mathbb{R}^t$` 构造（跨头求和 + 沿序列 L1 归一化）同段 ✓
- **F4** 式(4) 仅在被选中集合 $\mathcal{S}_t$ 上计算 → S1 Eq. 4 一致 ✓
- **N** `index_n_heads=64`、`index_head_dim=128`、`index_topk=2048`、`num_attention_heads=128`、`num_hidden_layers=61`、`hidden_size=7168`、`kv_lora_rank=512`、`qk_rope_head_dim=64` → S3 `config.json` 字段逐项命中；`k=2048` 另有 S1 原文 *"select 2048 key-value tokens for each query token."*；warm-up LR $10^{-3}$/1000 步/16 条 128K/2.1B token、sparse training LR $7.3\times10^{-6}$/15000 步/480 条 128K/943.7B token → S1 两阶段原文逐项命中；2 USD/GPU 小时、H800、128K 起点 → S1 §"Inference Costs" 与 §"Continued Pre-Training" ✓
- **构造示例** 贯穿例子（8 位置、$k=3$、$H^I=2$、$d^I=2$）与 warm-up 目标分布例（4 头、8 位置）均在正文与来源章节明确标为人为构造 ✓
- **辅助解释与类比边界** 三处教学解释（ReLU vs softmax 规约、Hadamard 动机、打分器只需排序）均声明论文未给出该论证或属推断 ✓ 无「教学解释当来源结论」的情况

### 数值复算

- 配对次数：$1+\cdots+8=36$、$1+\cdots+16=136$，136/36≈3.78（页面「涨了将近 4 倍」）✓
- token 量：warm-up $1000\times16\times131072=2.097\times10^{9}$ ≈ 2.1B；sparse $15000\times480\times131072=9.4372\times10^{11}$ ≈ 943.7B；比值 449.4 ≈ 450（页面「约 450 倍」）✓
- index score 表 8 行逐行复算全部命中（含 $s=3$ 头 1 点积 $-1$ 截零、$s=4$ 头 2 点积 $-2$ 截零）；降序 4.00/3.00/2.50 → 选中 $\{2,7,8\}$，与 2 章表格 DSA 行、4 章、7.3 节一致 ✓
- warm-up 目标分布：跨头逐位置和 $(0.60,0.80,0.15,0.12,0.18,0.30,1.05,0.80)$，总和 4.00；除以 4.00 得 $(0.1500,0.2000,0.0375,0.0300,0.0450,0.0750,0.2625,0.2000)$，各项和 1.0，最高位置 7（0.2625）、最低位置 4（0.0300）✓
- 手算与可运行代码输出交叉一致；代码输出与「预期输出」块一致 ✓

## 问题

- [轻微·格式] 5.1 与 5.2 节的式(3)、式(4)：两条公式后未按规范给出 `<ul>` 逐项定义符号（$t$、$p_{t,:}$、$I_{t,:}$、$\mathcal{L}^I$、$D_{\mathrm{KL}}$、$\mathcal{S}_t$ 散落在前后段落中，无固定符号表）｜引文依据：S1 Eq. 3/Eq. 4 原文中的符号与页面一致，但 `guides/concept/style-guide.md` §11 要求「公式后紧跟 `<ul>` 逐项定义每个符号」｜修复要求：在式(3)、式(4) 后各补一个 `<ul>`，逐项定义该式中出现的每个符号（含 $\mathcal{L}^I$ 的含义与 $\mathcal{S}_t$ 的定义），或明确引用首次定义位置｜修复：在 5.1 式(3)、5.2 式(4) 的公式后各补一个 `<ul>`：式(3) 逐项定义 $\mathcal{L}^I$、$t$、$p_{t,:}$、$I_{t,:}$、$\mathrm{Softmax}(I_{t,:})$、$D_{\mathrm{KL}}$；式(4) 逐项定义 $\mathcal{S}_t$、$p_{t,\mathcal{S}_t}$、$I_{t,\mathcal{S}_t}$、$\mathrm{Top\text{-}k}(I_{t,:})$，并说明其余符号含义同式(3)。公式交叉引用按规范写作「论文 Eq. 1/2/3」，不自行编号。｜复验：validate.py `validation ok`（退出码 0）；两处 `<ul>` 均紧跟在公式之后；全文未出现自编公式号「式(1)/式(3)/式(4)」，对论文公式的引用一律为 Eq. 编号。
- [轻微·技术] 3.2 节「便宜来自五处」表格「key 维度」行：单元格写作「$d^I = 128$（每头），全部头共用同一 key」，「每头」与「同一 key 共用」相互矛盾；本章问题第 3 题解答中「key 维度低（indexer 每头 128 维…）」同源｜引文依据：S2 `inference/model.py:446` `self.wk = Linear(self.dim, self.head_dim)` 产出不分头的单个 key，`:845`/正文 F1 符号表已写明 $\mathbf{k}^I_s$ 不带头下标；S3 `config.json` `index_head_dim=128`｜修复要求：把该单元格中 key 的「（每头）」删除或改写为「与每头 query 维度相同的 128 维」，使「全部头共用同一 key」与维度描述不冲突｜修复：3.2 表格「key 维度」单元格改为「$d^I = 128$（与每头 query 维度相同），全部头共用同一 key」；本章问题第 3 题解答同步改为「indexer 的 key 为与每头 query 维度相同的 128 维、全部头共用同一份」。｜复验：改写后与 S2 `inference/model.py:446` `self.wk = Linear(self.dim, self.head_dim)`（产出不分头的单个 key）及 F1 符号表「$\mathbf{k}^I_s$ 不带头下标」一致；单元格内不再同时出现「每头」与「共用」两种归属。
- [轻微·可读性] 6.2 节「FP8 index kernel 的四步」：四步本身只出现在 `<details>` 折叠块内的伪代码中，正文以「步骤 1/2/3/4」直接指代，折叠块收起时读者无法看到四步是哪四步｜引文依据：不适用（规范依据 `content-examples.md` A8「折叠块后一句恢复主线，不引用只有展开才见到的内容」）｜修复要求：把四步的名称（FP8 矩阵乘 → ReLU 加权 → 沿头求和 → 乘 key 侧 scale）写进正文，折叠块只保留对应伪代码｜修复：把四步名称写进 6.2 正文——「**FP8 矩阵乘**（算出每头对每个位置的点积）、**ReLU 加权**（负值截零后乘 query 侧权重）、**沿头求和**（跨 indexer 头累加）、**乘 key 侧 scale**（还原 FP8 量化的比例）」，并写明「四步的伪代码如下，本节后续正文直接以这四步的顺序指代」；折叠块内只保留伪代码。｜复验：折叠块收起时正文已完整给出四步名称，其后「步骤 1/2/3/4」的指代与本章问题第 2 题的解答不再依赖展开；validate.py `validation ok`。
- [轻微·可读性] 7.4 节首次出现术语「Elo」未解释（「以 ChatbotArena 作为间接评估，V3.1-Terminus 与 V3.2-Exp 在 2025 年 11 月 10 日取得的 Elo 分数相近」）；5.2 节「indexer 的输入从计算图上 detach」中 detach 亦为未展开的实现术语｜引文依据：不适用（规范依据 `check.md` §2.1「术语是否在首次使用时解释」；两处术语均可在不引入新概念的前提下用一句话说明）｜修复要求：7.4 节补充 Elo 的最小含义（两两比较得出的相对评分，分数越接近表示水平越接近）；5.2 节把「从计算图上 detach」改写为「切断 indexer 输入的梯度回传（detach）」｜修复：7.4 首次出现 Elo 处补最小含义——「（Elo 是按两两对战的胜负累计出的相对评分，两个模型的分数越接近表示水平越接近）」；5.2 改为「切断 indexer 输入的梯度回传（detach，即把该输入从反向传播的计算图上摘除）」；`overview.html` 的 ChatbotArena Elo 同步补「按两两对战胜负累计的相对评分」。｜复验：两处术语均在首次使用时就地解释，未引入新概念；validate.py 两个页面均 `validation ok`。
- [轻微·技术] 来源章节 C12 条：该项其余子依据均给出源码路径，唯「indexer 接收主注意力 q_c 的调用」只写「S4 中…」而未给路径与行号｜引文依据：该调用实际位于 S4 `vllm/model_executor/layers/mla.py:206`：`self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`（`q_c` 定义见同文件 `:175`），页面正文「由主注意力的 query 低秩 latent `qr` 再投影一层」的实质成立（S2 `inference/model.py:460` `q = self.wq_b(qr)`）｜修复要求：在 C12 中补上文件路径 `vllm/model_executor/layers/mla.py`，与其他条目保持同等定位精度｜修复：C12 条补上 S4 文件路径，改写为「S4 `vllm/model_executor/layers/mla.py` 中 `indexer_rope_emb` 的构造与注释，以及 `MLA` 转发时 indexer 接收主注意力低秩 latent `q_c` 的调用（`mla.py:206`；`q_c` 定义见同文件 `:175`）」。｜复验：C12 全部子依据（`inference/model.py` 的 `wq_b`/`wk`/`k_norm`、两处 rope 注释，S4 的 `indexer_rope_emb` 构造与 `q_c` 调用）现均带文件路径，定位精度与其他条目一致；正文「由主注意力的 query 低秩 latent 再投影」与 `mla.py:206` 的调用对应。
- [轻微·技术] 第 2 章「精确检索能力保留」「线性注意力…换来的是精确检索能力的损失」两处属本页推理，未标注为推断｜引文依据：S1 摘要与 §2.1 只声称 DSA 在压缩复杂度的同时保持长上下文性能，未对线性注意力的检索能力或 DSA 的检索能力给出该论断｜修复要求：按《辅助解释与类比边界》既有的处理方式，把这两句改为明确标注的对比性推理（例如「本页据此推断」），或删去「精确检索能力」表述只保留「被选中的位置仍按完整 softmax 注意力计算」这一可核对事实｜修复：第 2 章两处降级为明确标注的推断/可核对事实——线性注意力一句改为「…任意细节。这一比较是本页据两者机制推出的，论文只对比了 DSA 与 V3.1-Terminus，未对线性注意力的检索能力作论断」；DSA 一句改为「被选中的位置照旧按完整的 softmax 注意力计算（这一点可从机制直接核对）」；`overview.html` 的对应两处同步处理。｜复验：S1 摘要与 §2.1 未声称的「检索能力」论断已改为标注推断，保留的仅为机制事实；`guides/concept/content-examples.md` A8 的「正文完整」要求不受影响；validate.py 两个页面均 `validation ok`。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：修复（6 条轻微问题均可就地编辑解决，不涉及范围或大纲变更，无需返回规划）。核心结论未发现与来源不符之处：C1–C15、F1–F4 与全部 N 项数字均在标注位置取得原文片段并逐项命中（含 `select 2048 key-value tokens for each query token`、`totally aligned with the 128K long context extension data`、`is_neox_style=not getattr(config, "indexer_rope_interleave", False)`、`triu_(1)` 因果掩码与 `scatter_(-1, topk_indices, 0)` 的 −∞ 掩码写法）；手算与可运行代码实跑结果一致；机械验证全部通过。修复 6 条轻微问题后，本轮不存在遗留阻断/重要问题。

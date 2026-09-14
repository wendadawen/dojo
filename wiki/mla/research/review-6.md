<!-- review-meta
round: 6
page: wiki/mla/index.html
reviewed_content_sha256: 195ef838600bd74a
-->
# MLA 审查记录（第 6 轮）

- 页面版本：ae8e3e0ecb510ef01e951a0a1f6d1f034000d99e（index.html 工作树哈希）
- 审查时间：2026-09-14 17:07
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何文件）
- 页面类型：concept（`<meta name="dojo:type" content="concept">`），适用 `guides/concept/check.md`
- 已完整阅读章节：核心问题（5 问 + 5 解答）→ 1. MLA 压缩了什么——KV 联合压缩的核心机制 → 2. 推理时不重建 K/V——矩阵吸收 → 3. 为什么 RoPE 要解耦——位置编码与矩阵吸收的冲突 → 4. KV cache 到底减少了多少——与 MHA / GQA / MQA 对照 → 5. K3 的 Gated MLA——NoPE 与 full-rank output gate → 来源与范围说明（含全部折叠块与 callout）
- 外部来源核对方式：DeepSeek-V2（arXiv:2405.04434v5）与 Kimi K3 报告（arXiv:2607.24653v2）下载 HTML 原文逐段定位；官方 `moonshotai/Kimi-K3` 的 `config.json`、`modeling_kimi_linear.py` 用 HF API/raw 直接读取；DeepSeek LLM 报告（arXiv:2401.02954）表 2 与 `deepseek-llm-67b-chat` config 核对 67B 架构；vLLM `vllm.models.kimi_k3` 文档核对。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/mla/index.html` → `validation ok`；`<details>/<summary>` 各 25 个、配对完整；无「（待生成）」/占位符；`dojo:topics=注意力机制` 在 AGENTS.md 词表内；正文除 `.diagram` 流程箭头「→」外无 Unicode 数学字符（→ 为流程箭头，非数学符号，与全站图示惯例一致）；页面无 `<img>` 数据图，无需像素测量。

## 核对要点（本轮已逐条回源、全部通过）

- DeepSeek-V2 摘要「Compared with DeepSeek 67B … reduces the KV cache by 93.3%」——原文确认 baseline 为 DeepSeek 67B，非 MHA；页面第 4 章黄色 callout 的澄清成立。同配置 MHA 下 576/32768 ≈ 1/57（减少 98.2%）复算无误；按 67B（95 层、8 KV head）对 V2 MLA（60 层、576）复算 1 − 34560/194560 = 82.2% ≈ 82%，与页面一致。
- Table 1 公式（MHA 2n_h d_h / GQA 2n_g d_h / MQA 2d_h / MLA (d_c+d_h^R)l ≈ 9/2 d_h l，能力列 Strong/Moderate/Weak/Stronger）与原文逐格一致；d_c=4d_h、d_h^R=d_h/2 的设置一致；「等于 GQA 2.25 组」自洽（576/(2×128)=2.25）。
- V2 超参（l=60, d=5120, n_h=128, d_h=128, d_c=512, d_c'=1536, d_h^R=64）§3.1.2 原文核对无误。
- 公式编号：Eq.(7)=§2.1.1 MHA 逐头 softmax 输出、Eq.(9)(10)(11)=KV 联合压缩、Eq.(12)(13)=query 压缩、Eq.(14)–(17)=解耦 RoPE、Eq.(18)=√(d_h+d_h^R) 注意力、Eq.(19)=W^O 拼接输出，均与原文章节/编号对应；「吸收」句确在 §2.1.2 第一段（段起点 82333，吸收句 90228/90756），「RoPE 不兼容」句确在 §2.1.3 第一段。
- K3：`config.json` 直读确认 `mla_use_nope=true`、`mla_use_output_gate=true`、`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`num_attention_heads=96`、`hidden_size=7168`、`num_hidden_layers=93`、`full_attn_layers` 24 项、`kda_layers` 69 项——与 N4/N5 全部一致。
- 官方 `modeling_kimi_linear.py` 直读（该文件在官方仓库确实存在，HF API siblings 列出）第 358/396/398–401/403 行与 forward 段确认：`self.use_nope = config.mla_use_nope`、`assert self.use_nope`、`self.rotary_emb = None`、`g_proj = nn.Linear(hidden_size, num_heads*v_head_dim)`、`k_pass, k_rot = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim])` 后 `key_states = cat(k_pass, k_rot)` 无条件执行、gate 在 `o_proj` 之前 `attn_output * g_proj(hidden_states).sigmoid()`。N6 的三条断言（不旋转 / 无 `if self.use_nope` 跳过分支 / 分支仍缓存）全部成立；W^g ∈ R^{d_h n_h × d} 与「õ_t 是 W^O 输入端的拼接向量」两处推断与源码一致。
- K3 引文逐句比对：§2.1.2「applies No Position Encoding (NoPE) to all MLA layers」「The gate projection W_g is full rank, matching the new parameterization used by KDA in Kimi K3.」「This gate allows each token to modulate the channels read from global attention」「augments MLA with an input-dependent, channel-wise full-rank output gate」；§2.1.1 有 Full-rank gate 段；报告另有「encodes positional information implicitly through the recurrent gating and decay mechanism of KDA」——页面第 5 章各处归因与 [C5][C6][N7] 成立；Eq.(7) 即 gate 式（`S2.E7` 紧随 gate 段）。
- 构造示例全部复算无误：c_t^KV=(1,0,1)、W^{UK}ᵀq'=(1,1,0)、内积=1、重建 K 路径同样得 1、拼接后 q^Tk/√6=1/2.449≈0.408、σ(1)/σ(0)/σ(−1)=(0.7311,0.5,0.2689) 及 Hadamard 结果 (0.7311,1.0,0.8067,2.0)、K3 单层 MHA 2×96×128=24576、576/24576≈1/43、min(d_h n_h,d)=min(12288,7168)=7168。
- 前置概念页 `wiki/mqa-gqa`、`wiki/low-rank-projection`、`wiki/rope`、`wiki/linear-attention` 均存在；index↔overview 互链；overview 与 index 关键数字（1/57、98.2%、1/32、480 GB、config 标志）一致。

## 问题

- [轻微·可读性] 5.2 节（第 439 行）：「$\tilde{\mathbf{o}}_t$ … 即第一章公式里 $W^O$ 输入端的拼接向量 $[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$」把 $W^O$ 与拼接式的出处标为「第一章」，但第 1 章（1 节）只出现 $W^Q,W^K,W^V,W^{DKV},W^{UK},W^{UV},W^{DQ},W^{UQ}$，全页 $W^O$ 首次出现于第 2 章第 213 行，拼接式 $[\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$ 首次出现于第 2 章补充块第 219 行。｜引文依据：不适用（页内交叉引用）｜修复要求：将「第一章公式里」改为「第二章公式里（式 $W^O[\mathbf{o}_{t,1};\ldots;\mathbf{o}_{t,n_h}]$）」或直接写「$W^O$ 输入端的拼接向量」，使读者能定位到实际出处。｜修复：｜复验：
- [轻微·技术] 第 4 章（第 367 行）与来源说明 C1（第 496 行）、C7（第 502 行）、N1（第 518 行）、N2（第 519 行）共 5 处把 Table 1 记为「§2.1.4 Table 1」。｜引文依据：arXiv:2405.04434v5 的 HTML 中 Table 1 的锚点标题为「Table 1 ‣ 2.1.3 Decoupled Rotary Position Embedding ‣ 2.1 Multi-Head Latent Attention」，且表格元素位置（S2.T1，offset 113880）落在 §2.1.3 标题（96519）与 §2.1.4 标题（121557）之间；§2.1.4 正文仅写「We demonstrate a comparison of the KV cache per token among different attention mechanisms in Table 1.」（即 §2.1.4 引用 Table 1，而非收录 Table 1）。｜修复要求：5 处「§2.1.4 Table 1」统一改为「§2.1.3 Table 1（§2.1.4 讨论）」，或改为「§2.1.3 Table 1」。｜修复：｜复验：
- [轻微·技术] 开篇段（第 80 行，overview 同）：「BF16（2 字节/数）下约 3.75 MB/token，128K 上下文下整个 KV cache 约 480 GB」——两个数按二进制口径才自洽（1966080×2 B = 3.75 MiB；3.75 MiB×131072 = 480 GiB），但单位名写作十进制 MB/GB；读者按页面给的「3.75 MB」乘以 128K token 得 491.5 GB，与「480 GB」不闭合。｜引文依据：不适用（页内推导数字）｜修复要求：把单位统一为 MiB/GiB，或改数值为十进制（约 3.93 MB/token、约 515 GB），使两处数值与单位名互相吻合。｜修复：｜复验：
- [轻微·技术] 第 421 行引文截断丢掉首从句、第 440 行引号内改写了符号。｜引文依据：arXiv:2607.24653v2 §2.1.2 原文为「Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear [57] and applies No Position Encoding (NoPE) to all MLA layers.」；页面引作「Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to all MLA layers.」（省略首从句但未加省略号）；同节原文为「The gate projection $\mathbf{W}_{g}$ is full rank, matching the new parameterization used by KDA in Kimi K3.」，页面引作「The gate projection $W^g$ is full rank, …」（引号内把 $\mathbf{W}_g$ 换成 $W^g$）。｜修复要求：第 421 行在引文前加省略号（或补足首从句）；第 440 行引文内的 $W^g$ 还原为 $\mathbf{W}_g$，页面的 $W^g$ 记法在引文外用括注说明。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。四条轻微问题均不影响正确性与主线理解（1 条页内交叉引用、1 条表号归属章节、1 条单位名口径、1 条引文写法），按「修改范围限于问题位置及直接受影响的引用位置」就地修复即可，无需返回规划。核心论断（KV 联合压缩公式、矩阵吸收等价性、RoPE 解耦冲突、93.3% baseline 澄清、K3 NoPE 与 full-rank output gate 源码行为）已逐条回源并附原文片段，无阻断与重要问题。

统计：阻断 0 / 重要 0 / 轻微 4
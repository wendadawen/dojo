<!-- review-meta
round: 5
page: wiki/mla/index.html
reviewed_content_sha256: a955c6001558d5f2
-->
# MLA 审查记录（第 5 轮）

- 页面版本：af8dd94bde8efd404aa89a5b62e2f411293ca56c（index.html 工作树哈希，git 工作树干净）
- 审查时间：2026-09-13 21:46 CST
- 审查者：独立子代理（未参与写作与前序轮次，未读取本页 research/）
- 已完整阅读章节（含全部折叠块与图注）：核心问题（学习目标）；1. MLA 压缩了什么——KV 联合压缩的核心机制；2. 推理时不重建 K/V——矩阵吸收；3. 为什么 RoPE 要解耦——位置编码与矩阵吸收的冲突；4. KV cache 到底减少了多少——与 MHA / GQA / MQA 对照；5. K3 的 Gated MLA——NoPE 与 full-rank output gate（5.1/5.2）；来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；overview.html 全文。

## 来源核对（已核对通过项，附原文片段）

**DeepSeek-V2（arXiv:2405.04434, HTML v5）**
- §3.1.2 配置："the number of Transformer layers to 60 and the hidden dimension to 5120"；"the number of attention heads n_h to 128 and the per-head dimension d_h to 128"；"The KV compression dimension d_c is set to 512, and the query compression dimension d_c′ is set to 1536"；"we set the per-head dimension d_h^R to 64"。与 N1 的 d=5120, n_h=128, d_h=128, l=60, d_c=512, d_c'=1536, d_h^R=64 完全一致。
- §2.1.4 Table 1：MHA 2n_h d_h l、GQA 2n_g d_h l、MQA 2d_h l、MLA (d_c+d_h^R)l ≈ (9/2)d_h l；能力列 Strong / Moderate / Weak / Stronger；"its KV cache is equal to GQA with only 2.25 groups"。与第 4 章表格、行 369 的 d_c=4d_h、d_h^R=d_h/2 折算一致（4×128=512、128/2=64、4.5×128=576）。
- 公式编号：Eq.(1)–(8) MHA，Eq.(7) 为第 i 头注意力输出（行 133 引用正确）；Eq.(9)(10)(11) KV 压缩；Eq.(12)(13) query 压缩；Eq.(14)–(19) 解耦 RoPE，其中 Eq.(15) k_t^R=RoPE(W^{KR}h_t)（行 299 引用正确）、Eq.(18) 分母 sqrt(d_h+d_h^R)（行 294 引用正确）。编号全部对位无误。
- §2.1.2 第一段："since W^{UK} can be absorbed into W^{Q}, and W^{UV} can be absorbed into W^{O}"；"we even do not need to compute keys and values out for attention"。行 211 所引论文原文 "absorbed into W^Q" 准确。
- §2.1.3 第一段："RoPE is incompatible with low-rank KV compression"；"W^{UK} cannot be absorbed into W^{Q} any more during inference"。
- §2.1.3 第二段："the decoupled key should also be cached"，总 cache (d_c+d_h^R)l。行 299 论断成立。
- 摘要："Compared with DeepSeek 67B … saves 42.5% of training costs, reduces the KV cache by 93.3% …"。行 372/505 与 overview 的 baseline 归属（DeepSeek 67B，非 60 层同配置 MHA）成立。

**DeepSeek LLM（arXiv:2401.02954）Table 2**："67B 95 8192 64 8 4096 4608 3.2e-4 2.0T"；"the 67B model uses Grouped-Query Attention (GQA) instead of the traditional Multi-Head Attention (MHA)"。N3 的 95 层、8 个 KV head、d_h=8192/64=128、GQA 全部成立。

**Kimi K3（arXiv:2607.24653v2，2026-07-27 提交）**
- §2.1.1 有 "Full-rank gate" 段："Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear … to an input-dependent full-rank projection."（行 464 引用准确）
- §2.1.2 第 2 段："applies No Position Encoding (NoPE) to all MLA layers"；KDA 提供 position-sensitive mixing、MLA 提供 unrestricted global content interaction、免于 retune RoPE base/YaRN（行 423/427/428 成立）。
- §2.1.2 第 3 段：Eq. 7 y_t=W_o[Sigmoid(W_g x_t)⊙ō_t]；"The gate projection W_g is full rank, matching the new parameterization used by KDA in Kimi K3."；"This gate allows each token to modulate the channels read from global attention."（行 435/442/460 引文逐字对位）
- Table 1：93 层、69 KDA + 24 MLA、hidden 7168、96 heads（行 375/378/383 成立）。

**K3 官方 config.json（huggingface moonshotai/Kimi-K3, text_config）**：num_hidden_layers=93, hidden_size=7168, num_attention_heads=96, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, mla_use_nope=true, mla_use_output_gate=true, linear_attn_config.full_attn_layers 共 24 项（4,8,…,92,93）, use_full_rank_gate=true。与 N4、N5 及行 375–383 全部数字一致。

**K3 官方源码 modeling_kimi_linear.py**：self.rotary_emb = None（无条件，非 if 分支）；assert self.use_nope；kv_a_proj_with_mqa 输出宽度 kv_lora_rank + qk_rope_head_dim 且 torch.split(..., [kv_lora_rank, qk_rope_head_dim])；forward 中 k_rot = k_rot.expand(...)、key_states = torch.cat((k_pass, k_rot), dim=-1) 无条件执行，无 if self.use_nope 跳过分支；self.g_proj = nn.Linear(hidden_size, num_heads*v_head_dim) 且 g = self.g_proj(hidden_states).sigmoid(); attn_output = attn_output * g。N6 与行 386/387 的源码论断逐条成立。

**vLLM（N7）**：vllm/models/kimi_k3/ 存在，MLA 类 rotary_emb: RotaryEmbedding | None = None、assert mla_use_nope，fused MLA concat/kv-cache 算子不传 positions/cos_sin_cache 时即 NoPE fast path。N7 可核对部分成立。

**公式与数字复算**：2×128×128×60=1,966,080；2×128×128=32768；576/32768=0.01758≈1/57（减少 98.24%）；(512+64)×60=34,560、MQA=15,360；512/16384=1/32；2×96×128=24,576、576/24576≈1/43；构造示例（W^{DKV}h_t=(1,0,1)、q'=(1,1,0)、内积两算法同为 1、拼接后分母 sqrt6≈0.408、σ(1,0,-1,0)≈(0.7311,0.5,0.2689,0.5) 及逐元素积）全部可复算且结论相符；12288>7168=min、满秩⇒单射论证无误。

**机械项**：validate.py 返回 validation ok；全文无 Unicode 数学字符、无 alt 含 $...$；4 个概念链接（mqa-gqa、low-rank-projection、rope、linear-attention）与全部本地 libs 引用文件均存在；C1–C8、F1–F6、N1–N7 与正文 sup 引用双向对应、无孤立条目；两级问题块每题均有「解答：」折叠块；h2/h3 编号与固定小节命名合规。

## 问题

- [轻微·来源定位] 来源章节 C4、F5 与正文第 2 章（行 211）｜问题：把矩阵吸收结论标注为 DeepSeek-V2「§2.1.2 末段」，但该结论在 §2.1.2 第一段，§2.1.2 末段（第二段）讲的是 query 低秩压缩，不含吸收内容。｜引文依据：§2.1.2 第一段 "since W^{UK} can be absorbed into W^{Q}, and W^{UV} can be absorbed into W^{O}"；§2.1.2 末段 "we also perform low-rank compression for the queries, even if it cannot reduce the KV cache"｜修复要求：将 C4、F5 及行 211 的定位由「§2.1.2 末段」改为「§2.1.2 第一段」｜修复：｜复验：
- [轻微·来源定位] 第 5 章行 460 与来源章节 C6｜问题：行 460 把 "each token to modulate the channels read from global attention" 标为「K3 §2.1.2 末段」，C6 标为「§2.1.2 Eq.(7) 及其后一段」，但该句与 Eq.(7) 同在 §2.1.2 第三段；§2.1.2 末段（第四段）讲 FP32 注意力输出与训练 kernel 重叠。｜引文依据：K3 §2.1.2 共 4 段，第三段含 "This gate allows each token to modulate the channels read from global attention."，第四段为 "the attention output is kept in FP32 during training … redesigned to overlap … KV staging buffers"｜修复要求：把行 460 的「末段」改为「第三段（output gate 段）」，C6 的「其后一段」改为「同段」｜修复：｜复验：
- [轻微·表述] 行 125｜问题：过渡句「下面从 MHA 的 KV cache 说起——由此能看清 MLA 到底压缩了什么。」属元话语式引导（"下面……说起"），与规范列举的"下面来看…"同类。｜引文依据：不适用｜修复要求：删除该元话语引导句，直接以 MHA 的 KV cache 规模或第 1 章首句承接，不保留"下面/说起"式前导｜修复：｜复验：
- [轻微·技术/数字说明] 第 4 章黄色 callout（行 372）｜问题：用 DeepSeek 67B 的架构数字（95 层、8 个 KV head、d_h=128、GQA）作为摘要 93.3% 的对照说明，但按这组数字复算 MLA vs 67B 得 34,560/(2×8×128×95)=0.1776，即减少约 82.2%，与页面引用的 93.3% 不符；页面未说明 93.3% 不能由这组公开配置按 (d_c+d_h^R) 公式直接复算，读者照给出的数字验算会与引用值对不上。｜引文依据：DeepSeek-V2 摘要 "Compared with DeepSeek 67B … reduces the KV cache by 93.3%"；DeepSeek LLM 报告 Table 2 "67B 95 8192 64 8 …"（n_kv_heads=8）｜修复要求：在 callout 中说明 93.3% 是论文给出的系统级结果、其折算口径论文未公开（按公开的 67B 配置复算约为 82%），或删去易被当作折算依据的 67B 具体架构数字，只保留「baseline 是 DeepSeek 67B 而非同配置 MHA」这一有来源支持的结论｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。4 条均为轻微：2 条为来源段落定位标注不准（引用的 §2.1.2/§2.1.3 章节号正确、论断本身经原文片段核对成立，读者仍可在被引章节内定位），不影响核心结论与阅读主线；1 条为元话语式过渡句；1 条为 callout 说明性数字的折算口径缺失。核心结论（MLA 低秩联合压缩、矩阵吸收、RoPE 解耦、cache d_c+d_h^R 与 1/57、K3 NoPE 与 full-rank gate、24 层 MLA）全部逐条回源核对通过，公式与数字全部可复算，validate.py 通过。

统计：阻断 0 / 重要 0 / 轻微 4

> 本轮所列问题的处理结果见 `minor-fixes.md`。

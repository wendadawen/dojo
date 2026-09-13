<!-- review-meta
round: 7
page: wiki/kv-cache/index.html
reviewed_content_sha256: 30a8a836cc292b2a
-->
# KV cache 审查记录（第 7 轮）

- 页面版本：f780d9bcfb7b16a6356366540c6c9c593318fab1（工作树哈希）
- 审查时间：2026-09-13 21:43
- 审查者：独立子代理（编排者派发，未参与写作与前序轮次审查；未读取本页 research/ 下任何文件）
- 已完整阅读章节：引言与「核心问题」（4 条解答折叠块）→ 1. 注意力为什么需要缓存——K/V 与查询无关（含 4-token 示例表、逐步计算折叠块、本章问题）→ 2. prefill 与 decode——缓存产生的两个阶段（含 dg-flow 流程图、本章问题）→ 3. 缓存有多大——每 token 字节数公式（含代入折叠块、本章问题）→ 4. 为什么显存成为瓶颈——动态缓存与权重的争夺（含对照表、本章问题）→ 来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）。全部 details 折叠块与图注已展开通读。

## 来源核对（引文依据）

- C4 / N3（显存分布 65% / 30%）：vLLM §1 原文 "Approximately 65% of the memory is allocated for the model weights"，"Close to 30% of the memory is used to store the dynamic states of the requests"；Figure 1 图注 "Memory layout when serving an LLM with 13B parameters on NVIDIA A100"。与第 252 行「约 65% 显存是静态权重，留给 KV cache 的只有近 30%」一致。
- N1 / F1（OPT-13B 800 KB/token、1.6 GB）：vLLM §3 原文 "the KV cache of a single token demands 800 KB of space"，代入式 "2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2"，"the memory required to store the KV cache of one request can be as much as 1.6 GB"。与第 225、260 行一致。
- C5（既有系统完成即丢弃缓存）：SGLang §1 原文 "In existing systems, the KV cache of a request is discarded after processing is completed"，"which prevents KV cache from being reused across multiple generation calls"；vLLM §4.3 "Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests."。论断本身成立（见问题 2 的编号错位）。
- C2（两阶段）：Strata §2.1 原文 "LLM inference operates in two phases: prefill and decode."；"In the subsequent decode phase, the model generates tokens autoregressively, continually reusing and extending the KV cache."。与第 158 行一致。
- N4（40 GB ≈ 0.3M token）：Strata §1 原文 "40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B"，"which can be quickly consumed by a handful of documents or hundreds of conversation turns"。与第 254 行一致；本页按 128 KB/token 复算 $40\times 2^{30}/131{,}072\approx 0.33\text{M}$，与来源「约 0.3M」同量级。
- N2（Llama-3.1-8B 配置）：模型 config.json 取值为 num_hidden_layers=32、num_attention_heads=32、num_key_value_heads=8、hidden_size=4096、head_dim=128、max_position_embeddings=131072、torch_dtype=bfloat16。与第 225、238 行使用的「32 层 / 8 KV 头 / 头维 128 / 128K / bf16」逐一一致。
- 第 268 行 HiSparse 机制（arXiv:2608.07009v1）：HiSparse §3.1 原文 "For each request and layer, HiSparse reserves a fixed-size GPU cache"；§3.2 "Conceptually, there is a B-slot cache for each request and layer"；摘要 "keeps each request's full KV history in host memory"，"bounds its decode footprint with a small, fixed-size GPU cache"。页面「全量历史留在主机内存、显存里每请求每层只保留固定大小的缓存、解码显存不随上下文长度增长」逐点有据。
- 文献出处：vLLM = SOSP 2023（arXiv:2309.06180）✓；SGLang = NeurIPS 2024（arXiv:2312.07104）✓；Strata = OSDI 2026（作者公开出版页 zhiqiangxie.com/publications 列于 OSDI 2026，arXiv:2508.18572）✓。
- 算术复算（全部通过）：$2\times32\times8\times128\times2=131{,}072$ B $=128$ KiB；$20{,}000\times131{,}072=2{,}621{,}440{,}000$ B $=2.4414$ GiB $\approx2.44$ GB；$20{,}100\times128$ KB $=2.45$ GiB；$131{,}072\times128$ KB $=2^{34}$ B $=16$ GiB；$2\times5120\times40\times2=819{,}200$ B $=800$ KiB；$2048\times800$ KB $=1.5625$ GiB（对应 vLLM 原文 1.6 GB）；$8/32=1/4$，MHA 结构即 512 KB；$40$ GiB$/131{,}072\approx0.33$M；$1+2+3+4=10$，$1+\cdots+n=n(n+1)/2$，$20{,}000^2/2\approx2$ 亿。分项之和、乘积与标注值均相符。
- 机械项：`validate.py wiki/kv-cache/index.html` 返回 "validation ok"；页面无 `$...$` 出现在 alt 属性（正文/images 无带数学的 alt）；无 <pre>/<code> 可运行代码，第 3 项（代码执行）不适用；图示为 HTML dg-flow 结构，无等宽字符框线与 SVG `<text>` ASCII 公式；正文引用的 6 个前置概念页（standard-attention、prefix-caching、hetero-pd、paged-attention、strata、hisparse）在仓库中均真实存在；overview.html 与 index.html 双向链接；head 的 description/dojo:summary/dojo:type/dojo:topics/dojo:tag 齐备，summary 内公式可由 KaTeX 渲染。

## 问题

- [轻微·一致性] dojo:summary（index.html 第 7 行）：20K token 手册写作「约 2.4 GB」，而正文四处（第 90、221、225、238、252 行）与 overview.html 第 35 行均写「2.44 GB」，同一量纲的数字在本页 summary 与正文/overview 之间精度不一致｜引文依据：正文 $20{,}000\times131{,}072\,\text{B}=2{,}621{,}440{,}000$ B $=2.4414$ GiB，即 2.44 GB（2.4 为其一位小数舍入，非量值冲突，故不计为阻断）｜修复要求：将 dojo:summary 中的「约 2.4 GB」改为「约 2.44 GB」，与正文及 overview 统一｜修复：｜复验：
- [轻微·来源] 来源与范围说明·论断与来源（C）（index.html 第 298 行）：C5 的来源标注为「SGLang §1、§3、vLLM §4.3」，并将 C4、C5 一并挂在「vLLM 论文 §1 与 SGLang 论文 §3」之下；核对 SGLang §3 只描述 SGVM/RadixAttention 跨调用复用（"A key aspect of SGVM is its RadixAttention feature, which automatically reuses KV caches across multiple calls."），并不含「请求完成后丢弃缓存」的论断，该论断在 SGLang §1；C4（65%/30% 显存分布）出自 vLLM §1，与 SGLang 无关｜引文依据：SGLang §1 "In existing systems, the KV cache of a request is discarded after processing is completed"；vLLM §1 "Approximately 65% of the memory is allocated for the model weights" ｜修复要求：把 C5 中的 SGLang 标注由「§1、§3」改为「§1」，并把 C4 与 C5 的来源分列（C4 → vLLM §1；C5 → SGLang §1、vLLM §4.3），不再统一挂到「SGLang §3」；论断本身保留｜修复：｜复验：
- [轻微·可读性] 「核心问题」第 3 条解答（index.html 第 90 行）：MHA、GQA、bf16 三个缩写在页面正文中首次出现时未给中文全称，全称直到第 3 章才出现（第 223 行「MHA（多头注意力）」「GQA（分组查询注意力）」、第 225 行「bf16（bfloat16，每元素 2 字节）」）｜引文依据：不适用｜修复要求：在第 90 行 MHA、GQA、bf16 首次出现处补中文全称括注（MHA（多头注意力）、GQA（分组查询注意力）、bf16），使术语在首次使用时即得解释｜修复：｜复验：

（另记一处不影响判定的观察，不计入问题数：「来源与范围说明」的 C 小节从 C3 起列、无 C1，且顺序为 C3、C2、C4、C5；正文与来源编号双向对应完整，无悬空引用。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。本轮未发现核心结论错误、来源不支持或数字矛盾：全部事实性论断（65%/30% 显存分布、OPT-13B 800 KB/token 与 1.6 GB、0.3M token 容量、两阶段机制、完成后丢弃缓存、Llama-3.1-8B 配置、HiSparse 机制）均回源核对成功并留有原文片段；全部算式与分项之和复算一致；O(n²) 与「K/V 与查询无关」已按推断显式标注；表述维度通读未发现元话语、会话指代、调试叙事、临场评价或 AI 拼接腔。3 条轻微问题均为精度/编号/术语展开层面，不改变主线结论，接受理由为可用一句话局部修订关闭，不影响本页当前版本的使用与发布。
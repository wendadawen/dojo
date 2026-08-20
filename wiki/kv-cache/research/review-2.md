# KV cache 审查记录（第 2 轮）

- 页面版本：index.html 4cc68ac454c845a90d495bd9e969e8d63f2ca64a；overview.html db00ef5fee893e6552db79c1670f5b1704ef8824（工作树哈希）
- 审查时间：2026-08-20 11:17
- 审查者：独立子代理（reviewer-kv-cache-2）
- 输入材料：`wiki/kv-cache/index.html`、`wiki/kv-cache/overview.html`、`/tmp/strata-research/vllm-paper.txt`（arXiv:2309.06180v1 文本）、`/tmp/strata-research/sglang-paper.txt`（arXiv:2312.07104v2 文本）、`guides/concept/check.md`。Llama-3.1-8B 配置（32 层/8 KV 头/head_dim 128/bf16/128K 窗口）由编排者预先核实。未读取 `research/` 下任何文件。
- 已完整阅读章节（按顺序）：
  - index.html：标题与导语 → 核心问题（4 题及解答折叠块）→ 第 1 章 注意力为什么需要缓存（含 Attention 公式、$k_i=x_iW_K$ 公式、4 token 表格、「展开：4 token 例子的逐步 K/V 计算」折叠块、本章问题 2 题）→ 第 2 章 prefill 与 decode（含流程图、多轮对话算例、本章问题 2 题）→ 第 3 章 缓存有多大（含 F1 公式、GQA/MHA 说明、Llama-3.1-8B 与 OPT-13B 算例、「展开：128 KB/token 的完整代入与两个检查」折叠块、本章问题 2 题）→ 第 4 章 为什么显存成为瓶颈（含三个因素、量级判断、算例对照表、三处误解澄清、本章问题 3 题）→ 来源与范围说明（核心论断与来源/核心公式与来源/外部数字与实验条件/构造示例/辅助解释与类比边界/简化条件及其限制，共 6 小节）
  - overview.html：这是什么 → 为什么需要它 → 核心机制 → 关键结论与边界
- 机械验证：四个前置概念链接（../standard-attention、../paged-attention、../prefix-caching、../strata）与 ../../index.html、overview.html、本地 libs 资源全部存在；Unicode 数学字符扫描仅命中 UI 装饰字符（导航「←」、标题分隔符「·」、JS 模板字符串），无数学语境违规；按文本节点模拟 KaTeX auto-render 配对，overview.html 无未配对 `$`，index.html 有 1 处（见问题 2）。

## 问题

- [重要·技术] index.html 第 2 章正文与本章问题（"20,000 token $\times$ 128 KB = 2.5 GB"两处）、核心问题解答 3 / 第 3 章正文与本章问题 / 第 4 章正文与算例表 / overview.html（"20,000 token 文档约 2.56 GB"）：同一量（20,000 token 手册的 KV cache）全站出现 2.5 GB 与 2.56 GB 两种数值，且均不可与同页 16 GB 用同一口径复算。复算：每 token 131,072 B（2×32×8×128×2），20,000 token = 2,621,440,000 B = 2.44 GiB（二进制）或 2.62 GB（十进制）；2.56 GB 来自「二进制 KB × 10^6 KB/GB」混合换算（20,000×128 KB = 2,560,000 KB）。而 128K 窗口单请求缓存按二进制恰为 16 GiB（131,072×131,072 B = 16×2^30 B，页面写 16 GB），第 4 章 "16–128 GB 量级" 的 128 GB 亦为二进制口径（1M token = 128 GiB），"40 GB/128 KB ≈ 0.31M" 又是十进制 GB ÷ 二进制 KB。同一页三种换算口径并存、同一数值两种写法，数值示例不满足「可以复算」要求。｜引文依据：不适用（复算类；vLLM 论文 §3 的 800 KB 算例本身用 2×5120×40×2=819,200 B=800 KB 的纯二进制口径）。｜修复要求：全页（含 overview.html）统一换算口径并与 16 GB、128 GB 相容——建议统一按 1 GB = 2^30 B：20,000 token 改为「约 2.44 GB」（或 2.4 GB），第 2 章 "2.5 GB" 两处同步，第 4 章 "40 GB/128 KB ≈ 0.31M" 改为 "≈ 0.33M（约 0.3M）"；或另行选定口径但须使 2.44/2.62、16/16.78、0.31/0.33、128/137 各组数字按同一口径自洽，并将第 2 章 "（20,000 token $\times$ 128 KB = 2.5 GB）" 的等号与结果一并写入 $...$（与第 3 章 "$20{,}000\times 128\,\text{KB}=...$" 写法一致）。｜修复：｜复验：
- [重要·机械] index.html 第 1 章「关键观察」段（`<span class="callout-inline">` 内，约 L802）："（其输入 $x_i$ 由第 1..i$ 个 token 逐层计算得到）"——"1..i" 之后有一个未配对的 `$`（该文本节点内 `$` 为奇数个，经文本节点级配对模拟确认，全页仅此一处）。KaTeX auto-render 遇不成对定界符将放弃渲染，页面上会显示字面 "$" 字符，且区间 "1..i" 作为数学符号未置于 $...$ 内，违反「数学符号全部由 KaTeX 渲染」机械项与发布条件。此问题为第 1 轮补充「依赖全部前缀」说明时引入（该补充的语义本身自洽：$k_i$ 直接依赖 $x_i$ 与 $W_K$、$x_i$ 逐层依赖前缀，与 vLLM §2.1 "the KV cache of one token depends on all its previous tokens" 一致，无内容错误，仅排版缺陷）。｜引文依据：不适用。｜修复要求：改为「由第 $1..i$ 个 token」或「由第 $1\ldots i$ 个 token」，保证该文本节点内 `$` 成对；修复后重跑配对检查应为 0 处。｜修复：｜复验：
- [轻微·来源] index.html meta description、blockquote「主要依据」、来源与范围说明「核心论断与来源」中 C5 的位置标注 "SGLang NeurIPS 2024 §3.2"：所提供的 SGLang 论文文本（arXiv:2312.07104v2）中 §3 无编号小节 3.2，该位置定位不到。论断本身有原文支持（见下），仅标注位置不可定位。另 C5 的 vLLM 侧标注 "§1" 仅有 Figure 1 caption "The memory for the KV cache (red) is (de)allocated per serving request" 的弱支持，明确表述在 §4.2/§4.3。｜引文依据：SGLang §1 "In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls"；SGLang §3 首段 "Unlike existing systems that discard the KV cache after a generation request finishes, our system retains the cache for prompts and generation results in a radix tree"；vLLM §4.2 "Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests"。｜修复要求：将 meta description、blockquote 与来源说明中的 SGLang 位置改为可定位的 §1 与 §3（如所引为 NeurIPS 2024 相机就绪版，须先核对该版本实际编号再标注）；vLLM 侧补注 §4.2（Figure 1 caption 可保留为辅助）。｜修复：｜复验：
- [轻微·可读性] index.html 核心问题解答 4（首现 "40GB HBM 对 Llama-3.1-8B 只够缓存约 0.3M token"）及第 4 章正文、算例表（"40 GB HBM 缓存容量"）：缩写 HBM 首次使用时未解释。｜引文依据：不适用。｜修复要求：在首现处加简注，如「HBM（GPU 的高带宽显存）」。｜修复：｜复验：
- [轻微·可读性] index.html 第 3 章正文 "GQA（分组查询注意力）中多组查询头共享一组 KV 头" 及第 3 章本章问题解答 2 "GQA 让 $H_{\text{kv}}$ 组查询头共享一组 KV 头"：「多组查询头共享一组 KV 头」字面可读作「全部查询头共享同一组 KV 头」（即 MQA 情形，对应 $H_{\text{kv}}=1$），与同段 $H_{\text{kv}}=8$、$8/32=1/4$ 的表述矛盾，存在形成误解的歧义。｜引文依据：不适用。｜修复要求：改为无歧义表述，如「查询头分成 $H_{\text{kv}}$ 组，每组内的查询头共享同一个 KV 头」，两处同步修改。｜修复：｜复验：

## 来源核对记录（对照 vLLM/SGLang 原文；通过项不重复列为问题）

- C1（KV cache 定义）：通过。vLLM §1："Close to 30% of the memory is used to store the dynamic states of the requests. For Transformers, these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache [41], which represent the context from earlier tokens to generate new output tokens in sequence."
- C2（prefill/decode 两阶段）：标注位置 Strata §2.1 本轮无法核对（Strata 原文未提供）；论断内容有 vLLM §2.1 等价支持："The prompt phase takes the whole user prompt (𝑥1, . . . , 𝑥𝑛) as input and computes the probability of the first new token... the computation of the prompt phase can be parallelized using matrix-matrix multiplication operations"、"The autoregressive generation phase generates the remaining new tokens sequentially. At iteration 𝑡... only the new key and value vector 𝑘𝑛+𝑡 and 𝑣𝑛+𝑡 are computed at this iteration"、"this phase severely underutilizes GPU computation and becomes memory-bound"（后者同时支持第 2 章 decode 访存密集的表述）。
- C3（K/V 与查询无关，页面已标注推断）：通过。vLLM §2.1 Eq. 2："𝑞𝑖 = 𝑊𝑞𝑥𝑖, 𝑘𝑖 = 𝑊𝑘𝑥𝑖, 𝑣𝑖 = 𝑊𝑣𝑥𝑖" 及 "the key and value vectors at positions 1 to 𝑛+𝑡−1 are cached at previous iterations"。推断标注合规。
- C4（13B/A100 40GB 显存分布）：通过。vLLM §1："Fig. 1 (left) illustrates the memory distribution for a 13B-parameter LLM on an NVIDIA A100 GPU with 40GB RAM. Approximately 65% of the memory is allocated for the model weights... Close to 30% of the memory is used to store the dynamic states of the requests."
- C5（完成后丢弃缓存）：内容通过（引文见问题 3）；位置标注问题见问题 3。
- F1（每 token 字节数公式）：通过。vLLM §3："the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)"——MHA 形式一致；页面已将 GQA 推广形式（$H_{\text{kv}}\cdot d_{\text{head}}$ 替代 $d_{\text{hidden}}$）标注为推断并经算例复算验证，处理合规。复算 Llama-3.1-8B：2×32×8×128×2 = 131,072 B = 128 KB ✓。
- N1（OPT-13B 800 KB/token、单请求 1.6 GB）：通过。同上 §3 原文，及 "Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as 1.6 GB"。复算：819,200 B = 800 KB ✓；2048×800 KB = 1.6 GB ✓。页面"已被视为'大'（vLLM 论文原话）"与 §3 小节标题 "Large KV cache" 相符。
- N2（Llama-3.1-8B 配置）：编排者已核实（32 层/8 KV 头/head_dim 128/bf16/128K 窗口），本轮未重复核对。
- N3（65%/30%）：通过，引文同 C4（vLLM §1 Figure 1：Parameters 26GB, 65%；KV Cache >30%）。
- N4（40 GB ≈ 0.3M token）：标注 Strata §1 本轮无法核对（原文未提供）；页面自身复算 40×10^9/131,072 ≈ 0.31M 成立（口径一致性问题见问题 1）。
- N5（128K → 16 GB）：F1×N2 推算，二进制口径下精确（131,072 token × 131,072 B = 16×2^30 B）；口径一致性问题见问题 1。
- 4 token 例子：复算通过（无缓存 1+2+3+4=10 组、有缓存 4 组；$n(n+1)/2$；20,000 token → 200,010,000 ≈ 2 亿组）。
- GQA 比例：复算通过（$H_{\text{kv}}/H_q = 8/32 = 1/4$；MHA 对照 2×32×32×128×2 = 524,288 B = 512 KB，恰 4 倍）。
- 其余抽查：20,100/20,300 token 多轮算例自洽；两级问题块（核心问题 4 题 + 各章 2/2/2/3 题）均有解答折叠块、答案独立可读、核心问题答案均指明论证所在章节；折叠块收起后正文结论完整；「纯读取」的类比边界已在来源说明中声明；简化条件（sliding window/MLA/KV 量化/batching）已列明。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（2 个重要问题——数值口径不一致、未配对 `$`——修复并复验后，无阻断问题，页面质量满足继续第 3 轮审查的条件；轻微问题建议一并修复）


## Round 2 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | 2.5/2.56 GB 口径混乱 | 全页统一按 1 KB=1,024 B、1 GB=2^30 B：20,000 token 改"约 2.44 GB"（含 overview）、0.31M 改"$\approx 0.33\text{M}$（约 0.3M，Strata 原文口径）"、§3 公式后新增口径声明段 | validate.py ok；grep 旧值清零 |
| 重要 #2 | "由第 1..i$ 个"未配对 $ | 改 "由前 $1\ldots i$ 个 token" | 文本节点级配对检查 0 处 |
| 轻微 #3 | C5 定位 §3.2 不可定位 | SGLang 改 §1、§3；vLLM 补 §4.2 | 验证 |
| 轻微 #4 | HBM 首现未解释 | 核心问题 Q4 答案处加"（GPU 高带宽显存）" | 验证 |
| 轻微 #5 | GQA"多组共享一组"可误读为 MQA | 两处改"查询头分成 $H_{\text{kv}}$ 组、每组内的查询头共享同一个 KV 头" | 验证 |

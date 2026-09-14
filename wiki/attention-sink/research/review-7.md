<!-- review-meta
round: 7
page: wiki/attention-sink/index.html
reviewed_content_sha256: 7c3ef7a39071bfe2
-->
# 注意力汇聚点审查记录（第 7 轮）

- 页面版本：81040663fe37c7b1e3bcc4386b02fd70085e594c
- 审查时间：2026-09-13 21:08
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 现象：开头的位置拿到远超其重要性的注意力 / 2. 成因：softmax 必须把注意力分配完 / 3. 两种做法（3.1 形态一、3.2 形态二）/ 4. 对缓存与推理意味着什么（4.1、4.2、4.3）/ 来源与范围说明；含全部折叠块（4 个「解答：」、1 个「补充：」、1 个「展开：」、1 个「代码：」）与 2 处表格。外部来源：StreamingLLM（arXiv:2309.17453）HTML 全文逐句定位；DeepSeek-V4.1-Flash 官方材料（`wiki/deepseek-v4-1/research/official/README.md`、`official/inference/README.md`）、`wiki/deepseek-v4-1/research/measured.md` 产物登记。

## 核对依据（关键数值与片段）

- 份额与解析解复算：`share=e^{sink}/(W+e^{sink})`，$W=128$：$1/129=0.7752\%$、$e^2/(128+e^2)=5.4577\%$、$e^5/(128+e^5)=53.6928\%$、$e^8/(128+e^8)=95.8828\%$——与正文 0.78% / 5.46% / 53.69% / 95.88% 一致；解析解 $(v_1+v_2)/(2+e^{sink})$，$sink=-\infty/0/2$ 分别为 $(v_1+v_2)/2$、$/3$、$(v_1+v_2)/9.389$，与正文一致。
- 页面代码块（index.html L294–299）复制执行，输出逐字符等于页面「预期输出」（L303–306）。
- StreamingLLM Table 1（Llama-2-13B，PG19 首本书 65K token）：「0+1024」5158.07、「4+1020」5.40、「4"\n"+1020」5.60；Table 2 说明「Cache config x+y denotes adding x initial tokens with y recent tokens. Perplexities are evaluated on 400K tokens in the concatenated PG19 test set.」，模型为 Falcon-7B/MPT-7B/Pythia-12B/Llama-2-7B（前三个窗口 2048、Llama-2-7B 用 4096）；Table 3 位于 §4.2「Results of Pre-Training with a Sink Token」，为 160M 参数、按 Pythia-160M 代码库同条件训练、PG19 首个样本评测，sink 模型 1+1023 困惑度 18.01、vanilla 2+1022 为 18.05。
- 引文逐句定位：§3.1「beyond the bottom two layers… focuses on the initial tokens across all layers and heads」「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance…」「The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values…」「removing these initial tokens' KV will remove a considerable portion of the denominator…」「initial tokens are visible to all subsequent tokens… more easily trained to serve as attention sinks」「the absolute position of the starting tokens, rather than their semantic value, holds greater significance」；§1「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV…」；§3.1「Introducing four initial tokens generally suffices; further additions have diminishing returns.」；§3.2「StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial…」；附录 A「it does not extend the models' context window or enhance their long-term memory capabilities.」
- DeepSeek-V4.1-Flash：官方 README「a 40-layer Transformer organized as a 20-layer causal encoder followed by a 20-layer decoder」；窗口 128、每头一个 `attn_sink`（64 query 头）来自官方配置与 checkpoint 张量头；`research/measured.md`（本页）与 `wiki/deepseek-v4-1/research/measured.md`（登记 `ckpt/headers.json`）均存在，页面引用的路径可定位。
- 机械项：`python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` → `validation ok`；页面无 Unicode 数学字符；`alt` 中无 `$...$`；只 1 个 blue callout；h2/h3 编号连续；核心问题 4 条、每章「本章问题」均有独立「解答：」折叠块；index↔overview 互链；`../kv-cache/`、`../sliding-window-attention/`、`../standard-attention/` 前置概念页均存在。

## 问题

- [轻微·来源] 4.3 边界（index.html L315）：正文以 `[C4]` 支撑「要真正「记得更远」，需要窗口之外的另一路可见性（例如压缩后的全局 KV）」这一论断，但 `[C4]` 条目（L341）登记的内容只有 `attn_sink` 参数定义、稀疏注意力分母与 checkpoint 张量头，未登记「压缩后的全局 KV」的依据，读者无法从 `[C4]` 定位该例子的出处。｜引文依据：`[C4]` 原文「其注意力模块中 self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))，随稀疏注意力调用传入；稀疏注意力内核把汇聚点项加在分母上……真实 checkpoint 分片头：43 个 attn_sink 张量」——不含压缩全局 KV；该事实的实际依据在官方 README「Compressed Sparse Attention 2 (CSA2)… the decoder's global KV cache is projected from the final encoder hidden states … reduce the global KV cache footprint to 890 bytes per token」（`wiki/deepseek-v4-1/research/official/README.md`）。｜修复要求：把「压缩后的全局 KV」的依据补进 `[C4]`（或为该例子单列一条来源，指向官方 README 的 CSA2/全局 KV 说明），使该论断可按引文核对。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布

> 本轮所列问题的处理结果见 `minor-fixes.md`。

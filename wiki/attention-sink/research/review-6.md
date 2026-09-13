<!-- review-meta
round: 6
page: wiki/attention-sink/index.html
reviewed_content_sha256: a74c1f315ccade5e
-->
# 注意力汇聚点审查记录（第 6 轮）

- 页面版本：059e41a4f2d3e4a311c4f263dc86ee29e5207e63
- 审查时间：2026-09-13 20:11
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题、常见误解、1. 现象、2. 成因、3. 两种做法（3.1、3.2）、4. 对缓存与推理意味着什么（4.1、4.2、4.3）、来源与范围说明；含全部 details 折叠块与两个代码块。本页无图，无图注项。

核查说明：StreamingLLM 依据 arXiv:2309.17453 全文（ar5iv HTML）逐句定位，Table 1/2/3、§3.1–3.3、§4.2 与附录 A 均给出原文片段；DeepSeek-V4.1-Flash 依据仓库内官方材料 `wiki/deepseek-v4-1/research/official/README.md` 与 `.../official/inference/README.md`（技术报告 PDF、config.json、model.py/kernel.py、headers.json 均已不在仓库，页面现指向 `research/measured.md`，该文件与 `wiki/deepseek-v4-1/research/measured.md` 均存在）。share 代码块已复制执行，输出与「预期输出」逐字符一致（sink= 0.0/2.0/5.0/8.0 → 0.0078/0.0546/0.5369/0.9588）。`python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` 返回 `validation ok`；`dojo:topics=注意力机制`、`dojo:tag=KV cache` 均在 AGENTS.md 与 catalog_builder.py 的封闭词表内；两处概念内链 `../kv-cache/index.html`、`../sliding-window-attention/index.html` 均存在；index.html 与 overview.html 互相链接。

已核对通过（不列为问题）：[C1]「beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads.」与「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」；[C2] 的 SoftMax「prevents all attended tokens from having zero values … dump unnecessary attention values to specific tokens」、「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function (Equation 1) … significant shift in the distribution of attention scores」、「Due to the sequential nature of autoregressive language modeling, initial tokens are visible to all subsequent tokens … initial tokens are more easily trained to serve as attention sinks」，均逐字命中 §3.1；[C6]「it does not extend the models' context window or enhance their long-term memory capabilities.」（附录 A）逐字命中；[N1] Table 1「0 + 1024 (Window) 5158.07 / 4 + 1020 5.40 / 4"\n"+1020 5.60」及表注「Perplexities are measured on the first book (65K tokens) in the PG19 test set」逐项命中，[N1] 关于「1024/2048/4096 是各次实验窗口长度、Table 2 的 Llama-2-7B 组用 4096」的说明与论文 Table 2 一致；[N2] Table 2 四模型 400K token 对照与「Introducing four initial tokens generally suffices; further additions have diminishing returns.」命中；[N3] Table 3「Learnable Sink 1+1023 = 18.01」「Vanilla 2+1022 = 18.05」与 §4.2「we trained two language models, each with 160 million parameters, under identical conditions … employed the Pythia-160M codebase and followed its training recipe」命中（正文用「两个 160M 参数模型」与 §4.2 口径一致）；§3.2 位置口径句「StreamingLLM focuses on positions within the cache rather than those in the original text.」逐字命中。§3.2 公式与 §4.2 份额公式符号全文单义，份额数值复算一致；§3.2 展开块的解析解（q=0、两槽位 → (v1+v2)/(2+e^sink)，sink=2 时 2+e^2≈9.39）与出口结论一致；「整行可见槽位无效 → 分子 0、汇�poly点项趋无穷、输出全零」与 §3 本章问题 3、§4 结论一致。一级「核心问题」4 条、各章「本章问题」2/2/3/2 条均配解答折叠块，答案与正文结论一致且指明了完整论证所在章节。全文未出现「我们/你」等会话指代，未出现「下面来看/需要注意的是」式元话语与调试踩坑叙事，无 Unicode 数学字符，无「（待生成）」占位。

## 问题

- [重要·技术] index.html:250（第 3 章「本章问题」第 1 题解答）：把 StreamingLLM 的「可学习 sink token」实验（Table 3，配置 1+1023）当作「每头一个可学习标量」的「直接对照」，并写「差别就在于前者不需要靠真实位置来充当这个角色」。sink token 是 prepend 到每个训练样本的占位 token、带可学习 KV、占 1 个缓存槽（配置里的那个 1），与本节主张的「可学习标量以参数形式加在分母上、不占用任何缓存槽位」是两种不同机制。该实验只能支持「专用可学习 sink 减少了对多个初始位置的依赖」，不能支持标量形态的「不占缓存、不依赖任何 token」；且「不需要靠真实位置」与 sink token 本身就是一个固定位置相抵触。｜引文依据：论文 §3.3「one prepending a learnable placeholder token (Sink Token) in all training samples」；Table 3 行 Learnable Sink「1+1023 = 18.01」、Vanilla「2+1022 = 18.05」；同表注「Cache config x+y denotes adding x initial tokens with y recent tokens.」。｜修复要求：在该解答中说明 sink token 是占 1 个缓存槽的 prepend 占位 token，与只进分母的标量属两种机制，把该对照限定为「专用可学习 sink 可减少对多个初始位置的依赖」，并删除或改写「前者不需要靠真实位置来充当这个角色」。｜修复：｜复验：

- [轻微·技术] index.html:285、289 及文末「简化条件及其限制」：把「可见槽位个数 $W$」与 DeepSeek-V4.1-Flash 的「窗口大小 128」等同。该形态在第 3 章被描述为稀疏注意力，其一次调用的可见集并不止窗口；第 4.2 节与「简化条件」都没有交代 $W=128$ 只是取窗口大小这一构造取值，读者会把 0.78%–95.88% 当成在真实可见槽位集合上算出的份额。｜引文依据：本页 :285「$W$：可见槽位个数（窗口场景下即窗口大小）」与 :289「取 $W=128$（DeepSeek-V4.1-Flash 的窗口大小）」；[N5]「构造示例，$W=128$」。｜修复要求：在 4.2 节或「简化条件及其限制」中写明 $W=128$ 只是取该模型窗口大小作为构造取值、并非其实际可见槽位总数。｜修复：｜复验：

- [轻微·表述] index.html:66、129：以「本页」为主语的自我指代。:66「本页讨论的汇聚点是…」、:129「因此本页把「注意力分数高」与「信息重要」明确分开」，均可直接以主语陈述。｜引文依据：不适用｜修复要求：改为无「本页」主语的陈述，如 :66「汇聚点是注意力分配的去处…」、:129「因此「注意力分数高」与「信息重要」需要分开看…」。｜修复：｜复验：

- [轻微·来源] index.html:340（[C3]）：引号内首句「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV to anchor the attention computation and stabilize the model's performance.」是论文 §1 引言原文，不是标注的 §3.2。｜引文依据：该句在 arXiv:2309.17453 全文仅出现一次，位于 §1（「Based on the above insights, we propose StreamingLLM… with just 4 initial tokens sufficing…」段）；§3.2 以「we reintroduce a few starting tokens' KV in the attention computation」表述同一方法，不含该句。｜修复要求：把该引文的出处由「§3.2」改为「§1 引言（§3.2 同述该方法）」，或去掉引号改为对 §3.2 的概述。｜修复：｜复验：

- [轻微·格式] index.html:7（dojo:summary）：「43 层 $\times$ 64 头」与正文口径不一。正文 :90、:218、:227 统一写作「40 个主干层与 3 个 MTP 层」，summary 的「43 层」易被读成 43 个主干层。｜引文依据：不适用｜修复要求：summary 改为与正文一致的口径，如「主干 40 层 + 3 个 MTP 层，每层每头一个」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复
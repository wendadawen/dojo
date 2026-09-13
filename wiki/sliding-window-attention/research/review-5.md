<!-- review-meta
round: 5
page: wiki/sliding-window-attention/index.html
reviewed_content_sha256: 1236991566dd3a25
-->
# 滑动窗口注意力审查记录（第 5 轮）

- 页面版本：68bfdf6752412ee98772546a8adc6eaf2bdb1f9e（index.html 工作树哈希）
- 审查时间：2026-09-13 20:25
- 审查者：编排者派发的独立审查者（子代理，未参与写作与前序审查）
- 已完整阅读章节：核心问题、常见误解、1. 窗口把可见集切成「最近 W 个位置」（含 1.1、本章问题、$-1$ 槽位折叠块）、2. 省下的是打分次数与缓存字节——以及精度取舍（含 2.1/2.2/2.3、本章问题、40 层总量折叠块）、3. 层堆叠接力：感受野 k×W 与它的天花板（含 3.1/3.2/3.3、本章问题、StreamingLLM 对照数字折叠块）、4. 环形缓冲：槽位、覆盖与位置编码（含 4.1 代码折叠块、4.2/4.3、本章问题）、来源与范围说明（全部 C/F/N 小节、构造示例、辅助解释与类比边界、简化条件及其限制）；另读 overview.html

## 问题

- [轻微·可读性] 来源与范围说明·[C1]（L434）｜问题：[C1] 引文中的区间写法用 U+2212 减号「i−W」，与全页其余位置在公式里统一用 ASCII 连字符的写法（如 $[i-W+1, i]$）不同，同一变量/运算符全页写法不一致｜引文依据：原作语料原文为 "positions between i−W and i"（ar5iv 2310.06825 Figure 1 与 §2 原文即用连字符），本页为逐字引文；页内 L115/L161 等处写 $[i-W+1, i]$、$[i-W, i]$｜修复要求：把引文里的「i−W」改为与页内一致的 ASCII「i-W」（保持引文其余逐字不变），或明确该处为原文照录；不得改动引文语义｜修复：｜复验：
- [轻微·可读性] 3.1 节（L269–277）｜问题：公式 $\text{span}(k)=k\times W$ 的符号说明列了 $k$、$W$、$\text{span}(k)$，但同句正文使用的 $h_i$ 未在任何位置定义（首次出现的"第 $k$ 层位置 $i$ 的隐状态"与 $h_i$ 之间没有显式对应），读者须自行把 $h_i$ 关联到位置 $i$ 的隐状态；[C1] 引文里出现的 h_i 也不能替代定义｜引文依据：不适用（可读性）｜修复要求：在符号说明列表补一条「$h_i$：位置 $i$ 的隐状态」，或在该句首次使用处写成「位置 $i$ 的隐状态 $h_i$」后再引用 $h_i$｜修复：｜复验：
- [轻微·可读性] 2.1 节（L199、L201、表 L211–216）与 1.1 节表（L163）｜问题：窗口打分次数的计数口径两处不一致——1.1 节表格写「约 $N\cdot W$」，2.1 节公式写 $n_{\text{score}}(N)=N\times W$（等式，无「约」），对照表给出精确值 524 288 等。但位置 $i<W-1$ 的 query 可见槽位不足 $W$，真实计数应为 $\sum_{i=0}^{N-1}\min(i+1,W)$：$N=4096$、$W=128$ 时为 516 160，而非表中所列 524 288（差 8 128，约 1.5%）。页面把 $W$ 定义为「可见槽位数上限」（L206）已隐含这是上界，但公式与表格未标条件｜引文依据：不适用（可读性/口径）｜修复要求：把 2.1 节的 $N\times W$ 统一为「约」（与 1.1 节表格一致），或补一句条件——「按每个 query 满窗口 $W$ 计，$N\ge W$ 时成立；序列开头不足 $W$ 个槽位的 query 使真实计数略小于此」，表格数值保持现口径但标注为上限估算｜修复：｜复验：

## 已核对通过（无问题）的主要事实项

以下逐条回源核对，均一致（引文依据见括号内来源位置），供复验参考：

- [C1] Mistral 7B（arXiv:2310.06825）§2「Sliding Window Attention」：原文「The hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i.」「Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens」「At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens.」——页面引文逐字一致；「图注写明每个 token 最多关注上一层的 $W$ 个 token」（L181）对应 Figure 1 caption 原文「each token can attend to at most W tokens from the previous layer」。
- [C2] 同文 §2「Rolling Buffer Cache」：原文「The keys and values for the timestep i are stored in position i mod W of the cache」「a fixed size of W」，与页面 slot(i)=i mod W、覆盖最旧的表述一致（ar5iv 原文命中）。
- [C3] StreamingLLM（arXiv:2309.17453）：§1「StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV」；Table 2 caption「(1) Window attention (0+y) has a drastic increase in perplexity.」；§3.1「removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function」——三条引文均逐字命中。
- [C5] §3.2 原文「StreamingLLM focuses on positions within the cache rather than those in the original text. This distinction is crucial for StreamingLLM's performance.」——命中。
- [N1] Mistral $W=4096$、32 层、$4096\times32=131\,072$：论文 §2「using a window size of W=4096, we have a theoretical attention span of approximately 131K tokens」；Table 1 的 n_layers=32、window=4096 亦对应。
- [N2] 「32K 序列下缓存显存减少 8 倍（$32768/4096=8$）」：论文 §2 原文「On a sequence length of 32k tokens, this reduces the cache memory usage by 8x」。
- [N3] StreamingLLM Table 1（Llama-2-13B，PG19 65K）：「0+1024」5158.07、「4+1020」5.40、4 个换行符 5.60——数值与档位均一致（L316 正文「5.60 对 5.40」的新旧顺序正确）。
- [N4] Table 3（160M）：可学习 sink「1+1023」18.01、vanilla「2+1022」18.05——一致。
- 复算类数字逐项复算通过：$128\times512\times1\,\text{B}=65\,536\,\text{B}=64\,\text{KiB}$；$40\times64\,\text{KiB}=2560\,\text{KiB}=2.50\,\text{MiB}$；$3\times64\,\text{KiB}=192\,\text{KiB}=0.19\,\text{MiB}$，合计 2.69 MiB；$1\,048\,576\times512\times1\,\text{B}=512\,\text{MiB}$，40 层 20 GiB，比值 $1/8192$；打分次数表三行（$4096^2$、$131072^2$、$1048576^2$ 与 $N\times128$）与倍数 32×/1024×/8192× 全部等于 $N/W$；$40\times128=5120$，$5120/1\,048\,576=0.49\%$；$4096\times32=131\,072$；$4999-128+1=4872$、$5000-128+1=4873$、$4999-128=4871$、$W+1=129$；$N(N+1)/2 / (N\cdot W)\approx N/(2W)$。
- 代码块（L362–380）实际执行（Python 3）：输出与页面「预期输出」逐行一致（槽位序列 [4,5,6,7,0,1,2,3]、映射回位置 [12..19]、期望因果窗 [12..19]、位置%W [4,5,6,7,0,1,2,3]）。
- [N7] 索引一致性独立复现：按页面同一环形缓冲规则写脚本，$W=128$、序列长 300 与 $W=8$、序列长 20 两种情况，decode 槽位映射回的位置集合与 prefill 因果窗口 $[\max(0,p-W+1),p]$ 逐步比对，不符均为 0 处；另跑 $W=128$、序列长 1000 亦为 0 处，与 [N7] 表述一致。
- 与同站 DeepSeek-V4.1-Flash 说明页交叉核对一致：报告 §2「40 causal Transformer layers ... both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only.」对应本页「给除最前两层外的层再挂一路压缩后的全局 KV」；「KV 头 $1\times512$」对应本页「只有 1 个 KV 头、512 维」；「SWA KV 保持 FP8」对应本页 1 B/元素；「3 个 MTP 块」对应本页「3 个 DSpark 草稿层」；官方 README（仓库内 research/official/inference/README.md）述 40 层主干、1M 上下文、DSpark 推测解码、FP4 主 KV。
- 页面链接与资源：standard-attention、causal-mask、kv-cache、rope、attention-sink、deepseek-v4-1 六个站内页与 overview.html↔index.html 互链均有效；引用的 research/measured.md（页内相对路径）与 wiki/deepseek-v4-1/research/measured.md 均存在（均为 .md）；无「（待生成）」占位。
- 公式书写与结构：validate.py 返回 `validation ok`；全页数学符号由 KaTeX 渲染，标题/summary/正文/列表/表格无 Unicode 数学字符（validate 白名单内允许的 ×、–、1 处引文 U+2212 见问题 1）；图示为内联 SVG 与 HTML dg-flow 结构，SVG `<text>` 仅含纯数字位置下标，无 ASCII 公式；对比度/窄屏由共享 CSS 承担。
- 问题块：核心问题 4 条（3–5 区间内），四条均带「解答：」折叠块且末尾指向第 1/2/3/4 章；四章的本章问题各 2 条，均带解答折叠块、答案独立可读、与正文结论一致。
- 表述：通读全文（含折叠块与图注）未发现元话语（「本页将…」「下面来看…」「需要注意的是」）、会话指代（我/我们/你）、调试与复现踩坑叙事、临场评价或 AI 拼接腔；「本页」自称符合 style-guide §12，无需修改。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（三条轻微问题不影响正确性与主线理解，可选修复）

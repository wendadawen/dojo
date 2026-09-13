<!-- review-meta
round: 3
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: 2c22450daa887c6b
-->
# MQA 与 GQA 审查记录（第 3 轮）

- 页面版本：2b42ac4e5f76d7589be4bd154dbba10431e25e37（wiki/mqa-gqa/index.html）
- 审查时间：2026-09-13 19:09
- 审查者：独立子代理（未参与写作与前序轮次）
- dojo:type：concept → 适用 guides/concept/check.md
- 已完整阅读章节：引言与核心问题 / 最容易误解 / 1. 为什么 MHA 推理受内存带宽限制 / 2. MQA——所有 query 头共享一组 K/V / 3. GQA——在 MHA 与 MQA 之间插值 / 4. 手算对比 / 5. 边界与后续 / 来源与范围说明（含全部折叠块、图注、overview.html）
- 来源核对方式：Shazeer 2019（arXiv:1911.02150，ar5iv 全文）、Ainslie 2023（arXiv:2305.13245，ar5iv 全文）、DeepSeek-V2（arXiv:2405.04434，ar5iv 全文）、NVIDIA 官方数据手册数字。overview.html 一并通读。

## 问题

- [阻断·技术] index.html L198–201（第 1 章「本章问题」第 3 题及其答案），并涉及正文 L179：把 Shazeer 性能比值 $\Theta(n/d+1/b)$ 中 $n/d$ 项的系数写成 $h$；同一页 L245 又写"把性能比值 $\Theta(n/d+1/b)$ 里的 $n/d$ 项系数减为 $1/h$"，对同一量给出互相矛盾的说法（$h$ 与 $1/h$），且"系数是 $h$"不被来源支持——Shazeer 的 MHA 式不含系数 $h$，$h$ 只在 MQA 式中以 $n/(dh)$ 出现。｜引文依据：Shazeer 2019 §2.4.1 "the ratio of memory access to arithmetic operations is Θ(n/d+1/b)"；同文 §3.1 "the ratio of memory access to arithmetic operations is Θ(1/d+n/(dh)+1/b)" 与 "We have reduced the offensive n/d by a factor of h."｜修复要求：把 L201、L179 的表述由"$n/d$ 项的系数是 $h$／减少 $h$ 这个系数"改为"$n/d$ 项的系数为 1，MQA 把它减为 $1/h$（$n/d\to n/(dh)$）"，与 L245 及原文一致；并在正文补出 MQA 式 $\Theta(1/d+n/(dh)+1/b)$，使"减 $h$ 倍"有式可依。｜修复：｜复验：
- [重要·技术] index.html L168 及来源 [C2]（L520）、[F5]（L536）：把 MHA 的 $\Theta(n/d+1/b)$ 归到 Shazeer 2019 §3.1，按页面标注的「§3.1 性能分析」定位不到该式（§3.1 给的是 MQA 的 $\Theta(1/d+n/(dh)+1/b)$）。｜引文依据：Shazeer 2019 §2.4.1 为 MHA 式 $\Theta(n/d+1/b)$，§3.1 为 MQA 式 $\Theta(1/d+n/(dh)+1/b)$。｜修复要求：L168、[C2]、[F5] 的出处改为 §2.4.1（MHA 式）与 §3.1（MQA 式、减 $h$ 倍结论）分别标注。｜修复：｜复验：
- [轻微·格式] 来源章节 [N1]（L541）、[N2]（L542）：两条在正文无任何 `<sup>` 引用，正文引用的 N 只有 N3/N4/N5/N6，来源编号与正文的"双向对应"只成立单向。｜引文依据：不适用｜修复要求：给 Table 1（L327 段或 L340 来源行）补 `<sup>[N2]</sup>`、给 MQA 定性结论处（L210 或 L253）补 `<sup>[N1]</sup>`；或删除未被引用的条目。｜修复：｜复验：
- [轻微·表述] index.html L159、L206、L285、L369、L456：存在元话语与临场评价——L159"下面用一个构造示例把它落到可手算的程度"；L206"那能不能少存几份？下一章看 MQA 的极端做法"；L285"减得太狠……下一章看 GQA 如何插值"；L369"三种机制分别讲完了。下一章把它们并排放一起手算"；L456"连续谱清楚了……下一章点出区别"。"下一章看…"同一句式重复四段。｜引文依据：不适用｜修复要求：按 style-guide §8 改为 1–2 句陈述"本节结论与下节问题的关系"，删除"下一章看／讲完了／清楚了"等元话语与"减得太狠"这类临场评价。｜修复：｜复验：
- [轻微·表述] index.html L65 与 L552（来源章节·构造示例）：L65 把 $h=128,d_k=128,l=80,n=4096$ 称为"真实模型"，而 L552 同类推算自称"数字为教学推算，不代表真实模型"；且 L65 未给出 $d_k$，读者无法据 L65 复算约 21.5 GB。｜引文依据：不适用｜修复要求：统一为"真实规模量级示例"一类措辞，并在 L65 补 $d_k=128$ 或直接指向第 4 章的推算块。｜修复：｜复验：
- [轻微·来源] 来源 [C7]（L525）：把 "The converted checkpoint is then pre-trained for a small proportion α of its original training steps" 归到 §3.1，该句实际在 Ainslie 2023 §2.1（Uptraining）；§3.1（Experimental setup）给的是 "For α=0.05, training took approximately 600 TPUv3 chip-days."。｜引文依据：Ainslie 2023 §2.1 含 "pre-trained for a small proportion α of its original training steps"；§3.1 含 600 TPUv3 chip-days。｜修复要求：α 句出处改为 §2.1，600 chip-days 保留 §3.1。｜修复：｜复验：
- [轻微·来源] 来源 [N3]（L543）与正文 L342：正文 L342 把措辞写成确定结论"论文称其为 favorable middle ground"，而 [N3] 保留"原文措辞未逐一核对，仅取其选定 G=8 的事实"的不确定注记（check.md §2.2.5 不保留未确认注记）。核对结果：原文确有 "…We selected 8 groups as a favorable middle ground."。｜引文依据：Ainslie 2023 Figure 6 讨论 "increasing the number of groups from MQA only results in modest slowdowns initially… We selected 8 groups as a favorable middle ground."｜修复要求：删除 [N3] 的不确定注记，或改写为已核实的事实陈述。｜修复：｜复验：
- [轻微·来源] index.html L255–258 折叠块「补充：MQA 在工程实现上的"广播"问题」："标准注意力 kernel（如 FlashAttention）通常假设 Q、K、V 的头维度一致……要么把 K/V 在头维度上广播 $h$ 份……要么修改 kernel"等 kernel 机制描述无来源支持。｜引文依据：不适用｜修复要求：补 kernel 文档/源码出处，或明确标注为工程推断而非来源事实。｜修复：｜复验：
- [轻微·来源] overview.html L46："$G=8$ …已成主流 LLM 标配（Llama 2 70B、Llama 3、Mistral 等）"为无来源支持的事实判断，写成结论。｜引文依据：不适用｜修复要求：补来源（各模型 config/技术报告），或删去"已成主流 LLM 标配"的枚举。｜修复：｜复验：

## 复核通过项（无问题）

- 手算与算术全部复算一致：$2\times4\times64=512$、$512\times10=5120$、$5120\times2=10240\approx10$ KB；$2\times64=128$、$1280$ 元素、$2560\approx2.5$ KB；$2\times2\times64=256$、$2560$ 元素、$5120\approx5$ KB；真实规模 $2\times128\times128\times4096\times80\times2\approx2.15\times10^{10}$ B≈21.5 GB、GQA-8≈1.34 GB、MQA≈168 MB；$32768/576\approx57$、MHA/MQA≈128 倍=$h$。
- Table 1 数字与来源一致：MHA-Large 0.37s/46.0、MHA-XXL 1.51s/47.2、MQA-XXL 0.24s/46.6、GQA-8-XXL 0.28s/47.1；倍率 1.51/0.24≈6.3×、1.51/0.28≈5.4×、47.2−46.6=0.6、47.2−47.1=0.1。caption 为 per-TPUv4 chip，与 L327 一致。
- 引文逐条核对通过：Shazeer §1 内存带宽句、§3 "the different heads share a single set of keys and values"、abstract "incur only minor quality degradation"；Ainslie §2.2 GQA 定义句、abstract "uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA"、§2.1 均值池化句、Appendix A "multi-query attention can lead to training instability during fine-tuning…"、"We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention."、数据集清单（CNN/DailyMail、arXiv、PubMed、MediaSum、MultiNews、WMT EnDe、TriviaQA）。
- DeepSeek-V2 §2.1.4 Table 1 对照一致：MHA $2h d_k$、GQA $2G d_k$、MQA $2d_k$、MLA $(d_c+d_h^R)$；配置 $n_h=128,d_h=128,d_c=512,d_h^R=64$→576；"equal to GQA with only 2.25 groups, but can achieve stronger performance than MHA"；93.3% 的对比对象 DeepSeek 67B 确为 GQA（8 KV 头）。
- 投影形状与 Shazeer einsum 记法一致（MHA $P^Q,P^K\in\mathbb{R}^{h\times d\times d_k}$、$P^V,P^O\in\mathbb{R}^{h\times d\times d_v}$；MQA $P^K\in\mathbb{R}^{d\times d_k}$、$P^V\in\mathbb{R}^{d\times d_v}$）。
- 结构：h2 编号 1–5 连续、来源 h2 不编号；来源小节命名合规；每章「本章问题」、页面「核心问题」均有解答折叠块，答案指明完整论证章节；summary 前缀仅用「补充：」「解答：」；无「（待生成）」占位，无指向已移除 research/ 的路径；无 Unicode 数学字符违规（× → − 为 validate.py 明示排除的普通排版字符）；图示为 HTML 表格，无等宽框线图。
- 链接有效：../../index.html、overview.html、../standard-attention/index.html、../mla/index.html 均存在；overview 与 index 互链；`dojo:topics=注意力机制`、`dojo:tag=注意力` 均在词表内。
- `.dojo/scripts/validate.py wiki/mqa-gqa/index.html` → validation ok。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 7
- 处置：修复（阻断与重要问题须关闭后复验；轻微问题逐条处理或给出接受理由）

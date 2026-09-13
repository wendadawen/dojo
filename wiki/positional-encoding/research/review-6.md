<!-- review-meta
round: 6
page: wiki/positional-encoding/index.html
reviewed_content_sha256: e16d5c4c3fa59090
-->
# 位置编码基础审查记录（第 6 轮）

- 页面版本：e336f6f01a588fb55f59ab99aceec0181fafd6eb
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- dojo:type：concept
- 已完整阅读章节（按顺序）：head（description / dojo:summary / title）→ 引言 → 常见误解 → 核心问题（5 题含解答折叠块）→ 1. 为什么 Transformer 需要位置编码——注意力的排列等变性（含本章问题）→ 2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算（含「展开：d=4 手算」「补充：线性性质的和角公式推导」两折叠块与本章问题）→ 3. 可学习绝对位置编码（含「展开：Table 3 row (e)」折叠块与本章问题）→ 4. 相对位置编码——T5 的 bias 机制（含对比图、折叠块与本章问题）→ 5. 四类方案对比与 NoPE 选择（含 5.1 与本章问题）→ 来源与范围说明（C/F/N、构造示例、类比边界、简化条件及其限制）→ 全部脚本段
- 来源核对方式：Vaswani 2017（arXiv:1706.03762v7 全文 HTML）、T5（arXiv:1910.10683v3 全文 HTML）、RoPE（arXiv:2104.09864）、ALiBi（arXiv:2108.12409）、Kimi K3 报告（arXiv:2607.24653v2 全文 HTML）；本地 wiki 链接逐条存在性核对；d=4 手算与和角公式在本机复算。

## 问题

- [重要·技术] 全页「四类方案/四类」的口径自相矛盾（meta description 第 6 行、h1 第 58 行、核心问题 5 问句第 99 行、§5 章标题第 436 行、第 494 行 为 A 口径；引言第 65 行、核心问题 5 解答第 102 行、§1 第 142 行 为 B 口径）｜引文依据：meta「四类方案：绝对正弦、可学习绝对、相对 bias（T5）、旋转（RoPE）。K3 选 NoPE。」；第 65 行「按位置信号加在哪里可分成四类——…乘在 query/key 上的旋转（RoPE），以及完全不施加的 NoPE。这四类方案在注入点上分别是输入嵌入、注意力分数、query/key 旋转与不施加」；第 102 行「四类方案按位置信号注入点区分：绝对正弦/可学习加在输入嵌入上、T5 bias…、RoPE…、NoPE 不施加」｜问题：同一术语「四类」在页面内指代两个不同集合——A 口径：绝对正弦、可学习绝对、相对 bias、RoPE（NoPE 单列，共 5 项）；B 口径：绝对（正弦或可学习）、相对、RoPE、NoPE（NoPE 计入四类，正弦与可学习合并为一类）。最直接的矛盾在同一问题块内：第 99 行问句写「对比四类方案与 NoPE」（=5 项），其解答第 102 行却把 NoPE 计入四类（=4 项）。h1 又写「补充位置信号的四类方案」，而 NoPE「不施加/不补充」，与第 65 行把 NoPE 计入四类的写法冲突｜修复要求：把 meta、h1、正文与两级问题块统一到同一口径（例如全页统一为「绝对方案（正弦/可学习）、相对偏置、RoPE、NoPE 四类」，或统一为「四类显式编码 + NoPE」），并修正第 99/102 行问与答的口径一致｜修复：｜复验：
- [轻微·技术] C9 来源混装（来源与范围说明「论断与来源（C）」C9，第 505 行；正文第 494 行 [C9] 引用处）｜引文依据：C9「（RoPE 绝对构造相对效果；NoPE 依赖因果掩码）：Su et al. 2021, arXiv:2104.09864 §3.1；见 RoPE、NoPE。」；Su et al. §3.1 原文仅述旋转构造与相对效果（"the proposed RoPE encodes the absolute position with a rotation matrix"、内积只依赖相对位置）｜问题：C9 标题捆绑两条独立论断，却只给 Su et al. §3.1 一个来源，该来源只支撑「RoPE 绝对构造相对效果」；「NoPE 依赖因果掩码」出自 NoPE 论文（Kazemnejad et al. 2023, arXiv:2305.19466），本页 C9 未列此来源，仅以链接指向 NoPE 页｜修复要求：把「NoPE 依赖因果掩码」拆为单独 C 条目并标注其来源（NoPE 论文）｜修复：｜复验：
- [轻微·技术] 无来源支持的判断写成结论（§5 末段第 465 行；同类表述另见第 494 行末句）｜引文依据：不适用｜问题：第 465 行「这三类约束不存在同时满足它们的单一方案」以确定性结论写出，无来源支撑，且所列三条并非互相排斥的约束（decoder-only 架构 + RoPE + YaRN 可同时满足「架构」与「外推」两项），该断言与同页第 5 章表格中「RoPE 需 YaRN/PI 扩展」「decoder-only 适用 NoPE/RoPE」的表述关系不清｜修复要求：把该断言降级为明确标注的取舍总结（如「各方案在三个维度上各有取舍，需按架构与位置来源选择」），或删去「不存在同时满足它们的单一方案」｜修复：｜复验：
- [轻微·格式] 正文引用其他章节未使用章节标题（引言后段第 117 行）｜引文依据：第 117 行「RoPE（旋转位置编码，见 RoPE）与 NoPE（无位置编码，见 NoPE）在后文对比章节展开。」；style-guide §1「正文引用其他章节时使用章节标题」，同页第 119 行已用「"绝对正弦位置编码"一章」的正确写法｜问题：用模糊的「后文对比章节」而非实际章节标题，且与同页另一处的正确写法不一致｜修复要求：把「后文对比章节」改为对应章节标题「四类方案对比与 NoPE 选择」｜修复：｜复验：

## 已核对但未发现问题（备查）

- Vaswani 2017 Table 3 row (e)：正弦 base 25.8 / 可学习 base 25.7（newstest2013 dev），"nearly identical"；big 28.4 出自 Table 2 test 集 newstest2014——与正文第 301/302/306/517 行一致。波长 2π→10000·2π、$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数、"we hypothesized it would allow the model to easily learn to attend by relative positions" 均与 §3.5 原文一致。
- T5（Raffel 2019 §2.1）：bias 加在 softmax 之前；32 个桶、每头独立各层共享、max_distance 128 clamp、邻近距离每距离一桶+远距离对数桶——与第 359/363/364/365/403/415 行一致。
- Kimi K3 报告 §2.1.2/§3.4：「applies No Position Encoding (NoPE) to all MLA layers」「The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction」「avoids … retuning a RoPE frequency base or applying YaRN」「extrapolates directly to 1M-token contexts」——与第 457/459/461/487/506 行一致。Gated MLA 保留 RoPE 接口（rot 分量）的结构性说法与本仓库 wiki/mla 的官方源码核对结论一致。
- RoPE（Su 2021 §3.1）：按绝对位置旋转 q/k、内积只依赖相对位置、无学习参数——与第 453/479/505 行一致。ALiBi（Press 2021）：按距离成比例的线性负偏置、斜率为固定正数不学习、不分桶——与第 404 行一致。
- d=4 手算逐项复算：$\omega_0=1$、$\omega_1=0.01$；$PE_1\approx(0.8415,0.5403,0.0100,1.0000)$、$PE_2\approx(0.9093,-0.4161,0.0200,0.9998)$、$PE_3$ 行数字正确；高/低频维差值与「pos=1→100 差约 0.83」复算一致；旋转角度 $-\Delta\omega_i$ 与和角公式推导复算正确。core-questions 解答与正文数字一致。
- 结构项：pipeline 无占位符「（待生成）」；引用的 wiki/rope、wiki/nope、wiki/kimi-k3、wiki/mla、wiki/standard-attention 均存在；无 `<img>` 带内容 alt（仅 lightbox 空 alt），无 alt 内 `$...$`；`.dojo/scripts/validate.py` 返回 validation ok；dojo:topics「注意力机制」在 AGENTS.md 固定大类内；两级问题块命名与解答折叠块齐备，核心问题解答均指向完整论证章节；结构和弦图与全部本地 libs/CSS 类存在；未检出元话语（"本页将…""下面来看…""需要注意的是"）、会话指代（我/我们/你）、调试叙事或临场评价；"本页/本文"仅出现在「简化条件及其限制」范围说明中，属 style-guide §12 允许的自称用法。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复

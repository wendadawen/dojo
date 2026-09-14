<!-- review-meta
round: 10
page: wiki/positional-encoding/index.html
reviewed_content_sha256: 1554aa79e293959d
-->
# 位置编码基础审查记录（第 10 轮）

- 页面版本：506908d12c51499e3050f617f5216e147c737925
- 审查时间：2026-09-14 17:42
- 审查者：独立子代理（未参与写作，未读取本页 research/）
- 已完整阅读章节：标题 / blockquote.meta / 引言 / 核心问题（5 条含解答）/ 常见误解 / 「1. 为什么 Transformer 需要位置编码——注意力的排列等变性」/「2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算」/「3. 可学习绝对位置编码——把固定向量换成可学习参数」/「4. 相对位置编码——T5 的 bias 机制与"加在分数上"的本质区别」/「5. 四类方案对比与 NoPE 选择——为什么 K3 选 NoPE」（含 5.1）/「来源与范围说明」。全部 details 折叠块、图注与表格一并在内通读。
- 核对来源与版本：Vaswani et al. 2017, arXiv:1706.03762（ar5iv HTML 全文，§3.5 与 Table 3 row (E)、参考文献 [9]）；Raffel et al., T5，JMLR 第 21 卷条目 "(140):1−67, 2020"（jmlr.org/papers/v21），§2.1；Press et al. 2021, arXiv:2108.12409（ar5iv HTML）；Su et al. 2021, arXiv:2104.09864 v1（ar5iv HTML，§3 目录）；DeepSeek-V2, arXiv:2405.04434（ar5iv HTML，§2.1.3）；Kazemnejad et al. 2023, arXiv:2305.19466（§2/§8）；Kimi K3 技术报告 arXiv:2607.24653**v2**（arxiv.org/html/2607.24653v2，§2.1.2/§3.4）与官方 config.json（huggingface.co/moonshotai/Kimi-K3 `text_config`，`mla_use_nope: true`、`qk_rope_head_dim: 64`）。

## 问题

- [重要·技术] 第 5 章对比表「RoPE / 长度外推」单元格、第 5 章「选择位置方案的判断框架」段、5 章本章问题解答（第 1 题）：三处写作「RoPE 需 YaRN/PI 扩展」，但 YaRN 与 PI 在首次出现处未作解释，且全页来源列表（C/F/N 共 21 条）没有任何条目支撑「RoPE 需要 YaRN/PI 才能扩展长度」这一机制论断——这是把一个无来源的机制描述写成了结论。｜引文依据：Su et al. 2021（arXiv:2104.09864）§3.3/§3.4 只讨论 long-term decay 与 sequence-length flexibility，WebFetch 核对结论为「该文 does not mention position interpolation or YaRN」；本页 C9 只覆盖 §3.1/§3.2/§3.4 的旋转构造与相对效果，未覆盖外推手段。｜修复要求：二选一——补入 YaRN/PI 原文出处并在首次出现处用一句话解释这两个名称；或按 check.md §2.2 将三处降级/改写为有来源的表述，例如「RoPE 本身不提供长度外推，扩展上下文通常需配合位置插值一类方法（K3 §3.4 即因避免此类改动而选 NoPE）」。｜修复：采用前一种方案（补入出处 + 首次出现处解释）。新增来源条目 C14（Chen et al. 2023, arXiv:2306.15595 → PI；Peng et al. 2023, arXiv:2309.00071 → YaRN）；第 5 章总对比表 RoPE 行「长度外推」单元格改为「需 YaRN/PI 等扩展<sup>[C14]</sup>」；表后新增一句解释：PI（Position Interpolation，位置插值）把推理位置整体缩放回训练范围，YaRN 在此基础上按频率分组施加不同缩放（高频维度基本保持、低频维度插值），二者都是 RoPE 之上扩展上下文的独立工作；「判断框架」段与 5 章问题解答第 1 题两处同步补 <sup>[C14]</sup>，全页该论断三处写法一致。｜复验：PI 原文摘要（arXiv:2306.15595）以「RoPE 预训练模型超出训练长度线性外推会出现灾难性高注意力分数」为出发点提出位置插值；YaRN 全文（ar5iv 2309.00071）机制为「not to interpolate the higher frequency dimensions at all while always interpolating the lower frequency dimensions」，即按频率分组缩放，与页面新增解释一致。三处论断现有可定位来源；validate.py 返回 validation ok。

- [轻微·格式] 第 3 章「核心权衡」对比表与第 5 章总对比表的表格单元格：「约 $32 \times h$（桶数×头数，各层共享）」「桶数×头数（各层共享）」中的乘号以 Unicode 字符「×」直接出现在 `$...$` 之外，而同格另一处同一运算写作 `$32 \times h$`，一页内同义运算两种写法。｜引文依据：不适用（style-guide §11：数学运算符必须包在 `$...$`／`$$...$$` 内，要求覆盖表格单元格；同页同一写法需一致）。｜修复要求：两处「桶数×头数」改为「桶数 $\times$ 头数」。｜修复：第 3 章「核心权衡」表「可学习参数量」单元格与第 5 章总对比表「相对 bias（T5）」行「可学习参数」单元格两处均改为「桶数 $\times$ 头数」。｜复验：全文已无未包在 $...$ 内的「×」（第 457 行「桶数乘头数」为中文连词用法，非数学符号）；validate.py 返回 validation ok。

- [轻微·技术] blockquote.meta 主要依据与来源条目 C7：T5 记为「Raffel et al. 2019, …, JMLR」／「JMLR Vol 21」，年份与卷号不一致——JMLR 第 21 卷为 2020 年。｜引文依据：JMLR v21 目录页该文条目「(140):1−67, 2020」，全卷标注 2020。｜修复要求：统一为「Raffel et al. 2020, JMLR Vol 21」，或改为标注「arXiv:1910.10683（2019）」。｜修复：按前一种：blockquote.meta「主要依据」与来源条目 C7 均改为「Raffel et al. 2020, …, JMLR Vol 21」；正文其余 4 处（第 4 章 T5 引入处、式注 \text{Raffel §2.1}、C8、F4）同步由 2019 改为 2020，全页年份统一。｜复验：全页已无「Raffel … 2019」（grep 命中 0）；JMLR 第 21 卷为 2020 年，年份与卷号一致；validate.py 返回 validation ok。

- [轻微·技术] 「来源与范围说明」首段「可学习绝对把固定公式换成参数表<sup>[C6]</sup>」：上标 [C6] 指向的条目为「C6（可学习外推失败）：Vaswani 2017 §3.5；BERT、GPT-2 采用可学习」，不覆盖「用参数表代替固定公式」这一形式定义；该定义的实际来源条目为 F5（「可学习参数表：Vaswani 2017 §3.5 引用 [9] Gehring et al. 2017」）。上标与所引条目不对应。｜引文依据：C6 原文「可学习外推失败」；F5 原文「可学习参数表：Vaswani 2017 §3.5 引用 [9] Gehring et al. 2017」。｜修复要求：将该处 [C6] 改为 [F5]（或 [C5]）。｜修复：「来源与范围说明」首段「可学习绝对把固定公式换成参数表」的上标 [C6] 改为 [F5]。｜复验：F5 条目内容即「可学习参数表：Vaswani 2017 §3.5 引用 [9] Gehring et al. 2017」，与该形式定义对应；C6 仅余两处正确用法（第 112、319 行的「BERT、GPT-2 采用可学习」）；validate.py 返回 validation ok。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复

补充说明（本轮已核对通过、未列为问题的项）：正弦公式与 Vaswani §3.5 逐字一致；Table 3 row (E) 25.8/25.7、big Table 2 28.4、base $d_{model}=512$、参考文献 [9]=Gehring et al. 2017 arXiv:1705.03122 均与原文相符；$d_{model}=4$ 手算表（$pos=0\!\sim\!3$）、$\omega_0=1$/$\omega_1=0.01$、和角公式推导与 $2\sin1\cos1=\sin2$ 复算一致，旋转角 $-\Delta\omega_i$ 方向正确；T5 §2.1「标量偏置加到注意力 logit、32 桶、offset 128 后归入同桶、每头一套、各层共享」四条全部与原文相符；ALiBi「与距离成正比的静态非学习偏置、每头固定斜率、几何序列」与原文相符；K3 §2.1.2/§3.4「所有 MLA 层 NoPE、KDA 承担位置敏感近因感知混合、避免重调 RoPE 频率基或施加 YaRN、直接外推到 1M token」与原文相符，`mla_use_nope=true` 与 config.json 相符；NoPE「因果掩码破坏排列不变性、可无显式位置信息建模」与 Kazemnejad §2/§8 相符；C1–C13/F1–F5/N1–N3 正文上标与条目双向对应、无孤立编号；5 个站内链接（rope/nope/kimi-k3/mla/standard-attention）目录均真实存在；`.dojo/scripts/validate.py` 返回 validation ok；KaTeX 复核 `\text{…§…}` 与 `\text{（可学习）}` 可渲染（§ 仅触发 strict-mode 控制台 warn，输出保留该字符）。
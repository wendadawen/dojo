<!-- review-meta
round: 3
page: wiki/positional-encoding/index.html
reviewed_content_sha256: fedfe9431bf1a700
-->
# 位置编码基础审查记录（第 3 轮）

- 页面版本：70df6471265f9a45545f690e7961efaf915c49f3（`git hash-object wiki/positional-encoding/index.html`）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题（5 题）、引言、1. 为什么 Transformer 需要位置编码、2. 绝对正弦位置编码、3. 可学习绝对位置编码、4. 相对位置编码、5. 四类方案对比与 NoPE 选择、来源与范围说明；含全部 `<details>` 折叠块与图注
- 来源获取：Vaswani et al. 2017（arXiv:1706.03762，PDF 逐页与 ar5iv HTML 双向核对）、Raffel et al. 2019（arXiv:1910.10683v1 与 v4/JMLR 21）、Su et al. 2021（arXiv:2104.09864）、页面内链的 5 个概念页；`.dojo/scripts/validate.py` 返回 `validation ok`

## 问题

- [重要·技术] 正文 §4 与「来源与范围说明」C7/F4/meta：T5 相对位置偏置的引文位置写为「Raffel et al. 2019 §3.2」，该章节不含该机制，正确位置为 §2.1（Model）。｜引文依据：JMLR 版 §2.1 原文 "We use a simplified form of position embeddings where each 'embedding' is simply a scalar that is added to the corresponding logit used for computing the attention weights. For efficiency, we also share the position embedding parameters across all layers in our model, though within a given layer each attention head uses a different learned position embedding. … we use 32 embeddings for all of our models with ranges that increase in size logarithmically up to an offset of 128 beyond which we assign all relative positions to the same embedding."（arXiv v1 与 v4 两版均在 §2.1）；而 §3.2 标题为 "Architectures"，其 §3.2.1/§3.2.2 全文检索无 "position"/"relative" 字样（`grep 740-1020 行无命中`）。论断内容本身与来源一致，故非阻断，但按标注位置核对会「内容不符」。｜修复要求：把 meta、C7、F4 中的「§3.2」改为「§2.1」，并保留 T5 源码 `_relative_position_bucket` 的引用。｜修复：｜复验：

- [重要·技术] §1 第 3 段（`why-positional-encoding`，第 116 行）：「只是输出跟着挪了位置，而每个位置上的输出值不变」——后半句与排列等变的定义相反，也与本页另外三处表述矛盾。排列等变意味着输出随输入一同重排，固定索引上的输出值会改变（交换 token 1、2 后，索引 1 的输出由 $o_1$ 变为 $o_2$）。｜引文依据：不适用（自检）。同页矛盾证据：核心问题解答「仅输出随位置重排——即排列等变」（第 74 行）、本章问题解答「只是输出跟着挪了位置」（第 141 行）；另第 126/148 行用「对行重排不变」（invariant 措辞）与本章标题「排列等变性」（equivariant）混用。｜修复要求：将「而每个位置上的输出值不变」改为「每个位置的输出值随之一同重排（固定索引上的值会改变）」，并把第 126/148 行的「对行重排不变」统一为「对行重排等变（输出随重排一同置换）」。｜修复：｜复验：

- [重要·技术] §4 正文与对比表（第 401、412、434 行）：「相对 bias 与 FlashAttention 不兼容（需物化 $n\times n$ 偏置矩阵）」被写成无条件结论，无 `<sup>[Cx]</sup>` 引文，且所给理由是实现相关而非方法固有——偏置可在核内按 tile 计算，不必落到 HBM；官方 FA2 的 `alibi_slopes` 与 `score_mod` 接口、以及 FAT5 等实现均支持核内加性偏置。｜引文依据：不适用（页面未给来源）。核对参考：Dao-AILab/flash-attention issue #332（T5 `attention_bias` 属"未支持/待支持"）；FAT5/FlashT5 的 `fa2_rpe`/`fa2_bias` 与 Flash-Attention-with-Bias-Triton 均在核内加 bias。｜修复要求：限定为「原版 FlashAttention 不直接支持逐头学习的相对距离偏置（需在核外物化 $n\times n$ 偏置或改造核内 bias）」，或补一条可定位来源；并在核心问题解答（第 102 行）同步该限定。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明」C4/C5：段落序号与来源不符。C4（线性性质）标注「§3.5 第 4 段」、C5（learned 对比）标注「§3.5 第 5 段」，二者实际同属 §3.5 第 3、4 段。｜引文依据：ar5iv §3.5 共 4 个 `<p>`：第 3 段含 "The wavelengths form a geometric progression from 2π to 10000·2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos."；第 4 段为 "We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E))."（C3/N2 标「第 3 段」是对的）。｜修复要求：C4 改为「§3.5 第 3 段」、C5 改为「§3.5 第 4 段」。｜修复：｜复验：

- [轻微·技术] C6 与 C8 引用的「iclr-blogposts 2025 综述」不可定位：无标题、作者、URL 或编号，无法按 check.md 第 2.2 节打开核对。｜引文依据：不适用（页面仅有该名称，无链接；全页 `grep -oE 'https?://'` 无任何外部 URL）。｜修复要求：删除该来源（C6 主论断已由 Vaswani 2017 §3.5 + BERT/GPT-2 原论文支撑，C8 已由 Vaswani §3.5 与 Raffel §2.1 支撑），或补出可访问的完整出处。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」中 F1、N3 有定义但正文从不引用（正文上标仅出现 C1–C10、F2–F5、N1、N2），违反 style-guide §6「与来源章节双向对应」。｜引文依据：不适用（`grep -oE '\[(C|F|N)[0-9]+'` 统计：F1、N3 各 0 次）。｜修复要求：在公式（第 172 行）补 `<sup>[F1]</sup>`、在「Vaswani 2017 base 模型为 512」（第 178 行）补 `<sup>[N3]</sup>`；或合并删除与 C2 重复的 F1。｜修复：｜复验：

- [轻微·表述] 全页存在重复的固定过渡句式与临场评价：第 160/283/329/356 行「下一章看…」（4 次）、第 195 行「下面用构造示例把公式落到可复算程度」、第 110 行「这个问题会在下一章手算回答」。另有口语化收尾「没有银弹，只有权衡」在第 474、505 行各一次。｜引文依据：不适用。｜修复要求：过渡句改写为「前一节结论 → 下一节问题」的关系陈述（style-guide §8 禁固定句式），保留不超过一处；删除「没有银弹，只有权衡」，改为陈述权衡结论（如「三类约束不存在同时最优的方案」）。｜修复：｜复验：

- [轻微·表述] §5 本章问题解答（第 496 行）两处把「场景」当术语使用：「MLA 压缩潜变量是 RoPE 不适用的典型场景，双向编码器是 NoPE 不适用的典型场景」。｜引文依据：不适用。｜修复要求：改为描述适用条件的陈述，如「RoPE 不适用于对压缩潜变量施加旋转的场景」→「对压缩潜变量施加旋转会使 MLA 矩阵吸收失效，故 RoPE 不适用于此类结构」。｜修复：｜复验：

- [轻微·格式] callout 组件与颜色偏离 style-guide：`callout-purple` 用于「教学解释」（§3 规定紫色为「源码验证/深度分析」）；「构造示例」用 `callout-yellow`（§4 规定示例应标「计算示例/代码示例/构造数据」并按用途标记，§3 黄色为「注意事项/边界条件/易误解澄清」）；三处「常见误解」（第 254/325/460 行）用 `callout-yellow` 内嵌，未使用 §2 的 `misconceptions` 组件（红色左边框）。另标签收尾标点不一致（「教学解释</strong>。 」「构造示例</strong>。 」「常见误解</strong>：」）。｜引文依据：不适用。｜修复要求：按 style-guide §2/§3/§4 归位——教学类比改用不占语义色的容器或改为普通段落，示例统一改标「构造数据」，常见误解改用 `misconceptions` 组件；统一标签收尾标点。｜修复：｜复验：

- [轻微·技术] 数学符号全页不统一：位置变量在 §2 用 $pos$（$x_{pos}$、$PE_{pos}$），§3 起改用 $m$（$x_m$、$PE_m$），§4 正文用 $i,j$ 而图注与 §5 用 $m,n$（图中 $x_m\to W^Q\to q_m$、$b_{\text{bucket}(m-n)}$，正文 $\text{score}(i,j)=q_i\cdot k_j+b_{\text{bucket}(i-j)}$）。｜引文依据：不适用。｜修复要求：选定一套（建议正文统一 $i,j$ 表 query/key 位置、$pos$ 表绝对位置），图注与 §5 同步改写，全页保持一致。｜修复：｜复验：

- [轻微·技术] §4「补充：T5 分桶函数的方向性说明」（第 408 行）两处细节：① 方向标注与页内定义相反——页面已定义 $i$ 为 query 位置、$j$ 为 key 位置，故 $i-j>0$ 是「query 在后、key 在前」，但补充块把 $i-j>0$ 写成「query 在前、key 在后」；② 「邻近距离（如 $|i-j|\le 8$）每距离一桶」与 T5 源码不符，源码 `max_exact = (num_buckets//2)//2 = 8`，精确桶为 $|i-j|<8$（即 0–7）。｜引文依据：T5 源码 `_relative_position_bucket`：`num_buckets //= 2`（32→16，双向时）、`max_exact = num_buckets // 2`（=8）、`is_small = relative_position < max_exact`。｜修复要求：交换两处方向标注；「$\le 8$」改为「$<8$（0–7）」。｜修复：｜复验：

- [轻微·技术] 第 408 行补充块为 T5 分桶方向性给出的动机「原因是语言中'看前面'和'看后面'的依赖模式不同（自回归解码时 query 只能看前面的 key）」无来源支持，属推断写成机制归因。｜引文依据：Raffel §2.1 仅述 "we use 32 embeddings … ranges that increase in size logarithmically…"，未解释方向性动机；T5 源码以 `bidirectional` 标志切换，未陈述该理由。｜修复要求：删除该动机句，或降级为「一种可能的解释」并注明为本页推断。｜修复：｜复验：

- [轻微·技术] 第 409 行 ALiBi 描述（$-m_h\cdot|i-j|$、单调下降、外推性好）只给了「Press et al. 2021」姓名，无 C/F/N 上标、无可定位条目，属无引文的机制描述。｜引文依据：不适用（页面未给条目；简化条件「简化 3」亦声明不展开）。｜修复要求：为该段补一条可定位来源（Press et al. 2021, arXiv:2108.12409 的提出处），或把整段改为「本页不展开、见来源」的一句。｜修复：｜复验：

- [轻微·技术] 第 217 行「低频维区分远距离位置（$pos=1$ 和 $pos=100$ 的 dim 2 才差约 1）」与同页第 232 行的数值不一致：$\sin(1)-\sin(0.01)=0.8415-0.0100=0.8315$，应约为 0.83。｜引文依据：不适用（自算）。｜修复要求：把「才差约 1」改为「才差约 0.83」。｜修复：｜复验：

- [轻微·技术] 第 240 行把式 $\big(\begin{smallmatrix}\cos(k\omega_i)&\sin(k\omega_i)\\-\sin(k\omega_i)&\cos(k\omega_i)\end{smallmatrix}\big)$ 称为「旋转（角度 $k\omega_i$）」：按标准逆时针约定该矩阵对应 $-k\omega_i$，词语与矩阵未给约定。｜引文依据：不适用（自核；Vaswani §3.5 只述 "linear function"）。｜修复要求：给矩阵下标出旋转方向，或改为「是一个旋转（按逆时针约定对应角度 $-k\omega_i$）」。｜修复：｜复验：

## 核对通过的关键项（无问题）

- Vaswani 2017 §3.5 Eq.(3)(4) 正弦公式、$PE_{pos+k}$ 线性性质、波长 "$2\pi$ 到 $10000\cdot 2\pi$" 表述与来源一致（原文 "The wavelengths form a geometric progression from 2π to 10000 · 2π."）。
- Table 3 row (E)「positional embedding instead of sinusoids」base BLEU 25.7、base 行 25.8，与页面「25.8 对 25.7 BLEU，nearly identical」一致；经 pymupdf 坐标级核对，该表**无 big 列组**，row (E) 仅 base 值，页面「仅对 base 做 learned 消融、big 只用正弦」成立；big 28.4 出自 Table 2（test newstest2014），标注正确。
- $d_{model}=4$、$pos=0$–$3$ 手算表与 $PE_1$、$PE_2$ 逐步代入全部复算通过（$\omega_0=1$、$\omega_1=0.01$；$0.99955\to0.9996$ 等舍入均正确）；和角公式推导与 $k=1$ 数值验证（$2\sin1\cos1=\sin2$）通过。
- T5「每头独立、各层共享」「32 个桶」「超出 128 处 clamp 到同一桶」与 §2.1 原文逐句一致；score 形式 $q_i\cdot k_j+b_{\text{bucket}(i-j)}$ 对应原文 "added to the corresponding logit used for computing the attention weights"（softmax 之前）成立。
- K3 部分（`mla_use_nope=true`、KDA 递归门控与衰减承载位置、1M 直接外推、§2.1.2/§3.4）与所引 <a href="../../wiki/kimi-k3/index.html">Kimi K3</a>、<a href="../../wiki/mla/index.html">MLA</a> §3、<a href="../../wiki/nope/index.html">NoPE</a> 页一致；RoPE「绝对构造、相对效果」与 Su et al. 2021 摘要及 §3.1 Formulation（Eq.(11) 相对内积）一致。
- 页面链接 5 个概念页全部存在，无「（待生成）」占位；overview 与 index 双向互链；`dojo:type=concept`、`dojo:topics=注意力机制`（属词表）、`dojo:tag=位置编码` 有效；结构图为 HTML 结构（`dg-stack`/`dg-layer` 类在 `libs/dojo-concept.css` 中均有定义），无 `<text>` ASCII 近似；两级问题块命名、数量与「解答：」折叠块齐备且答案自足；`validate.py` 通过。

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 11
- 处置：修复（本轮无需返回规划；无阻断项，3 项重要问题按上述要求就地修正后即可进入收尾，轻微项除「iclr-blogposts 不可定位」必须删改外其余按条修复）

# DFlash 2

- 描述：DFlash 2 是在块扩散草稿器 DFlash 上增加路径选择器与两抽头动态卷积的第二代草稿器，一次前向并行猜出一整块 token 并从中串出一条连贯路径，解码无损、输出与目标模型一致。
- 摘要：块内每个位置保留 top-16 候选，轻量选择器给相邻候选对打分、一次并行算完，再沿分数贪心走出一条路径；骨干里插入两抽头动态卷积，让每个位置读到前一位的表示，压住块末端的准确率衰减。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 资料

- Inco AI，《DFlash 2: Keep Drafting Parallel》，2026-08-18，[链接](https://inco.ai/blog/dflash2/)
- Inco AI，Qwen3.8-27B-DFlash2 模型卡，[链接](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- Inco AI，Muse-Glimmer-30B-DFlash2 模型卡，[链接](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2)
- Inco AI，Qwen3.8-27B-DFlash2-GGUF 模型卡（GGUF，一种量化模型文件格式），[链接](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)

## 短页

### 它是第二代的块扩散草稿器

投机解码把生成拆成起草与验证两步：草稿器（drafter）先逐 token（词元，模型处理的最小文本单位）猜一串，目标模型（target）再验证这一串。块扩散草稿器把起草也变成一次前向——一次并行猜出整块位置上的 token。DFlash 2 就是这类草稿器的第二代，2026-08-18 发布，发布物是配套的草稿模型权重（模型参数）而非独立语言模型。它在块扩散上加了两个组件：在候选之间选一条路径（路径选择器），以及在草稿模型里插两抽头动态卷积。解码无损，贪心输出与目标模型完全一致。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L6；model-card/qwen3.8-27b-dflash2.md L23-L36；blog/inco-ai-dflash2-keep-drafting-parallel.txt L34-L49

### 逐位置独立挑不够

块扩散草稿器在每个位置各自取自己最可能的一个 token，位置之间不协调，块内可能挑出不连贯的一串。但证据显示正确答案大多已经在候选里：只取最可能的一个，首个位置对的占 85.4%（条件于前文全部正确）；若每个位置保留 16 个候选，正确答案落在其中的比例升到 99.5%。另一处余量是块越靠后正确率越低——连从候选里完美挑对，正确率也从 99.5% 掉到末位的 87.8%，这一处靠「挑」修不了。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L141-L143；blog/inco-ai-dflash2-keep-drafting-parallel.txt L149-L158；blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L237

### 选路径补卷积，并行照旧

针对两处余量，DFlash 2 各加一个组件。选路径：每个位置保留 top-16 候选，给每一对相邻候选打分（草稿器自己对后一个的打分，加上两个候选各自 256 维嵌入——把 token 映成向量的匹配，匹配由当前位置的上下文门（决定匹配的哪些维计入分数）加权），所有相邻对一次并行算完，再从最后已验证的 token 出发，沿分数走出一条连贯路径，串行只在这一步。补卷积：在草稿模型的每个注意力与 FFN（前馈网络）子层前后插入两抽头动态卷积，让每个位置读到前一位的表示、首位读到上一个已验证的 token，压住块末端的衰减。两个组件合起来让每次验证多出约 20% 的输出，只增加约 1.3% 的循环延迟（一次起草加验证的耗时）。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L45；blog/inco-ai-dflash2-keep-drafting-parallel.txt L194-L214；blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L323；blog/inco-ai-dflash2-keep-drafting-parallel.txt L347-L349；blog/inco-ai-dflash2-keep-drafting-parallel.txt L375-L378

## 长页

[投机解码](wiki/speculative-decoding)是常见推理软件栈的一部分：一个小模型（草稿器）先猜出一块 token（词元，模型处理的最小文本单位），目标模型再一次前向（把整段输入整体算一遍、得出输出）把这整块一起验证。猜得准，一次前向就换回多个 token；猜不准，这一块就作废。以往草稿器自己也一个 token 一个 token 地猜，起草这一步仍是串行的。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L34-L40

块扩散草稿器把起草也改成一次前向：草稿器一次并行猜出整块位置上每个位置的 token（因为整块同时预测，称为块扩散），再把这整块交给目标模型一次验证。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L35-L39；model-card/qwen3.8-27b-dflash2.md L27-L28

DFlash 2 保留块扩散的一次前向结构，加了两个组件：从草稿器每位置的候选里选一条连贯路径（路径选择器），以及在草稿模型的骨干（串起各层的 transformer 主体）里插入两抽头动态卷积。它让每次验证多产出约 20% 的输出、只增加约 1.3% 的循环延迟（一次起草加验证周期的耗时），且输出与目标模型一致，接受长度（每轮验证平均接受的 token 数）比第一代更长。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L48；blog/inco-ai-dflash2-keep-drafting-parallel.txt L375-L378；model-card/qwen3.8-27b-dflash2.md L23-L36；model-card/qwen3.8-27b-dflash2.md L83-L84

DFlash 2 于 2026-08-18 发布，发布形态是技术博客加两份草稿模型权重（模型参数）。它本身不是独立语言模型，只在投机解码服务里为目标模型起草。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L6；blog/inco-ai-dflash2-keep-drafting-parallel.txt L395-L398；model-card/qwen3.8-27b-dflash2.md L23-L25

### 两个剩余问题

块扩散草稿器在每个位置各自取自己最可能的 token，位置之间没有协调。这样留下两处余量，各自对应一个组件：一处是块内正确 token 其实已经在候选里，只是没被选中；另一处是块越靠后准确率越低。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L49；blog/inco-ai-dflash2-keep-drafting-parallel.txt L145-L152

#### 选择空间：答案已经在候选里

在一个五层草稿器上按块内位置逐位统计：只取每位置最可能的一个 token（top-1 正确率），首个位置是 85.4%；若每个位置保留前 16 个候选，正确答案落在其中的比例（top-16 正确率）是 99.5%。这两项指标都沿块下降，top-1 到最后一个位置降到 72.9%，top-16 降到 87.8%。每位置保留 top-16 时，多数正确答案已经在候选里，缺的只是把对的那个挑出来。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L149-L158

::: 逐位置 top-1 与 top-16 正确率（五层 Qwen3-4B 是一个目标模型、GSM8K 是小学数学题基准、条件于前文全部正确）
| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 接受长度 |
|---|---|---|---|---|---|---|---|---|
| top-1 正确率 | 85.4% | 80.3% | 79.4% | 78.3% | 77.5% | 75.9% | 72.9% | 4.27 |
| top-16 正确率 | 99.5% | 97.3% | 94.8% | 92.6% | 90.8% | 89.4% | 87.8% | 6.79 |
接受长度一列按行分：top-1 那一行的 4.27 是基线，top-16 那一行的 6.79 是假想总能挑对时的结果。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L154-L161
:::

假想一个总能从 top-16 里挑对的选择器，也就是 oracle（完美选择），接受长度能从 4.27 提到 6.79。这段差距是纯选择空间。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L160-L161

#### 后缀衰减：连 oracle 也在掉

连 oracle 也掉：完美选择下正确率仍沿块下降。这说明候选池本身在后段枯竭，任何选择器都修不了。DFlash 2 把这种现象叫后缀衰减，并归为骨干的问题。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L232-L237

一个疑点是容量：五层骨干可能太小，保不住跨块的依赖。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L239-L241

### 路径选择器

路径选择器把「逐位置独立挑」换成「在候选之间串一条路径」：每个位置保留 top-16 候选，给每一对相邻候选打分，再从最后已验证的 token 出发，沿分数走出一条路径——贪心每步取分数最高的后继，或从这些分数里采样。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L194-L214

#### 相邻对打分

选择器只用相邻候选对打分，依据是一条设计前提：一个候选合不合适，主要取决于紧邻它前一个 token，所以给相邻对打分就够了。它和另一条路线（串行修正头）不同：串行修正头是一个逐位置重算整个词表分布的修正模块（词表即全部 token 的集合）。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L145-L146；blog/inco-ai-dflash2-keep-drafting-parallel.txt L196-L198

对一对相邻候选 $(a,b)$（$a$ 是前一个位置的候选，$b$ 是当前候选），分数是两项相加：
$$S_t(a,b)=U_t(b)+\langle A(a)\odot H(h_t),\,B(b)\rangle$$
第一项 $U_t(b)$ 是草稿器自己对 $b$ 的打分。第二项衡量 $b$ 接在 $a$ 之后的连贯程度：$A$ 和 $B$ 各给一个 token 一个 256 维的嵌入（把 token 映成一个 256 维向量），其中 $A(a)$ 先与上下文门 $H(h_t)$ 逐维相乘，再与 $B(b)$ 求内积；$H(h_t)$ 由该位置的隐状态 $h_t$（该位置在草稿器里算出的那个向量）算出，用来决定匹配的哪些维计入分数。两个嵌入只有 256 维，远小于隐状态，所以第二项是在相邻候选间做一次低秩双线性匹配。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L198-L207

::: 手算一次选择器怎么走出一条路径
以下数字为自行构造，为方便手算把 top-16 缩到 top-2、块长缩到 3；位置按 1 起编号（各表的 0 起是另一套约定）。
1. 三个位置的单点评分 $U_t(b)$：位置 1 的候选取 1.0 与 0.5；位置 2 取 0.4 与 0.9；位置 3 取 0.6 与 0.3。
2. 从最后已验证的 token 进位置 1（只有这两个候选）：匹配项取 0.1 与 0.4。位置 1 的分数 $S_1=1.0+0.1=1.1$、$0.5+0.4=0.9$。
3. 位置 1 到位置 2 的匹配项取 0.5、0.1、0.2、0.7，四个分数 $S_2=0.4+0.5=0.9$、$0.9+0.1=1.0$、$0.4+0.2=0.6$、$0.9+0.7=1.6$。
4. 位置 2 到位置 3 的匹配项取 0.3、0.4、0.8、0.1，四个分数 $S_3=0.6+0.3=0.9$、$0.3+0.4=0.7$、$0.6+0.8=1.4$、$0.3+0.1=0.4$。
5. 贪心走路径：位置 1 取分数最高的候选（1.1）；位置 2 在它的后继里取分数最高的（1.0 胜过 0.9）；位置 3 再取分数最高的后继（1.4 胜过 0.4），得到一条路径。
6. 若把位置 2 到位置 3 的匹配项里 0.1 改到 1.2，对应分数升到 $0.3+1.2=1.5$，超过原来的 1.4，位置 3 就会改选那条更连贯的边——匹配项把连贯性显式算了进去。
依据：自行构造
:::

沿分数走路径时，采样用拒绝采样（按接受-拒绝修正分布的采样法）修正草稿分布，使最终输出与目标模型的分布一致。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L213-L214

#### 为什么便宜

打分是全并行的：所有位置的所有相邻对一次算完，不需要额外的骨干或 LM head（把隐状态映成词表上打分的输出层）前向。唯一串行的工作是最后沿预计算的分数走路径，代价随块长线性。打分又只在 256 维嵌入上做，不碰词表大小的输出，这正是它比串行修正头开销小的地方——后者要逐位置重算整个词表的分布。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L144-L146；blog/inco-ai-dflash2-keep-drafting-parallel.txt L204-L214

::: 选择器单项的接受长度与开销（五层 Qwen3-4B、GSM8K；开销相对朴素 DFlash；M 为百万；$T$ 为采样温度）
| 方法 | 参数 | 延迟 | 接受长度（$T=0$） | 接受长度（$T=1$） |
|---|---|---|---|---|
| DFlash | — | — | 4.27 | 3.78 |
| 加串行修正头 | +77.8M | +9.6% | 4.49 | 4.08 |
| 加路径选择器 | +2.0M | +0.6% | 4.61 | 4.25 |
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L216-L221
:::

单独用选择器（不含卷积）时，接受长度从 4.27 提到 4.61（$T=0$）。选择器仍有上限：oracle 是 6.79。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L216-L227

### 两抽头动态卷积

后缀衰减出在草稿模型的骨干上，单靠选候选修不了，需要在骨干侧补一个局部模块。补法的算子是一个短卷积：只取两个抽头，一个看当前位置，一个看前一位（背景见[深度卷积](wiki/depthwise-conv)）。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L237；blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L308

#### 诊断：注意力的两份工作失衡

加深主要改善靠后的位置。三层、五层、十五层草稿器在首位几乎相同，沿块逐渐拉开。但加深是无差别的：把 5 层加到 15 层，多出来的十个注意力块（注意力机制见[标准注意力](wiki/standard-attention)）在每个位置都加容量，连早期那些可增益不多的位置也加，代价是参数变多、循环延迟也涨。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L239-L255

另一处线索在注意力。它有两份工作：读块之前的上下文，以及建块内的依赖。但它在第二份上花得越来越少——块内 token 分到的注意力份额从第 1 层的 30% 掉到第 5 层的 8%，剩余份额也越来越集中在少数几个注意力头（注意力里并行的一路子单元）上。按这个诊断，把块内那份工作交给一个专门的短程模块，注意力只管读上下文。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L257-L262

::: 层数与卷积的逐位置 top-1 正确率对照（Qwen3-4B、GSM8K、$T=0$、条件于前文全部正确；卷积模型不含选择器）
| 草稿器 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 3 层 | 85.21% | 79.26% | 77.18% | 75.75% | 73.96% | 70.4% | 64.97% |
| 5 层 | 85.39% | 80.31% | 79.39% | 78.27% | 77.39% | 76.03% | 72.86% |
| 15 层（3 倍参数） | 86.42% | 81.61% | 80.68% | 80.34% | 80.59% | 79.66% | 78.73% |
| 5 层加卷积（加 3% 参数） | 85.83% | 80.94% | 79.98% | 79.68% | 79.73% | 79.43% | 77.61% |
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L248-L255
:::

#### 卷积怎么做

在每个草稿器层的每个注意力子层与 FFN（前馈网络）子层（子层即层内的一个计算单元）的前后，各插入一个两抽头动态卷积。它的公式是
$$\operatorname{Conv}_{k}(x)_t=k_{t,0}\odot x_t+k_{t,1}\odot x_{t-1}$$
其中 $x_t$ 是当前位置的表示，$x_{t-1}$ 是前一位的表示（首位改读上一个已验证 token 的表示），$k$ 是这组卷积系数的记号，$k_{t,0}$ 与 $k_{t,1}$ 是随内容变化的两个动态系数，$\odot$ 是逐维相乘（每个特征通道各自相乘）。每个系数由一个可学的基核（训练中更新的基础卷积核）加上一小项修正组成，修正由当前隐状态算出，每 16 个特征通道共享一个修正。它块内局部、无状态，直接插进 DFlash，不改注意力、LM head 或验证流程。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L313-L321；blog/inco-ai-dflash2-keep-drafting-parallel.txt L340-L345

因为信息只跨一位，块内所有位置仍能并行计算，一边算一边把前一位的信息传过来。收益上，只加 16.5M 参数（3%）的五层加卷积版本接近 15 层版本：它减小了后缀衰减，末位 top-1 从 5 层基线的 72.86% 提到 77.61%，而 15 层是 78.73%；卷积只让循环延迟涨 0.7%，多出的十个 transformer 层却要涨 15.2%。第 4、5 层的块内注意力平均份额也从 5 层基线的 9.4% 降到加卷积后的 0.5%，与卷积接手了局部工作、注意力回到读上下文一致。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L248-L255；blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L323；blog/inco-ai-dflash2-keep-drafting-parallel.txt L347-L355

### 组合效果

把选择器和卷积合起来（DFlash 2），在 Qwen3.5-4B（另一个目标模型）上，五项基准的每请求平均接受长度是 5.97，比第一代 DFlash 的 4.92 多 1.05 个 token（21%）。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L364-L378

在 MATH-500（一个数学题基准）上，DFlash 2 的条件接受率（在该位置前文都正确的前提下、该位置被接受的比例）一路稳在约 86% 到末位，各基线在块末位比它低 6 到 9 个百分点。另两个目标模型上方向一致：Qwen3.8-27B（块 8，官方草稿器是 MTP）与 Muse Glimmer（块 16，官方草稿器是官方 DFlash）的均值都比各自的官方草稿器与社区草稿器更高。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L380-L382；blog/inco-ai-dflash2-keep-drafting-parallel.txt L385；blog/inco-ai-dflash2-keep-drafting-parallel.txt L402-L430

::: 三个目标模型的逐基准接受长度与 Qwen3.5-4B 逐位置条件接受率
对照方法：MTP 是目标模型原生的多 token 预测路径，DSpark 是串行修正头路线的对照草稿器（Qwen3.8-27B、Muse 两表用的是社区版），DFlash 是第一代草稿器。五个基准：GSM8K 与 MATH-500 是数学题，HumanEval 与 MBPP 是代码题，MT-Bench 是对话。Qwen3.5-4B 的采样为 thinking 开启（输出长推理链）、温度 1.0、top-p 0.95、top-k 20（每步按累计概率 0.95 在前 20 个候选里采样）、presence penalty 1.5、无损拒绝采样。
Qwen3.5-4B（每请求平均接受长度）：
| 基准 | MTP | DFlash | DSpark | DFlash 2 |
|---|---|---|---|---|
| GSM8K | 4.78 | 4.99 | 5.69 | 6.20 |
| MATH-500 | 5.04 | 5.42 | 6.20 | 6.76 |
| HumanEval | 4.84 | 5.43 | 5.80 | 6.28 |
| MBPP | 4.16 | 4.49 | 4.96 | 5.41 |
| MT-Bench | 3.90 | 4.26 | 4.77 | 5.20 |
| 均值 | 4.54 | 4.92 | 5.49 | 5.97 |
Qwen3.8-27B（块 8、默认采样）：
| 基准 | MTP | DSpark | DFlash 2 |
|---|---|---|---|
| GSM8K | 5.02 | 4.36 | 5.46 |
| MATH-500 | 4.72 | 3.92 | 5.28 |
| HumanEval | 3.91 | 3.30 | 4.39 |
| MBPP | 3.99 | 3.51 | 4.79 |
| MT-Bench | 3.74 | 3.01 | 4.10 |
| 均值 | 4.28 | 3.62 | 4.80 |
Muse Glimmer（块 16、默认采样）：
| 基准 | DFlash | DSpark | DFlash 2 |
|---|---|---|---|
| GSM8K | 5.43 | 5.45 | 6.57 |
| MATH-500 | 5.39 | 5.01 | 6.56 |
| HumanEval | 4.11 | 4.33 | 5.66 |
| MBPP | 3.74 | 4.02 | 5.30 |
| MT-Bench | 3.52 | 3.59 | 4.42 |
| 均值 | 4.44 | 4.48 | 5.70 |
Qwen3.5-4B 在 MATH-500 上的逐位置条件接受率（位置 0 到 14）：
| 方法 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MTP | 84.57% | 80.23% | 79% | 78.42% | 78.63% | 78.17% | 77.36% | 77.74% | 77.91% | 76.96% | 78.06% | 77.4% | 77.49% | 77.48% | 77.85% |
| DFlash | 88.35% | 77.7% | 77.8% | 79.45% | 80.3% | 81.12% | 81.22% | 81.07% | 81.29% | 80.28% | 80.64% | 80.29% | 79.56% | 78.77% | 77.48% |
| DSpark | 87.24% | 84.59% | 83.79% | 83.63% | 83.6% | 83.27% | 82.97% | 82.54% | 82.21% | 82.39% | 81.58% | 80.7% | 81.35% | 80.57% | 79.86% |
| DFlash 2 | 88.3% | 85.3% | 84.98% | 84.88% | 85.41% | 85.3% | 85.36% | 85.13% | 85.95% | 85.99% | 86.41% | 86.46% | 86.43% | 86.02% | 86.48% |
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L364-L373；blog/inco-ai-dflash2-keep-drafting-parallel.txt L399-L400；blog/inco-ai-dflash2-keep-drafting-parallel.txt L402-L409；blog/inco-ai-dflash2-keep-drafting-parallel.txt L410-L411；blog/inco-ai-dflash2-keep-drafting-parallel.txt L417-L426；blog/inco-ai-dflash2-keep-drafting-parallel.txt L385-L389
:::

### 适用条件与边界

收益取决于并发。单卡 H200（一种 GPU）上，Qwen3.8-27B 在并发 1 时吞吐是自回归解码（一次前向产一个 token 的朴素解码）的 2.67 到 3.43 倍；并发 8 降到 2.27 到 2.85 倍；并发 32 收窄到 1.01 到 1.45 倍，其中 MT-Bench 只有 1.01 倍。这些区间取各任务加速比的最小到最大。低并发、结构化任务（如 GSM8K、MATH-500）收益最大，高并发对话任务的收益趋近于零。
依据：model-card/qwen3.8-27b-dflash2.md L67-L127；blog/inco-ai-dflash2-keep-drafting-parallel.txt L431-L432

::: Qwen3.8-27B 与 Muse Glimmer 的并发×任务吞吐（每格为输出 tok/s，即每秒输出 token 数，与相对自回归的加速比）
Qwen3.8-27B，并发 1：
| 任务 | 自回归 | MTP | DSpark | DFlash 2 |
|---|---|---|---|---|
| GSM8K | 68.9 | 178.5 (2.59×) | 185.3 (2.69×) | 236.1 (3.43×) |
| MATH-500 | 69.0 | 172.8 (2.51×) | 174.5 (2.53×) | 230.7 (3.34×) |
| HumanEval | 69.0 | 151.9 (2.20×) | 159.9 (2.32×) | 214.6 (3.11×) |
| MBPP | 69.0 | 153.1 (2.22×) | 163.3 (2.37×) | 226.9 (3.29×) |
| MT-Bench | 68.9 | 134.9 (1.96×) | 137.6 (2.00×) | 184.0 (2.67×) |
Qwen3.8-27B，并发 8：
| 任务 | 自回归 | MTP | DSpark | DFlash 2 |
|---|---|---|---|---|
| GSM8K | 467.2 | 1,022.1 (2.19×) | 1,040.8 (2.23×) | 1,328.7 (2.84×) |
| MATH-500 | 480.0 | 1,023.5 (2.13×) | 1,025.8 (2.14×) | 1,368.3 (2.85×) |
| HumanEval | 483.4 | 934.2 (1.93×) | 956.5 (1.98×) | 1,291.5 (2.67×) |
| MBPP | 478.0 | 938.1 (1.96×) | 974.1 (2.04×) | 1,328.0 (2.78×) |
| MT-Bench | 480.5 | 835.2 (1.74×) | 802.3 (1.67×) | 1,090.2 (2.27×) |
Qwen3.8-27B，并发 32：
| 任务 | 自回归 | MTP | DSpark | DFlash 2 |
|---|---|---|---|---|
| GSM8K | 1,329.8 | 1,381.1 (1.04×) | 1,506.5 (1.13×) | 1,922.5 (1.45×) |
| MATH-500 | 1,505.8 | 1,415.6 (0.94×) | 1,429.0 (0.95×) | 1,951.8 (1.30×) |
| HumanEval | 1,546.5 | 1,296.8 (0.84×) | 1,330.1 (0.86×) | 1,799.0 (1.16×) |
| MBPP | 1,507.7 | 1,314.9 (0.87×) | 1,361.3 (0.90×) | 1,886.8 (1.25×) |
| MT-Bench | 1,507.4 | 1,159.7 (0.77×) | 1,115.5 (0.74×) | 1,525.3 (1.01×) |
Muse Glimmer，并发 1 / 8 / 32 的 DFlash 2 加速比依次为 3.08–4.62 倍 / 2.74–3.99 倍 / 1.15–1.68 倍（区间取各任务加速比的最小到最大）。
依据：model-card/qwen3.8-27b-dflash2.md L94-L127；model-card/muse-glimmer-30b-dflash2.md L98-L131
:::

使用上还有两条限制：每个目标模型要配套一个草稿器，DFlash 2 随 Qwen3.8-27B 与 Muse Glimmer 各发布一份；配套引擎里 vLLM 与 llama.cpp（两个推理引擎）的支持还在 PR 分支（尚未合并的代码分支），未进稳定版本。
依据：blog/inco-ai-dflash2-keep-drafting-parallel.txt L395-L398；blog/inco-ai-dflash2-keep-drafting-parallel.txt L54-L99

三点容易混的地方：其一，3.43 倍是并发 1、GSM8K 那一格，不是普适加速；其二，选择器只修「从候选里挑」，修不了后缀衰减——oracle 也会衰减，那是骨干的问题，靠卷积修；其三，两抽头卷积不是加深网络，它是插在每个子层前后的局部无状态算子，只覆盖「当前位置加前一位」的短程依赖。
依据：model-card/qwen3.8-27b-dflash2.md L101-L107；blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L237；blog/inco-ai-dflash2-keep-drafting-parallel.txt L344-L355

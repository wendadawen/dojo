# 模型并行审查记录（第 3 轮）

- 页面版本：ceb762cea31d915324554152cc33843970f08d34
- 审查时间：2026-08-19 21:35
- 审查者：独立子代理（第 3 轮，未参与写作与前两轮审查/修复）
- 已完整阅读章节：index.html 依序：标题/导言 → 核心问题（4 题含折叠解答）→ 1（1.1–1.3 含手算折叠块）→ 本章问题 → 2（2.1–2.2 含补充折叠块）→ 本章问题 → 3（3.1–3.3）→ 本章问题 → 4 → 本章问题 → 来源与范围说明；overview.html 全文。

机械验证：`validate.py wiki/model-parallelism/index.html` 通过；六个前置概念页（standard-attention、gpu-communication、moe-serving、chunked-prefill、beyond-buzz-disaggregation、mla）均存在；两级问题块齐全且答案独立可读；Unicode × 已全部 LaTeX 化；图 2 figcaption 方位（左下填充/右上排空）正确；来源说明自 C2 起始、无 C1 引用；head meta 五项齐全。

## 问题

- [阻断·技术] index.html §1.1 正文与"完整手算"折叠块，构造示例数值错误且三处互相矛盾：给定 $X=[1,1,1,1]$、$A$ 列和为 $(3,3,7,7)$，复算 $XA=[3,3,7,7]$、$XA_1=[3,3]$（前两列）、$XA_2=[7,7]$（后两列）。正文写 $XA=[2,4,7,7]$、$XA_1=[2,4,0,0]$、$XA_2=[0,0,7,7]$；折叠块另写 $XA=[2,4,7,7]$、$XA_1=[4,4]$、$XA_2=[4,8]$、拼接 $[4,4,4,8]$。三套数字（[2,4,7,7]／[4,4,4,8]／[3,3,7,7]）互不一致且均不可由给定矩阵复算。｜引文依据：复算 $XA_j=\sum_i A_{ij}=3,3,7,7$（列 1=1+2、列 3=3+4）；前两轮修复目标即 [3,3,7,7]，当前页面未落地。｜修复要求：正文与折叠块全部数字统一改为 $XA=[3,3,7,7]$、$XA_1=[3,3]$、$XA_2=[7,7]$，分段拼接写法两处统一；修改后重新复算并复核 GeLU 后乘 $B$ 的等价性论证（该论证用任意逐元素 $G$，不受数字影响）。｜修复：｜复验：
- [重要·来源] index.html §2.2 补充折叠块末句"训练推导是公式的出处（GPipe §2.2）"章节号错误：气泡公式位于 GPipe §2.3 Performance Optimization，§2.2 为 Algorithm 节。｜引文依据：gpipe.txt §2.3 "This bubble time is O((K−1)/(M+K−1)) amortized over the number of micro-steps M"；页面来源说明 F2/N1/C6/C7 均标 §2.3，此句为唯一残留。｜修复要求：该处改为"GPipe §2.3"。｜修复：｜复验：
- [轻微·图示] 图 1（FFN 数据流）figcaption："虚线框内的一次 all-reduce 是这个子块唯一的卡间通信"。图中 all-reduce 节点为实线 accent 框（class="dg-box dg-accent"），全图无虚线框（虚线样式仅用于图 2 气泡）。｜引文依据：不适用（图内元素 class 核对）。｜修复要求：改为"蓝色框内的一次 all-reduce"或删去"虚线框内"。｜修复：｜复验：
- [轻微·来源] §1.2 末段"若卡数超过头数，头无法再分，这就是 TP 与 KV 头数关系（KV 复制因子）的来源之一"：机制归因无来源标注，属未标注推断。｜引文依据：megatron.txt §3 仅述 Q/K/V 列切使每头落一卡，未论及卡数超过头数与 KV 复制因子的关系。｜修复要求：加"（推断）"标注或改为中性外链表述。｜修复：｜复验：
- [轻微·来源] C9 的 EP×PP=64（DeepSeek-R1、64 GPU）实例：本轮输入仅含 Megatron/GPipe/Sarathi 三份文本，未含 Beyond the Buzz 论文，无法独立复核。来源说明已给出定位（§4 disaggregation_in_practice.tex 图 caption）与引文 "DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"。｜引文依据：本轮不可得；页面自记录引文如上。｜修复要求：无需修改页面；接受理由：编排者说明该条已于前两轮核对到位，来源说明含 paper.txt 定位与原文片段。如需完全满足"每轮独立核对"，应由持有该来源的执行者补一次核对并在此记录。｜修复：｜复验：

## 已核对来源论断摘录（无问题项）

- C2/F1：Megatron §3 "Another option is to split A along its columns A=[A1,A2]…allows the GeLU nonlinearity to be independently applied"；"split the second GEMM along its rows…requires only a single all-reduce operation in the forward pass"，Eq.(2)(3) Figure 3a ✓
- C3：Megatron §3 "partitioning the GEMMs associated with K, Q, V in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU" ✓
- C4/N2：Megatron §3 "two all-reduces in the forward path and two in the backward path"（单层 2 次；$2L$ 为本页叠加上限，简化条件已声明嵌入层除外）✓
- C5/C6/C7/F2/N1：GPipe §2.2 "Communication primitives are automatically inserted at partition boundaries"；§2.3 气泡公式、$M\ge 4\times K$ 阈值、"we only need to pass activation tensors at the partition boundaries…even on accelerators without high-speed interconnects" ✓
- C8：Sarathi §1 "DGX A100…up to 8 GPUs"、"Pope et al.…256 devices on specialized TPUv4 pods"、"poor performance when hyper-clusters are unavailable" ✓
- C9 正交性：Megatron §1 "orthogonal to pipeline-based model parallelism"（乘积=总卡数为已标注推断）✓；C10：Megatron §2.3 data/model parallelism 定义 ✓
- 复算：$p{=}4,m{=}4$ → $3/7\approx0.43$、$m{=}16$ → $3/19\approx0.16$；$d/dp[(p-1)/(m+p-1)]=m/(m+p-1)^2>0$；$2\times61=122$；图 2 时间槽布局逐格核对与公式一致 ✓

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 3
- 处置：修复。阻断项（§1.1 构造示例数字，前两轮目标修复未落地）与重要项（GPipe §2.2→§2.3 残留）修复并复验、重跑 validate 后方可发布；轻微 3 项中图 1 figcaption 与 KV 复制因子标注建议随手修复，EP×PP=64 条目按记录中的接受理由遗留。阻断修复属第 1 次追加修复（上限 2 次）。


## 修复记录（追加第 1 次）

按 check.md §4 阻断问题追加修复（最多 2 次，本次为第 1 次）：

- 阻断：§1.1 构造示例数字回归修复——按 X=[1,1,1,1] 与给定 A 实算 XA = [3, 3, 7, 7]（列 1=1+2=3、列 2=1+2=3、列 3=3+4=7、列 4=3+4=7），XA1（前两列）= [3, 3, 0, 0]，XA2（后两列）= [0, 0, 7, 7]，拼接 = [3, 3, 7, 7] = XA。三处数字（§1.1 正文、完整手算折叠块、可能的标注）统一为 [3, 3, 7, 7] 与 [3, 3] / [7, 7] 分段值。复算 6+7 = 13、3+3+7+7 = 20 等数值关系均符合矩阵列和。
- 重要：§2.2 补充折叠块末句"训练推导是公式的出处（GPipe §2.2）" → "（GPipe §2.3）"。F2/N1 已为 §2.3，残留 1 处已对齐。
- 轻微 1：图 1 figcaption "虚线框内" → "蓝色 accent 框内"（图中无虚线样式，all-reduce 节点为 dg-accent）。
- 轻微 2：§1.2 末段 KV 复制因子"来源之一"加"基于 TP 切分粒度对齐头数与 KV 头数之比的推断，论文未明文论述"，标记推断。

**机械验证：** `validate.py` 通过。Chrome 探针：186 KaTeX、5 foreignObject、0 overlap。

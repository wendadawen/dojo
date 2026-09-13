<!-- review-meta
round: 5
page: wiki/model-parallelism/index.html
reviewed_content_sha256: a7e634be664ffaba
-->
# 模型并行审查记录（第 5 轮）

- 页面版本：3f6746066f4510d464c134d862f449e26d61d296
- 审查时间：2026-09-13 20:20
- 审查者：独立子代理
- 适用规范：guides/concept/check.md（dojo:type=concept），对照 guides/concept/style-guide.md
- 已完整阅读章节：核心问题（4 条）→ 1. 一层怎么切：张量并行（1.1 FFN、1.2 Attention、1.3 一层两次通信、本章问题）→ 2. 整摞层怎么分：流水线并行与气泡（2.1 推导、2.2 摊薄与推理的约束、本章问题）→ 3. 两种切法的通信代价（3.1 通信结构对比、3.2 组合部署、3.3 与数据并行/专家并行的边界、本章问题）→ 4. TP 与 PP 的取舍（本章问题）→ 来源与范围说明（C/F/N/构造示例/辅助解释/简化条件），含全部折叠块与三处图注。
- 来源核对方式：WebFetch 抓 arXiv 原文（1909.08053 Megatron-LM、1811.06965 GPipe、2308.16369 SARATHI、2506.05508 Beyond the Buzz），PDF 转文本后逐条定位原文片段。

## 核对结论（逐条来源）

- Megatron-LM §3 原文：`[Y1 , Y2 ] = [GeLU(XA1 ), GeLU(XA2 )]`（Eq.3）、`only a single all-reduce operation in the forward pass (g operator) and a single all-reduce in the backward pass (f operator)`、attention `partitioning the GEMMs associated with key (K), query (Q), and value (V ) in a column parallel fashion`、`using only two all-reduces in the forward path and two in the backward path` → 支撑 C2/C3/C4/F1/N2（每层前向 2 次 all-reduce）。✓
- GPipe §2.3 原文：`This bubble time is O((K-1)/(M+K-1)) ... negligible when M ≥ 4 × K`、`we only need to pass activation tensors at the partition boundaries ... even on accelerators without high-speed interconnects`、`Figure 2c assumes partitions are evenly balanced` → 支撑 C5/C6/C7/F2/N1 与「slot 等长/负载均衡」简化条件。✓
- SARATHI §1 原文：`In servers with high bandwidth connectivity such as NVIDIA DGX A100, tensor-parallelism ... on up to 8 GPUs`、`Pope et al. [39] show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods`、`can result in poor performance when hyper-clusters are unavailable` → 支撑 C8。✓
- Beyond the Buzz §4 图 5 caption 原文：`Prefill performance is shown for DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)`，正文 `expert parallelism within the NVLink domain is consistently preferred`、`benefits of disaggregation are most pronounced for prefill-heavy workloads`、`benefits of disaggregated inferencing become more pronounced with larger models` → 支撑 C9 的 EP×PP=64 实例与第 4 章的 Beyond the Buzz 描述。✓
- 手算复算：XA=[3,3,7,7]，XA1=[3,3]、XA2=[7,7]；G(XA1)B1+G(XA2)B2 = [3g3,3g3,7g7,7g7] = G(XA)B（B=A）✓；2×61=122 ✓；p=4,m=4→3/7≈0.43、m=16→3/19≈0.16 ✓；d/dp[(p-1)/(m+p-1)] = m/(m+p-1)^2 > 0 ✓。SVG 时间槽（p=4,m=4）七列、每 stage 4 忙 3 空，与 GPipe 排程一致 ✓。
- 机械项：validate.py 返回 `validation ok`；正文无 research/ 文件路径引用；7 个引用前置页（standard-attention、gpu-communication、moe-serving、beyond-buzz-disaggregation、chunked-prefill、pp-load-balancing、mla）均真实存在；8 个本地 libs 资源存在；无 Unicode 数学字符直写；C/F/N 上标双向对应。

## 问题

- [重要·技术] 第 351 行第三张图（TP 4 × PP 2 组合部署）的图注与图布局自相矛盾：图内 `.dg-stack` 按「节点 1」在前（上）、「节点 2」在后（下）自上而下排列，图注却写「自下而上是数据流方向：请求先流经节点 1 的 stage，再进入节点 2」——节点 1 在顶部、请求先经节点 1，方向应为自上而下。｜引文依据：`libs/dojo-concept.css:626` `.dg-stack { display: flex; flex-direction: column; }`（同处注释「纵向堆叠，从上到下表示层级或阶段」）+ 页面第 342–349 行 `<div class="dg-layer">` 节点 1 先于节点 2 + 图注「自下而上是数据流方向：请求先流经节点 1 的 stage，再进入节点 2」。｜修复要求：使方向描述与图层序一致——将「自下而上」改为「自上而下」，或调换两 `.dg-layer` 的顺序使节点 1 在下；二选一后图注与图必须同向。｜修复：｜复验：
- [轻微·表述] 第 201 行：「这里出现一个问题：模型有 $p$ 个 stage……卡与卡之间怎么协作？答案是……」为叙事化设问式元话语（先抛问题再自答），非必要引导语。｜引文依据：页面第 201 行原文。｜修复要求：改为直接陈述（如「$p$ 个 stage 协作的方式是一起跑同一个 step：第 1 个 micro-batch 逐级前进……」），去掉「这里出现一个问题」「答案是」。｜修复：｜复验：
- [轻微·表述] 第 166 行：「值得强调的是切分粒度的对齐」中的「值得强调的是」为强调型元话语（与 check.md 例举的「需要注意的是」同类）。｜引文依据：页面第 166 行原文。｜修复要求：删除该引导语，直接从「切分粒度的对齐：切在头的边界上……」起句。｜修复：｜复验：
- [轻微·表述] 第 159 行折叠块：「注意这个论证对任何逐元素函数 $G$ 都成立」中的「注意」为元话语。｜引文依据：页面第 159 行原文。｜修复要求：去掉「注意」，改为直陈（如「这个论证对任何逐元素函数 $G$ 都成立」）。｜修复：｜复验：
- [轻微·表述] 第 287 行折叠块：「只要求读者知道训练推导是公式的出处」以「读者」为陈述对象，属对读者的元话语（style-guide §12 要求不对读者作人称称呼）。｜引文依据：页面第 287 行原文「本页聚焦推理，只要求读者知道训练推导是公式的出处（GPipe §2.3）」。｜修复要求：改为机制表述，如「训练场景的推导是公式出处（GPipe §2.3），本页不重复」。｜修复：｜复验：
- [轻微·格式] 第 70 行同段内「本页」「本文」混用（style-guide §12 要求自称统一为其中之一）。｜引文依据：页面第 70 行「本页取广义用法，把两者都算作模型并行。本文讨论的是推理 serving 场景」。｜修复要求：统一为「本页」或「本文」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复（1 条重要 + 5 条轻微逐条修复后，本页可发布；来源论断均已回源核对，无阻断项）
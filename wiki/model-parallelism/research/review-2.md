# 模型并行（Model Parallelism）审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 66dda204db08483cf49ae12a7802cb47962d0b0b
- 审查时间：2026-08-19 21:25
- 审查者：独立子代理（第 2 轮，未参与写作与第 1 轮）
- 已完整阅读章节：index.html 全文（核心问题 → 第 1 章 1.1/1.2/1.3 → 第 2 章 2.1/2.2 → 第 3 章 3.1/3.2/3.3 → 第 4 章 → 来源与范围说明，含全部折叠块）；overview.html 全文。
- 来源核对：Megatron-LM（/tmp/beyond-buzz-research/megatron.txt）、GPipe（gpipe.txt）、Sarathi（sarathi.txt）逐条定位核对；Beyond the Buzz 文本未在本次输入中提供，无法独立核对（见问题 6）。

## 逐条核对结果

- C2 ✓：Megatron §3 Eq.(2) 行切 "Y = GeLU(X1A1+X2A2) … will require a synchronization point before the GeLU function"；Eq.(3) 列切 "[Y1,Y2]=[GeLU(XA1),GeLU(XA2)]"。页面一致。
- C3 ✓：§3(b) "partitioning the GEMMs associated with key (K), query (Q), and value (V) in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU… parallelized along its rows"。
- C4 ✓：§3 "only two all-reduces in the forward path and two in the backward path (see Figure 4)"。
- C5 ✓：GPipe §2.1/§2.2 "consecutive groups of layers can be partitioned into cells… Communication primitives are automatically inserted at partition boundaries… split a mini-batch… into smaller micro-batches"。
- C6/F2 ✓：GPipe §2.3 "This bubble time is O((K-1)/(M+K-1))"；3/7≈0.43、3/19≈0.16、m=1→(p-1)/p、d/dp 导数 m/(m+p-1)²>0 均复算正确；时间槽图（p=4,m=4）逐格核对与 7 slot 结构一致。
- C7 ✓：GPipe §2.3 "we can achieve efficient scaling performance even on accelerators without high-speed interconnects"。
- C8 ✓：Sarathi §1 "tensor-parallelism can enable deployment of an LLM on up to 8 GPUs [DGX A100]… Pope et al… up to 256 devices on specialized TPUv4 pods. However… poor performance when hyper-clusters are unavailable"。
- C9 ✓（正交性）：Megatron §1 "This approach is orthogonal to pipeline-based model parallelism"；组合正交性页面已标"为推断"。
- C10 ✓：Megatron §2.3 "data parallelism where a training minibatch is split across multiple workers, and model parallelism in which the memory usage and computation of a model is distributed across multiple workers"。
- F1 ✓：Megatron §3 Figure 3a 及正文 "split the second GEMM along its rows… a single all-reduce operation in the forward pass"。
- N1 ✓：GPipe §2.3 "negligible when M ≥ 4 × K"；§3 实验重现。训练实验条件已在页面标注。
- N2 ✓：Megatron §3；61 层×2=122 复算正确。
- 修复项 2（Megatron §3）✓；修复项 4（EP×PP=64 来源）形式到位（来源节引图 caption 原文，见问题 6）；修复项 5（取舍表降级）✓ 已标"分析性判断…非 Sarathi 原文直接论断"；修复项 7（句不通）✓ 通读无残留病句。
- 链接与资源：6 个概念页链接、overview 互链、KaTeX/Prism 本地库均存在。head 含 description、dojo:summary/type/topics/tag。

## 问题

- [重要·技术] index.html §1.1 正文及"完整手算"折叠块：构造示例数字不可复算且两处互相矛盾。X=[1,1,1,1]、A 对角块如页面所设，XA 应为 [3,3,7,7]（各列求和 1+2、1+2、3+4、3+4），正文（第 838 行附近）写 "XA=[2,4,7,7]、XA_1=[2,4,0,0]、XA_2=[0,0,7,7]"；折叠块（第 842 行附近）另写 "XA_1=[4,4]、XA_2=[4,8]，拼接 [4,4,4,8]"。正确值：XA_1=[3,3]、XA_2=[7,7]。第 1 轮修复项 1 未到位。｜引文依据：复算 X·A_{:,j}=Σ_i A_{ij}：col1=1+2=3、col2=3、col3=3+4=7、col4=7｜修复要求：正文与折叠块统一改为 [3,3,7,7]（分段 [3,3]、[7,7]），并保证两处数字一致、可手算复现｜修复：｜复验：
- [重要·技术] index.html §2.2 折叠块（第 972 行）：气泡公式出处写 "GPipe §2.2"，实际公式位于 GPipe §2.3 Performance Optimization（"As illustrated in Figure 2c… This bubble time is O((K-1)/(M+K-1))"）；§2.2 Algorithm 仅含 "This sequence of operations is illustrated in Figure 2c"，无公式。与来源节 F2 条目（§2.3）自相矛盾。｜引文依据：gpipe.txt §2.3 段 "This bubble time is O( (K-1)/(M+K-1) ) amortized over the number of micro-steps M"｜修复要求：第 972 行 "GPipe §2.2" 改为 "GPipe §2.3"｜修复：｜复验：
- [轻微·格式] index.html 第 1023 行 "总卡数 = TP 维度 × PP 维度"、overview.html 第 55 行 "总卡数 = TP × PP，…EP × PP = 64"：正文残留 Unicode ×，第 1 轮修复项 6 未彻底。｜引文依据：不适用（check.md §2.2 第 9 条要求无 Unicode 数学字符）｜修复要求：3 处 × 改为 $\times$（overview 已加载 KaTeX）｜修复：｜复验：
- [轻微·格式] index.html 来源与范围说明：条目列 "C1–C4…：Megatron-LM…"，但正文全篇无任何 [C1] 引用点。｜引文依据：grep '\[C1\]' index.html 零命中｜修复要求：删除来源节中的 C1 或恢复正文 C1 引用点，使编号一一对应｜修复：｜复验：
- [轻微·可读性] index.html 图 2 figcaption：'右下三角是排空期'方位错误。排空期气泡位于 stage 1、2（图顶部行）的后期 slot，即右上三角；左下（填充期）表述正确。｜引文依据：不适用（按图内实线/虚线框位置直接核对）｜修复要求：'右下三角' 改为 '右上三角'｜修复：｜复验：
- [轻微·技术] index.html 来源节 C9 条目引 Beyond the Buzz（arXiv:2506.05508）§4 图 caption "DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"：该论文文本不在本轮提供的来源中，无法独立定位核对。｜引文依据：无（来源未提供）｜修复要求：发布前由可访问该来源的执行者复核引文与位置，或在条目上明确标注"待复核"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复。第 1 轮 7 项修复中 5 项到位，构造示例数字未修复到位且正文与折叠块矛盾，GPipe 章节号残留 1 处 §2.2。修复后需重跑数字复算与 `.dojo/scripts/validate.py`，再进入第 3 轮。


## 修复记录

第 2 轮所有问题已修复。重要/阻断问题逐条对应：

### model-parallelism
- 重要：构造示例数字修正——按 X=[1,1,1,1] 与给定 A 实算 XA=[3,3,7,7]、XA1=[3,3,0,0]、XA2=[0,0,7,7]（列 1+2=3、列 2=3、列 3+4=7、列 4=7）。两处（§1.1 正文 + 完整手算折叠块）统一改为 [3,3,7,7]。复算验证通过。
- 重要：§2.2 折叠块（972 行）"GPipe §2.2 在 F2 给出气泡占比" → "GPipe §2.3 在 F2 给出气泡占比"。复算：GPipe §2.3 原文 "This bubble time is O((K-1)/(M+K-1))" 与 F2 一致。
- 轻微：Unicode × 换 $\times$（"总卡数 = TP 维度 × PP 维度"、"TP × PP"、"EP × PP = 64"）；来源说明中 "C1–C4" 改为 "C2–C4"（C1 实际未在正文引用，对齐论断与正文标记）；图 2 figcaption "右下三角" → "右上三角"（排空期在 stage 1/2 后期 slot 即图右上角）。

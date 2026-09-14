<!-- review-meta
round: 7
page: wiki/model-parallelism/index.html
reviewed_content_sha256: 9fb714c84ca54b2e
-->
# 模型并行审查记录（第 7 轮）

- 页面版本：ecb19cad679472c64c3efe4ec7d661990d21739d
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题；1. 一层怎么切：张量并行（1.1 FFN：第一段列切，第二段行切／1.2 Attention：按头切，通信藏在投影里／1.3 一层两次通信／本章问题）；2. 整摞层怎么分：流水线并行与气泡（2.1 气泡占比 $(p-1)/(m+p-1)$ 的推导／2.2 摊薄与推理的约束／本章问题）；3. 两种切法的通信代价：TP 圈在机内，PP 跨机（3.1 通信结构对比／3.2 组合部署：TP $\times$ PP／3.3 与数据并行、专家并行的边界／本章问题）；4. TP 与 PP 的取舍（本章问题）；来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制）。全文含折叠块、图注、SVG 图内文字与 head 元数据；overview.html 一并对照。

## 来源核对（引文依据留档）

- Megatron-LM §3（arXiv:1909.08053）：原文确认第一段 GEMM 按列切、第二段按行切（Eq.(3)、Figure 3a），"This is advantageous as it removes a synchronization point"；行切优先方案 "will require a synchronization point before the GeLU function"；MLP 子块前向 "only a single all-reduce operation in the forward pass (g operator)"，整层 "all the GEMMs in a transformer layer can be done with only two all-reduces in the forward path and two in the backward path"。页面 [C2][C3][C4][F1][N2] 与之一致。
- Megatron-LM §2.3/§B.1：数据并行 "a training minibatch is split across multiple workers"、"all GPUs within a data parallel group hold the same model parameters"。页面 [C10] 背景表述一致；引用位置偏宽（"每张卡一份完整模型"实际落在 §B.1），不构成错误。
- GPipe §2.3（arXiv:1811.06965）：气泡 "This bubble time is O((K-1)/(M+K-1))"，页面 $p\leftrightarrow K$、$m\leftrightarrow M$ 后即为 $(p-1)/(m+p-1)$；阈值 "In our experiments, we found the bubble overhead to be negligible when M≥4×K"；通信 "we only need to pass activation tensors at the partition boundaries"、"we can achieve efficient scaling performance even on accelerators without high-speed interconnects"。页面 [C5][C6][C7][F2][N1] 与之一致，且均标注为训练实验/出处。
- Sarathi §1（arXiv:2308.16369）："tensor-parallelism can enable deployment of an LLM on up to 8 GPUs"（例举 NVIDIA DGX A100）、"Pope et al. show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods"、"can result in poor performance when hyper-clusters are unavailable"。页面 [C8] 与之一致。
- Beyond the Buzz §4（arXiv:2506.05508）：Fig.5 caption "DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"；abstract "disaggregation is most effective for prefill-heavy traffic patterns and larger models"；"we simulate the prefill and decode pools separately, allowing each to independently optimize"。作者 Mitra et al. 与 arXiv 号已核对。页面 [C9] 与之一致。
- 图内数值：FFN 数据流图与流水线时间槽图按像素测量核对——$p=4,m=4$ 时间槽图列数 7（slot 1–7，x 间距 78px）、每 stage 实心框 4 个/虚线框 3 个，左下三角=填充期、右上三角=排空期，与图注 $m+p-1=7$、$3/7\approx0.43$ 一致。
- 计算复算：$XA=[3,3,7,7]$（列和 1+2=3、3+4=7）与 $XA_1=[3,3]$、$XA_2=[7,7]$ 相符；$B=A$、$B_i$ 为 $2\times4$；$(p-1)/(m+p-1)$：$3/7\approx0.4286\to0.43$、$3/19\approx0.1579\to0.16$；$\mathrm{d}/\mathrm{d}p[(p-1)/(m+p-1)]=m/(m+p-1)^2>0$；$2\times61=122$。均与正文一致。
- 交叉页核对：beyond-buzz-disaggregation 页确有"KV 头数时 KV 复制而非切分"的带宽公式（支撑 §1.2 的推断指向）；chunked-prefill 页确有 CPP"等大的块顺带消掉流水线气泡"；pp-load-balancing 页确有"token 维度切分/batch 维度配平"；moe-serving 页覆盖 prefill/decode 与 KV cache；gpu-communication 页有 NVLink 与跨机带宽的量级对比（支撑 §3.1"低一个量级"）。所引 7 个前置概念页均真实存在。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`；`dojo:topics`（并行与通信, 推理系统）在词表内；head 的 description 为纯文本、dojo:summary 的 LaTeX（$\frac{p-1}{m+p-1}$、$\times$）可渲染；全页无 `$...$` 之外出现在标题/summary/正文/表格的 Unicode 数学字符；alt 属性仅 lightbox 空 alt，无 `$...$`；SVG 公式均在 `<foreignObject>` 内。以上均无问题，故不列条。

## 问题

- [轻微·来源] 来源与范围说明·构造示例（第 429–431 行列表）｜§1.3 两处（第 170、192 行）使用的"61 层模型"在本页被明确标注为"构造示例（层数自设）"，但"构造示例"小节只收录矩阵数字、气泡 $p/m$ 取值、组合部署图三项，未列出该 61 层示例，示例清单与正文标注不一致，后续维护无从得知该参数的来源与用途｜引文依据：不适用（页面内部一致性）｜修复要求：在"构造示例"小节补一条"61 层模型（$2L=2\times61=122$ 次 all-reduce，层数自设，用于说明计数随层数线性）"；或删去第 170、192 行的"构造示例"标注，改用与矩阵示例一致的中性表述，使清单与正文一致｜修复：｜复验：
- [轻微·格式] 来源与范围说明·论断与来源（C）（第 409–413 行）｜来源编号序列自 C2 起，全页既无 C1 的定义也无 `[C1]` 上标引用，C 系列存在空号（C2–C10），编号不连续｜引文依据：不适用（页面内部一致性）｜修复要求：将 C 系列编号整体前移一位（C2→C1 … C10→C9）并同步替换正文全部 `[Cx]` 上标，或在"论断与来源（C）"补出 C1 的定义使序列连续｜修复：｜复验：

## 结论

- 处置：可发布（本轮无阻断、无重要问题；2 条轻微均为清单/编号一致性，不影响正确性与主线理解，不阻碍发布。需注意本轮以外的既有审查记录不在本审查范围内）
- 统计：阻断 0 / 重要 0 / 轻微 2

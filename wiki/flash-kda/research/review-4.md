<!-- review-meta
round: 4
page: wiki/flash-kda/index.html
reviewed_content_sha256: 9ca55a3d401c3e5b
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 4 轮）

- 页面版本：af6c79669e9ee82a8bdcf7c3a78d0bcf77385fc5（wiki/flash-kda/index.html 工作树 blob）
- 审查时间：2026-09-13 19:39
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：页面级前置块（主要依据、引言、核心问题、最容易误解）→ 1. 串行状态 vs GPU 并行 → 2. FlashKDA → 3. 设备内 context parallelism → 4. KCP（4.1–4.5，含 4.5 的「展开：ground truth 4 步 + KCP 分解 3 步」折叠块）→ 5. KDA 解码（5.1–5.3）→ 来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制／全文总结）；并同读 overview.html。
- 核对用来源：Kimi K3 Technical Report 正文（§2.1.1 Eq.1/Eq.5、§5.1.1、§5.1.2 Eq.17、§5.4.2）、moonshotai/Kimi-K3 config.json、NVIDIA H100 SXM5 规格。本页正文无外部超链接，来源以章节号与引文标注，审查按章节号逐条回原文核对。

## 回源核对结果（通过项，留档）

- C1：报告 §5.1.1 原文"The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime. We design a dedicated kernel for each regime."——与页面引文逐字一致。
- C2：报告 §5.1.1"We therefore develop FlashKDA, a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation. The kernel decomposes the work into token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently... FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention."——一致。
- C3：报告 §5.1.1"Tensor parallelism partitions heads across devices but never shortens the recurrence... The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward. An automatic SM-level context-parallel (CP) planner... merges them to recover each segment's exact initial state... entirely intra-device and incurs no cross-device communication."——一致。
- C4：报告 §5.1.2"This direct summation, however, is insufficient for KDA... KDA's delta rule applies the token-dependent matrix M_t to the incoming state before adding the current write." / "we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities..." / "KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling."——一致；Ring Attention 归因正确（报告 ref [72] 即 Liu/Zaharia/Abbeel, Ring Attention）。
- C5：报告 §5.4.2"the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step... if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back. Maintaining a state snapshot... would also multiply state traffic... the state after any accepted draft prefix... is fully determined by the projected inputs of the draft tokens... rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [25]... Because the projection caches never leave the decode stage, prefix caching and prefill–decode disaggregation operate on the same payload as in non-speculative serving."——逐句一致；"sub-linearly"/"below state-caching baselines"/融合 kernel 覆盖"short convolution, input normalization, gating, the KDA recurrence, and output normalization"均有原文支撑。
- F1/F2：报告 §2.1.1 Eq.1 为 S_t=(I−β_t k_t k_t^⊤)Diag(α_t)S_{t−1}+β_t k_t v_t^⊤；页面写成 M_t:=(I−β_t k_t k_t^⊤)Diag(α_t) 后等价。Eq.17 第一行为 S_t^{[i+1]}=S̃_t^{[i+1]}+M_{t←1}^{[i+1]}S_{T_i}^{[i]}，第二行展开式与页面第 393 行等价（页面把前导因子 M^{[i]} 并入连乘 Π_{l=j+1}^{i}，上下标经 i+1→i 重排后一致）；prefix scan 递推 S←M_{T_j←1}^{[j]}S+S̃_{T_j}^{[j]} 与报告一致。
- F3/N4：报告 §2.1.1 Eq.5 α^h_t=exp(g^h_t)∈(e^{g_min},1)，g_min=−5；config.json linear_attn_config.gate_lower_bound=−5.0——页面的 (e^{-5},1) 正确。
- N1/N2：config.json linear_attn_config.head_dim=128、num_heads=96，顶层 num_attention_heads=96、num_hidden_layers=93；69 层 KDA 由 kda_layers（23 组×3=69）与报告 Table 1「69 KDA + 24 MLA」确认。
- N3：H100 SXM5 为 132 SM（全 GH100 144 SM，SXM5 启用 132）。
- 手算示例：用 numpy 独立复算 4 步 ground truth S4=[[5.5,7],[8.5,10]]、M_{T←1}^{[1]}=M_{T←1}^{[2]}=0.5I、S̃_T^{[1]}=[[1,2],[3,4]]、S̃_T^{[2]}=[[5,6],[7,8]]、prefix scan 重组=[[5.5,7],[8.5,10]]、误用直接求和=[[6,8],[10,12]]——全部与页面一致，分项与合计相符。
- 功能与格式：python3 .dojo/scripts/validate.py wiki/flash-kda/index.html 返回 validation ok；无 Unicode 数学字符（仅 —、“”、→、✓、✗ 等排版符）；图为 HTML 结构（dg-flow/dg-stack），无框线字符图；两级问题块命名与「解答：」折叠块齐全；overview.html 与 index.html 互链；引用的 KDA / GPU 执行模型 / 线性注意力 概念页均存在；无「（待生成）」占位。

## 问题

- [重要·技术] §4.4「一次 all-gather + prefix scan，通信量固定」（index.html 第 429 行）及 overview.html「4. 关键结论与边界」第 1 条：KCP 通信量被写成"KCP 只 all-gather 固定大小的状态（config.json 中约 32KB[N1]）"。｜引文依据：同页 §4.4 第 403 行为"交换的是固定大小的两个片段——状态 $\tilde S\in\mathbb{R}^{d_k\times d_v}$ 与累积转移 $M\in\mathbb{R}^{d_k\times d_k}$（K3 中 $d_k=d_v=128$，故两者尺寸相同）"；K3 报告 §5.1.2 原文"Each rank first computes M_{T_i←1}^{[i]} and eS_{T_i}^{[i]} locally, then exchanges both tensors with one all-gather"。两个片段各为 128×128×2 bytes≈32KB，一次 all-gather 合计约 64KB，页面写 32KB 少算了一半（漏掉 M 片段）；且 32KB 是由 config.json 的 head_dim=128 推得，config.json 并未直接给出字节数，写成"config.json 中约 32KB"把推导值说成来源字段。｜修复要求：把第 429 行与 overview.html 对应句改为"每个片段约 32KB、一次 all-gather 合计约 64KB（M∈R^{128×128} 与 S̃∈R^{128×128}，bf16）"，或明确限定为只计状态片段并给出合计值；同时把"config.json 中约 32KB"改为"由 config.json 的 head_dim=128 推得约 32KB"。｜修复：｜复验：
- [轻微·技术] §2 末（第 246 行）："重叠调度解决了 chunk 内的空转"与本章/上章对空转位置的定位不符。｜引文依据：§1 表格首行"训练 / prefill｜chunkwise 形式里 intra-chunk 计算与 cross-chunk 状态传播交替，SM 在传播时空转"；§2 第 205 行"算 chunk $t$ 的 intra-chunk 时，cross-chunk 的 $S[t]\to S[t+1]$ 没法开始"。空转发生在 cross-chunk 传播期（chunk 之间），并非 chunk 内部。｜修复要求：改为"重叠调度解决了传播期（chunk 间）的 SM 空转"。｜修复：｜复验：
- [轻微·技术] 引言（第 70 行）："Kimi K3 用 69 层 KDA 替换 softmax 注意力"。｜引文依据：报告 Table 1"Attention-Layer Composition … 69 KDA + 24 MLA"；报告 §2.1"Each block contains 3 KDA layers followed by 1 Gated MLA layer"，§5.1 限定为"KDA replaces the growing key–value cache of softmax attention"。K3 为 69 KDA 与 24 Gated MLA 的 3:1 混合，不是把 softmax 注意力全部替换。｜修复要求：改为"用 69 层 KDA 替换大部分 softmax 注意力，把随序列增长的 KV cache 压成固定大小的递归状态（其余 24 层为 Gated MLA 全局注意力）"或等价的限定表述。｜修复：｜复验：
- [轻微·表述] 第 582 行"四套方案覆盖了训练/prefill、长 prefill、跨设备与解码四个 regime，下面汇总各条论断的来源与适用范围。"；第 133 行"K3 报告 §5.1.1 开门见山："；第 333 行"但它对 KDA 不成立——这一章讲清楚为什么，以及 K3 的 KCP 怎么解决。"｜引文依据：不适用。｜修复要求：删去预告式元话语与对来源的临场评价——第 582 行改为直接进入来源章节（如"各条论断的来源与适用范围如下"）；第 133 行把"开门见山"改为"指出"；第 333 行改为引用小节标题（如"原因见 4.2 KDA 为什么不能"）。｜修复：｜复验：
- [轻微·技术] 来源章节「外部数字与实验条件（N）」N1（第 624 行）："状态大小约 32KB（$128\times128\times2$ bytes，bf16）"。｜引文依据：config.json linear_attn_config 仅给出 head_dim=128，未给出状态存储精度；报告 §5.1.1 仅写 S∈R^{d_k×d_v}，亦未给出精度。bf16 是页面自行假设却按来源事实陈述，且按 bf16 恰好为 32768 bytes（精确值），与"约"字自相矛盾。｜修复要求：改为"按 bf16 估算约 32KB"并注明精度为估算前提；或删去 dtype 断言，只写"128×128 的状态矩阵（约 32KB，按 2 bytes/元素估算）"。｜修复：｜复验：
- [轻微·可读性] 引言与 h2（第 70 行"四个执行 regime"、123 行"四个 regime 瓶颈不同"、137 行表头"执行 regime"）：术语"regime"首次出现即用于正文与章节标题，全文未给出中文释义。｜引文依据：不适用。｜修复要求：在首次出现处加最小解释，如"执行 regime（同一模型在训练、长 prefill、跨设备、解码等阶段的不同执行形态）"，后文沿用该词。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复（关闭 1 条重要 + 5 条轻微后，重跑 `.dojo/scripts/validate.py` 复核；核心结论与来源一致性无阻断问题，来源论断均已在原文定位到引文依据）

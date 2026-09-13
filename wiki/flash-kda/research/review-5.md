<!-- review-meta
round: 5
page: wiki/flash-kda/index.html
reviewed_content_sha256: a67f97e7d09ec6e7
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 5 轮）

- 页面版本：7166bca125de35f6c2394748972eec9e348c997c（工作树）
- 审查时间：2026-09-13 20:11
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. 串行状态 vs GPU 并行 → 2. FlashKDA → 3. 设备内 context parallelism → 4. KCP（4.1–4.5，含「展开」折叠块）→ 5. KDA 解码（5.1–5.3）→ 来源与范围说明；含全部「本章问题」解答折叠块
- 机械项：`.dojo/scripts/validate.py wiki/flash-kda/index.html` 返回 `validation ok`

## 来源核对（逐条回源，记录原文片段/关键数值）

- [C1] §5.1.1 原文核对一致：“The serial dependence of the KDA state is at odds with the GPU’s preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime. We design a dedicated kernel for each regime.”（§5.1.1 标题为 “KDA Kernels across Regimes”，含子节 “Chunkwise kernel for training and prefill”“Intra-device context parallelism for long-context prefill”）
- [C2] §5.1.1 原文核对一致：“We therefore develop FlashKDA [14], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation... token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently... FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention [141].”
- [C3] §5.1.1 原文核对一致：“Tensor parallelism partitions heads across devices but never shortens the recurrence... The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward. An automatic SM-level context-parallel (CP) planner... partitions the sequence across the SMs of a single rank, evaluates the segment transitions in parallel, and merges them to recover each segment’s exact initial state. In contrast to the cross-device KCP of §5.1.2, this parallelism is entirely intra-device and incurs no cross-device communication.”
- [C4] §5.1.2 原文核对一致：“This direct summation, however, is insufficient for KDA... KDA’s delta rule applies the token-dependent matrix Mt to the incoming state before adding the current write...”；“we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities, a cumulative transition acting on the incoming state and a state generated locally from zero”；“KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling.”；另核对 “These rank-level updates compose associatively, so the incoming state of each rank can be recovered by a prefix scan... starting from S = 0 and applying S ← M S + S̃ at each fragment.”
- [C5] §5.4.2 原文核对一致：“the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step... if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back. Maintaining a state snapshot for each draft position would enable rollback, but would also multiply state traffic — a cost that dominates at the large batch sizes typical of online serving. The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself. We therefore cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [25].”；“Verification latency grows sub-linearly with the number of tokens verified and remains below that of state-caching baselines.”；“a single fused kernel covering short convolution, input normalization, gating, the KDA recurrence, and output normalization.”；“Because the projection caches never leave the decode stage, prefix caching and prefill–decode disaggregation operate on the same payload as in non-speculative serving.”
- [F1]/[F2] §2.1.1 Eq.1 原文核对一致：“St = [I − βt kt kt⊤] Diag(αt) St−1 + βt kt vt⊤，õt = S⊤t qt”；§5.1.2 复述 “Mt := (I − βt kt kt⊤)Diag(αt)”。§5.1.2 Eq.17 第一行 “Mt←1[i+1] := ∏ Mr ∈ Rdk×dk，St[i+1] = S̃t[i+1] + Mt←1[i+1]S Ti[i]”；第二行求和式展开为本页 index.html:393 的重编号等价形式（逐项推导验证成立：∑_{j=1}^{i}(∏_{l=j+1}^{i}M^[l])S̃^[j] = S T[i]，j=i 时为空积 = I）。
- [F3] §5.1.2 原文含 “additive recurrence of vanilla linear attention”，与页面 $s_i = s_{i-1}+\phi(k_i)v_i^\top$ 一致。
- [N2] K3 config.json 核对：`linear_attn_config.num_heads = 96`、`num_attention_heads = 96` → 页面“96 个 head”成立。
- [N1] config.json 核对：`head_dim = 128`、`v_head_dim = 128` → $d_k=d_v=128$ 成立；128×128×2 bytes = 32768 B ≈ 32KB，bf16 估算成立。
- [N4] §2.1.1 核对：`gmin = −5`，`αht = exp(gth) ∈ (e^{gmin},1)`，config.json `gate_lower_bound = -5.0` → 页面“$\alpha$ 由 scaled sigmoid 产出，范围 $(e^{-5},1)$”成立。
- 层级混合核对：report 附录表 “Attention-Layer Composition: 69 KDA + 24 MLA”，`num_hidden_layers = 93`、`full_attn_layers` 24 项、3:1 混合 → 页面“69 层 KDA / 其余 24 层为 Gated MLA”成立。
- [N3] H100 SXM5 = 132 SM：与 NVIDIA 规格一致，页面已标注为非 K3 报告数据。
- 4.5 手算示例：逐步复算全部正确。ground truth S1=[[1,2],[0,0]]、S2=[[1,2],[3,4]]、S3=[[5.5,7],[3,4]]、S4=[[5.5,7],[8.5,10]]；M[1]=M[2]=0.5I；S̃[1]=[[1,2],[3,4]]、S̃[2]=[[5,6],[7,8]]；prefix scan 重组得 [[5.5,7],[8.5,10]] 与 S4 一致；误用直接求和得 [[6,8],[10,12]] ≠ 正确值。构造参数（$\mathrm{Diag}(\alpha_t)=I$、标量 $\beta=0.5$）已在正文与「构造示例」双重标注为教学构造。

## 问题

- [重要·技术] §5 图（index.html:522-545，t1–t4 节点）：图中 t2 标注“验证 ✗ 被拒绝”，t3 却标注“验证 ✓”，t4 标“?”；MTP 投机解码按位置自左向右验证，首个被拒绝位置之后的所有 draft token 一律丢弃，t3/t4 不可能被判接受，该图给出的接受/拒绝序列在投机解码下不成立，会让读者误以为拒绝之后仍可接受后续 draft。｜引文依据：K3 报告 §5.4.2 “if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token”；本页正文 index.html:520 亦写“部分 draft token 被拒绝——状态停在了一个‘不该停’的位置”。｜修复要求：删去 t3 的“验证 ✓”与 t4 的“验证 ?”，或整图去掉逐位置的接受/拒绝标注、只保留“状态随每步原地更新一路推进到 $S_d$”的陈述，使图意与被拒绝即截断的语义一致。｜修复：｜复验：
- [轻微·表述] §5.3 末尾 index.html:582：“各条论断的来源与适用范围如下。”——该句以“如下”承诺紧随其后列出来源与适用范围，但紧跟的是 h3「本章问题」及其问题块，来源小节在更下方；这是遗留的元话语式前向指引，在当前位置没有直接对象。｜引文依据：不适用｜修复要求：删除该段，或将其移入下方「来源与范围说明」小节开头。｜修复：｜复验：
- [轻微·符号] §4.1 图（index.html:348-361）用 rank 0 / rank 1 / rank 2（首个 rank 记为 0），而 §4.4 图（index.html:411-424）与 §4.5 手算（index.html:442-469）用 rank 1 / rank 2（首个 rank 记为 1）；同一元素“rank”两处编号起点不一致，读者会在两张图之间错位映射。｜引文依据：不适用｜修复要求：统一 rank 编号起点（建议均改为 1 起，与 Eq.17 的 $S_T^{[i]}$ 定义一致），或在 §4.1 图注中显式标注该图为 0 起编号。｜修复：｜复验：
- [轻微·来源] 来源说明 [N1]（index.html:624）：“精度为估算前提，config.json 与报告均未给出状态存储精度。”——与来源不符：K3 config.json 顶层给出 `"dtype": "bfloat16"`，本页 32KB 估算正是按 bf16。｜引文依据：config.json `"dtype": "bfloat16"`。｜修复要求：改为与事实一致的表述，如“config.json 给出 dtype=bfloat16，本页据此按 2 bytes/元素估算；报告未单独给出递归状态的存储精度”。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（重要问题修复后即可发布；核心公式、数字与手算示例经逐一回源核对无误，未发现阻断级问题）
<!-- review-meta
round: 7
page: wiki/flash-kda/index.html
reviewed_content_sha256: 3e60c56e2d5522dd
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 7 轮）

- 页面版本：07728023fa8c290575b752ac512ae9beace29557
- 审查时间：2026-09-13 21:47
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：引言 / 核心问题 / 最容易误解 → 「1. 串行状态 vs GPU 并行——为什么 KDA 在四个 regime 瓶颈不同」→「2. FlashKDA——把 chunk 内计算与 chunk 间状态传播重叠」→「3. 设备内 context parallelism——单 rank 的 SM 级切序列」→「4. KCP——为什么不能直接求和，以及 M+S̃ 分解」（含 4.1–4.5）→「5. KDA 解码——投影输入缓存与状态重建」（含 5.1–5.3）→「来源与范围说明」
- 来源获取：Kimi K3 Technical Report 正文取自 arXiv:2607.24653 HTML 版（https://arxiv.org/html/2607.24653v1 ，本地解析后逐段比对 §2.1.1 / §5.1.1 / §5.1.2 / §5.4.2）；config.json 取自 https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json 。

## 核对依据（关键引文与数值）

- §5.1.1 原文逐字：`The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime. We design a dedicated kernel for each regime.` / `We therefore develop FlashKDA [14], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation. The kernel decomposes the work into token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently, and substantially outperforms the Triton reference implementation. FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention [141].` / `Tensor parallelism partitions heads across devices but never shortens the recurrence... The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward. An automatic SM-level context-parallel (CP) planner... merges them to recover each segment's exact initial state. In contrast to the cross-device KCP of §5.1.2, this parallelism is entirely intra-device and incurs no cross-device communication.` —— 与页面 [C1][C2][C3] 引用一致。
- §5.1.2 原文逐字：`This direct summation, however, is insufficient for KDA. Recall from Eq. 1 that KDA updates its state as St = Mt St−1 + βt kt v⊤t , where Mt := (I − βt kt k⊤t ) Diag(αt ). KDA's delta rule applies the token-dependent matrix Mt to the incoming state before adding the current write. Consequently, the effect of a local sequence segment depends on the state entering that segment...` / `we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities...` / `KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling.` —— 与 [C4][F2] 一致。
- 报告 Eq.17：`M[i+1]^{t←1} := ∏_{r←1}^{t} Mr ∈ R^{dk×dk}`，`S[i+1]^t = S̃[i+1]^t + M[i+1]^{t←1} S[i]^{Ti}`，第二行 `= S̃[i+1]^t + M[i+1]^{t←1} ∑_{j=1}^{i}(∏_{l←j+1}^{i} M[l]^{Tl←1}) S̃[j]^{Tj}`。页面第 393 行 `S_T^{[i]} = S̃_T^{[i]} + ∑_{j=1}^{i-1}(∏_{l=j+1}^{i} M_{T←1}^{[l]}) S̃_T^{[j]}` 经验算等于把 Eq.17 第二行的前置因子 `M[i]^{Ti←1}` 折入连乘、并把该式应用到「离开 rank i」的同一等式，代数等价，成立。
- §2.1.1 Eq.1：`St = (I − βt kt k⊤t ) Diag(αt ) St−1 + βt kt v⊤t , õt = S⊤t qt`；α∈(0,1)^{dk}、β∈(0,1)。报告 §5.1.2 提及 softmax CP 交换 KV block 的引用为 [73]（Ring Attention）。
- §5.4.2 原文逐字：`the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step.` / `if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back.` / `Maintaining a state snapshot for each draft position would enable rollback, but would also multiply state traffic — a cost that dominates at the large batch sizes typical of online serving.` / `The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself. We therefore cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [25].` / `...inside a single fused kernel covering short convolution, input normalization, gating, the KDA recurrence, and output normalization. Verification latency grows sub-linearly with the number of tokens verified and remains below that of state-caching baselines.` / `Because the projection caches never leave the decode stage, prefix caching and prefill–decode disaggregation operate on the same payload as in non-speculative serving.` —— 与 [C5] 与 5.1–5.3 节各段一致。
- config.json（HF 原始文件）：`dtype: bfloat16`；`text_config.num_hidden_layers = 93`；`num_attention_heads = 96`；`linear_attn_config = {head_dim: 128, num_heads: 96, kda_layers: [69 项], full_attn_layers: [24 项], gate_lower_bound: -5.0}`。页首「69 层 KDA + 24 层 Gated MLA（合计 93）」、[N2] 96 head、单 head 状态 `128×128×2 B ≈ 32KB` 均与之一致。
- 手算示例逐步复算（d_k=d_v=2、Diag(α)=I、β=0.5、单位 k）：M1=M3=[[0.5,0],[0,1]]、M2=M4=[[1,0],[0,0.5]]；顺序递推 S4=[[5.5,7],[8.5,10]]；KCP 片段 M_{T←1}^{[1]}=M_{T←1}^{[2]}=0.5I、S̃_T^{[1]}=[[1,2],[3,4]]、S̃_T^{[2]}=[[5,6],[7,8]]；prefix scan 重组得 [[5.5,7],[8.5,10]]，与 ground truth 相同；误用直接求和得 [[6,8],[10,12]]。全部数字与页面一致，无 a×b≠积、分项之和≠合计、算式与结论不符的情况。
- 页面内部一致性：description / dojo:summary / 正文 / 表格 / 折叠块中「32KB、64KB、96 head、132 SM、69/24 层」等数字各处一致，无同页互相矛盾。

## 问题

- [轻微·技术] 「来源与范围说明·外部数字与实验条件（N）」[N4] 及「构造示例」：把 α 的来源写为 scaled sigmoid。原文分两步——scaled sigmoid 产出的是逐 token log-decay g，α 是 g 的指数。｜引文依据：报告 §2.1.1 Eq.5 `g_t^h = g_min Sigmoid(e^{A_h} z_t^h) ∈ (g_min, 0)^{d_k}`，`α_t^h = exp(g_t^h) ∈ (e^{g_min}, 1)^{d_k}`（g_min = −5）。｜修复要求：把「$\alpha$ 由 scaled sigmoid 产出」改为「log-decay $g$ 由 scaled sigmoid 产出、$\alpha=\exp(g)$」或等义表述；范围 $(e^{-5},1)$ 保留（与 Eq.5 一致）。｜修复：｜复验：
- [轻微·技术] 「来源与范围说明·外部数字与实验条件（N）」[N1]：将 $d_k=d_v=128$ 一并归因 config.json。config 的 `linear_attn_config` 只给单一 `head_dim: 128`（KDA head 维度），未单列 d_v；$d_v=128$ 是由 head_dim 推得（页面 overview.html 即写作「config.json 的 head_dim=128 推得」，措辞更准）。｜引文依据：config.json `linear_attn_config = {"head_dim": 128, "num_heads": 96, ...}`；`v_head_dim: 128` 属 MLA（全局注意力）配置，非 KDA。｜修复要求：把 [N1] 改为「config.json 的 `linear_attn_config.head_dim = 128`（本页取 $d_k=d_v=128$）」，或将该等式标注为推断；32KB 的 bf16 估算说明保留。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布

（说明：本轮对页面全部事实性论断、5 个引用编号 [C1]–[C5]、2 个公式编号 [F1][F2]、4 个数字 [N1]–[N4]、以及 4.5 手算示例的每一步矩阵运算均回源核对，未发现定位不到、来源不支持、实验条件被写成无条件论断、或推断被包装成来源结论的情形。两处轻微均为「来源表述精度」问题，不影响核心结论，故不阻断发布。）

> 本轮所列问题的处理结果见 `minor-fixes.md`。

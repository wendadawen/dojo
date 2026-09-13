<!-- review-meta
round: 3
page: wiki/kda/index.html
reviewed_content_sha256: 689488f9911432ee
-->
# Kimi Delta Attention（KDA）审查记录（第 3 轮）

- 页面版本：8f33f1d93a614431716de201d2be0f5cf90a7042
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么 K3 需要 KDA——KV cache 爆炸与线性注意力的"记不清" / 2. KDA 的递归核心——delta rule 加 channel-wise forget gate / 3. K3 的关键改动——lower-bounded decay（3.1 negative-softplus、3.2 scaled sigmoid、3.3 对照与手算、3.4 对角 tile）/ 4. 参数化与 full-rank output gate（4.1 full-rank output gate）/ 5. chunkwise 并行形式（5.1 累积衰减、5.2 chunk 内并行、5.3 lower-bound 收益）/ 6. KDA 在 K3 中的配置与边界（6.1–6.4）/ 来源与范围说明（含全部折叠块）
- 来源获取方式：官方技术报告 PDF（github.com/MoonshotAI/Kimi-K3 `k3_tech_report.pdf`，pdftotext 解析）§2.1.1、§5.1.1/§5.1.2、参考文献列表、Fig.3；官方 `config.json`（huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）；前置页 wiki/delta-rule、wiki/linear-attention。
- 机械项：`.dojo/scripts/validate.py wiki/kda/index.html` 返回 `validation ok`；overview.html 与 index.html 互链；前置概念页 delta-rule、linear-attention 均存在；无"（待生成）"占位；公式手算全部复算通过（见下）。

## 复算记录（用于来源核对）

- 报告 Eq.1：`St = (I − βt kt kt⊤) Diag(αt) St−1 + βt kt vt⊤`，`õt = S_t^⊤ q_t`；页 §2 逐字一致，"先衰减再擦写"的算子顺序（Diag 在最右）成立。
- 报告 Eq.2/Eq.3/Eq.4/Eq.5/Eq.6 与页 §4/§5/§3 各公式逐字一致；V_e[t]=U[t]−W[t]S[t]、Tril 保留对角、Γ 行向堆叠均一致。
- 1M KV 估算：1048576×96×128×2×2 = 51,539,607,552 bytes = 48.0 GB；页 "≈5.15×10^10 bytes ≈48 GB" 正确。
- 手算：Softplus(1)=1.3133→α=0.2689；Sigmoid(1)=0.7311→g=−3.6553→α=0.02585；e^{−5}=0.0067379；e^{80}=5.5406×10^34；均与页面一致。
- config.json 逐字段：num_hidden_layers=93、kda_layers 69 项（1,2,3,5,…,89,90,91）、full_attn_layers 24 项（4,8,…,88,92,93）、gate_lower_bound=−5.0、head_dim=128、num_heads=96、short_conv_kernel_size=4、use_full_rank_gate=true、hidden_size=7168、max_position_embeddings=1048576；6.2 表格与 6.1 层布局（22 个完整 3:1 块=88 层 + 89–91 KDA + 92/93 Gated MLA）全部正确。

## 问题

- [重要·技术] §1（`id="why-k3-needs-kda"`，第 172 行；同判据见第 116 行核心问题解答、第 209 行）：把"单头状态"写成"单层状态"。页面写"K3 单层 KDA 的状态 $S \in \mathbb{R}^{128 \times 128}$ 只占 $128\times128\times2\approx32$ KB"，并在核心问题答案里写"（单层约 32 KB）"。但 $S\in\mathbb{R}^{d_k\times d_v}$ 是**单头**状态（页 §2 自己也写"KDA 的单头递归"，且符号表把 $S_t$ 定义为单步矩阵状态）。K3 每层 `num_heads=96`，单层 KDA 状态应为 96×128×128×2 = 3 MB，页面数值低估 96 倍，且与 §1 的 softmax 估算（每层 96 头、单层 48 GB）量纲不可比，构成页内两处互相矛盾。｜引文依据：报告 §2.1.1 "For clarity, we first describe a single attention head, ... and recurrent state St ∈ R^{dk×dv}"；config.json `linear_attn_config.num_heads = 96`；复算 96×32 KB = 3 MB。｜修复要求：把该处及其在核心问题答案、第 209 行的表述改为单头口径（"单头状态约 32 KB"）或补上单层总量（"单头约 32 KB、单层 96 头合计约 3 MB"）；两者取其一并全页统一，使与 §2"单头递归"及 config 的 num_heads 一致。｜修复：｜复验：

- [重要·技术] §4（第 485 行，ShortConv 条目）：把 ShortConv 归因给 K3，与来源不符。页面写"ShortConv … 这是 K3 相对原始 DeltaNet 的小改动，让 key 不只看当前 token"。报告把整条参数化链（含 ShortConv、Swish、L2Norm）明确归于"沿用 Kimi Linear"，K3 相对 Kimi Linear 的改动只有 lower-bounded decay 与 full-rank gate 两处（页 §1 亦然）；把 ShortConv 说成 K3 的改动属来源不支持的归因。｜引文依据：报告 §2.1.1 "Following Kimi Linear [64], KDA parameterizes the per-head quantities as qth, kth = L2 Norm(Swish(ShortConv(W_{q/k}^h xt)))…"；同段"Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear"（K3 改动只在 gate 与 decay 映射）。｜修复要求：改为"ShortConv 是 KDA 从 Kimi Linear 继承的参数化步骤，相对原始 DeltaNet 增加了局部时序信息"，删除"这是 K3 相对原始 DeltaNet 的小改动"这一归因，或改为有来源支持的表述。｜修复：｜复验：

- [重要·技术] 正文与来源章节（第 105、533、565、719、749 行）：Kimi Linear 的引文编号错误。页面统一用"[63]"（含"UT 变换的完整推导 K3 报告指向 [63]"），但 K3 报告将 Kimi Linear 编为 **[64]**，其 [63] 是 PerceptionBench。按页面自述"K3 报告指向 [63]"，读者按页标注回查报告会定位到错误条目。｜引文依据：报告 §2.1.1 正文 "KDA extends the delta-rule recurrence [106, 140] with a channel-wise forget gate [64]"、"Following Kimi Linear [64]"；报告参考文献列表 [63] = "Kimi Team. PerceptionBench…"，[64] = "Kimi Team et al. Kimi Linear: An Expressive, Efficient Attention Architecture. 2025. arXiv: 2510.26692"。｜修复要求：把全页对 Kimi Linear 的 "[63]" 改为 "[64]"，或在页面内建立自己的引文编号表并给出对应关系；不得保留与所引报告不符的编号。｜修复：｜复验：

- [轻微·格式] 来源与范围说明（第 723 行）：h3 命名不符合固定命名。页面写 `<h3>外部数字与实验条件</h3>`，固定名称为"外部数字与实验条件（N）"（同页"论断与来源（C）""公式与来源（F）"都带字母）。｜引文依据：guides/concept/style-guide.md §1 "来源章节…下的 h3 使用固定命名…论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例…"；同库 wiki/rmsnorm/index.html:364、wiki/deepseek-moe/index.html:477 均为"外部数字与实验条件（N）"。｜修复要求：改为"外部数字与实验条件（N）"。｜修复：｜复验：

- [轻微·可读性] §5.1（第 543 行）：构造示例对前文例子的引用不成立。页面写"把递归核心章的 3 步迷你序列当作一个 $C=3$ 的 chunk，取 $\alpha_1=(1,1,1,1)$…$\alpha_3=(0.7,0.6,0.95,0.4)$"，但 §2 的例子只演示了第 1、2 步（$k_1,v_1$ 与 $k_2,v_2$），$\alpha_1$、$\alpha_3$ 均为此处新引入且未标注为新增；且 $\alpha_1=(1,1,1,1)$ 落在页 §3 自己规定的 $\alpha\in(e^{-5},1)$ 之外。｜引文依据：不适用（页内一致性）。｜修复要求：把 §5.1 的引用改为"在递归核心章 2 步示例基础上扩展为 $C=3$ chunk"，明确 $\alpha_1$、$\alpha_3$ 为本示例新增；$\alpha_1$ 注明是为便于手算取的理想值（真实 $\alpha<1$）或改用区间内取值。｜修复：｜复验：

- [轻微·表述] 正文段落与 §3 开头（第 158、333 行）：存在元话语。"下面是一个教学估算，不是工程基准，只为感受量级。"与"K3 在这两段映射上做了一处关键改动，是本页的核心点。"前者是"下面是…"式引入，后者是对本页结构的自我指涉。｜引文依据：不适用。｜修复要求：删去元话语框架，直接给出内容（如"这是一个教学估算，只用于感受量级：…"），删去"是本页的核心点"。｜修复：｜复验：

- [轻微·表述] §3.4 末尾（第 423 行）：无来源支持的评价写成结论。"在长序列训练里，这个权衡是值得的"是页面自行给出的价值判断，报告未作此结论（报告只陈述 lower-bound 消除了 position-pair 路径这一工程后果）。｜引文依据：报告 §2.1.1 "This finite range allows both diagonal and off-diagonal tiles to use dense Tensor Core matrix multiplications, eliminating the position-pair diagonal path."（无"值得"类评价）。｜修复要求：删除"这个权衡是值得的"，或改为标注为推断并给出成立条件。｜修复：｜复验：

- [轻微·技术] `<head>` description（第 6 行）：声称的来源与页内不符。description 写"基于 K3 报告 §2.1.1 + config.json + 源码核对"，但页内 blockquote.meta 的"主要依据"只列报告与 config.json，正文、[C]/[F]/[N] 各条也无任何官方源码路径与行号。｜引文依据：页面第 6 行与第 105 行；页内无源码引用条目。｜修复要求：删除"源码核对"，或补上具体源码路径/行号并登记到来源章节。｜修复：｜复验：

- [轻微·技术] §6.4 适用边界（第 671 行）：无来源支持的推断。"若用 FP32 训练，negative-softplus 也能用，但 FP32 的显存与速度代价更高"报告未述，属页面的外推。｜引文依据：报告 §2.1.1 只说明 Kimi Linear 用 log 空间 + 16-token tile 控制数值范围，未比较 FP32 方案。｜修复要求：删除该句，或明确标注为推断并说明依据。｜修复：｜复验：

- [轻微·格式] 来源章节 [C9]（第 711 行）：同一变量两种写法。全页把下界写作 `$g_{\min}$`（第 354、359、391 行等），此处却写纯文本 `gmin=-5`。｜引文依据：guides/concept/style-guide.md §11"同一变量在页面中保持同一种写法"。｜修复要求：把 `gmin=-5` 改为 `$g_{\min}=-5$`。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 7
- 处置：修复（三条重要问题关闭后可发布；轻微问题按上列要求逐条修复；核心机制、公式、config 数值、chunkwise 形式与手算示例经复核与来源一致，无阻断问题）

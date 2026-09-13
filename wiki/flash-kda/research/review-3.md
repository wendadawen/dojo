<!-- review-meta
round: 3
page: wiki/flash-kda/index.html
reviewed_content_sha256: b502faf762ce23a7
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 3 轮）

- 页面版本：bd48f832660a0c1af8ca41bf6e6468684e757717（wiki/flash-kda/index.html，862 行）
- 审查时间：2026-09-13 19:05
- 审查者：独立子代理（第 3 轮，未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 串行状态 vs GPU 并行 / 2. FlashKDA / 3. 设备内 context parallelism / 4. KCP（含 4.1 vanilla CP、4.2 KDA 不能求和、4.3 分解、4.4 all-gather 流程、4.5 手算验证及折叠展开块）/ 5. KDA 解码（5.1–5.3）/ 来源与范围说明 / 各章「本章问题」与全部解答折叠块 / overview.html
- 来源获取：Kimi K3 Technical Report = arXiv:2607.24653。已下载并解出正文：v2（arxiv.org/html/2607.24653v2，1520019 B）、v1（…v1，1366070 B），用 MathML 的 `alttext`（LaTeX 原文）核对公式。K3 config.json = huggingface.co/moonshotai/Kimi-K3/raw/main/config.json（真实下载核对）。H100 SM 数、CUTLASS 为通用规格。`.dojo/scripts/validate.py` 通过。
- 本轮核对到的关键原文（v2，行号为本轮解出文本）：
  - §5.1.1：“The serial dependence of the KDA state is at odds with the GPU’s preference for wide, uniform parallelism…”；“We therefore develop FlashKDA [13], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation.”；“Tensor parallelism partitions heads across devices but never shortens the recurrence…An automatic SM-level context-parallel (CP) planner…entirely intra-device and incurs no cross-device communication.”
  - §5.1.2 Eq.17 与 “M_t := (I − β_t k_t k_t^⊤) Diag(α_t)”；“KCP requires only a fixed-size all-gather…achieves linear compute scaling.”
  - §5.4.2：“…managing the evolving recurrent state…cannot be trivially rolled back…maintaining a state snapshot…would multiply state traffic…cache only these projected inputs, rebuild the states of accepted tokens on-chip…concurrent work ReplaySSM [23]…projection caches never leave the decode stage…”
  - config.json：`linear_attn_config.head_dim=128`、`num_heads=96`、`gate_lower_bound=-5.0`、`kda_layers` 长度 69；`dtype=bfloat16`。

## 问题

- [阻断·技术] wiki/flash-kda/index.html 第 127 行（§1 正文）、第 371 行（§4.2 正文）、第 615 行（来源 [C4] 引文）：$M_t$ 的定义丢了括号，与 K3 报告 Eq.1/§5.1.2 不符，且被写进“原文”引文。页面写 $M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$，来源是 $M_t := (\mathbf{I}-\beta_t k_t k_t^\top)\operatorname{Diag}(\alpha_t)$——来源中 $\mathrm{Diag}(\alpha_t)$ 乘的是整个 $(\mathbf{I}-\beta_t k_t k_t^\top)$，页面版把它并入了 $k_t k_t^\top$，等于丢掉了对入状态的通道级衰减，语义不同。｜引文依据：报告 MathML alttext `\mathbf{S}_{t}=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})\mathbf{S}_{t-1}+\beta_{t}\bm{k}_{t}\bm{v}_{t}^{\top}` 与 `\mathbf{M}_{t}:=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})`；KDA 概念页 wiki/kda/index.html 第 239 行也写 $(I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1}$ 并强调“$\mathrm{Diag}(\alpha_t)$ 位于乘法链最右侧”。第 444 行“$M_t = I - 0.5\,k_t k_t^\top$”与 §4.5 全部矩阵均建立在此错误定义上。｜修复要求：把 127、371 两处公式与 [C4] 引文改为 $(\mathbf{I}-\beta_t k_t k_t^\top)\operatorname{Diag}(\alpha_t)$；并按修正后的定义重算 §4.5（第 444、451、457–458、466、474–482 行）的 $M_1\ldots M_4$、$\tilde S$ 与 ground truth（现参数下 $M_1 M_2$ 会退化，需重选教学参数或注明）；修正后重新核对来源。｜修复：｜复验：
- [重要·技术] 第 613 行 [C2] 与第 616 行 [C5] 引文中的方括号编号与报告不符，且全页未标注所引报告版本。页面写 “FlashKDA [14]”“flash-linear-attention [141]”“ReplaySSM [25]”。｜引文依据：报告 v1 为 “FlashKDA [13]”“flash-linear-attention [140]”“ReplaySSM [23]”；v2 为 “FlashKDA [13]”“flash-linear-attention [142]”“ReplaySSM [23]”——三处编号在 v1、v2 下均对不上，[14]/[141]/[25] 是两者都不匹配的值。｜修复要求：把引文编号改为与所引版本一致（并在页首或来源节注明“依据 v1/v2”），或删去方括号编号只保留文字。｜修复：｜复验：
- [重要·技术] 第 298 行（§3）：“设备内 CP：…SM 之间通过 shared memory / L2 协作，不走网络。”来源未支持“shared memory / L2”这一机制。｜引文依据：报告 §5.1.1 原文只说 “…partitions the sequence across the SMs of a single rank, evaluates the segment transitions in parallel, and merges them to recover each segment’s exact initial state. In contrast to the cross-device KCP of §5.1.2, this parallelism is entirely intra-device and incurs no cross-device communication.”，未提 shared memory 或 L2；页面把该推断写成事实。｜修复要求：删除“SM 之间通过 shared memory / L2 协作”，或降级为明确标注的推断。｜修复：｜复验：
- [重要·格式/写作] 第 411、423、457、466、482、485、620 行：§4.5 与来源 [F2] 使用记号 “$M累积$”，第 620 行另用 “$S_{入}$”，与 §4.3 第 385 行定义的 $M_{t\leftarrow 1}^{[i+1]}$、$S_T^{[i]}$ 不是同一写法，违反 style-guide 第 11 节“同一变量在页面中保持同一种写法”；且“累积”“入”等中文字符进入 `$...$`，由 KaTeX 渲染为数学文本。｜引文依据：不适用（页面内自相矛盾）。定义见第 385 行 “$M_{t\leftarrow 1}^{[i+1]} := \prod_{r\leftarrow 1}^{t} M_r \in \mathbb{R}^{d_k \times d_k}$”，使用见第 457 行 “$M累积_1 = M_2 M_1 = …$”。｜修复要求：把 §4.5 与 [F2] 的 “$M累积$”统一为 $M_{t\leftarrow 1}^{[\,]}$/$M_{T\leftarrow 1}^{[\,]}$，“$S_{入}$”统一为 $S_T^{[i]}$，消除公式内中文字符。｜修复：｜复验：
- [轻微·技术] 第 403 行（§4.4）与第 509 行（§4 本章问题答案）：“交换的是固定大小的状态（$S\in\mathbb{R}^{d_k\times d_v}$ 及同形状的 $M$）”。按 §4.3 第 385 行定义 $M_{t\leftarrow 1}\in\mathbb{R}^{d_k\times d_k}$，并非与 $S\in\mathbb{R}^{d_k\times d_v}$ 同形状（只在 K3 的 $d_k=d_v=128$ 下尺寸数值相等）。｜引文依据：报告 §5.1.2 “$\mathbf{M}_{[i+1]}^{t\leftarrow 1}\in\mathbb{R}^{d_{k}\times d_{k}}$” 与 $\mathbf{S}\in\mathbb{R}^{d_{k}\times d_{v}}$。｜修复要求：写明 $M$ 为 $d_k\times d_k$，或补一句“K3 中 $d_k=d_v=128$ 故两者同尺寸”。｜修复：｜复验：
- [轻微·表述] 元话语/过程叙事：第 70 行“本文讲解这四套方案——…”；第 174 行“本章先建立「冲突根源 + 四 regime 定位」的框架，后续四章各解一个 regime。”；第 453 行“下面用 KCP 分解重组，看能否得到同样的值。”均为对讲解行为的旁白，非内容陈述。｜引文依据：不适用。｜修复要求：改为直接陈述（如手算处直接写“用 KCP 分解重组的结果如下”），删除“本文讲解…”“本章先建立…”“下面…看能否…”式旁白。｜修复：｜复验：
- [轻微·表述] 五处章节过渡套用同一句式“本章讲了 X…但 Y——下一章讲 Z”：第 176、246、304、487、582 行。style-guide 第 8 节要求章节衔接“不使用固定句式”。｜引文依据：不适用。｜修复要求：改写各过渡句，使其互不同构，或直接在段末陈述下一节要解决的具体问题。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 3 / 轻微 3
- 处置：修复。阻断项（$M_t$ 定义丢括号）与重要项（引文编号、【shared memory/L2】无来源、记号不一致）关闭后方可进入下一轮复验或发布。
- 说明：本轮另核对通过的要点——69 层 KDA（config.json `kda_layers` 长度 69）、96 head（`linear_attn_config.num_heads=96`）、$d_k=d_v=128$ 与约 32KB（128×128×2 B，bf16）、$\alpha\in(e^{-5},1)$（`gate_lower_bound=-5.0` 与报告 §2.1.1 scaled sigmoid）、Eq.17 分解与 §4.5 prefix scan 数值（在页面自定的 $M_t$ 下自洽可复算）、C1/C3/C4/C5 其余引文文字、五个 regime 方案与四个核心问题答案均与报告一致；`.dojo/scripts/validate.py` 通过。

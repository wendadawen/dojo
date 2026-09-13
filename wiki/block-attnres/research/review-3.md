<!-- review-meta
round: 3
page: wiki/block-attnres/index.html
reviewed_content_sha256: e2138256a3561b9b
-->
# Block AttnRes 概念页审查记录（第 3 轮）

- 页面版本：index.html `adc61ee0a35624ec819f8ab63fd4c3ba54ba5da8`；overview.html `a14078a988922f6388a27ecdf9cedb03061487e9`
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节（含全部折叠块与图注）：核心问题；1. 标准残差在深度上的瓶颈——为什么需要 AttnRes；2. Full AttnRes 的公式——pseudo-query 如何检索前序层；3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$；4. K3 的具体配置——8 块×12 层、9 个候选、加权三次；5. softmax kernel 中的 RMSNorm——为什么不能直接用内积；来源与范围说明；overview.html 全文。
- 来源获取：Kimi K3 Technical Report（arXiv 2607.24653，HTML 全文逐句检索）与 HuggingFace 官方 `config.json`（`https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`）。

## 核对通过（要点）

- config.json 实测：`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`attn_res_block_size=12`、`linear_attn_config.full_attn_layers` 24 个索引、`kda_layers` 69 个索引 —— 与页面配置表逐项一致（`full_attn_layers` 最大索引 93，93 层全覆盖）。
- 报告 §2.2 Eq.(8)(9)(10) 原文与页面公式逐符号一致：$k_i=v_i=\{h_1\,(i{=}0);\,f_i(h_i)\,(1\le i\le l-1)\}$、$\phi(q_l,k_i)=\exp(q_l^\top\mathrm{RMSNorm}(k_i))$、$\alpha_{i\to l}=\phi/\sum_j\phi$、$V=\{[b_0..b_{n-1}]^\top\,(i{=}1);\,[b_0..b_{n-1},b_n^{i-1}]^\top\,(i\ge2)\}$、$b_n=\sum_{j\in B_n}f_j(h_j)$、$b_0=h_1$。
- 报告原句逐字核对通过："Standard residual connections compress all prior information into a single state $h_l$ over depth — a bottleneck reminiscent of RNNs over time."；"Since network depth is modest (L<100), the $O(L^2d)$ arithmetic … the practical overhead is the $O(Ld)$ memory …"；"Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(Nd)"；"The final output layer then aggregates all N block representations."；"the RMSNorm prevents layers with large-magnitude outputs from dominating the weights"；"each module to selectively retrieve representations"；"three complementary dimensions: sequence length, network depth, and model width"；Table 1 "69 KDA + 24 MLA"。
- 三处手算复核全部通过：Full AttnRes 6 候选 $h_6\approx[0.703,0.703]$、Block AttnRes 4 候选 $h_6\approx[1.059,1.241]$、加 RMSNorm 后权重 $[0.2058,0.2620,0.2704,0.2620]$ 与 $h_6\approx[1.004,1.056]$，逐位与页面一致；方向判断（$b_2$ 较 $b_1$ 更贴近 $q_6$）成立；$\exp$ 比值（7.4 / 148 / 54.6）正确。
- `.dojo/scripts/validate.py wiki/block-attnres/index.html` 返回 `validation ok`；head 的 `description`/`dojo:summary`/`dojo:type=concept`/`dojo:topics=注意力机制`/`dojo:tag=注意力` 均在允许词表内；无 "（待生成）" 占位，无指向 `research/` 的失效路径；前置页 `residual-connection`、`kimi-k3-dataflow` 均存在，`modeling_kimi_linear.py` 与 `self_attention_res_norm/proj` 等字段名在 `kimi-k3-dataflow` 中可查得。

## 问题

- [阻断·来源] 正文 §4 原文引用块（index.html 第 584 行）、C6（第 829 行）、N2（第 847 行）、"未核对与遗留问题"（第 878 行）及 overview.html 第 4 节：AttnRes 原 preprint 的引文编号写成 [58]，与报告不符（应为 [60]），且该编号出现在一段标注为"§2.2 原文"的英文引文内部，等于改动了引文。｜引文依据：报告 §2.2 实为 "Empirically, N≈8 recovers most of the benefit across model scales **[60]**; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."；报告参考文献 [60] = "Kimi Team (2026) Attention residuals. Note: Preprint. Cited by: §1, §2.2, §2.2, §2, §5.2.2, §5.4.2, Abstract."，而 [58] = "Kimi Team (2025) Kimi k2: open agentic intelligence."（即页面"原 preprint [58]（Kimi Team, 2026）"的年份/身份也与 [58] 不符）。｜修复要求：把四处正文引文中的 [58] 一律改为 [60]，并同步修正"原 preprint [58]（Kimi Team, 2026）"的编号；overview.html 对应句同步修改。｜修复：｜复验：

- [阻断·来源] 第 286 行 与 来源说明 F5（第 841 行）：同一 RMSNorm 定义的引用编号两处互相矛盾，且都与报告不符。正文写"K3 报告引用 [148]"，F5 写"K3 报告引用 [146]"。｜引文依据：报告 §2.2 原文 "The attention weights follow a softmax kernel φ(**q**,**k**)=exp(**q**ᵀRMSNorm(**k**)) **[55, 147]**"；参考文献 [147] = "B. Zhang and R. Sennrich (2019) Root mean square layer normalization. Advances in NeurIPS 32."（[146] 与 [148] 分别是其他文献）。｜修复要求：正文与 F5 统一改为 [147]，并保留报告"与 [55] 并列引用"的写法可选；改后重核来源。｜修复：｜复验：

- [阻断·来源] 第 653 行："例如 K3 报告 §6.4 提到的 nano-kpu 原型使用 block size=2（K3 报告第 2208-2210 行确认）"——章节定位错误，§6.4 内容与该论断不符。｜引文依据：报告 §6.4 "Cost Efficiency" 全节讲推理成本对比（"Beyond scores, we examine inference cost efficiency by comparing score against per-task cost across four suites…"），无 nano-kpu；nano-kpu 出现在第 7 章 "7 Case Studies" 的 "Chip design" 小节："Kimi K3 designed an inference-chip prototype for a nano model following the same architecture — hybrid KDA and NoPE-MLA attention, **Block AttnRes with a block size of two**, sigmoid-based MoE routing with one shared expert…"。｜修复要求：把章节号由 §6.4 改为"第 7 章 Case Studies（Chip design）"；行号 2208-2210 无法定位，一并删除或改为可复核的定位（引该小节标题）。｜修复：｜复验：

- [重要·来源] overview.html 第 48 行："K3 实现预分配 9 个槽位适配最大值。"——无来源支持的 K3 实现细节被写成结论，且与本页 index.html 第 596 行的自我声明相冲突。｜引文依据：报告全文无 AttnRes 槽位"预分配"表述（`preallocat` 检索 0 命中，`slot` 4 处均为冗余专家槽 / VPP 缓冲 / KV checkpoint，与 AttnRes 无关）；index.html 第 596 行明写"具体槽位分配的实现细节（预分配与屏蔽策略）属于工程实现，K3 报告未展开，本文不推测"。｜修复要求：删除"K3 实现预分配 9 个槽位适配最大值"，或改为明确标注的推断（如"若实现按最大候选数预分配槽位"）并去掉对 K3 的断言。｜修复：｜复验：

- [轻微·表述] 第 239、346、466、502 行（另 118、225 行同类）：出现元话语"需要强调的是""需要强调的一个细节""但要注意""这里要注意"，以及"本文要回答：…""先用一个对照表固定两者的区别"。｜引文依据：不适用。｜修复要求：删除或改写为直接陈述（如把"需要强调的是：AttnRes 替代的是…"改为"AttnRes 替代的是…"；"需要强调的一个细节：候选数随 block index 增长"改为"候选数随 block index 增长"；"但要注意：""这里要注意："直接删除冒号前缀并入正文）。｜修复：｜复验：

- [轻微·格式] 第 296、300、472、732 行：手算块以固定引入句"构造示例。"开头。｜引文依据：不适用（`guides/concept/style-guide.md` §4 要求示例按用途标记为"计算示例/代码示例/构造数据"，不使用固定引入句）。｜修复要求：改为规范允许的标记（如"构造数据"）或直接进入输入描述。｜修复：｜复验：

- [轻微·技术] 第 726 行：数值稳定说明"$\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k)-m)$，$m$ 为 batch 内最大值"中 $m$ 的取值域不准确——softmax 归一化的最大值是取在候选（求和下标 $j$）上的，与 batch 无关。｜引文依据：不适用（通用数值稳定技术，非来源论断）。｜修复要求：把"$m$ 为 batch 内最大值"改为"$m$ 为全部候选 $\phi(q_l,k_j)$ 的最大值"或等价表述。｜修复：｜复验：

- [轻微·技术] 全页符号一致性：第 1 章用 $F_l$（如 $h_{l+1}=h_l+F_l(h_l)$、$h_L=h_1+\sum_{l=1}^{L-1}F_l(h_l)$），第 2 章起改用 $f_i(h_i)$ 表示同一"层变换输出"（$k_i$、块内求和、伪代码均用 $f$），且层下标一变 $l$、一变 $i$。｜引文依据：报告统一写作 $f_j(h_j)$（§2.2 "layer outputs are reduced to a single representation by summation, $b_n=\sum_{j\in B_n}f_j(h_j)$"）。｜修复要求：统一为报告写法 $f_i$（或第 1 章改用 $f$），层下标全页统一。｜修复：｜复验：

## 结论

- 统计：阻断 3 / 重要 1 / 轻微 4
- 处置：修复。三条阻断均为来源保真问题（两处引文编号与报告不符、其中一处同页自相矛盾；一处章节定位错误），消除后按本规范第 4 节复验并重跑 `.dojo/scripts/validate.py`。其余为表述与符号一致性问题，不推翻核心结论。

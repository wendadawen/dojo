<!-- review-meta
round: 7
page: wiki/latent-moe/index.html
reviewed_content_sha256: 81b24ad4e4de55f9
-->
# LatentMoE 审查记录（第 7 轮）

- 页面版本：e5632fdbae8686cbde45a929473a2306172bae72
- 审查时间：2026-09-14 17:00
- 审查者：独立子代理（未参与写作，未参与本页前序审查；未读取 research/ 任何文件）
- 适用规范：guides/concept/check.md（dojo:type = concept）
- 已完整阅读章节（含折叠块、图注、代码块、overview.html）：引言与核心问题 / 最容易误解 / 1 扩大专家池的代价 / 2 LatentMoE 的层结构 / 3 压缩比与 Reinvestment / 4 压缩下限与投影成本 / 5 与 Stable LatentMoE 的关系 / 来源与范围说明

## 来源核对（逐条回源，含关键引文）

外部来源均以 WebFetch / curl 抓原文核对（本地无这些来源文件）：

- **公式 F2 / 结构 [C1]**：K3 报告 Eq. 11 原文 `u=∑i∈𝒯k(x)pi·E_i^routed(W↓x)`、`y=∑j E_j^shared(x)+W↑RMSNorm(u)`（arXiv:2607.24653 §2.3）；通用形（去 RMSNorm）与页面 F2 一致。论文 §3 标题为 "LatentMoE Architecture"，§2 为 "LatentMoE Core Design Principles"，与 [C1][C2] 标注的节名一致。
- **router 在 d 维 [C7]**：论文 §3 原文 "The routing weights p′=Softmax(W_r′·x) are computed from the original token x∈ℝ^d" 及 "all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension d"，与页面引文逐字一致。
- **缩小比例 [C3/F3]**：Figure 1 caption 原文 "reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"，与页面 C3/F3 一致。
- **reinvestment [C4/F4]**：§3 定义 N′=α·N（ℓ-MoE_eff）、K′=α·K（ℓ-MoE_acc），"the default LatentMoE configuration (a.k.a., ℓ-MoE_acc)"；§4.1 原文 "We recommend ℓ-MoE_acc for Pareto-optimal accuracy versus inference cost."，与页面"论文推荐 ℓ-MoE_acc 作为准确率与推理成本 Pareto 最优的变体"一致。
- **五条设计原则**：论文 §2 确为五条；III "requires preserving the effective nonlinear budget, K·m"（论文记 U_eff ∝ K·m）；IV "There exists a task-specific feature rank r_eff that imposes a lower limit on the reduction of d"；V "Scaling both the number of experts N and top-k per token K enhances model quality"。页面折叠块译文一致。原则 II 的"all-to-all 数据量 ∝ 激活专家数 × d"亦见于 §2（volume ∝ N/EP·t_exp·d，等价 t_total·K·d/EP）。
- **9% 投影成本 [N5]**：§4.3.1 原文 "Relative to native Kimi-K2-1T, Kimi-K2-1T-LatentMoE introduces additional computation due to latent projection operators. In our projections, native Kimi-K2-1T remains close, within up to ~9% of Kimi-K2-1T-LatentMoE, indicating that projection overhead is small"，节号与标题 "Projected Serving Impact at Trillion-Parameter Scale" 与 [N5] 一致。
- **16B 消融 [N4]**：§4.1 原文 "16B total parameters with 2B active, which we use for conducting ablation studies"；"model quality is preserved for compression ratios α≤4. Consequently, we adopt α=4"；Figure 4 "reducing d without scaling the expert count leads to significant quality degradation"。页面的"压缩比到 4 时质量保持（此时专家数已按 α 放大）"与之一致。
- **N6（登记未引用）**：§4.3.1/Figure 7 原文 "requires an additional ∼350B parameters"、"yields a 1.24×–3.46× projected slowdown"，与 N6 一致；已核对正文确实未直接引用 [N6]，页面自述与事实相符。
- **K3 配置 [N1]**：官方 config.json（moonshotai/Kimi-K3）实测 hidden_size=7168、routed_expert_hidden_size=3584、num_experts=896、num_experts_per_token=16、num_shared_experts=2；稀疏度 896/16=56，压缩比 7168/3584=2，全部与页面一致。
- **Nemotron-3 Super/Ultra [N2/N3]**：[R]（sebastianraschka.com/blog/2026/nemotron-3-ultra-latent-moe.html）原文确认 Super 4096→1024→4096、120B/12B；Ultra 8192→2048→8192、550B/55B、512 路由专家、top-22、1 共享专家、路由中间维度 5120、共享中间维度 10240。与页面 [N2][N3] 一致。论文 [L] 摘要与 §1 均只出现 "has been adopted by the flagship Nemotron-3 Super and Ultra models"，未给规格，与页面"仅在摘要与 §1 确认"一致。
- **可运行代码**：本地实跑页面代码（python3），输出与"预期输出"逐行一致：top-8 32768/8192（4.0x）、top-16 65536/16384（4.0x）、C(8,2)=28、C(32,8)=10518300、增长倍数 375653.57、投影参数 8388608。
- **交叉一致性**：d=4096/k=8→32768、k=16→65536、d/ℓ=4、N′=32/k′=8、组合数 28→10518300、d=7168/ℓ=3584/N=896/k=16/N_s=2/稀疏度 56/2.8T/4x(Nemotron)/2x(K3)/9% 在正文、折叠解答、表格、图注、overview.html 之间全部一致；overview 与 index 相互链接；被引概念页 moe-serving / deepseek-moe / stable-latent-moe 均真实存在且无"（待生成）"；stable-latent-moe 的 K3 数字与 Eq. 11 与本页一致，无跨页矛盾。
- **机械项**：alt 属性无 `$...$`；标题/摘要/列表/表格无 Unicode 数学字符（仅代码块字符串内的 → 属于代码，非公式）；无"本页/我们/你/需要注意的是/值得注意"等元话语与会话指代；无 img 图，结构图为 HTML 折叠框（无等宽 ASCII 框线）；`.dojo/scripts/validate.py wiki/latent-moe/index.html` 返回 "validation ok"。

## 问题

- [轻微·表述] 5. 与 Stable LatentMoE 的关系（正文第 480 行）：句子"这使 K3 的 Eq. 11 在 $W_\uparrow$ 作用于 $\mathrm{RMSNorm}(u)$ 而非 $u$ 本身"语法破碎，"在"疑为"中"之误，读作"Eq. 11 在 W↑ 作用于…"缺主谓搭配；同章本章问题第 3 题解答（第 558 行）同一意思写作"K3 的 Eq. 11 因 Normalized LatentMoE 在 $W_\uparrow$ 前加了 RMSNorm，上投影作用于 $\mathrm{RMSNorm}(u)$"，语序正确，可据此统一。｜引文依据：不适用（可读性）｜修复要求：将该句改为"这使 K3 的 Eq. 11 中 $W_\uparrow$ 作用于 $\mathrm{RMSNorm}(u)$ 而非 $u$ 本身"或等价通顺表述。｜修复：｜复验：
- [轻微·格式] 来源与范围说明·简化条件（2）（第 591 行）："见 DeepSeekMoE §3.3"指向被链回的 wiki 页 deepseek-moe，但该页只有 §1–§5，其 §3 为"共享专家隔离"、无 3.3 子节（负载均衡 L_ExpBal/L_DevBal 仅在该页 §3 的折叠块中一句带过，标注为"论文 §3.3"）；按本页自身惯例（如"moe-serving §3"指被链页内 §3），此指针在目标页内无法定位。｜引文依据：不适用（页内指针）｜修复要求：写明所指为"DeepSeekMoE 论文 §3.3"或改指 deepseek-moe 页内实际含该内容的章节/折叠块。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。本轮逐条回源核对（论文 Figure 1 caption、§2 五原则、§3 变体定义、§4.1 消融、§4.3.1 的 9% 与 N6，K3 报告 §2.3 与 official config.json，[R] Nemotron 规格），全部与来源一致；代码实跑输出一致；validate.py 通过；无阻断与重要问题。遗留 2 处轻微（一处语法、一处页内指针）不影响正确性与主线理解，接受理由为纯表述/指针层面、不改变任何事实或数字，建议随下一批轻微项清理。
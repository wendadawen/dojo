<!-- review-meta
round: 3
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 30f4703b3a1bf2d9
-->
# Quantile Balancing 审查记录（第 3 轮）

- 页面版本：a51b04add5f4a72328aa0567cdace14417e38740
- 审查时间：2026-09-13 19:09
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题、最容易误解、1. auxiliary-loss-free 路由——bias 做什么、不做什么（含本章问题）、2. DeepSeek-V3 的固定步长更新——为什么 896 专家时失效（含本章问题）、3. QB 的核心机制——从一次前向推导下一个 bias（3.1–3.9、手算代码折叠块、本章问题）、4. QB 为什么好——从平衡分配到对偶的视角（4.1–4.3、「补充：从对偶目标到 coordinate minimizer」折叠块、本章问题）、5. 训练时的直方图估计与推理冻结（5.1–5.3、直方图代码折叠块、本章问题）、来源与范围说明
- 来源获取：Kimi K3 Technical Report（本地全文 k3_report_text.txt，对应 PDF kimi_k3_tech_report.pdf）§2.3.3、Appendix C、Appendix D；HuggingFace moonshotai/Kimi-K3 config.json（WebFetch）；DeepSeek-V3 Technical Report arXiv:2412.19437（WebFetch v1/v2）

## 已核对来源（无问题，供复验）

- Eq.13（F1）：报告 §2.3.3 "T_i = argtopk(s_i+b), p_{i,j} = s_{i,j}/Σ_{r∈T_i} s_{i,r}, j∈T_i" 与页面一致。
- sign 更新（F2）：报告 §2.3.3 "b^{(t+1)}_j = b^{(t)}_j + γ sign(ℓ̄ − ℓ^{(t)}_j) [30]" 与页面一致；DeepSeek-V3 原文（§2.1.2）为 overload 减 γ、underload 加 γ，页面 C2 的等价改写正确。
- Eq.14（F3）：报告 "b̃^{(t+1)}_j ← −quantile_{1−k/n}(s_{:,j} − α^{(t)}), b^{(t+1)} ← b̃^{(t+1)} − mean(b̃^{(t+1)})1" 与页面一致。
- 对偶目标 Eq.23（F4）：报告 "L(α,β) := Σ max(0, s_{i,j} − α_i − β_j) + k Σ α_i + (mk/n) Σ β_j" 与页面一致。
- coordinate minimizer Eq.25–26（F5）：报告 "α*_i = quantile_{1−k/n}(s_i − β)"、"β*_j = quantile_{1−k/n}(s_{:,j} − α)" 与页面一致。
- subgradient Eq.27（F6）：报告 "∂L/∂β_j = mk/n − Σ χ(s_{i,j} − α_i − β_j > 0)" 与页面一致；页面 4.3 "等价于 mk/n − ℓ_j" 成立。
- 收敛（C1）：报告 Appendix C "equilibrates within a few update steps even for nearly 10^3 experts" 与页面"近 10^3 专家时几步内收敛"一致。
- 直方图（C14/F7）：报告 Appendix D "histogram the required bias r_{i,j} := α_i − s_{i,j}"、"r falls in [b_min − 1, b_max + 1]"、"w = (b_max − b_min + 2)/B"、"select the first bin whose cumulative count reaches ⌈q⌉"、"b̃_j = b_min − 1 + (β_j + clip((q−c_j)/h_j, 0, 1)) w" 与页面 5.1 及代码逐项一致。
- 误差与通信（N2/N3）：报告 Appendix D Properties "with B = 1000 this is at most a few 10^{-3}, and we observe no measurable residual load imbalance"、"one integer all-reduce of nB values per layer per step, independent of m … below 1% of the cost of exchanging the raw margins" 与页面一致。
- EMA（C16）：报告 "maintaining an exponential moving average of the estimated quantiles across steps reduces batch-to-batch sampling noise" 与页面一致。
- 896/16 与 K2 384（N1/C17）：config.json 实测为 num_experts 896、num_experts_per_token 16、topk_method "noaux_tc"、moe_router_activation_func "sigmoid"；报告 Table 1 为 Routed Experts 384→896、Experts Active per Token 8→16。页面数字全部正确。
- 代码（check.md 2.2-3）：实际执行折叠块中的 Python 代码，输出与页面「预期输出」逐行一致（初始 loads [4,3,1,0]、b_tilde ['-0.2','-0.1','-0.0','+0.1']、mean -0.05、b_new ['-0.15','-0.05','+0.05','+0.15']、QB 后 loads [2,2,2,2]、改变 token T4/T5/T8）。
- 手算示例：8×4 分数矩阵的 α、margin、分位数、mean-centering、新路由各表数字全部复算通过；loads (4,3,1,0)→(2,2,2,2) 正确。
- validate.py：`validation ok: wiki/quantile-balancing/index.html`（退出码 0）。

## 问题

- [阻断·技术] 2. 章正文（第 219 行）与 2 章「本章问题」解答（第 231 行）：sign 更新所需步数"约 50000 步"与同处给出的算理不符。解答原文"每步 bias 只变化 $0.001$，要在 $0.05$ 量级上产生实质影响需要约 $50000$ 步"，而 $0.05 / 0.001 = 50$，自洽结果应为约 50 步（50000 步对应的是把负载差 50 当成了 bias 变化量，量纲不一致）。｜引文依据：本处为构造数据、无外部来源；自洽核算 0.05 ÷ 0.001 = 50 ≠ 50000。｜修复要求：把 219、231 两行的步数改为与 0.05/0.001 自洽的"约 50 步"（若坚持 50000 步，则须把需要产生的 bias 变化量一并改为与之自洽的数值），改后重新核对两处数值一致、且与"每步 0.001"的倍数关系成立。｜修复：｜复验：

- [重要·技术] blockquote.meta（第 98 行）、来源章节 C2（第 747 行）、F2（第 754 行）：DeepSeek-V3 报告章节号标为 §3.3，该位置并不包含 auxiliary-loss-free 内容，读者按标注定位会落空。｜引文依据：DeepSeek-V3 Technical Report (arXiv:2412.19437) §3.3 标题为 "FP8 Training"（3.3.1 Mixed Precision Framework 等）；auxiliary-loss-free 负载均衡在 §2.1.2 "DeepSeekMoE with Auxiliary-Loss-Free Load Balancing"（原句 "decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded"）。v1、v2 两个版本均为该编号。｜修复要求：把三处 DeepSeek-V3 章节号由 §3.3 改为 §2.1.2；改后按新位置重新核对原文措辞与页面 C2「overload 减 γ / underload 加 γ 等价于 b_j + γ·sign(ℓ̄ − ℓ_j)」的对应关系。｜修复：｜复验：

- [轻微·技术] 4.3 节（第 603 行）："（因 $b = -\beta$，对 $\beta$ 的梯度下降等价于对 $b$ 的 sign 更新，推导见折叠块……）"中的"推导见折叠块"指向 4.2 节的「补充：从对偶目标到 coordinate minimizer」，该折叠块只推导 β 的 coordinate minimizer，不含 b=−β 的符号换算这一步，读者按指引找不到所述推导。｜引文依据：不适用（页面内部指向不成立）。｜修复要求：或在该折叠块补写 b=−β 的换算（对 β 的梯度下降 ↔ 对 b 的 sign 更新），或删除"推导见折叠块"改为直接给出这一步换算。｜修复：｜复验：

- [轻微·格式] 5.1 节（第 688 行"其中 $\beta_j$ 是选中 bin 的索引"、第 671 行代码注释）与 4.2/4.3 节（第 582、592 行）：符号 $\beta_j$ 一符二义——4 章用 $\beta_j$ 表示专家侧对偶乘子（$b_j=-\beta_j$），5 章又用 $\beta_j$ 表示直方图选中 bin 的索引，全页同一写法承载两种含义。｜引文依据：不适用（符号一致性，style-guide §11）。｜修复要求：5.1 的 bin 索引改用不冲突的符号（如 $\hat{j}$ 或 $p_j$），全节统一；4 章双乘子 $\beta_j$ 保持不变。｜修复：｜复验：

- [轻微·表述] 全文多处元话语：第 159 行"在讲 QB 之前，先理解它所处的路由框架……这里只说 QB 直接依赖的部分"；第 263 行"下面逐步拆解。"；第 311 行"公式讲清了，现在用手算例子验证每一步。"；第 359 行"注意被选中的专家……"。｜引文依据：不适用。｜修复要求：改为直接陈述或删除——第 263 行整句删除（3 章流水线图后可直接进入 3.1），第 311 行改为直接给出示例输入，第 159 行删去"在讲……之前""这里只说……"的提示语。｜修复：｜复验：

- [轻微·表述] 章节过渡句在各章末尾重复同一固定句式"本章说明了 X——但 Y——下一章讲 Z"（第 179、223、537、620、712 行）。｜引文依据：不适用（核对 style-guide §8"不使用固定句式，也不为形式完整而添加过渡"）。｜修复要求：保留各章结论到下一章问题的衔接关系，但改写为不同措辞，避免同一模板连续出现。｜修复：｜复验：

- [轻微·技术] 2 章标题（第 201 行）"为什么 896 专家时失效"、description 与第 112、221 行把 sign 更新在 896 专家下的表现写成"失效"。｜引文依据：K3 报告 §2.3.3 原文为 "Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer. Imbalanced routing slows expert-parallel training and may leave some experts poorly trained"，并把固定步长规则的问题表述为 γ "trades off slow adaptation against load oscillation"；来源未称 sign 更新"失效"。｜修复要求：把"失效"改为与来源一致的"更难维持均衡"，或明确写出"收敛慢、均衡附近震荡、部分专家训练不足"这些具体表现；H2 标题、description、第 112/221 行同步。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复（阻断项与重要项关闭前不得发布）

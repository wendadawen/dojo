<!-- review-meta
round: 6
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 1401cf8269a83e0e
-->
# Quantile Balancing 审查记录（第 6 轮）

- 页面版本：a48fad7fcb2810bb5355420bc75af8f470b2297f
- 审查时间：2026-09-13 21:21
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作与任何前序轮次）
- 已完整阅读章节：核心问题、最容易误解、「1. auxiliary-loss-free 路由」（含本章问题）、「2. DeepSeek-V3 的固定步长更新」（含本章问题）、「3. QB 的核心机制」（3.1–3.4 与 3.5–3.9 构造示例、代码块与预期输出、本章问题）、「4. QB 为什么好——从平衡分配到对偶的视角」（含折叠推导、对比表、本章问题）、「5. 训练时的直方图估计与推理冻结」（含直方图代码块、误差/成本、推理冻结表、本章问题）、「来源与范围说明」、结尾总结段
- 来源获取：arXiv:2607.24653v2 全文 HTML（§2.3.3、Appendix C Eq.20–27、Appendix D、Table 1、参考文献表）、HuggingFace moonshotai/Kimi-K3 config.json、arXiv:2412.19437 DeepSeek-V3 技术报告 §2.1.2

## 问题

- [重要·技术] 「来源与范围说明」C2 行与 F2 行：两处都写"K3 报告 §2.3.3 引用 [30]"，把 K3 报告对 sign 更新的引文编号标错。K3 报告 §2.3.3 与 Appendix C 对该规则的引用编号是 [27]，而 [30] 是另一条完全无关的条目（DeepSWE benchmark）。｜引文依据：K3 §2.3.3 原文 "The original method updates 𝒃 with the fixed-step rule b_j^(t+1) = b_j^(t) + γ sign(ℓ̄ − ℓ_j^(t)) [27], for which γ trades off slow adaptation against load oscillation."；同节前文 "Kimi K3 adopts auxiliary-loss-free routing [27]"。参考文献表：[27] = "DeepSeek-AI … (2024) DeepSeek-v3 technical report. … 2412.19437 （Cited by: Appendix C, §2.3.3 …）"；[30] = "(2026) DeepSWE benchmark (Website) Datacurve"。｜修复要求：将 C2、F2 两行中的"引用 [30]"改为"引用 [27]"；其余关于 DeepSeek-V3 §2.1.2 与 sign 更新方向的表述经核对无误（§2.1.2 章标题为 "DeepSeekMoE with Auxiliary-Loss-Free Load Balancing"，原文 "decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded"）。｜修复：｜复验：
- [轻微·表述] <meta name="description"> 末句"QB 一步到位无学习率"：该措辞可读作"一步即收敛到均衡"，而正文 §3.9 与「最容易误解」已明确写"一步收敛不是一般保证""QB 一步就能完美均衡任何负载"是误解；description 与正文的这处口径不一致。来源实际表述为 "equilibrates within a few update steps even for nearly 10^3 experts"。｜引文依据：不适用（正文自身口径）｜修复要求：把 description 中"QB 一步到位无学习率"改为不与正文冲突的措辞（如"QB 直接求解、无学习率类超参数"），使 description 不再暗示一步收敛。｜修复：｜复验：
- [轻微·表述] 结尾总结段（末段）与「5. 训练时的直方图估计与推理冻结」末段：以"本文"为主语的自我指代与元话语——"本文的论断、公式与数字的来源出处及简化条件的边界，见下文「来源与范围说明」"、"本文从 auxiliary-loss-free 路由出发，说明了……开篇「核心问题」列出的五个问题已在前述各章完整作答"。｜引文依据：不适用｜修复要求：删去以"本文"为主语的自我指代句，把"来源出处见…"并入「来源与范围说明」的标题承接或删除；末段改为直接陈述结论（去掉"本文…说明了""已在前述各章完整作答"这类描述文档结构的元话语）。｜修复：｜复验：
- [轻微·技术/表述] 「5.1 直方图估计机制」代码块输入行"输入：每 rank 的本地 margins r_{i,j}（前向传播产出）"：把 r_{i,j} 称作 margins，与同段以及本页 §5.1 正文的定义 r_{i,j} = α_i − s_{i,j} = −margin（required bias）冲突，也与该代码块自己的注释"r = α - s = -margin"冲突。｜引文依据：K3 Appendix D 原文 "we histogram the required bias r_{i,j} := α_i − s_{i,j}"｜修复要求：把该行改为"输入：每 rank 的本地 required bias r_{i,j} = α_i − s_{i,j}（前向传播产出）"，使 r 的命名与正文/注释单义。｜修复：｜复验：

## 核对说明（本轮已回源、判定为无问题的项）

- Eq.13（T_i = argtopk(s_i+b)、p_{i,j}=s_{i,j}/Σ_{r∈T_i}s_{i,r}）、"b 不进入 mixture weight/梯度目标"、"b_j ← b_j + γ·sign(ℓ̄−ℓ_j)"、"Top-(k+1) 取 cutoff"、"旧 bias 只经 cutoff 进入 margin"、"因果性：batch 不用自己推导的 bias"、"推理冻结 bias"：均与 §2.3.3 原文逐句一致。
- Eq.14（b̂_j ← −quantile_{1−k/n}(s_{:,j}−α^(t))；b ← b̂ − mean(b̂)·1）：与 §2.3.3 Eq.14 及 Appendix D 一致。
- 对偶目标（F4）与 Eq.23 完全一致：min_{α,β} ℒ := Σ_{i,j} max(0, s_{i,j}−α_i−β_j) + kΣα_i + (mk/n)Σβ_j；coordinate minimizer 为 (1−k/n) 分位数（Eq.25–26）；∂ℒ/∂β_j = mk/n − Σχ(s_{i,j}−α_i−β_j>0) 即"目标负载减实际负载"（Eq.27）；b=−β 的符号关系、"SignSGD 恢复 DeepSeek-V3 的固定步长更新""几步内平衡近 10^3 专家"——引文编号（Eq.23/25–26/27）与 Appendix C 一致。
- 直方图（§5）：r:=α−s 为 required bias、b̂_j 为 r 的 k/n 分位数、分箱范围 [b_min−1, b_max+1]、w=(b_max−b_min+2)/B、每步重算、本地 scatter-add + 一次整数 all-reduce、从 pooled counts 线性插值 b̂_j = b_min−1+(u_j+clip((q−c_j)/h_j,0,1))w、B=1000 时误差"几个 10^-3"、通信 nB 且与 m 无关、<1% raw-margin 成本、EMA 降采样噪声——逐条与 Appendix D 一致。
- config.json：num_experts=896、num_experts_per_token=16、moe_router_activation_func="sigmoid"、topk_method="noaux_tc" 均与页面一致；K2 384 → K3 896 与 Table 1（Routed Experts 384 / 896）一致。
- 构造示例（m=8,n=4,k=1）：分数矩阵、cutoff、8×4 margin 表、四组降序 margins 与"第 3 大"、b̃=(-0.2,-0.1,0.0,0.1)、mean=-0.05、b=(-0.15,-0.05,0.05,0.15)、新 biased score 表与新负载 (2,2,2,2)、改道 T4/T5/T8——逐步复算无误；与 K3 Fig.5（m=8,n=4,k=1、负载 (4,3,1,0)→(2,2,2,2)、q=2、(q+1)=第 3 大）一致。
- 页面代码块实际执行，输出与页面"预期输出"逐行完全一致（loads=[4,3,1,0]→[2,2,2,2]，changed=T4(E1->E3), T5(E1->E4), T8(E2->E4)）。
- 章3「本章问题」第 1 题（把 T1 的 E3 分数改为 0.75）：cutoff 仍 0.7、E3 在 T1 的 margin 变为 +0.05、E3 降序第 3 大仍 0.0、b̃_3=0——复算无误。
- 页面内链 ../moe-serving/index.html、../aux-loss-free-routing/index.html 均真实存在；overview.html 与 index.html 互相链接；`python3 .dojo/scripts/validate.py wiki/quantile-balancing/index.html` 返回 "validation ok"；正文/列表/表格/标题无 Unicode 数学字符（仅伪代码块内出现 ∈ × ≥ ⌈⌉ α 等，属代码清单，非渲染正文）；无图片 alt 含 `$...$`；引用文件路径均为仓库中真实存在者。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（1 条重要 + 3 条轻微待关闭；技术结论与全部数字/公式经回源复核无误，无阻断问题）
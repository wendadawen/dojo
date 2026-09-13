<!-- review-meta
round: 7
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 1ed52b3ac028fd45
-->
# Quantile Balancing 审查记录（第 7 轮）

- 页面版本：579bf2456aa66a614f19480f23ecfe40d36c209a（wiki/quantile-balancing/index.html，工作树）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查；未读取本页 research/ 下任何文件）
- 已完整阅读章节：页面开头（主要依据、引言、核心问题、最容易误解）→ 1. auxiliary-loss-free 路由——bias 做什么、不做什么（含本章问题）→ 2. DeepSeek-V3 的固定步长更新——为什么 896 专家时更难维持均衡（含本章问题）→ 3. QB 的核心机制——从一次前向推导下一个 bias（3.1–3.9、代码折叠块、本章问题）→ 4. QB 为什么好——从平衡分配到对偶的视角（4.1–4.3、补充折叠块、本章问题）→ 5. 训练时的直方图估计与推理冻结（5.1–5.3、代码折叠块、本章问题）→ 来源与范围说明

## 来源核对（本轮实际抓取原文定位）

- 路由与 QB 更新：K3 报告 arXiv:2607.24653 §2.3.3 原文 “𝒯_i=argtopk(𝒔_i+𝒃), p_{i,j}=s_{i,j}/∑_{r∈𝒯_i}s_{i,r}”（Eq.13）、“b̂_j^{(t+1)}←−quantile_{1−k/n}(𝒔_{:,j}−𝜶^{(t)})”、“𝒃^{(t+1)}←𝒃̂^{(t+1)}−mean(𝒃̂^{(t+1)})𝟏”（Eq.14）、“The margins subtract the biased cutoff α_i^{(t)} from the raw score s_{i,j}, so the old bias enters the update only through the cutoffs”。页面 §1/§3 的 Eq.13/14、符号表与文字一致。
- “几步内收敛”：Appendix C 末段原文 “This view explains both why QB requires no learning-rate-like hyperparameter and why it equilibrates within a few update steps even for nearly 10^3 experts.” —— 页面 §3.9、§4.3 表与「最容易误解」的 [C1] 引用成立（重点复核项，确认有来源，非无据）。
- 对偶推导：Appendix C Eq.20 “max_{x_{i,j}∈{0,1}} Σ x_{i,j}s_{i,j} s.t. Σ_j x_{i,j}=k, Σ_i x_{i,j}=mk/n（assumed integral）”、Eq.23 “min_{α,β} ℒ(α,β):=Σ_{i,j}max(0,s_{i,j}−α_i−β_j)+kΣ_iα_i+(mk/n)Σ_jβ_j”、Eq.25/26 “α_i*=quantile_{1−k/n}(s_i−β)”“β_j*=quantile_{1−k/n}(s_{:,j}−α)”、Eq.27 “∂ℒ/∂β_j=mk/n−Σ_iχ(s_{i,j}−α_i−β_j>0)” 及 “A SignSGD step on this objective recovers the fixed-step sign update … up to the sign convention b=−β”。页面 §4.1–4.3 与折叠块的逐式推导、b_j←b_j+γ·∂L_j/∂β_j 的取负推导均与之一致。
- 直方图：Appendix D 原文 “r_{i,j}:=α_i−s_{i,j}，the QB target b̂_j of Eq.14 is exactly the (k/n)-quantile of r_{:,j}”、“Every r_{i,j} therefore falls in [b_min−1, b_max+1]”、“w=(b_max−b_min+2)/B”、“with B=1000 this is at most a few 10^{−3}, and we observe no measurable residual load imbalance”、“one integer all-reduce of nB values per layer per step, independent of m, which in our configuration is below 1% of the cost of exchanging the raw margins”、“maintaining an exponential moving average of the estimated quantiles across steps reduces batch-to-batch sampling noise”。页面 §5.1–5.3、N2/N3、C16 与之一致；恢复公式 ã 与 “b̂_j=b_min−1+(β_j+clip((q−c_j)/h_j,0,1))w” 一致。
- 896/16 与 K2：config.json 实测 `num_experts: 896`、`num_experts_per_token: 16`、`moe_router_activation_func: "sigmoid"`、`topk_method: "noaux_tc"`；K3 报告 Table 1 “Routed Experts 384 → 896”“Experts Active per Token 8 → 16”。页面 §1/§2 与 C17/N1 一致。
- DeepSeek-V3 sign 更新：K3 报告 §2.3.3 写作 “b_j^{(t+1)}=b_j^{(t)}+γ·sign(ℓ̄−ℓ_j^{(t)}) [27]”；DeepSeek-V3 报告（arXiv:2412.19437）§2.1.2 原文为离散式（“decrease the bias term by γ if … overloaded”“increase it by γ if … underloaded”，γ=0.001）与 sign 式等价。页面 C2/F2 已明确写出两者的等价关系，成立。
- 代码：§3 的 Python 脚本实际执行，输出与页面「预期输出」逐行一致（初始 loads=[4,3,1,0]；E1/E2/E3/E4 第 3 大 margin=+0.2/+0.1/+0.0/−0.1；b̃=(−0.2,−0.1,−0.0,+0.1)，mean=−0.05，b=(−0.15,−0.05,+0.05,+0.15)；新 loads=[2,2,2,2]；改变 T4/T5/T8）。
- 手算表逐格复算无误：分数矩阵 → α（如 T1 α=0.7）；margin 表（T1: +0.0/−0.4/+0.1/−0.2）；分位数表（E3 8 个 margin 降序 +0.1,0.0,0.0,0.0,−0.1,−0.2,−0.5,−0.5，第 3 大 0.0）；mean-centering 与验证表（T6→E2、T7→E1、T4/T5/T8 改道）全部自洽。
- 链接与结构：../moe-serving/index.html、../aux-loss-free-routing/index.html 均存在且标题与链接文字相符；无「（待生成）」占位；.dojo/scripts/validate.py 返回 validation ok；KaTeX/Prism/主题 CSS 均为本地资源；核心问题与各章本章问题均有解答折叠块且答案独立可读、指明完整论证章节。

## 问题

- [轻微·技术] §4.2 折叠块「从对偶目标到 coordinate minimizer」：用 `$m_i = s_{i,j} - \alpha_i$` 记 margin，与全文 `$m$`（token 数）同形异义，且与同一式中的 `$\frac{mk}{n}$` 同屏出现。｜引文依据：K3 报告 Appendix C 直接写 $s_{i,j}-\alpha_i$，未引入 $m_i$；Eq.24/27 中 m 恒为 token 数（“the token count m spans millions of tokens”）。｜修复要求：将该符号改记作 `$\mu_i$`，或直接写 `$s_{i,j}-\alpha_i$`，使 `$m$` 全文单义。｜修复：｜复验：
- [轻微·技术] §4.3 对比表「超参数」行把 QB 写作「无超参数」。｜引文依据：K3 报告 Appendix C “why QB requires no learning-rate-like hyperparameter”（限定语为 learning-rate-like）；且页面 §5 的直方图 bin 数 $B$ 本身是设计参数。正文其余各处均写「无学习率类超参数」，仅表格放宽。｜修复要求：该格改为「无学习率类超参数」，与正文及来源一致。｜修复：｜复验：
- [轻微·格式] 目录项截断产生悬空 `$`。页面内联脚本以 `h.textContent.slice(0, 30)` 生成目录文字：h3「3.3 步骤 3：取 $(1-k/n)$ 分位数得 $\tilde{b}$」被截为「3.3 步骤 3：取 $(1-k/n)$ 分位数得 $\ti」（第 27 个字符处落单 `$`，无配对），h3「3.1 步骤 1：Top-$(k+1)$ 路由得 cutoff $\alpha_i$」被截为「…cutoff cuto」。正文标题自身渲染正常。｜引文依据：不适用（页面自身脚本）。｜修复要求：缩短这两个 h3 标题，或在生成目录时按 `$` 成对边界截断，使目录项不含悬空 `$`、不截断到公式中间。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（三条轻微问题均不影响正确性与主线理解；来源论断逐条有引文依据，公式可复算，代码输出与页面一致）
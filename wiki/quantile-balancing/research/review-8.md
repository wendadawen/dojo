<!-- review-meta
round: 8
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 7975b970fce50c22
-->
# Quantile Balancing 审查记录（第 8 轮）

- 页面版本：f829a29151f5d686bce39daa97fc0f270b8bfe5a（index.html 工作树哈希）
- 审查时间：2026-09-14 17:10
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：页面开头（meta/引言）、核心问题、最容易误解、1. auxiliary-loss-free 路由、1 本章问题、2. DeepSeek-V3 的固定步长更新、2 本章问题、3. QB 的核心机制（含 3.1–3.9 各小节、表格、代码折叠块与图注）、3 本章问题、4. QB 为什么好（含对偶目标折叠块）、4 本章问题、5. 训练时的直方图估计与推理冻结、5 本章问题、来源与范围说明、总结段

## 来源核对依据

- 官方材料：Kimi K3 Technical Report（`https://github.com/MoonshotAI/Kimi-K3/raw/main/k3_tech_report.pdf`，并比对了 arXiv:2607.24653 v1 与 v2）与 `https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`；DeepSeek-V3 Technical Report（arXiv:2412.19437）。
- §2.3.3 原文片段：「The original method updates b with the fixed-step rule b_j^(t+1) = b_j^(t) + γ sign(ℓ̄ − ℓ_j) [30]」；Eq.13「T_i = argtopk(s_i + b), p_{i,j} = s_{i,j} / Σ_{r∈T_i} s_{i,r}」；Eq.14「b̃_j^(t+1) ← − quantile_{1−k/n}(s_:,j^(t) − α^(t))；b^(t+1) ← b̃^(t+1) − mean(b̃^(t+1))·1」；Fig.5「m = 8 tokens, n = 4 routed experts, and k = 1 … produces loads (4, 3, 1, 0) … (c) The retained choices yield the balanced load (2, 2, 2, 2)」。
- Appendix C：Eq.20 最大分数均衡分配、Eq.23 对偶目标「min L(α,β) := Σ max(0, s_{i,j} − α_i − β_j) + k Σ α_i + (mk/n) Σ β_j」、Eq.25/26「α_i* = quantile_{1−k/n}(s_i − β)」「β_j* = quantile_{1−k/n}(s_:,j − α)」、Eq.27「∂L/∂β_j = mk/n − Σ χ(s_{i,j} − α_i − β_j > 0)」；「equilibrates within a few update steps even for nearly 10^3 experts」。
- Appendix D：「we histogram the required bias r_{i,j} := α_i − s_{i,j}」「(k/n)-quantile of r_:,j」「range … [b_min − 1, b_max + 1]」「w = (b_max − b_min + 2)/B」「select the first bin whose cumulative count reaches ⌈q⌉ and interpolate linearly」「with B = 1000 this is at most a few 10^−3」「below 1% of the cost of exchanging the raw margins」「maintaining an exponential moving average … reduces batch-to-batch sampling noise」。
- Table 1：「Routed Experts — Kimi K2: 384 / Kimi K3: 896」。
- config.json：`num_experts: 896`、`num_experts_per_token: 16`、`moe_router_activation_func: "sigmoid"`、`topk_method: "noaux_tc"`。
- DeepSeek-V3 §2.1.2「DeepSeekMoE with Auxiliary-Loss-Free Load Balancing」：「we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded」。
- 公式与示例复算：页面手算代码（第 403–529 行折叠块）在 Python 3 下实际执行，输出与页面「预期输出」逐行一致（初始 loads=[4,3,1,0]；E1–E4 第 3 大 margin = +0.2/+0.1/+0.0/−0.1；b_tilde = [−0.2,−0.1,−0.0,+0.1]；b_new = [−0.15,−0.05,+0.05,+0.15]；QB 后 loads=[2,2,2,2]；改变 token = T4(E1→E3), T5(E1→E4), T8(E2→E4)）。分数矩阵 α、margins、分位数、mean-centering、新路由各表均已逐格复算无误；3 本章问题第 1 题的改写结果（E3 降序 +0.05,0,0,0,−0.1,−0.2,−0.5,−0.5，第 3 大 0.0，b̃_3 不变）亦复算正确。
- 机械项：`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 `validation ok`；dojo:topics=模型结构 在 AGENTS.md 固定大类表内；概念链接 `../moe-serving/index.html`、`../aux-loss-free-routing/index.html` 均真实存在；页面 21 个 details（5 核心问题 + 12 本章问题解答 + 2 补充 + 2 代码）齐全；无 img/alt 含 `$...$`；无 Unicode 数学字符出现在标题、summary、正文段落、列表或表格（`<pre><code>` 内的 α、×、⌈⌉、∈、b̃ 属 style-guide §11 豁免的代码块）；所有 `.dg-flow/.dg-node/.callout-yellow/.highlight-row` 等类均在 libs/dojo-concept.css 中定义。

## 问题

- [阻断·技术] 「来源与范围说明 › 论断与来源（C）」中 C2 条与「公式与来源（F）」中 F2 条：页面称 K3 报告的 sign 更新出处是「K3 报告 §2.3.3 引用 [27]」，与所引版本（arXiv:2607.24653 v1/v2 及官方 GitHub PDF）的文献表不符。K3 §2.3.3 引用的是 [30]（DeepSeek-V3），[27] 是无关论文。｜引文依据：K3 报告 §2.3.3 原文「The original method updates b with the fixed-step rule b_j^(t+1) = b_j^(t) + γ sign(ℓ̄ − ℓ_j) [30]」「Kimi K3 adopts auxiliary-loss-free routing [30]」；该报告文献表 `[27] Soham De et al. Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.`、`[30] DeepSeek-AI et al. DeepSeek-V3 Technical Report. 2024. arXiv: 2412.19437 [cs.CL].`（v1、v2 与 GitHub main 三个版本一致，均为 [30]，不存在编号漂移）。｜修复要求：把 C2、F2 两处的「[27]」改为「[30]」，或删去该编号只保留「K3 报告 §2.3.3 引用 DeepSeek-V3（[30]）」。｜修复：回源核对结论为「不成立」，但已做防复发处理。按 **arXiv:2607.24653v2** 文献表逐条核对：[27] = DeepSeek-AI 等《DeepSeek-V3 Technical Report》（arXiv:2412.19437），[26] = Soham De 等 Griffin，[30] = DeepSWE benchmark（Datacurve）；§2.3.3 原句「the fixed-step rule … [27]」「the update takes effect only in the next step [27]」均为 [27]；v1（arXiv HTML）与 v2 一致，无 v1/v2 漂移。审查所述「v1/v2 与 GitHub main 三者一致，均为 [30]」与原文不符：官方 GitHub main PDF 编号确实不同（[27] = Griffin、[30] = DeepSeek-V3、[31] = DeepSWE），审查实际读的是该 PDF 版。本页与姊妹页 `wiki/kimi-k3/index.html` 均以 arXiv:2607.24653v2 为准（该页 F6 亦写「[27] 对应 DeepSeek-V3 technical report」），编排者亦已就此编号争议裁定「[27]→[30] 为误判」（`wiki/kimi-k3/research/ref-numbering-ruling.md`）。故保留 [27]，并把 C2、F2 两处改为「K3 报告 §2.3.3 引用 DeepSeek-V3（arXiv:2607.24653v2 文献表编号 [27]）」——同时点名被引对象与版本，编号不再依赖版本。｜复验：

- [重要·可读性] 「1. auxiliary-loss-free 路由」末尾的「本章问题」第 1 题：题面问「如果两个专家的 bias 同时加 $0.1$，Top-k 选择会变吗？」，解答却按「所有专家的 bias 同时加常数」作答并断言「都不变」。只给两个专家各加 $0.1$ 会抬高它们相对其余专家的排序，Top-k 集合一般会改变，因此按题面字面理解，解答的「Top-k 集合不变」为假——题目与解答互相矛盾。｜引文依据：不适用（同页第 1 章正文写的是「如果所有专家的 bias 同时加常数 $c$，则 $s_{i,j} + b_j + c = (s_{i,j} + b_j) + c$，Top-k 选择不变」，与解答一致、与题面不一致）。｜修复要求：把题面改为「如果所有专家的 bias 同时加 $0.1$」（与正文与解答保持一致），或改写解答分别说明「全体平移不变、仅子集偏移会改变 Top-k」。｜修复：采纳（改题面）。题面「如果两个专家的 bias 同时加 $0.1$」改为「如果所有专家的 bias 同时加 $0.1$」，与本章正文「如果所有专家的 bias 同时加常数 $c$，…Top-k 选择不变」及解答「所有专家的 bias 同时加常数」一致。｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 0
- 处置：修复

## 备注（本轮未构成问题的核对项）

- 手算示例设定（$m=8,n=4,k=1$，$(4,3,1,0)\to(2,2,2,2)$）与 K3 报告 Fig.5 一致，且页面已在「构造示例」「辅助解释与类比边界」「简化条件及其限制」中明确标注为人为构造、不可外推。
- 「近 $10^3$ 专家时几步内收敛」在正文、summary、overview 之间一致；896 专家、top-16、`noaux_tc`、sigmoid 与 config.json 一致；K2 384 专家与 Table 1 一致。
- 对偶目标、coordinate minimizer、sign 次梯度的符号推导（$b=-\beta$、$b_j \leftarrow b_j + \gamma\,\partial L_j/\partial\beta_j$）自洽且与 Appendix C Eq.23/25–27 吻合。
- 表述维度通读（含全部折叠块与图注）未发现元话语、以「本页」为主语的自我指代、会话指代、调试叙事或临场评价。

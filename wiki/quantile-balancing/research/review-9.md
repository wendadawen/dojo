<!-- review-meta
round: 9
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: e61ae6e1c7a40969
-->
# Quantile Balancing 审查记录（第 9 轮）

- 页面版本：2a2cd9be50883f758d171370c690d5ab4b64334e（wiki/quantile-balancing/index.html 工作树哈希）
- 审查时间：2026-09-14 17:43
- 审查者：独立子代理（未参与写作，未读取本页 research/）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. auxiliary-loss-free 路由 → 2. DeepSeek-V3 的固定步长更新 → 3. QB 的核心机制（3.1–3.9，含代码折叠块）→ 4. QB 为什么好（4.1–4.3）→ 5. 训练时的直方图估计与推理冻结（5.1–5.3）→ 来源与范围说明（含论断/公式/数字三小节、构造示例、类比边界、简化条件）

## 核对来源版本

- Kimi K3 Technical Report = **arXiv:2607.24653v2**，同时核对两种渲染：LaTeX/PDF 版（arxiv.org/pdf/2607.24653v2）与 arXiv HTML 版（arxiv.org/html/2607.24653v2）。逐条定位 §2.3.3 Eq.13–14、Appendix C Eq.20–27 + Algorithm 1、Appendix D、Fig.5、Table 1。
- DeepSeek-V3 Technical Report = **arXiv:2412.19437v2** §2.1.2。
- HuggingFace **moonshotai/Kimi-K3** config.json（text_config 段）。
- 页内 Python 代码用 python3 实际执行。

### 关键论断核对（原文片段）

- Eq.13 路由：K3 §2.3.3 "$T_i=\mathrm{argtopk}(s_i+b),\ p_{i,j}=s_{i,j}/\sum_{r\in T_i}s_{i,r}$"；"Because $b$ is omitted from $p_{i,j}$, it regulates dispatch without altering the mixture weights or the gradient-based optimization of the router." 与页面 §1 一致。
- sign 更新：K3 §2.3.3 "The original method updates $b$ with the fixed-step rule $b_j^{(t+1)}=b_j^{(t)}+\gamma\,\mathrm{sign}(\bar\ell-\ell_j^{(t)})$ [30], for which $\gamma$ trades off slow adaptation against load oscillation." 与页面 §2 公式、γ 两难表述一致；DeepSeek-V3 §2.1.2 原文确认方向（overloaded 减 γ、underloaded 加 γ，γ 名为 "bias update speed"）。
- 896/16：K3 §2.3.3 "Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer"；Table 1 "Routed Experts 384 / 896"、"Experts Active per Token 8 / 16"；config.json `num_experts: 896`、`num_experts_per_token: 16`、`moe_router_activation_func: "sigmoid"`、`topk_method: "noaux_tc"`。页面 §2、C17、N1 一致。
- Eq.14：K3 §2.3.3 "$\hat b_j^{(t+1)}\leftarrow-\mathrm{quantile}_{1-k/n}(s_{:,j}-\alpha^{(t)})$；$b^{(t+1)}\leftarrow\hat b^{(t+1)}-\mathrm{mean}(\hat b^{(t+1)})\mathbf 1$"，与页面 §3.3/§3.4 公式逐字一致；"the old bias enters the update only through the cutoffs"、"a batch is never routed with a bias derived from itself" 与 §3.2/因果性 callout 一致。
- Fig.5：K3 "with m = 8 tokens, n = 4 routed experts, and k = 1 … produces loads (4, 3, 1, 0) … yield the balanced load (2, 2, 2, 2)"，与页面构造示例设定及 §3.9 结论一致。
- 收敛步数：K3 Appendix C "why it equilibrates within a few update steps even for nearly $10^3$ experts"，与页面 [C1]、核心问题答案、§4.3 表格一致。
- 对偶：K3 Eq.23 与页面 §4.2 目标函数逐字一致；Eq.25 "$\alpha_i^*=\mathrm{quantile}_{1-k/n}(s_i-\beta)$"、Eq.26 "$\beta_j^*=\mathrm{quantile}_{1-k/n}(s_{:,j}-\alpha)$" 与页面表述一致；Eq.27 "$\partial L/\partial\beta_j=mk/n-\sum_i\chi(s_{i,j}-\alpha_i-\beta_j>0)$" 与页面 §4.3 一致；"A SignSGD step on this objective recovers the fixed-step sign update … up to the sign convention $b=-\beta$" 与页面 §4.3/折叠块推导一致；Eq.20 平衡分配、b-matching 整性、Algorithm 1 均与页面 §4.1 一致。
- 直方图：Appendix D 分箱范围 "$s_{i,j}\in(0,1)$ … $\alpha_i$ … lies in $(b_{\min},1+b_{\max})$ … every $r_{i,j}$ therefore falls in $[b_{\min}-1,b_{\max}+1]$"、bin width "$w=(b_{\max}-b_{\min}+2)/B$"、恢复式 "$\hat b_j=b_{\min}-1+(\beta_j+\mathrm{clip}(\frac{q-c_j}{h_j},0,1))w$"、"the target rank is exactly the target load $q=mk/n$ … select the first bin whose cumulative count reaches $\lceil q\rceil$"、Properties "with $B=1000$ this is at most a few $10^{-3}$, and we observe no measurable residual load imbalance"、"below 1% of the cost of exchanging the raw margins"、"an exponential moving average of the estimated quantiles … reduces batch-to-batch sampling noise"，全部与页面 §5.1–5.3 及 [C14,F7]/N2/N3/C16 一致。
- 代码：页面折叠块中的 Python 实际执行，输出与"预期输出"逐行一致（初始 loads=[4,3,1,0]，b_tilde=(-0.2,-0.1,-0.0,+0.1)，b_new=(-0.15,-0.05,+0.05,+0.15)，改动 token T4/T5/T8，新 loads=[2,2,2,2]）。
- 手算三表 + 新路由表 + mean-centering 全部按 Eq.14 流程复算一致；§3.5 分数矩阵的 Top-1/Top-2 与 α 列自洽。
- 引用编号回文献表逐条确认：页面 C2/F2 标注 "arXiv:2607.24653v2 文献表编号 [27]"。经核对，v2 的 **HTML 渲染**文献表中 [27] = "DeepSeek-AI, A. Liu, B. Feng, …"（含 2412.19437），与页面一致；v2 的 **PDF/LaTeX 渲染**中同一文献为 [30]（[27] 是 Griffin）。两种渲染均出自 v2，页面标注对其一为真，故不列为问题，仅在此登记版本差异。
- 机械项：validate.py 返回 "validation ok"；页面所有本地链接（含 ../aux-loss-free-routing/index.html，目录名确为 aux-loss-free-routing）均存在；无 "（待生成）" 占位；无 Unicode 数学字符（→ 为站内既有叙事写法，54 页共用，不在禁列）；无 img，故无 alt 含 $...$；结构图 dg-flow 为 HTML 结构、无等宽字符框线、公式在 div 中由 KaTeX 渲染，无脚本时可读；页面级核心问题 5 题、章节级本章问题 13 题全部有解答折叠块且独立可读。

## 问题

- [轻微·技术] 「来源与范围说明」C2（index.html:742）：以「原文：」引出 `$b_j \leftarrow b_j - \gamma$` / `$b_j \leftarrow b_j + \gamma$` 两条箭头式，但 DeepSeek-V3 §2.1.2 并无该公式，只有文字表述，「原文」标签与来源体裁不符。｜引文依据：arXiv:2412.19437v2 §2.1.2 "we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded, where γ is a hyper-parameter called bias update speed"（该节无编号公式、无 sign 形式、无 ℓ 符号）；箭头公式实为 K3 §2.3.3 "$b_j^{(t+1)}=b_j^{(t)}+\gamma\,\mathrm{sign}(\bar\ell-\ell_j^{(t)})$"。｜修复要求：把「原文：」改为「即：」，或直接引上句英文原文；方向语义与来源一致，公式内容不动。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（仅 1 条轻微标注问题，不影响正确性与主线；本页第 9 轮残余阻断 / 重要均为 0）
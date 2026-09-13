<!-- review-meta
round: 5
page: wiki/gated-deltanet/index.html
reviewed_content_sha256: eac4f71a00e0b8c5
-->
# Gated DeltaNet 审查记录（第 5 轮）

- 页面版本：2e4ce8a698e589252d23b1a7784eaf91ea563753（git hash-object wiki/gated-deltanet/index.html）
- 审查时间：2026-09-13 20:17 CST
- 审查者：独立子代理（编排者派发，未参与写作与前序审查）
- 依据规范：guides/concept/check.md（head 中 dojo:type=concept）
- 已完整阅读章节：核心问题、最容易误解、1. DeltaNet 与 Mamba2 各自缺什么、2. Gated DeltaNet 的公式与符号（2.1 α_t 与 β_t 的职责分工、2.2「先衰减后擦写」的等价解读）、3. 手算 Gated DeltaNet 一步更新（3.1 三模型对比）、4. 退化关系与并行训练算法（4.1 退化、4.2 并行训练算法含代码折叠块）、5. 与 KDA 的关系及实验效果（5.1、5.2、5.3）、来源与范围说明；并通读 overview.html 与全部折叠块、图注。
- 机械核验：`.dojo/scripts/validate.py wiki/gated-deltanet/index.html` 返回 validation ok；dojo:topics=注意力机制、dojo:tag=注意力 均在 AGENTS.md／catalog_builder.py 词表内；前置链接 delta-rule、linear-attention、kda、qwen3-5-dataflow 均存在；手算三步步逐一复算与正文一致，代码块实跑输出与「预期输出」完全一致（含两处退化 True）。

## 问题

- [阻断·技术] §4.2 折叠块「补充：WY 表示与 chunkwise 形式的作用（不展开推导）」：给出的 P^{[t]}_r 公式比来源多了一个因子 (γ^i/γ^r)，与来源公式不符且数学上不等价｜引文依据：arXiv:2412.06464v1 §3.2 原文为 "We adapt the WY representation in Eq. 4-5 to incorporate the decay term as below, 𝐏_{[t]}^r = γ_{[t]}^r ( 𝐈 − ∑_{i=1}^r 𝐰_{[t]}^i 𝒌_{[t]}^{i⊺} )"（KaTeX 源：\gamma_{[t]}^{r}\left(\mathbf{I}-\sum_{i=1}^{r}\mathbf{w}_{[t]}^{i}\bm{k}_{[t]}^{i\intercal}\right)）；带 γ^r/γ^i 因子的是紧邻的另一个公式 Eq.9：𝐇_{[t]}^r = ∑_{i=1}^r \frac{\gamma_{t}^{r}}{\gamma_{t}^{i}} 𝐮_{[t]}^i 𝒌_{[t]}^{i⊺}。页面写成 P^{[t]}_r = γ^r(I − ∑_{i=1}^r (γ^i/γ^r) w^i k^{i⊺})｜修复要求：删去 P 公式求和号内的 (γ^i/γ^r)，恢复为 P^{[t]}_r = γ_{[t]}^r(I − ∑_{i=1}^r w_{[t]}^i k_{[t]}^{i⊺})｜修复：｜复验：

- [重要·技术] §5.2 表头与「来源与范围说明 N3」：把论文 Table 2 的 "Avg." 列标为「常识推理 7 项平均」，但该列是对 8 项准确率（LAMBADA acc + 7 项常识推理）取平均，对页面列出的 7 项单独求平均得不到标注数值｜引文依据：arXiv:2412.06464v1 Table 2 列头 ['Model','Wiki.','LMB.','LMB.','PIQA','Hella.','Wino.','ARC-e','ARC-c','SIQA','BoolQ','Avg.']；Gated DeltaNet 行 '16.42 12.17 46.65 72.25 55.76 57.45 71.21 38.39 40.63 60.24 55.32' → 7 项（72.25,55.76,57.45,71.21,38.39,40.63,60.24）均值 = 56.56，8 项（再加 46.65=LAMBADA acc）均值 = 55.32；Mamba2 行 '16.56 12.56 45.66 71.87 55.67 55.24 72.47 37.88 40.20 60.13 54.89' → 7 项均值 56.21 ≠ 标注 54.89｜修复要求：§5.2 表头与 N3 改为与来源一致的口径——写明该列是论文 "Avg."（含 LAMBADA 准确率），或改列 7 项实算均值 56.56（GDN）／56.21（Mamba2）并注明口径｜修复：｜复验：

- [重要·技术] §2 符号表、§2.1、§5.1、最容易误解、简化条件：把 Gated DeltaNet 的 α_t 写成 σ(W_α x_t)，来源未给出该取值形式；论文全文 0 处 sigmoid，并注明 α 沿用 Mamba2 的参数化，官方实现用的是 Mamba2 的 softplus 门（默认分支），不是 σ(W_α x_t)｜引文依据：arXiv:2412.06464v1 全文 grep "sigmoid" = 0，§3.3 脚注 "We use Mamba2's parameterization for α but omit it for brevity"；论文 §3.1 仅给 "the data-dependent gating term α_t ∈ (0,1) controls state decay"；NVlabs/GatedDeltaNet lit_gpt/gated_delta_net.py："gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)"（默认 use_mamba_gate=True），非默认分支为 "gk = F.logsigmoid(gk) / self.gate_logit_normalizer"；fla/layers/gated_deltanet.py 同样用 self.A_log、dt_bias、注释 "Inverse of softplus"；Kimi K3 Technical Report §2.1.1 "Following GDN and Mamba-2, Kimi Linear uses the negative-Softplus mapping g = −e^A Softplus(z) ∈ (−∞,0)"｜修复要求：把 α_t 的取值形式改为来源支持的口径——保留「α_t ∈ (0,1) 数据相关」（论文 §2.2／§3.1），参数化注明沿用 Mamba2（负 softplus／指数衰减），或明确标注为教学简化；§5.1 的对照「GDN 用 sigmoid → KDA 改 scaled sigmoid」应改为「GDN 沿用 Mamba2 的负 softplus（无下界） → KDA 用 scaled sigmoid（有下界）」，与 K3 报告图 3 一致｜修复：｜复验：

- [轻微·表述] §3 正文与 §4.1 折叠块：「注意 $S_2$ 在两个正交方向各存了一个关联」「注意完整 Mamba2 还有 $+v_t k_t^\top$ 写入项」（另 §4.1 章节问题解答内一处）为对读者的祈使式元话语（与「需要注意的是」同类）｜引文依据：不适用｜修复要求：删去「注意」，改为直陈句（如「$S_2$ 在两个正交方向各存一个关联」；「完整 Mamba2 另有 $+v_t k_t^\top$ 写入项」）｜修复：｜复验：

- [轻微·格式] §5.2 两个实验表的表头单元格内直接出现 Unicode 箭头 ↑／↓（「Wiki PPL↓」「LMB PPL↓」「常识推理 7 项平均↑」），未用 LaTeX 书写｜引文依据：不适用（guides/concept/style-guide.md 第 107 行：页面任何位置的数学运算符与关系符须包在 $...$ 内）｜修复要求：表头方向记号改为 KaTeX 行内公式（$\downarrow$／$\uparrow$）或改用文字（「越低越好」「越高越好」）｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 2
- 处置：修复（阻断与重要须全部关闭后再发布；B 位于折叠补充块内但为公式与来源不符，按规则计阻断）

## 已复现、未列入问题的核验（供下轮参考）

- 手算主例三步（S_1=[[1,0],[0,0]]、S_2=I、S_3=[[0,0],[1,0.5]]）、§3.1 三模型对比表、§3.1 本章问题解答的 S_t=[[0.5,1],[0,0]] 均逐一复算正确。
- 代码块实跑输出与页面「预期输出」逐字符一致，含两处退化 np.allclose 为 True。
- Table 2 的 PPL（16.42/12.17、16.56/12.56、17.71/16.88、18.53/18.32、16.07/12.12、15.91/12.55）与 Table 3 的 S-NIAH-2/3 4K（18.6/56.2/92.2 与 22.4/4.6/27.6）与论文一致。
- K3 config.json（full_attn_layers 24 层、kda_layers 69 层、gate_lower_bound=−5.0）、Qwen3.5-397B-A17B config.json（full_attention_interval=4、60 层、attn_output_gate=true）、Qwen3-Next 3:1 与 80B/A3B 均与页面一致。
- Kimi K3 Technical Report §2.1.1 Eq.1、Eq.5 与页面 §5.1 的 KDA 公式、$S_t \in \mathbb{R}^{d_k \times d_v}$、scaled sigmoid 下界 $g_{\min}=-5$、$\alpha > e^{-5} \approx 6.7\times10^{-3}$、full-rank output gate 全部核对一致；「needle 落在真实文本中」有论文原文支撑（"needles are grounded in real-world text data"）。

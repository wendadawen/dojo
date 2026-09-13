<!-- review-meta
round: 4
page: wiki/latent-moe/index.html
reviewed_content_sha256: 86bca6285633e8d2
-->
# LatentMoE 审查记录（第 4 轮）

- 页面版本：3b6a908edae9e0ab009b0dde2cac4a64a77c87b8
- 审查时间：2026-09-13 19:42
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 扩大专家池的代价 / 2. LatentMoE 的层结构 / 3. 压缩比与 Reinvestment / 4. 压缩下限与投影成本 / 5. 与 Stable LatentMoE 的关系 / 来源与范围说明（含全部折叠块、两处数据流图与全部表格）。overview.html 一并通读。
- 来源获取：arXiv:2601.18089（ar5iv/alphaxiv 全文，逐节核对 Figure 1 caption、§2 五条设计原则、§3 两变体定义、§4.1 消融、§4.3.1 万亿参数投影）；Kimi K3 技术报告 arXiv:2607.24653 §2.3（ar5iv 全文）；官方 config.json（moonshotai/Kimi-K3，经检索核对）；Sebastian Raschka "Latent MoE" 与 "Nemotron 3 Ultra and Latent MoE Scaling"。页面代码已实际执行。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/latent-moe/index.html` → validation ok；页面内嵌 Python 代码实跑输出与页面「预期输出」逐行一致（dispatch 32768/8192、65536/16384；C(8,2)=28、C(32,8)=10518300、增长 375653.57；投影参数量 8388608）；无 research/ 死链，无「（待生成）」占位。

## 问题

- [重要·技术] 来源章节 F4（及 C4）：reinvestment 公式「$N \to \alpha N$、$k \to \alpha k$」标注来源为 [L] Figure 1 caption，但该 caption 原文只给出 $d/\ell$ 缩小因子，不含 $N$、$k$ 的放大关系。｜引文依据：[L] Figure 1 caption 全文为 "Standard MoE vs. LatentMoE architectures. In LatentMoE, tokens are projected from the model hidden dimension d into a smaller latent dimension ℓ for expert routing and computation, which reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ."；放大关系实际出处为 [L] §3："ℓ−MoE_acc … K′=α⋅K and N′=α⋅N"、"We recommend ℓ−MoE_acc for Pareto-optimal accuracy versus inference cost."｜修复要求：把 F4 与 C4 中 reinvestment 的来源锚点由 "Figure 1 caption" 改为 [L] §3（ℓ-MoE_acc 定义）；Figure 1 caption 只保留给 C3/F3 的 $d/\ell$ 论断。｜修复：｜复验：

- [重要·技术] §3「压缩比与 Reinvestment」末段 / §4 约束三及其本章问题解答、核心问题第 4 条：把 reinvestment 写成「论文给出的策略是…同时按同一因子 $\alpha=d/\ell$ 放大 $N$ 与 $k$」，并称 [N4] §4.1 消融证明「同时放大 $N$ 与 $k$ 才能保持甚至提升质量」。但 [L] §4.1（Figure 3、Figure 4）的消融配置是 ℓ-MoE_eff——只放大专家数 $N$（$N'=\alpha N$），$K$ 保持不变；「同时放大 $N$ 与 $k$」属 §3 的 ℓ-MoE_acc 变体。页面未提及存在两个变体，把推荐变体当作论文唯一策略。｜引文依据：[L] §4.1 "systematic ablation studies on 16B total parameter models (2B active)"、"compression ratios up to α = 4 preserve model quality"，Figure 4 对比曲线为 $N'=\alpha N$ 与不放大 $N$；[L] §3 "ℓ−MoE_eff … Keeps K constant while scaling total experts by α" 与上条 ℓ-MoE_acc 定义。｜修复要求：约束三改为「§4.1 消融（ℓ-MoE_eff）显示只放大 $N$ 即可补偿压缩损失；§3 的 ℓ-MoE_acc 进一步同时放大 $N$ 与 $k$，是论文推荐的准确率变体」，并把 §3「论文给出的策略是…同时放大 $N$ 与 $k$」补上两变体区分。｜修复：｜复验：

- [轻微·格式] §4 五条设计原则折叠块、正文与来源章节：同一对象两种编号写法并存——正文用阿拉伯数字（§1「设计原则 1、2」、§4 约束一「论文设计原则 4」、折叠块「原则 1、2…原则 3…原则 5…原则 4」），§3 用罗马数字（「设计原则 III」「设计原则 V」）。｜引文依据：不适用（原论文统一用 Roman numerals I–V）。｜修复要求：全页统一，建议统一为论文的 I–V。｜修复：｜复验：

- [轻微·格式] 正文与来源章节出现裸 Unicode 数学符号，未包入 `$...$`：第 440 行「top-$k$ × 专家中间维度」、第 597 行「非线性容量 = top-$k$ × 专家中间维度」、第 594 行「$N$ 约 3.5×、$k$ 为 2×」「激活 8 个 → 896 个」、第 403 行 summary「组合数 28 → 10518300」。同一表达式在表格（第 333 行）写作 `$k \times$ 中间维度`，与第 440 行写法不一致。｜引文依据：不适用｜修复要求：`×` 改为 `$\times$`、`→` 改为 `$\to$`（或改写为中文「变为 / 涨到」），全页统一写法。｜修复：｜复验：

- [轻微·技术] N5 数字精度：页面写「其投影分析估计这部分额外计算约 9%」，原文为上界形式 "within up to ∼9%"，页面把「至多约 9%」写成点估计「约 9%」。｜引文依据：[L] §4.3.1 "Relative to native Kimi-K2-1T, Kimi-K2-1T-LatentMoE introduces additional computation due to latent projection operators. In our projections, native Kimi-K2-1T remains close, within up to ∼9% of Kimi-K2-1T-LatentMoE, indicating that projection overhead is small…"｜修复要求：改为「至多约 9%」或「不超过约 9%」（§4 约束二、callout 与本章问题解答三处同步）。｜修复：｜复验：

- [轻微·表述] 通读全文（含折叠块）发现的会话残留与元话语：§1 末「但 LatentMoE 如何用投影把路由专家搬进更窄的隐空间——下一章讲层结构。」、§3 首「结构看清楚了。那把路由宽度从 $d$ 压到 $\ell$ 到底省了多少…」、§2「用一张数据流图把两支的分工与投影位置摆清楚：」、§3 末「压缩比听起来越大越好——但 $\ell$ 能不能无限压小？」。｜引文依据：不适用｜修复要求：改为陈述句衔接，去掉「下一章讲…」「看清楚了」「摆清楚」「听起来」这类现场解说口吻；其余段落未见元话语、自我指代（本页/本文）、会话指代（我/我们/你）或调试叙事，数学符号（除上条所列）均由 KaTeX 渲染。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复

### 已核对且成立的来源论断（本轮未列为问题）

- C1/C7：路由分支两端加共享 $W_\downarrow$/$W_\uparrow$、路由专家在 $\ell$ 维、router 与共享专家仍在 $d$ 维——[L] §3 原文 "$p'=\mathrm{Softmax}(W_r'\cdot x)$ … computed from the original token $x$" 与 "all operations outside the routed experts…continue to operate in the original hidden dimension d"，与 C7 引文一致。
- C3/F3：路由部分开销按 $d/\ell$ 缩小——Figure 1 caption 原文支持。
- C5：特征秩下限（设计原则 4）、不 reinvestment 会掉点（§4.1）——均定位到原文。
- C6/N1：K3 $d=7168$、$\ell=3584$（0.5×）、896 专家、top-16、2 共享专家、稀疏度 $N/k=56$、2.8T——[K3] §2.3/Table 1 与官方 config.json 一致；「激活爆炸」「近千专家负载失衡」「RMSNorm / SiTU-GLU / Quantile Balancing」与 §2.3 原文（"exploding internal activations in the routed branch"、"balancing the load of nearly 10³ experts"、"inserts RMSNorm between expert aggregation and the up-projection"）一致；「报告未给出该压缩比取值依据」经核对 §2.3 未发现取值论证。
- N2/N3：[R] Raschka 两文核对——Super $4096\to1024\to4096$、120B/12B；Ultra $8192\to2048\to8192$、550B/55B、512 路由专家、top-22、路由中间维度 5120、共享中间维度 10240、4× 压缩，逐项一致。
- N4：16B 总参 / 2B 激活消融、$\alpha\le4$ 质量保持——原文一致（仅其「与 $k$ 同时放大」的归属见重要第 2 条）。
- N6（登记于来源章节、正文未引用）：1.24×–3.46×、约 350B 额外参数——[L] §4.3.1 一致。
- 构造示例算术：$k\times d$（8×4096=32768、16×4096=65536）、$k\times\ell$（8192、16384）、缩小 4×；$\binom{8}{2}=28\to\binom{32}{8}=10518300$（约 37.6 万倍，页面写「约 37 万倍」）；投影参数量 $2\cdot4096\cdot1024=8388608$——全部复算一致，代码实跑输出与页面预期输出逐行一致。

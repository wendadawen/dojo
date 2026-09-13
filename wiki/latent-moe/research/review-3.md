<!-- review-meta
round: 3
page: wiki/latent-moe/index.html
reviewed_content_sha256: 735ea93ccaf2d3de
-->
# LatentMoE 审查记录（第 3 轮）

- 页面版本：ff3c91cdfc9a6d2917d83c7d3def7d95540532d2
- 审查时间：2026-09-13 19:05
- 审查者：独立子代理（未参与写作，未参与前序审查与修复）
- 已完整阅读章节（按顺序）：核心问题（学习目标）、最容易误解、1 扩大专家池的代价、2 LatentMoE 的层结构、3 压缩比与 Reinvestment、4 压缩下限与投影成本、5 与 Stable LatentMoE 的关系、来源与范围说明（C / F / N / 构造示例 / 辅助解释 / 简化条件），含全部折叠块与图注
- 来源核对方式：arXiv HTML 全文（[L] 2601.18089v1、[K3] 2607.24653v1）、huggingface.co/moonshotai/Kimi-K3/raw/main/config.json、sebastianraschka.com 博文；代码示例已在本机 python3 实际执行并逐行比对输出

## 问题

- [阻断·技术] 来源说明 C7（对应正文 §2 "第二，router 仍从全宽 $x \in \mathbb{R}^d$ 计算门控决策$^{[C7]}$"）｜问题：C7 用英文引号给出 [L] §2 的"原文"，但该句在论文中不存在，属伪造引文（把来源没有的话写成来源原话）｜引文依据：[L] 全文检索 "gating decisions" 与 "hidden representation" 均为 0 次命中；[L] §3 实际原文为 "The routing weights p′=Softmax(W_r′·x) are computed from the original token x∈ℝ^d using a learnable weight matrix W_r′∈ℝ^{N′×d}" 与 "all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension d"｜修复要求：删去 C7 的伪引号英文句，改引上列 §3 原文并注明正确章节号（门控定义在 §3，不在 §2）；"router 在 d 维工作"这一机制本身有来源支持，论断可保留，但不得再以不存在的引文为依据｜修复：｜复验：

- [重要·技术] 来源说明 N4、C5，及正文 §4 两处 $^{[N4]}$｜问题：16B 消融（压缩比到 4 质量保持、只压缩不 reinvestment 会掉点）的定位写成 [L] §3；该实验在 §4.1｜引文依据：[L] §4.1 "Results in Figure 3 indicate that model quality is preserved for compression ratios α≤4"；[L] §4.1 "reducing d without scaling the expert count leads to significant quality degradation, validating the expert scaling strategy employed by LatentMoE"（Figure 4）；[L] §3 标题为 "LatentMoE Architecture"，不含消融结果｜修复要求：N4、C5 及正文 [N4] 的定位由 §3 改为 §4.1（可附 Figure 3、Figure 4）｜修复：｜复验：

- [重要·技术] 来源说明 N6｜问题：把 1.24×–3.46× 加速与"多约 350B 参数"定位为 [L] Figure 2 与 §4；Figure 2 是与本论断无关的 roofline 分析，正确位置为 §4.3.1 与 Figure 7｜引文依据：[L] §4.3.1 "Across the projected Pareto frontier (Figure 7), Kimi-K2-1.35T is approximately 1.24×–3.46× slower than Kimi-K2-1T-LatentMoE"，同节 "corresponding to an increase of (1.35−1.0) T ≈ 0.35 T ≈ 350 B parameters"；[L] Figure 2 标题为 "Roofline Analysis for serving Qwen3-235B-A22B"｜修复要求：N6 定位改为 [L] §4.3.1 与 Figure 7｜修复：｜复验：

- [重要·技术] 来源说明 N2，及折叠块"补充：Nemotron-3 Super/Ultra 的 LatentMoE 配置"的来源行｜问题：Nemotron-3 Super 的规格（4096→1024→4096、120B 总参/12B 激活）标注来源为 [L] §4，但论文不含该规格；这些数字实际出自 [R] Raschka｜引文依据：[L] 全文 "Nemotron" 仅出现于摘要与参考文献（"has been adopted by the flagship Nemotron-3 Super and Ultra models"，无宽度与参数量规格）；[R] "Super uses 4096 -> 1024 -> 4096 pathways" 与 "Super (120B total, 12B active)"｜修复要求：N2 与折叠块的 "[L] §4" 归因改为 [R]，或删除 [L] §4；Ultra 一行（N3）本就只归 [R]，保持不变｜修复：｜复验：

- [重要·技术] §5 正文（"K3 的压缩比是 $d/\ell = 7168/3584 = 2$，比 Nemotron-3 的 4x 保守。这并非 K3 不想压得更窄，而是 2.8T 规模与极端稀疏下，再压窄会放大激活爆炸的风险——这正是三件稳定化要兜住的问题。"）；同说法在 §5 本章问题解答中重复｜问题：把无来源支持的因果推断写成来源结论。[K3] §2.3 只给出维度表 "Latent MoE Dimension 3584 (0.5×)"，未解释为何取 2x，也未说再压窄会放大激活爆炸；且按原文，激活爆炸的成因是"近乎四次连乘的矩阵乘法链 + 2.8T 规模"，与 ℓ 的大小无关，该推断本身也不成立｜引文依据：[K3] §2.3 "This ill-conditioned structure, combined with the 2.8-trillion-parameter scale, produces exploding internal activations in the routed branch"；该节无关于压缩比取值的任何论证｜修复要求：删除"并非 K3 不想压得更窄…"的因果句（正文与本章问题解答两处），或降级为明确标注的推断并给出支持来源；否则只保留"K3 取 2x、Nemotron-3 取 4x"的事实对照｜修复：｜复验：

- [轻微·可读性] §0 引言（"本页以 NVIDIA 论文 arXiv:2601.18089 为主源，讲清…本页要回答：…"）、§1 开头（"先回顾标准 MoE 层的结构"）、§2（"注意几个容易看错的地方"）、§5 结尾（"回顾本页的学习目标：…这就是 LatentMoE 的完整图景。"）｜问题：元话语与导航式表述（"本页以…为主源""本页要回答""回顾本页"），属规范第 12 项要排除的表述｜引文依据：不适用｜修复要求：改为直接陈述内容，删去"本页/回顾本页/先回顾"式元话语与收尾抒情句｜修复：｜复验：

- [轻微·技术] 来源说明 C1｜问题：LatentMoE 正式定义（F2 对应结构）定位为 [L] §2，但正式定义在 §3，§2 只是设计原则｜引文依据：[L] §3 标题 "LatentMoE Architecture"，给出 ℓ-MoE_eff（Eq. 1）与 ℓ-MoE_acc（Eq. 2）；[L] §2 为 "LatentMoE Core Design Principles"｜修复要求：C1 的 "[L] §2" 改为 §3（Figure 1(b) 可保留）｜修复：｜复验：

- [轻微·技术] §4 折叠块"补充：NVIDIA LatentMoE 论文的五条设计原则"的列表括注｜问题：括注为页面自加，个别与 [L] 原文用词不符｜引文依据：[L] 设计原则 II 原文 "minimizing the data volume of all-to-all operations. This volume is proportional to… the routed hidden dimension and active expert count"，未出现 "tokens"；原则 I 原文无 "accuracy per parameter matters" 字样｜修复要求：括注改为与原文一致的表述，或删去自加括注｜修复：｜复验：

- [轻微·技术] §4 正文与 N4（"16B 参数规模的消融实验"）｜问题：未写明激活规模，读者可能误以为 16B 激活｜引文依据：[L] §4 "16B total parameters with 2B active, which we use for conducting ablation studies"｜修复要求：改为"16B 总参 / 2B 激活"或"16B/2B 消融实验"｜修复：｜复验：

- [轻微·可读性] §0 引言首段（"一个 MoE 层有 256 个路由专家、每 token 激活 8 个，想扩到 896 个专家、每 token 激活 16 个"）｜问题：该构造未登记为构造示例，且其放大倍数（256→896 约 3.5×、8→16 为 2×）与页面后文 reinvestment"按同一因子 α 同时放大 $N$ 与 $k$"不一致，易被误读为同一策略｜引文依据：不适用｜修复要求：在引言明示这是假想的规模放大，或在"构造示例"小节登记并说明两处放大倍数不同、不属 reinvestment 策略｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 4 / 轻微 4
- 处置：修复（本轮存在 1 个阻断、4 个重要问题，需按上表逐条修复并复验后进入后续轮次）
- 说明：机械项通过——本地链接（moe-serving / deepseek-moe / stable-latent-moe）均存在，无"（待生成）"占位，页面未指向已移除的 research/ 路径，`python3 .dojo/scripts/validate.py wiki/latent-moe/index.html` 返回 success；§1、§3 两张 dispatch 表与组合数（$8\times4096=32768$、$16\times1024=16384$、$\binom{8}{2}=28$、$\binom{32}{8}=10518300$、$2\cdot4096\cdot1024=8388608$）经本机 python3 实跑复算，与页面"预期输出"完全一致。核心公式 F2 与 [K3] Eq. 11（去 RMSNorm）一致，符号 $d,\ell,N,k,p_i,T_k$ 全文一致。除上表所列外，[L] Figure 1 caption、五条设计原则、$U_\text{eff}\propto K\cdot m$、"压缩比 α≤4 质量保持"、投影额外约 9%、K3 config.json（$d=7168,\ell=3584,N=896,k=16,N_s=2$、稀疏度 56）均已定位到原文片段并核对相符。

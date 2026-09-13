<!-- review-meta
round: 3
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 4bad4cfa3622067a
-->
# Kimi K3 审查记录（第 3 轮）

- 页面版本：75c7e467251242b6da08ed734202da323c8a7d58（`git hash-object wiki/kimi-k3/index.html`，工作树）
- 论文版本：arXiv:2607.24653v2（2026-08-07 修订版）
- 审查时间：2026-09-13 19:06 CST
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 核对来源：论文 HTML 原文 <https://arxiv.org/html/2607.24653v2>（下载后逐节比对，缓存于 /tmp/k3paper.html）与官方 config.json <https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json>
- 已完整阅读章节（含全部折叠块与表注）：核心问题；1. 三维度信息流——K3 架构总览；2. 序列维度——KDA + 混合注意力；3. 深度维度——Block AttnRes；4. 宽度维度——Stable LatentMoE；5. 原生视觉——MoonViT-V2；6. 训练——数据、scaling law、Muon、长上下文扩展；7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE；8. 基础设施——3T 训练 + 1M RL + 推理；9. 性能与评价；10. 独立评价——系统性设计、开源里程碑与边界；来源与范围说明
- 机械项：`.dojo/scripts/validate.py wiki/kimi-k3/index.html` 返回 `validation ok`（含数学字符与结构图检查）；`overview.html` 与 `index.html` 相互链接；14 个前置概念链接（kda、mla、nope、block-attnres、stable-latent-moe、situ-glu、quantile-balancing、moonvit-v2、per-head-muon、mopd、mxfp4-qat、eagle-speculative、flash-kda、moonep、gpu-execution-model）全部真实存在；无「（待生成）」占位；无指向已移除 `research/` 路径的链接；8 个本地 libs 资源全部存在。
- 已核对通过的核心数字（引文依据为原文片段/数值）：Table 1 全部 18 行（61→93 层 ↑52%、1.04T→2.78T ↑167%、32.6B→104.2B ↑220%、7,168=7,168、–→3584(0.5×)、2,048→3,072 ↑50%、384→896 ↑133%、8→16 ↑100%、1→2 ↑100%、64→96 ↑50%、1 = 1、160K=160K、128K→1M 8×、MLA→Hybrid KDA–MLA、SwiGLU→SiTU-GLU、61 MLA→69 KDA + 24 MLA、1 layer=1 layer、–→401M、–→27 layers）；Table 2 全部摘录行（编程/智能体/视觉三表逐值一致）；config.json 的 num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、moe_intermediate_size=3072、routed_expert_hidden_size=3584、num_attention_heads=96、max_position_embeddings=1048576、attn_res_block_size=12、latent_moe_use_norm=true、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、mla_use_nope=true、first_k_dense_replace=1、vt_num_hidden_layers=27、full_attn_layers=[4,8,…,92,93]（24 项）、kda_layers（69 项，与页面折叠块逐项一致）；算式复算 2.78/1.04≈2.67、104.2/32.6≈3.20、896/16=56、384/8=48、896/384≈2.33、12×7+9=93、4×23+1=93、β₁β₂=4×25=100 全部成立；§5.3.2 的 133 ms / 6.5× / 51,219,741 / 1,505,678、§5.4.1 的 512-token 边界、§6.3 四项第三方数字、§6.4 的 $2.03 与 38% 均与原文一致；会话指代（我/我们/你）全文未出现。

## 问题

- [阻断·技术] §9「本章问题」第 3 题解答（第 612 行）、§9.4 折叠块（第 589 行）、§10.2 第 2 条（第 637 行）：三处断言「Artificial Analysis 报更低 [Terminal-Bench 2.1] 分数」，该论断无来源支持，且与同页 §9.4 折叠块的自述直接矛盾｜引文依据：论文 §6.3 Third-Party Evaluation 只报告四项——"Intelligence Index v4.1 of 57.1, ranking fourth of 580"、"Vals Index (74.7%), … second of 39"、"WebDev Arena (1,678 Elo …) ranks first of 99"、"Agent Arena … ranks fourth of 37 (9.1)"，全文无 Terminal-Bench 的第三方分数；页面 §9.4 折叠块自己也写「§6.3 第三方评估中 AA 仅覆盖 Intelligence Index v4.1，未涉及 Terminal-Bench；AA 官网具体数字以官网为准」，与同句前半段的「Artificial Analysis 报更低分数」互相否定｜修复要求：删除三处「AA 报更低分数」的断言。若保留「绝对分数受 harness 影响」的提示，只能引用论文可核实的依据——§6.1.3 原文 "On Terminal-Bench 2.1, we report the best score across harnesses for all models."，即官方 88.3 是三 harness 取最优；不得引入任何未登记的第三方数字。

- [重要·技术] §3 第 2 段（第 285 行）与 §3「代价与边界」（第 293 行）：把 $N \approx 8$ 的经验结论标注为「论文 §2.2 引 [57]」｜引文依据：论文 §2.2 原文 "Empirically, $N\approx 8$ recovers most of the benefit across model scales [60]；for Kimi K3, we partition its layers into 8 blocks with 12-layer size"；参考文献 [60] = Kimi Team, "Attention residuals", Preprint, 2026，而 [57] = Kimi Team, "Kimi linear: an expressive, efficient attention architecture", arXiv:2510.26692——引文编号错位｜修复要求：两处 [57] 均改为 [60]。

- [重要·一致性] `<head>` 的 description（第 6 行）与正文题录块（第 103 行）对论文版本的说法互相矛盾｜引文依据：description 写「基于 arXiv:2607.24653v1 技术报告、官方 config.json 与源码三层核对」；正文题录块写「arXiv 预印本，arXiv:2607.24653v2，2026-08-07」；页面链接 <https://arxiv.org/abs/2607.24653> 解析到 v2（v2 修订于 2026-08-07）｜修复要求：把 description 的「v1」改为「v2」，使页面自述的审校基线与正文、链接目标一致。

- [轻微·格式] §1 表（第 152-177 行）的「变化」列改写了论文 Table 1 的 Δ 列，但表注仍称「原文 Table 1」｜引文依据：论文 Table 1 中 Attention Mechanism / Activation Function / Attention-Layer Composition / Latent MoE Dimension 四行的 Δ 列均为「–」；页面「变化」列分别写成「重构 / 替换 / 重构 / 新增」｜修复要求：或在表注中注明该列是本文对变化的归类、不等同原文 Δ 列，或改回与原文一致的「–」。

- [轻微·格式] 「来源与范围说明」的编号不连续，且 F 序列在正文无任何引用｜引文依据：第 696-703 行 F 列表为 F1、F2、F5、F6、F7、F8、F9、F10（缺 F3、F4）；第 708-712 行 N 列表为 N1、N2、N5、N6-N9、N10（缺 N3、N4）；全文 sup 引用只出现 [C1]-[C13]、[N1][N2][N5][N6]，无任何 [F*]、无 [N10]｜修复要求：补齐缺失编号或整体重排为连续编号；删除既无定义引用也无正文使用的空号。

- [轻微·格式] 同一数学符号在页面内有 KaTeX 与 Unicode 两种写法｜引文依据：第 285 行 `<code>O(L²d)</code>` 用 Unicode 上标 ²；第 337 行 `<code>softcap(x, β) = β tanh(x/β)</code>` 用 Unicode β；第 452 行 `<code>π_teacher</code>`、`<code>π_θ</code>` 用 Unicode π；而同页第 285/305 行的 `$N \approx 8$`、`$O(Ld)$`、第 337 行的 `$\beta_1=4$`、`$\beta_2=25$`、`$|f| \le \beta_1\beta_2 = 100$` 均由 KaTeX 渲染｜修复要求：同一变量统一写法——β、π 一律走 KaTeX（如 $\beta$、$\pi_{\text{teacher}}$），幂次改为 `$O(L^2d)$`。

- [轻微·表述] 元话语、目录式过渡与把「场景」当术语，散布在正文、折叠块与表注｜引文依据：不适用｜修复要求：逐条改写下列位置并删除同类写法——第 108 行「本文逐维度拆解……」、第 150 行「本页先从 Table 1（§3.2）看……」、第 180 行「注意三个关键变化」、第 203 行「接下来逐一展开每个维度。构造示例。」、第 290 行「注意：block 划分是……」、第 397 行「本节回答这些训练方法如何……」、第 484 行「本节简述各类基础设施创新」、第 539 行「注意 SWE-Marathon 用 H20-calibrated 分支」、第 621 行「本章是本文对 K3 的独立评价」。章末 9 处「本章讲 X。但……——下一章讲 Y。」（第 227、277、319、367、393、442、480、518、617 行）改为直接陈述衔接（例：「93 层深网络的信息稀释由 Block AttnRes 处理」），不再报幕。第 643 行标题「10.3 适用场景与位置」及第 645、647、667、669、670 行的「场景」改为「用途 / 条件」。「构造示例。」作为段首残句（第 203、253、295、343 行）改为完整句。

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 4
- 处置：修复。阻断与重要问题须全部关闭后方可发布；轻微问题按第 4 节逐条处理，遗留的须给出接受理由。
- 已排除项：Table 1/Table 2 数值、config.json 全部字段、8 项公式编号（Eq.1/5/8-10/11/12/14/15/16）、§3.1-§3.4、§4.1-§4.1.4、§5.1-§5.4、§6.1.3/§6.1.4/§6.3/§6.4 与 §8 的机制描述与归因均逐条回到原文核对，未发现其他不符；§2.4 与 §8 的英文直引与原文逐字一致；核心问题 5 题与各章「本章问题」均有解答折叠块且答案与正文一致，核心问题答案均指明完整论证所在章节；页面无可运行代码（正文无代码块），第 2.2 节第 4 条不适用。
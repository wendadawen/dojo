<!-- review-meta
round: 5
page: wiki/latent-moe/index.html
reviewed_content_sha256: f2e52045cf109a7c
-->
# LatentMoE 审查记录（第 5 轮）

- 页面版本：aa0422a1cf1d385c6ef55ef41491fbc84e28d206（wiki/latent-moe/index.html，git hash-object）
- 审查时间：2026-09-13 20:18
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 扩大专家池的代价——标准 MoE 的全宽路由开销 / 2. LatentMoE 的层结构——共享分支全宽、路由分支隐空间 / 3. 压缩比与 Reinvestment——省下的预算花到哪里 / 4. 压缩下限与投影成本——为什么不能无限压窄 / 5. 与 Stable LatentMoE 的关系——LatentMoE 的稳定化变体 / 来源与范围说明（含全部折叠块、图注与代码块）；overview.html 一并通读。

## 来源核对摘录（供复验）

- [L] Elango et al., 2026, arXiv:2601.18089（"LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"，作者首位 Venmugil Elango）。Figure 1 caption 原文："...reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"。§3 变体定义与页面一致：ℓ-MoE_eff 按 α=d/ℓ 放大 N、K 不变；ℓ-MoE_acc（recommended）放大 N'=αN 与 K'=αK。§4.1（16BT-2BA）："Model quality is preserved for compression ratios α≤4"，Figure 3（compression ratio）、Figure 4（expert scaling）为该节图。§4.3.1："...requires an additional ∼350B parameters...yields a 1.24×–3.46× projected slowdown across the frontier."，Figure 7 为 Pareto 前沿图。C7 原文："The routing weights p'=Softmax(Wr'·x) are computed from the original token x∈R^d..." 与 "...all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension d."。设计原则 I–V 原文与页面基本一致（见问题 3）。
- [K3] Kimi K3 技术报告 k3_tech_report.pdf §2.3：Eq. 11 为 `u=Σ_{i∈Tk(x)} p_i E_i^routed(W↓x)`、`y=Σ_j E_j^shared(x)+W↑ RMSNorm(u)`（与页面 F2「去掉 RMSNorm」及 §5 表述一致）；"sparsity of 56"；"the routed path composes W↓, a gated multi-branch expert feed-forward network, and W↑ ... produces exploding internal activations in the routed branch"；"the 2.8-trillion-parameter scale"；"balancing the load of nearly 10^3 experts"；三件稳定化为 RMSNorm-before-up-projection / SiTU-GLU / Quantile Balancing。官方 config.json（moonshotai/Kimi-K3）：hidden_size=7168、routed_expert_hidden_size=3584、num_experts=896、num_experts_per_token=16、num_shared_experts=2、latent_moe_use_norm=true。页面 N1 的 K3 数字全部核对通过。
- [R] Sebastian Raschka：2026 博客仅有 "Nemotron 3 Ultra and Latent MoE Scaling"（Super 4096→1024→4096、120B/12B；Ultra 8192→2048→8192、550B/55B、512 routed experts、top-22、1 shared、中间维度 5120/10240；原文 "The down- and up-projections add work, so the 4x bottleneck does not imply a 4x speedup for an MoE layer or for the complete model."）。未定位到题为 "Latent MoE" 的文章（见问题 2）。

## 问题

- [重要·技术] 引言第 3 段（第 97 行）：「围绕 LatentMoE 有三个核心问题」与页面「核心问题」块条目数矛盾。｜引文依据：引言"围绕 LatentMoE 有三个核心问题：它用什么结构变化缓解路由开销？省下的预算花到哪里？压缩比能不能无限放大？"；而 `<section class="learning-goals">` 的「核心问题」块实际含 5 个 `<li>`（另有"共享/路由分支宽度与完整层公式""与 Stable LatentMoE 的关系"两条）。同页两处对核心问题的条数不一致。｜修复要求：将引言改为与实际一致（如"围绕 LatentMoE 有五个核心问题"并补齐到 5 条的概述，或去掉数量词只作正文线索概述），使引言所列问题与「核心问题」块条目一一对应。｜修复：｜复验：

- [重要·来源] 来源与范围说明「论断与来源（C5）」与「外部数字与实验条件（N2）」：两处引用的来源 `[R] Sebastian Raschka, "Latent MoE"` 无法定位。｜引文依据：Raschka 2026 博客归档无标题为『Latent MoE』的文章（推测 URL /blog/2026/latent-moe.html 返回 HTTP 404）；C5 所指"投影不等于整体加速"与 N2 的 Nemotron-3 Super 规格，实际出自其文章 "Nemotron 3 Ultra and Latent MoE Scaling"（该文原文："The down- and up-projections add work, so the 4x bottleneck does not imply a 4x speedup..."；"For Super, the routed path is 4096 -> 1024 -> 4096"；"Super has 120 billion total and 12 billion active parameters."）。｜修复要求：把 C5、N2 两处来源标题更正为可定位的 "Nemotron 3 Ultra and Latent MoE Scaling"（或删除该引用）；正文论断本身保留，因内容确在该文。｜修复：｜复验：

- [重要·来源] §4 折叠块「补充：NVIDIA LatentMoE 论文的五条设计原则」原则 II（第 439 行）：页面把 all-to-all 数据量写成"正比于路由隐空间宽度与激活专家数"，与源文不符。｜引文依据：[L] §2 原则 II 原文 "Improving performance in throughput-oriented MoE deployments requires minimizing the data volume of all-to-all operations. This volume is proportional to: Mcomm ∝ N/EP·texp·d"（即 ∝ 激活专家数 K × 表示宽度 d，d 为全宽，非隐空间宽度 ℓ）；页面 §1 对标准 MoE 同样写作 ∝ k·d，与本处括注自相矛盾。｜修复要求：把原则 II 括注改为"正比于激活专家数与表示宽度 $d$"以与源文一致；若需说明 LatentMoE 中该宽度被压到 $\ell$，另起一句限定，不写成原则本身的表述。｜修复：｜复验：

- [轻微·格式] §1 标准 MoE 公式（第 149 行）：`$$y(x)=\sum_{i\in S_k(x)} g_i(x)\cdot E_i(x)$$` 即来源章节登记的 F1，但正文未带 `<sup>[F1]</sup>`，来源章节与正文未双向对应。｜引文依据：不适用（style-guide §6 要求正文上标"与来源章节双向对应"；来源章节已登记 F1，正文该公式无 `[F1]` 上标）。｜修复要求：在该公式处加 `<sup>[F1]</sup>`，与 §1 其他来源引用写法一致。｜修复：｜复验：

- [轻微·格式] head 内联 `<style>`（第 48–57 行）：`.diagram{...}` 规则在页面无任何元素使用（页面仅用 `class="flow-diagram"`），属死 CSS，且与 `libs/dojo-concept.css` 已定义的 `.diagram` 重复；结构图用页面自定义 `.flow-diagram`/`.fd-*`，未用组件库结构 A/B（`.diagram`/`.dg-flow`/`.dg-stack`）。｜引文依据：不适用（`grep 'class="diagram"'` 命中 0；`class="flow-diagram"` 命中 2；libs/dojo-concept.css 已定义 .diagram/.dg-flow/.dg-stack，且未定义 fd-*）。｜修复要求：删除未使用的内联 `.diagram` 规则；结构图改用组件库结构 A/B，或保留自定义结构图样式并说明理由。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 2
- 处置：修复。核心来源核对通过：LatentMoE 论文（Figure 1 caption、§3 变体、§4.1 16B 消融、§4.3.1 9%/1.24×–3.46×/350B、五条设计原则）与 K3 配置（d=7168、ℓ=3584、N=896、k=16、Ns=2、稀疏度 56、2.8T、Eq. 11、三件稳定化）均与出处一致；代码块实际执行输出与页面逐字一致（C(8,2)=28、C(32,8)=10518300、增长 375653.57、2816？否——2·4096·1024=8388608）；`.dojo/scripts/validate.py` 返回 validation ok；dojo:type=concept、dojo:topics=模型结构、dojo:tag=MoE 均在词表内；前置概念页 moe-serving / deepseek-moe / stable-latent-moe 均存在，无占位。关闭上述 3 个重要与 2 个轻微后即可发布。
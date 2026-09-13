<!-- review-meta
round: 4
page: wiki/moonep/index.html
reviewed_content_sha256: 717e9a27c986a690
-->
# MoonEP 完美均衡专家并行审查记录（第 4 轮）

- 页面版本：dddc2b042b8599c6（index.html 工作树 sha256 前 16 位）
- 审查时间：2026-09-13 19:44
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作与前序审查）
- 来源获取：Kimi K3 技术报告 arXiv:2607.24653 full HTML（`https://arxiv.org/html/2607.24653`，落盘 1.52 MB 后逐段抽取正文与附录 E 原文）；MoonEP 仓库 `https://github.com/MoonshotAI/MoonEP`（页首声明不引用源码，仅确认仓库存在与报告脚注一致）；内链 `wiki/moe-serving/index.html`、`wiki/gpu-execution-model/index.html` 均存在
- 已完整阅读章节：引言与核心问题 → 1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路 → 2. 冗余专家的界——$E/R$ 上界与基本紧性 → 3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程（含 3.1–3.4、两处配图、伪代码折叠块） → 4. MoonEP 的边界——解决与不解决，以及与 ECHO/UltraEP/DeepEP 的区别（含 4.1–4.4 与对比表） → 来源与范围说明（含全部折叠块与问题块），全文含折叠内容逐段通读

## 已核对来源（无问题项，供发布条件留档）

- C1：§5.2.1 开头原文「In conventional EP schemes, token loads are imbalanced across ranks. The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation.」——页面「计算不均衡降低吞吐 / 激活形状动态变化造成显存碎片」两条与原文一致。
- C2：§5.2.1「MoonEP preserves the overall computation flow of conventional schemes such as DeepEP and additionally introduces online planning and migration of redundant experts.」——与 1 章、4.4 节的「保留总体计算流、dispatch 前多一步在线规划」一致。
- C3：§5.2.1「MoonEP requires every rank to receive exactly $S\times K$ tokens, where $S$ is the sequence length and $K$ is the number of experts selected per token」——「每 rank 恰好收到 $S\times K$」一致；$S$ 释为「每 rank 本地序列长度」是全页唯一自洽读法（总量守恒 $R\cdot S\times K$ 才等于全网 pair 数），已复算。
- C4：§5.2.1 Online planning「Computing the exact optimum at every training step is prohibitively expensive... compute exact solutions offline with integer linear programming (ILP) for representative cases as references and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the $E/R$ upper bound.」——「prohibitively expensive」「近最优 / 开销可忽略 / 总尊重 $E/R$ 上界」「离线 ILP 作参考」逐句一致。
- C5：§5.2.1「In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks.」——3.4 节两步流程一致。
- C6：§5.2.1 Zero-copy communication「We implement a fused permute/unpermute operator in which the planning kernel precomputes the destination of every token, so tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies.」——3.3 节一致。
- C7：同段「Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size $S\times K\times R$, whereas MoonEP requires only a fixed $S\times K$ buffer owing to the perfect balance.」——页面三处 $S\times K\times R \to S\times K$ 的表述一致，且都保留了「最坏不均衡」限定并说明报告未涉非最坏情况（复算：省 $R$ 倍成立）。
- C8：§5.2.1 Sync-free execution with static shapes「...the host must synchronize with the device at every layer to obtain the actual computation shapes before launching the expert computation, stalling the pipeline between layers... This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead.」——3.2 节一致。
- C9：§5.2.1 Expert-GEMM scheduling and overlap「Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed, and a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers... For the shared experts, we dispatch their GEMMs to a separate stream so that they overlap with other kernels.」——4.2 节「不解决 rank 内 per-expert 偏斜 / 独立 scheduler / shared expert 独立 stream」一致（该段确为 §5.2.1 最后一小节）。
- C10：§5.2.1「In contrast, prior work such as ECHO [138] and UltraEP [133] presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists within the cap, and the cap itself requires manual tuning while still leaving residual imbalance.」——正文 4.3 节（用「或」）一致。
- F1：§E「the planning objective is to minimize the maximum number of redundant experts on any rank, i.e., $M(I)=\min_P\max_r\{m_r(P)\}$」——一致。
- F2（Theorem 1）：§E「$M(I)\leq E/R$ always holds」；构造性引理原文「every EP rank receives exactly the same number of tokens ($S\times K$), and the remote tokens of each rank come from only one other EP rank」「terminates after at most $R-1$ fills; meanwhile, each rank is filled at most once」「these tokens belong to at most $E/R$ local experts on rank $s$, hence $m_r(P^*)\leq E/R$」——页面正文与「补充」折叠块的复述与终止性/同源性论证逐点一致（手算 $E=4,R=2$：情形 A 需 1 个冗余 ≤ 2；情形 B $M=1=\lceil 4\cdot1/4\rceil<E/R=2$，均正确）。
- F3（Theorem 2）：§E「there exist router outputs for which $M=\lceil E(R-1)/R^2\rceil\approx E/R$」「each expert receives $SKR^2/(E(R-1))$ tokens... at least $\lceil E(R-1)/R^2\rceil$ redundant experts」——一致；报告该显示式确编号 (28)，页面「即公式 (28)」正确。
- N：abstract「2.8T parameter Mixture-of-Experts model」、§2.3 Stable LatentMoE「896 routed experts with 16 active experts per token」、§5.2 标题「5.2 Infra for 3T-class Pre-Training」、§2.3.3「Kimi K3 adopts auxiliary-loss-free routing. Load balancing is implemented by adding an expert-specific bias $b_j$...」——页面的 $E=896$、$K=16$、2.8T、3T 级、auxiliary-loss-free + Quantile Balancing 四处数字与归因全部命中。
- §5.2.2「all GPU memory is allocated on the main compute stream and managed within a single memory pool, avoiding multi-stream fragmentation and host-bound overhead」——4.2 节的对照说明一致。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`；引言「构造示例」$S\times K=4$、$S\times K\times R=8$、$8+0=8$ 复算无误；两级问题块齐全且核心问题四条均指向对应章节；页内无「（待生成）」、无 research/ 路径引用、无 Unicode 数学字符；libs/ 本地资源（katex、auto-render、prism、dojo-concept.css）全部存在；overview.html 与 index.html 互链。

## 问题

- [重要·技术] §4.1 第三句、核心问题 Q4 解答、4.2 节末句、全文总结：四处把「routed-expert 激活形状动态变化造成的显存碎片」写成「随完美均衡的静态形状一并消除」，并以 `<sup>[C1]</sup>` 归到报告；报告 §5.2.1 只把碎片列为 rank 间不均衡的后果，从未声明 MoonEP 消除碎片，且同一节明说「per-expert token counts within each rank remain skewed」——rank 内激活形状仍随层变化，「消除」是由「静态形状」反推的推断却被当作来源结论陈述。｜引文依据：「The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation.」；「Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed」｜修复要求：删除四处「碎片随完美均衡消除 / 静态形状消除了这一动态性」的断言，或统一改为明确标注的推断（写明推理链「报告称碎片由激活形状动态变化引起、MoonEP 使每 rank 总量固定为 $S\times K$」，并说明 per-expert 形状仍随层变化这一前提），不得再挂在 [C1] 之下作为来源事实｜修复：｜复验：
- [重要·技术] §4.3 对比表「是否有界保证」列：ECHO 行填「否，预设冗余数」、UltraEP 行填「否，施加 token cap」，把两种机制分别指派给具体方法；报告只有一句合并表述，未做该区分（同节正文 4.3 用「预设冗余专家数或施加 per-rank token cap」是对的，表格与正文口径不一致）。｜引文依据：「In contrast, prior work such as ECHO [138] and UltraEP [133] presets the number of redundant experts or imposes a per-rank token cap.」｜修复要求：两行改为报告口径的并列写法（如「否，预设冗余数或 token cap」），或在该列标注「本页推断」并给出推断依据｜修复：｜复验：
- [轻微·技术] §4.2 末段括注「（具体算法报告未公开，本页不展开）」：前一分句刚陈述了报告内容（launch 前调参、系数离线 autotuning 标定），随后又称报告未公开该算法；报告实际披露了参数选择方式，只是没给完整公式。｜引文依据：「A lightweight heuristic selects these parameters using an analytical cost model of hardware metrics, with key coefficients calibrated through offline autotuning.」｜修复要求：改为「报告只给出启发式类别（解析硬件代价模型 + 离线 autotuning 标定系数），未给出完整公式」，删除「报告未公开」的说法｜修复：｜复验：
- [轻微·表述] §2 第三段「只要求最坏那个 rank 别太离谱」：口语化临场评价（全仓库仅本页出现该词），与全页客观陈述的语域不一致。｜引文依据：不适用｜修复要求：改为中性表述，如「只要求最坏 rank 的冗余数不超过给定值」｜修复：｜复验：
- [轻微·格式] 来源与范围说明 下的 `<h3>全文总结</h3>`：style-guide 第 3/11 节规定该章节 h3 使用固定命名（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制），「全文总结」不在其中；仓库其余页面（glu、swiglu、delta-rule、newton-schulz、positional-encoding、moonvit-v2、flash-kda）都把它写成正文段落而非标题。｜引文依据：不适用｜修复要求：去掉该 h3 标签，把内容并入段落并以「全文总结：」开头，与其余页面一致｜修复：｜复验：
- [轻微·格式] §2 显示式写成 `$$...\quad\text{(F1)}$$` 自行编号；正文两处（1 章构造示例、本章问题解答）以「公式 F4」指代总量守恒，但全页没有显示该公式。style-guide 第 11 节要求公式不自行编号、引用论文 Eq. 编号，且来源编号应与正文双向对应。｜引文依据：不适用｜修复要求：删去 `\text{(F1)}` 自编号，改用 `<sup>[F1]</sup>` 上标引用以与 [C]/[N] 口径一致；把总量守恒写成一行显示公式，或删去「（公式 F4）」的指代｜修复：｜复验：
- [轻微·技术] §3.3 伪代码内部不一致：状态区声明 `冗余专家权重缓存 redund_w[E/R]（最多 E/R 个槽位）`，步骤 2 却用 `redund_w[r].pull(home[e], e)` 以目标 rank $r$ 索引；当 $R\neq E/R$（真实配置 $E=896$、$R=64$ 时 $E/R=14$）会越界，语义也把「本地槽位容量」与「rank 编号」混为一谈。｜引文依据：不适用｜修复要求：改为按本地槽位编号索引（如 `redund_w[slot]`），或在状态区注明 redund_w 按目标 rank 索引且容量为 $R$ 项，二者取其一使伪代码自洽｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 5
- 处置：修复

（核对说明：本轮未发现事实性阻断问题——C1–C10、F1–F4 的定位与原文逐条命中，$E=4,R=2$ 两种情形的数字全部可复算，$E=896$/$K=16$/2.8T/3T-class 与报告一致，代码块已声明为伪代码故静态审查，页面功能与本地资源正常。两条重要问题均为「推断/未标注归因被写成来源结论」，需按上述修复要求处理后方可发布。）
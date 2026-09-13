<!-- review-meta
round: 3
page: wiki/low-rank-projection/index.html
reviewed_content_sha256: d2e7e83d1f1581bd
-->
# 低秩分解审查记录（第 3 轮）

- 页面版本：8bc60e6a3b9fac922bcfe9f0aaaf5d0dcf395f0b（git hash-object wiki/low-rank-projection/index.html）
- 审查时间：2026-09-13 19:06
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（5 题）；1. 大矩阵为什么贵——参数与存储的成本；2. 矩阵的秩——能被压缩多少由什么决定；3. SVD 与最优低秩近似——误差由奇异值决定；4. LoRA——把权重更新参数化为低秩；5. MLA——把 KV 压进低维潜向量；6. 适用边界——低秩不是万能的；来源与范围说明（含全部折叠块、构造示例、简化条件与图注）
- 机械验证：`python3 .dojo/scripts/validate.py wiki/low-rank-projection/index.html` → `validation ok`；`dojo:type=concept`、`dojo:topics=数学基础`、`dojo:tag=数学与数值` 均在 `catalog_builder.py` 词表内；overview.html 与 index.html 互链；`../../wiki/standard-attention/`、`../../wiki/mla/`、`../../wiki/kda/` 三个前置概念页真实存在，无「（待生成）」占位；全页数学符号均由 KaTeX 渲染（无 Unicode 数学字符进入标题/summary/正文/列表/表格）；`×`、`→` 仅出现在 HTML 结构图 flow-box 的盒间运算符位置。
- 复核数字：12288²=150,994,944；2×24576=49152；49152/150994944=0.0326%；2dr=49152；512/32768=1/64（1−1/64=98.44%，残留 1.5625%）；15.6/110.6=14.10%，34.6/860.2=4.02%（对应减少 85.9%/96.0%）；√1.25≈1.118；√(4+0.01)≈2.002；√(2.9²+2.8²)≈4.031；2×4096×8/4096²=16/4096≈0.39%。以上算式与页面标注一致。

## 问题

- [阻断·技术] §5「MLA——把 KV 压进低维潜向量」第二段（KV 比例说明处）：把 DeepSeek-V2 Table 7 描述成「DeepSeek-V2 MoE 与不同 MHA 基线（含 DeepSeek 67B 等）的整体对照，基线的 $n_h, d_h, d_c$、层数与模型规模均不同」，与来源不符。该表位于**附录 C.2**，对照的是**同一 backbone** 的 MoE 模型分别配 MHA 与 MLA（消融），不含 DeepSeek 67B；论文明确说这批模型 `share the same architecture except for the attention mechanisms`，故「含 DeepSeek 67B 等」与「层数与模型规模均不同」两处来源不支持。｜引文依据：arXiv:2405.04434v2 附录 C.2「In Table 7, we show the evaluation results for MoE models equipped with MLA and MHA, respectively, on four hard benchmarks... Also, two small MoE models and two large MoE models respectively share the same architecture except for the attention mechanisms.」；Table 7 表头「Small MoE w/ MHA / w/ MLA」「Large MoE w/ MHA / w/ MLA」，KV Cache per Token (# Element) 行 = 110.6K / 15.6K / 860.2K / 34.6K，表注「Comparison between MLA and MHA on hard benchmarks.」｜修复要求：按原文改写该句——Table 7 是同一 MoE 骨干分别配 MHA 与 MLA 的模型级对照（Small MoE 15.8B/15.7B、Large MoE 250.8B/247.4B，层数相同），删除「含 DeepSeek 67B 等」与「层数与模型规模均不同」；保留「这些比例是整个模型（含多层与解耦 RoPE 项）的比值，不能与同配置单层 1/64 直接等价」这一结论，并说明差异来源是 MHA 变体的注意力头配置与全模型多层统计口径。｜修复：｜复验：

- [重要·格式] `<head>` 的 `description`：「LLM 中用于 LoRA 微调、MLA KV 压缩、KDA 衰减门低秩生成。基于线性代数教材 + LoRA 论文核对。」与正文不符——正文（§6 与核心问题 5）讲的是 **K3 把 Kimi Linear 的低秩「输出门」改为 full-rank**，不是「衰减门低秩生成」，方向正好相反，且看板描述会随首页目录展示；「基于线性代数教材」也不在页末《主要依据》列出的来源（LoRA 论文、DeepSeek-V2 论文、Wikipedia SVD/Low-rank approximation）中。｜引文依据：正文 §6「Kimi Linear 的注意力门控用 low-rank 形式（$W_g$ 低秩分解），K3 把它改为 full-rank（配置项 `use_full_rank_gate = true`）」；页面《主要依据》仅列 LoRA / DeepSeek-V2 / Wikipedia 三处来源丨kda 概念页把该门称为「full-rank output gate（输出门）」。｜修复要求：把 description 第三项改为与正文一致的表述（如「KDA/K3 输出门由 low-rank 改 full-rank 的实例」），并把来源说明改为实际使用的三处来源；不要保留未出现在页面来源清单里的「线性代数教材」。｜修复：｜复验：

- [轻微·技术] 来源说明 [N4]（§「外部数字与实验条件」）：位置标注「§2.1.4 Table 7」错误。Table 7 在附录 C.2，§2.1.4 内只有 Table 1。｜引文依据：论文中 Table 7 出现两次，均在「C.2 Comparison Between MLA and MHA」小节；§2.1.4「Comparison of Key-Value Cache」正文为「We demonstrate a comparison of the KV cache per token among different attention mechanisms in Table 1.」｜修复要求：把 [N4] 的「§2.1.4 Table 7」改为「附录 C.2 Table 7」。｜修复：｜复验：

- [轻微·技术] §5 末段与本章问题 3 解答：「$d_c$ 的选择是压缩率与表达力的权衡」「DeepSeek-V2 选 $d_c = 512$ 是在压缩率和表达力之间权衡的结果」——论文未陈述该权衡，属本页推断写成来源结论。｜引文依据：论文只在 §2.1.4 给「For DeepSeek-V2, $d_c$ is set to $4d_h$」、§3.1.2 给「The KV compression dimension $d_c$ is set to 512」，全文无 trade-off / balance 关于 $d_c$ 的讨论（`trade-off` 仅出现在 4.4 节的 alignment tax 语境）。｜修复要求：降级为明确标注的推断（如「本页据 $d_c \ll d_h n_h$ 判断其取值为压缩率与表达力的折中，论文未展开该权衡」）或删除「权衡的结果」这一归因。｜修复：｜复验：

- [轻微·技术] §5：「MLA 的低秩压缩是**有损的**<sup>[C3]</sup>」「丢掉了 $d_h n_h - d_c$ 个方向的信息<sup>[C3]</sup>」——[C3] 原文只陈述 low-rank joint compression 这一机制，未陈述有损性或与 MHA 数值不等价。｜引文依据：[C3] 引文「The core of MLA is the low-rank joint compression for keys and values to reduce KV cache」（arXiv:2405.04434v2 §2.1.2）。｜修复要求：把 [C3] 标在「低秩联合压缩」处，「有损 / 数值不等价」标注为本页由 $d_c$ 瓶颈推得的分析。｜修复：｜复验：

- [轻微·技术] §2：「现实中 LLM 的权重矩阵几乎都是满秩的（$d \times d$ 矩阵秩为 $d$），无法无损分解」——关于真实模型权重的经验论断无来源支持，且未标注为一般性论断。｜引文依据：不适用（页面《主要依据》与 [C1]–[C4] 均未涵盖此论断）。｜修复要求：补来源或改写为条件式表述（如「若权重矩阵满秩（一般 $d \times d$ 随机矩阵几乎必然满秩），则无法无损分解」，并说明本页只讨论数学性质）。｜修复：｜复验：

- [轻微·表述] 开篇导语（第 115 行末）：存在路标式元话语。「本文从"大矩阵为什么贵"出发，讲清秩与可压缩性、SVD 给出的最优低秩近似及其误差公式，再落到 LoRA 微调和 MLA 的 KV 压缩两个 LLM 实际应用，最后说明适用边界。」｜引文依据：不适用｜修复要求：删除这句全篇路线图（内容已由各章开头/结尾的衔接句承担），或改成不叙述本页结构的引入。｜修复：｜复验：

- [轻微·表述] §4「这里要区分一个容易混淆的点：LoRA **不是在对 $W_0$ 做低秩近似**」与 §5「这里要强调一点：MLA 的低秩压缩是**有损的**」——「这里要区分/要强调一点」属元话语提示语。｜引文依据：不适用｜修复要求：直接给出命题（如「LoRA 并不近似 $W_0$：$W_0$ 冻结完整保留，被低秩参数化的是更新量 $\Delta W$」），删去提示语。｜修复：｜复验：

- [轻微·表述] §1「代入 LLM 的规模感受一下。」与 §2「这就是低秩分解的威力。」——前者为口语化祈使，后者为临场评价。｜引文依据：不适用｜修复要求：改为客观陈述（如「代入 LLM 的规模：」「参数量降至原来的约 0.033%」）。｜修复：｜复验：

- [轻微·可读性] §3 首次给出 SVD 定义处：「$U$ 是 $m \times m$ 正交矩阵」——「正交矩阵」为首次出现且未解释，而该性质（正交变换不改变 Frobenius 范数）正是后方折叠块「为什么截前 k 大奇异值就是最优的」所依赖。｜引文依据：不适用｜修复要求：在首次出现处加一句解释（正交矩阵 $Q$ 满足 $Q^\top Q = I$，列向量两两正交且为单位长度），或链接到已有前置概念页。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 7
- 处置：修复。核心论断、公式与数字（GPT-3 $d^2$、LoRA 的 0.01%/10,000 倍/rank 1–2、DeepSeek-V2 的 $n_h{=}128,d_h{=}128,d_c{=}512$ 与 93.3%、Table 7 的 110.6K/15.6K/860.2K/34.6K、Eckart-Young-Mirsky 与误差公式、MLA Eq.(9)(10)(11)、K3 `use_full_rank_gate`）已回到 LoRA 论文、DeepSeek-V2 论文、Wikipedia 与 K3 config 核对，除上列第 1 条对 Table 7 的性质描述外均与来源一致；数值示例（diag(3,1,0.5)、diag(5,2,0.1)、$d{=}4096,r{=}8$）已复算无误。修完第 1 条（阻断）与第 2 条（重要）后可发布。

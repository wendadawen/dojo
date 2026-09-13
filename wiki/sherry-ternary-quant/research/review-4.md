<!-- review-meta
round: 4
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: a42ad9581116c0c3
-->
# Sherry 稀疏三值量化（STQ1_0）审查记录（第 4 轮）

- 页面版本：index.html（工作树）MD5 `1749210f27c42d8c7e52828915734df1`
- 审查时间：2026-09-13
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：blockquote.meta / 引言 / 核心问题 / 常见误解 / 1. 三值量化的打包困境（含本章问题）/ 2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1 合法组合怎么数、2.2 5 bit 的内部结构、2.3 一组权重的编码手算、本章问题）/ 3. STQ1_0 字节布局——1.3125 bpw 的账（3.1 42 字节的构成、3.2 stride-16 分组、本章问题）/ 4. 量化决策——缩放系数与零位怎么选（4.1 缩放系数的三种选法、4.2 零位的两种选法、4.3 误差实测含全部代码折叠块、本章问题）/ 5. 为什么快——SIMD 解码与实测（5.1 解码不需要位重排、5.2 实测数字、本章问题）/ 6. 训练侧的 weight trapping 与 Arenas——本页边界（本章问题）/ 来源与范围说明

## 核对方式与已核实条目（摘要）

- 代码：`wiki/sherry-ternary-quant/index.html` 第 354–434 行的 Python 代码在本地实际执行（`python3`，Python 3），输出与页面「预期输出」（第 438–450 行）逐行一致：`amax + argmin 零位 d=1.1743 加权SSD=117.5101`；`WLS + argmin d=0.5152 加权SSD=19.6010`；`WLS + imatrix 3 轮 d=0.5381 加权SSD=17.7533`；`重要性全 1 时……一致: True`；`贯穿示例 …… SSD=0.0300`；`同一组用 amax d=0.7: SSD=0.0600`；`索引 32B + 符号 8B + fp16 scale 2B = 42B`；`bpw = 1.3125`。
- 论文（arXiv:2601.07892v1，经 arXiv HTML 抓取）：Table 4 八行数字与页面第 507–514 行逐格一致（0.7B：34.01/1360.0、132.13/256.56、116.83/233.44、148.27/205.50；3B：7.55/6190.0、41.87/873.65、38.80/846.01、45.55/712.40）；Table 1 的 Tequila 1.67→0.519 与 Sherry 1.25→0.519（1B）、0.576 与 0.567（3B）一致，ARC-Challenge 两规模反超（1B 0.309 对 0.305、3B 0.364 对 0.346）成立；「§4.2 3B 快 18%」原文成立；「10B tokens / UltraFineWeb」「PIQA、ARC-e、ARC-c、HellaSwag、WinGande 五任务零样本」「3 次运行」原文成立；「3:4 稀疏 ER<750，总维度 4096，坍缩程度与二值量化相当」原文成立；Arenas 公式 `Y = XTα + λ_t XW`、`λ_t` 退火到 0、附录 D 的 L2 最优性与 C.1 的 M∈{2^k}、附录 C 的 2:4「12 unique states」、vpshufb 单指令查找表 16 项上限，均原文成立。
- llama.cpp PR #22836：M4 Pro「12 cores: 8 P + 4 E, 24 GB」、8 线程、`-ngl 0`、llama-bench 表四行（含 Q1_0 336.25 MiB、`768.47 ± 14.75`、`109.62 ± 16.93`）与页面第 491–494 行一致；stride-16（`w0, w16, w32, w48`）、`vld1q_s8` 加载与 `vdotq_s32`、Q8_K 顺序存储激活、`qs[32]`/`sign[8]`/fp16 `d`、TQ1_0 的 1.6875-bit 3-way packing、NEON 用 `vqtbl2q` 的 32 项表、ACL 2026 接收，均原文成立；「~3 pp 精度差距」PR 原文为「per the Sherry paper (Fig. 6)」，与页面「按论文结果」相符（论文 Fig. 6 确含 binary / 3:4 / 1.67-bit 三档）。
- HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡：`213.66 GiB`、`2.38 bpw`、加权最小二乘 `d = Σ(w·sel·x)/Σ(w·sel²)`、imatrix 增量代价 `w[j]*(x[j]^2 - (|x[j]|-d)^2)`、`42 = 2 + 32 + 8` 与 `1.3125 bpw`、`−89.7%` 与再降 `−4.1%`、`gate`/`up` 投影走 STQ1_0（29 层）等，均原文成立；tencent/Hy4-preview 模型卡「770B total parameters, of which 49B are activated per token」支持页面的 770B 规模。
- 机械项：`python3 .dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html` 返回 `validation ok`；页面无 `research/` 路径引用；全文 Unicode 数学字符清零（唯一 `Σ` 在代码块注释内，规范允许）；`../hy4-preview-lite/index.html`、`../mixed-precision-quant/index.html` 两个前置概念页均存在；index.html 与 overview.html 互链；两级问题块命名与「解答：」前缀正确，5 条核心问题答案均在末尾指明完整论证所在章节。

## 问题

- [阻断·来源] §5.2 实测数字（index.html 第 519 行末句）：`在 Hy4 的官方算子对比中，STQ1_0 在 bit 最低的同时速度与 IQ1_M 基本持平、明显快于同样冲低比特的 IQ1_S<sup>[C15]</sup>`｜引文依据：**无法给出**。页面仅在「来源与范围说明」按名称引用「腾讯混元官方技术文章「Hy4 preview 轻量版」（2026-09-01）」，未给 URL；本轮逐一抓取可访问的官方材料均无此内容——HF `AngelSlim/Hy4-preview-GGUF` 模型卡渲染页与 raw README 只列 `Q4_K_M / UD-IQ1_M / STQ1_0` 三个文件且明示「No operator speed comparison table」，通篇未出现 `IQ1_S`；`tencent/Hy4-preview` 模型卡、GitHub `Tencent/AngelSlim` 仓库页与 README、`docs/source/_extra/hy4_preview_gguf_guideline.md` 亦无 STQ1_0 与 IQ1_M/IQ1_S 的速度对比。该论断在可核对的来源中定位不到，也无引文片段可记录。｜修复要求：补上该文章的完整 URL 与含「IQ1_M / IQ1_S 速度」的原文片段（同时补进 blockquote.meta 的主要依据列表）；若无法补出，删除此句，或改写成明确标注的推断（例如「按 Hy4 官方材料的定性描述（未给出可核对的对比数据）……」）。｜修复：｜复验：

- [重要·来源] §5.2 第一段（index.html 第 485 行）：`把同一个社区复现的 Sherry-1B checkpoint 统一转成 bf16 再分别量化为四种格式`｜引文依据：PR #22836 原文「we used the community-reproduced Sherry-1B reference checkpoint [MoraxGeo/Sherry-1B-1.25bit-per-channel], converted it to bf16 GGUF once, and then re-quantized the same weights into **all three ternary formats** (`STQ1_0` / `TQ1_0` / `TQ2_0`)」，以及「Compared to the 1-bit binary `Q1_0` **baseline**, STQ1_0 only trades a ~6% size increase」。同一 bf16 权重重量的受控对比只覆盖三种三值格式，Q1_0 是另列的参照基线，不是这条再量化链路的产物。页面写成「四种格式」，会让读者以为 Q1_0 行与该 bf16 基线同源、四行完全可比。｜修复要求：把该句改为「统一转成 bf16 后重新量化为三种三值格式（STQ1_0 / TQ1_0 / TQ2_0），并另列 llama.cpp 的 1-bit 基线 Q1_0 作参照」，并保留「PR 未给出 Q1_0 的 bpw」说明。｜修复：｜复验：

- [轻微·表述] §4 开篇 callout（index.html 第 302 行）：`三套做法对应不同场景，不要混用结论`——以祈使句直接称呼读者，且「场景」被当作术语使用（第 584 行「PTQ 场景的精度表现」同）｜引文依据：不适用（`guides/concept/style-guide.md` §12：不直接使用第二人称称呼读者）｜修复要求：改为不指向读者的陈述，如「三种做法的适用条件不同，各自的结论不能互相套用」，并把「场景」换成「适用条件/场合」。｜修复：｜复验：

- [轻微·表述] §4.3（index.html 第 345、349 行）：`一个容易看漏的事实：`、`结论先行：`、`下面的代码对一个构造的 256 权重块完整实现三种做法`、`（下面的代码实测验证了这两点）`——元话语与临场评价｜引文依据：不适用（`guides/concept/check.md` §2.2 第 12 项禁止元话语与临场评价）｜修复要求：改为直接陈述，如删去「一个容易看漏的事实：」「结论先行：」，代码引导改为「代码对一个构造的 256 权重块实现三种做法……」等不含预告语气的写法。｜修复：｜复验：

- [轻微·可读性] §1 对照表 2:4 行（index.html 第 136 行）：`2:4 稀疏三值 | 约 1.25（模式用不满）`——按正文给出的「12 种合法模式」直接换算应为 4 bit/组 = 1.0 bit/权重，表格里的 1.25 只有「沿用与 3:4 相同的 5 bit 容器」才成立，而这一前提全文未说明，与 §1 正文「填不满 16 项查找表，仍有 bit 浪费」的论述也没有衔接｜引文依据：不适用｜修复要求：注明该 1.25 的折算前提（沿用 5 bit 容器、16 个索引槽只用 12 个），或把该单元格改为「4 bit 索引 12/16 未用满」这类不产生歧义的表述。｜修复：｜复验：

- [轻微·表述] §6 标题与正文（index.html 第 539、541 行）：h2 `6. 训练侧的 weight trapping 与 Arenas——本页边界` 与 `本页只给出最小说明并划定边界`——副标题用「本页边界」描述写作安排，属页面自我描述而非本章结论｜引文依据：不适用（`guides/concept/style-guide.md` §1：h2 副标题说明本章的问题或结论）｜修复要求：副标题改为本章的问题或结论，如「——为什么 Hy4 轻量版用不到」；正文首句改为「Sherry 论文还有一半内容在训练侧，本章给出最小说明」一类不作页面自我描述的写法。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复
- 说明：本轮除上述条目外，全部事实性论断、公式与数字均可回源定位并已被原文片段或可复算的代码输出核实；页面结构与功能（公式渲染、折叠交互、目录锚点、本地资源、validate.py）正常。第 4 轮需先关闭阻断与重要条目后再进入复验。

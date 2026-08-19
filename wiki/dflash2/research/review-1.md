# DFlash 2 审查记录（第 1 轮）

- 页面版本：index.html `c05201311e2c4942321b1a0d1bd7430272d7db20`；overview.html `469121a5e9468e8f0e2589535896e4716ae1eb8d`
- 审查时间：2026-08-19 18:26
- 审查者：独立审查者（未参与写作与前序修复）
- 已完整阅读章节：核心问题（含全部解答折叠块）、1. 两个剩余问题、2. 路径选择器（2.1–2.4）、3. 两抽头卷积（3.1–3.3）、4. 组合效果（4.1–4.3）、5. 端到端与边界（5.1–5.3）、来源与范围说明（六小节）；overview.html 全文
- 核对来源：`/tmp/dflash-research/dflash2-blog.txt`（博客全文快照）、`/tmp/dflash-research/qwen38-modelcard.md`（HF 模型卡完整快照）、在线 https://inco.ai/blog/dflash2/ （仅用于核对 Muse 草稿器仓库名与 batch size 表述）
- 机械验证：`.dojo/scripts/validate.py wiki/dflash2/index.html` 返回 `validation ok`；`../dflash/index.html`、`../block-diffusion/index.html`、`../speculative-decoding/index.html` 均存在；index ↔ overview 互链有效

## 核对确认无误的关键数字（抽样记录）

- Table 1：Recall@1 七位 85.4/80.3/79.4/78.3/77.5/75.9/72.9%、Recall@16 七位 99.5/97.3/94.8/92.6/90.8/89.4/87.8%、oracle 4.27→6.79——页面与博客一致
- Table 2：+77.8M/+9.6%/4.49/4.08（DSpark）与 +2.0M/+0.6%/4.61/4.25（选择器）、40×/16×——一致
- Figure 2 数据表：3L/5L/15L/5L+conv 全部抽查位置（首/3/末位）一致；+16.5M(3%)、+0.7%、15.2%、9.4%→0.5% 一致
- Table 3：Qwen3.5-4B 全 24 格 + 均值 5.97/4.92/5.49/4.54、+1.05(21%)/+0.48——一致
- Table 4：Qwen3.8-27B 全格 + 均值 4.80/4.28/3.62——与博客及 HF 模型卡均一致
- Table 5：Muse Glimmer 全格 + 均值 5.70/4.44/4.48——一致
- HF 模型卡吞吐：并发 1/8/32 × 5 任务 × 3 方法全部 45 格逐格核对一致；MTP 并发 32 的 0.77×/0.94× 一致
- 构造示例的分数分解（0.40−0.30=0.10 等）手算复算通过
- 发布形态（博客+HF 模型、无独立论文）、C14 分析性推断标注、生态数字厂商宣称标注、C15/C14 存在——已确认

## 问题

- [重要·技术] index.html 2.4 节构造示例（含第 1 章末网格）：构造数字自相矛盾——U 表中位置 2 的 $U_2(\text{is})=0.55$ 高于 $U_2(\text{decoding})=0.40$，则独立 top-1（按自身 logit 取 argmax）应选 `is`、不产生 stutter；但同节结论与第 1 章网格均称「独立 top-1 输出 decoding decoding parallel」｜引文依据：不适用（页面内部构造数据自相矛盾；来源章节「构造示例」小节声明这些数字为人为设定）｜修复要求：重设构造数字使 $U_2(\text{decoding})>U_2(\text{is})$（独立 top-1 选 `decoding` 产生 stutter），选择器靠双线性项使 $S(\text{decoding},\text{is})>S(\text{decoding},\text{decoding})$；同步更新第 1 章网格、2.4 两张表、走路径推导与来源章节「构造示例」描述，保持每格可手算复算｜修复：｜复验：
- [重要·来源] index.html 2.3 节对比表第三行标签「DFlash + 路径选择器（DFlash 2）」：该行数据是"仅选择器、不含卷积"，不是 DFlash 2 完整体（DFlash 2 = 选择器 + 卷积，另加 16.5M/0.7%），标签会让读者把 4.61 当作 DFlash 2 的完整开销与收益｜引文依据：博客 Table 2 标题 "Acceptance length with path selection alone (no convolution)"｜修复要求：标签改为「DFlash + 路径选择器（不含卷积）」，并在表下注明完整 DFlash 2 另含卷积（+16.5M、+0.7%）｜修复：｜复验：
- [重要·来源] index.html 4.2 节正文「两个目标模型上 DFlash 2 都拿到完整 token 以上的领先」与第 4 章本章问题第 2 题解答「都拿到相对各基线的完整 token 以上领先」：来源只支持对 DSpark 的该表述；Qwen3.8-27B 上 DFlash 2 对 MTP 仅 +0.52（4.80 vs 4.28），不足一个完整 token｜引文依据：博客 "on both models, DFlash 2 averages more than a full token ahead of DSpark"（仅限 DSpark）｜修复要求：两处改为「领先 DSpark 一个完整 token 以上；对 Qwen3.8-27B 的 MTP 领先 0.52、对 Muse Glimmer 官方 DFlash 领先 1.26」｜修复：｜复验：
- [重要·来源] index.html 5.1 节末「Muse Glimmer 在批大小 1 的吞吐区间为 3.1–4.6×[N9]」：来源未给出批大小条件——batch size 1 仅出现在 Qwen3.8-27B 的表述中，Muse Glimmer 区间未注明并发，且博客明言「The model cards break the speedups down by task and concurrency」，3.1–4.6× 可能是跨条件区间｜引文依据：博客 "SGLang serves at 2.7–3.4× ... at batch size 1"（仅 Qwen3.8）；"and 3.1–4.6× on Muse Glimmer"（无 batch size 限定）｜修复要求：删除「批大小 1」限定，改为「博客给出 Muse Glimmer 吞吐区间 3.1–4.6×（未注明并发条件，模型卡按任务与并发细分）」，N9 条目同步修改｜修复：｜复验：
- [重要·技术] index.html 2.4 节走路径下标与 2.1 节定义不一致：2.1 定义 $b$ 位于位置 $t$、$a$ 位于 $t-1$（即 $S_t$ 的 $t$ 为当前位），示例位置按 1/2/3 编号，则位置 2 的相邻对应为 $S_2$、位置 3 为 $S_3$；页面写「位置 2：$S_1(\text{decoding},\ldots)$」「位置 3：$S_2(\text{is},\ldots)$」，且同行分解使用 $U_2$，同一行内 $S$ 与 $U$ 下标错位｜引文依据：不适用（F1 符号一致性；博客公式为 $S_t(a,b)$）｜修复要求：将走路径两行的 $S_1$ 改为 $S_2$、$S_2$ 改为 $S_3$，或在 2.4 开头声明位置从 0 编号并全节统一｜修复：｜复验：
- [重要·图] index.html 3.2 节 SVG 图：① `<text>` 内使用 ASCII 公式写法 `Conv(x)<sub>1</sub>`（SVG 命名空间不渲染 HTML `<sub>`，且规范禁止 `<text>` 内 ASCII 近似公式，要求公式置于 `<foreignObject>` 由 KaTeX 渲染）；② 已验证 token 到位置 1 的线段重复绘制两次（同坐标 130,180→170,80 的实线与虚线叠加）；③ 图注称「前一位=虚线、自身=实线」，但 位置1→Conv2、位置2→Conv3 两条「前一位」线为实线（dg-line 无 dasharray），线型语义与图注不符｜引文依据：不适用（style-guide 第 11 节「`<text>` 内无 ASCII 近似写法」、check.md 2.2 第 10 条）｜修复要求：三个 `Conv(x)` 标签改为 `<foreignObject>` 内 `<div class="dg-label">$\mathrm{Conv}(x)_t$</div>` 形式；删除重复线段；统一虚线=前位抽头、实线（accent）=自身抽头并与图注一致｜修复：｜复验：
- [重要·来源] index.html 3.3 节对照表「参数」列：3L 标「1×」、5L 标「2×」为页面发明数字——来源只支持 15L「3× more params」（相对 5L）与 conv「+3%」；且 1×/2×/3× 的线性标度与 15L=3×（相对 5L）参照系自相矛盾（按层数线性应为 0.6×/1×/3×）｜引文依据：博客 Figure 2 数据表仅含 "DFlash 15L (3× more params)"、"DFlash 5L + conv (+3% params)"，无 3L/5L 参数倍数｜修复要求：删除 3L/5L 的参数倍数单元格（改为「—」或文字说明「15L 参数为 5L 的 3 倍」），全表参照系统一为相对 5L｜修复：｜复验：
- [轻微·来源] index.html 5.2 节与 C14 条目两处「Anco AI」：公司名错误，应为 Inco AI（来源作者为 Inco AI）｜引文依据：博客标题行 "Inco AI"；模型卡 "incoai/Qwen3.8-27B-DFlash2"｜修复要求：两处改为 Inco AI｜修复：｜复验：
- [轻微·来源] index.html 4.1 节「沿块稳在 84.85–86.48%」：区间下界数字抄写错误，Figure 5 数据中 DFlash 2 除首位外的最小值为 84.88（位置 3），非 84.85｜引文依据：博客 Figure 5 数据行 "DFlash 2 88.3 % 85.3 % 84.98 % 84.88 % …"｜修复要求：改为「84.88–86.48%」｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章「「首位命中率」Recall@1 与「首位 top-16 命中率」Recall@16」：「首位」会被读成"块内第一位置"，而指标含义是候选列表第一位（top-1/top-16）逐位置统计｜引文依据：博客 Table 1 标题 "Recall@1 (how often the top pick is right) and Recall@16 (how often the right token is in the top 16) at each draft position"｜修复要求：改为「top-1 正确率 Recall@1」与「top-16 命中率 Recall@16」（与同章表格行标签一致）｜修复：｜复验：
- [轻微·来源] index.html 引言/核心问题 1/overview「（Qwen3.8-27B-DFlash2、Muse-Glimmer-30B-DFlash2，Apache 2.0）」：Apache 2.0 许可只在 Qwen 草稿器模型卡中可核对（license: apache-2.0），Muse 草稿器许可未在允许来源中出现｜引文依据：模型卡 YAML "license: apache-2.0"（仅 Qwen3.8-27B-DFlash2）；博客/模型卡无 Muse 草稿器许可信息｜修复要求：许可声明限定为 Qwen3.8-27B-DFlash2（如「Qwen 草稿器 Apache 2.0」），或补入 Muse 草稿器模型卡来源后再作双侧声明｜修复：｜复验：
- [轻微·来源] index.html blockquote「arXiv:2602.06036v2, ICML 2026」与 C1 条目「GitHub z-lab/dflash README 引用块」：arXiv 编号与 GitHub README 引用块均无法在博客或模型卡快照中定位（模型卡仅有 Chen/Liang/Liu 的 ICML 2026 bibtex 与 GitHub 链接本身）｜引文依据：模型卡 Citation 节 bibtex（无 arXiv 编号）；Header 行 "[GitHub](https://github.com/z-lab/dflash)"（无 README 内容）｜修复要求：删除 arXiv 编号，或注明编号出处；C1 引文改为模型卡可见内容（GitHub 链接 + bibtex），删除「README 引用块」表述｜修复：｜复验：
- [轻微·来源] index.html 5.3 节「所有数字来自厂商自测（单卡 H200/B200）」与简化条件⑤「单卡 H200/B200」：B200 不在来源中，模型卡实验环境仅 H200｜引文依据：模型卡 "Runtime: SGLang on one NVIDIA H200"｜修复要求：两处删除 B200，改为「单卡 H200」｜修复：｜复验：
- [轻微·来源] index.html 简化条件⑤「社区初步复测（llama.cpp PR 构建、RTX 5090 等）的数据点有限且与厂商环境不同」：社区复测数据点无任何可定位来源，属无来源论断｜引文依据：不适用（博客与模型卡均无 RTX 5090 / 社区复测内容）｜修复要求：删除该句，保留「N8 并发表为厂商自测、无第三方独立复现」即可｜修复：｜复验：
- [轻微·格式] index.html 来源章节前两个 h3 命名「核心论断与来源」「核心公式与来源」：不符合 style-guide 固定命名「论断与来源（C）」「公式与来源（F）」｜引文依据：不适用（style-guide 第 1 节 h3 固定命名清单）｜修复要求：两个 h3 改为固定命名｜修复：｜复验：
- [轻微·格式] index.html N 条目中的 N10：「本号原为 DFlash 论文 N10，与本页不同主题，不引用」是面向写作过程的遗留说明，读者无上下文，且造成 N 编号跳号（N9→N11）｜引文依据：不适用｜修复要求：删除 N10 条目，将 N11 重命名为 N10 并同步正文全部 `[N11]` 引用；或至少删除「本号原为…不引用」的规划性文字｜修复：｜复验：
- [轻微·格式] index.html 5.3 节生态句使用 `<sup>[厂商宣称]</sup>` 而非 `<sup>[C15]</sup>`：破坏 C/F/N 编号与来源章节的双向对应（C15 在来源节定义、正文却以非编号标记引用）｜引文依据：不适用（style-guide 第 6 节：正文使用 `<sup>[Cx]</sup>` 上标引用）｜修复要求：改为 `<sup>[C15]</sup>`，C15 条目内保留「标注为厂商宣称」说明｜修复：｜复验：
- [轻微·来源] index.html C13（已发布两款草稿器）在正文中无任何 `<sup>[C13]</sup>` 引用：来源章节条目与正文单向脱节；5.3 节「当前已发布 Qwen3.8-27B、Muse Glimmer 两款」处应引用｜引文依据：博客 "Two Drafters, Out Today"｜修复要求：在该句后加 `<sup>[C13]</sup>`（或核心问题 1 解答处）｜修复：｜复验：
- [轻微·可读性] index.html 2.4 节相邻对分数表引言「（构造数据，正文是辅助理解路径选择的工作方式，不代表实测）」：语句不通（"正文是辅助理解"主谓混乱）｜引文依据：不适用｜修复要求：改为「（构造数据，用于辅助理解路径选择的工作方式，不代表实测）」｜修复：｜复验：
- [轻微·可读性] index.html 2.4 节「从最后已验证 token（假设是「.」或上周期 bonus）出发」：「bonus」首次出现未解释（投机解码中指接受后目标模型顺带产出的下一 token）｜引文依据：不适用｜修复要求：加括号简注（如「bonus，即验证顺带产出的下一 token」）或删去该词｜修复：｜复验：
- [轻微·可读性] index.html 5.1 节「每格是相对自回归基线的吞吐倍数（绝对 tok/s 略；模型卡原始表含绝对数）」：「略」字歧义（省略？粗略？），且下表并发 1 块实际附有基线 tok/s 行，前后不一｜引文依据：不适用｜修复要求：改为「绝对 tok/s 见模型卡原始表；下表并发 1 块附自回归基线 tok/s 供换算」｜修复：｜复验：
- [轻微·格式] index.html 5.1 节表格加粗语义不一致：并发 1/8 块中加粗表示该列最优（DFlash 2 行），并发 32 的 MTP 行对 <1 的值（0.94/0.84/0.87/0.77）加粗表示警示，同一表内两种含义｜引文依据：不适用｜修复要求：取消 MTP 行的加粗，<1 的情况在表下要点列表中已有文字说明（「并发 32 时 MTP 与 DSpark 多数任务降到 1× 以下」），无需加粗标记｜修复：｜复验：
- [轻微·问题块] index.html 核心问题 1 的解答末尾未指明完整论证所在章节（其余四题均有「详见『…』」），不符合页面级答案须指明论证位置的要求｜引文依据：不适用（style-guide 第 9 节）｜修复要求：解答末尾补「详见『1. 两个剩余问题——候选池里有答案、块末端在漏气』」｜修复：｜复验：
- [轻微·格式] index.html 正文多处以「第 2 章」「第 3 章」「（见第 3 章）」「需要第 3 章的卷积修骨干」等形式引用其他章节（第 1 章两份表格、2.3 末段、2.4、第 1/2/4 章本章问题解答）：style-guide 要求正文引用其他章节使用章节标题而非编号代号｜引文依据：不适用（style-guide 第 1 节「正文引用其他章节」）｜修复要求：全部改为章节标题引用（如「见『3. 两抽头卷积——给块末端补上「看见前一位」的通道』」，可截断至可辨识长度）｜修复：｜复验：
- [轻微·来源] index.html C5 条目引文未覆盖「选比预测便宜」这一表述本身：2 章开篇以 `<sup>[C5]</sup>` 支撑「选比预测便宜」设计原则，但 C5 引文只有局部性论证句｜引文依据：博客 "Choosing is cheaper than predicting."（Table 2 后正文）｜修复要求：C5 引文补充该原句，或在 2 章开篇同时引用 N1｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章末段「<code>parallel</code> 与 <code>now</code> 的 Recall@1 都已经从首位掉了几个百分点」：Recall@1 是数据集级逐位置指标，不作用于单个候选 token，构造示例 token 与真实指标混写会产生错误归因｜引文依据：不适用（Table 1 的 Recall@1 按位置统计，与具体 token 无关）｜修复要求：改写为「位置 2、3 的 Recall@1 已低于首位（真实数据 80.3%/79.4% vs 85.4%），后段候选池本身在变差」，把构造 token 与真实指标分开表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 7 / 轻微 19
- 处置：修复。无阻断问题；页面主线（两个组件的机制、来源编号体系、核心数字表）与博客/模型卡核对基本一致，self-link 与 validate.py 均通过。7 条重要问题集中在：构造示例自相矛盾（2.4/第 1 章）、Table 2 行标签误作 DFlash 2、「完整 token 领先」超范围泛化、Muse 吞吐批大小条件无据、$S_t$ 下标错位、SVG 图公式/线型三处不符、参数倍数列无来源。逐条修复并复验后进入第二轮全量审查；重要问题全部关闭前不得发布。
